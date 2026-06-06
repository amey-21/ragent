# src/ingestion/embedder.py

"""
CONCEPT: Embedding pipeline
----------------------------
1. Load a local sentence-transformer model (no API calls, no cost)
2. Take our 810 Chunk objects
3. Embed each chunk's text → 384-dimensional vector
4. Store in ChromaDB with metadata for filtering

WHY CHROMADB?
ChromaDB is a local vector database — it stores vectors on disk and
lets you query by similarity. Perfect for development. In production
you'd swap to Pinecone or Weaviate for scale, but the code change is
minimal — just the client initialization.

WHAT GETS STORED PER CHUNK:
- id         : "get_refund_qa_0"  (must be unique)
- embedding  : [0.21, -0.54, ...] (384 floats)
- document   : the raw text       (returned at query time)
- metadata   : {source, topic, strategy, chunk_index}
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.ingestion.chunker import Chunk

# ── Config ─────────────────────────────────────────────────────────────────

CHROMA_DIR = "data/chroma_db"       # where ChromaDB persists to disk
COLLECTION_NAME = "support_agent"   # name of our vector collection
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 22MB, 384 dims, fast on CPU

# ── Clients (initialized once, reused) ────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """
    PersistentClient saves to disk — data survives restarts.
    (vs EphemeralClient which lives only in memory)
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_or_create_collection(client: chromadb.PersistentClient):
    """
    get_or_create_collection: if the collection exists, return it.
    If not, create it. Safe to call multiple times.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity (not euclidean)
    )


def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model. First run downloads ~22MB.
    Subsequent runs load from cache instantly.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Model loaded — embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ── Core functions ─────────────────────────────────────────────────────────

def embed_chunks(chunks: list[Chunk], batch_size: int = 64) -> None:
    """
    Embed all chunks and upsert into ChromaDB.

    UPSERT = insert if new, update if already exists.
    Safe to run multiple times — won't create duplicates.

    We batch the embeddings (64 at a time) because:
    - Sentence transformers are faster in batches than one-by-one
    - Avoids memory spikes on large corpora

    Args:
        chunks: list of Chunk objects from our chunker
        batch_size: how many chunks to embed at once
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    model = get_embedding_model()

    print(f"\nEmbedding {len(chunks)} chunks into ChromaDB...")
    print(f"Collection: '{COLLECTION_NAME}' at {CHROMA_DIR}")

    total_embedded = 0

    # Process in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        # Extract text for embedding
        texts = [chunk.text for chunk in batch]

        # Embed the batch — this is the heavy computation
        # show_progress_bar gives a tqdm progress bar per batch
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()  # ChromaDB needs plain Python lists, not numpy arrays

        # Prepare ChromaDB inputs
        ids        = [chunk.chunk_id for chunk in batch]
        documents  = [chunk.text for chunk in batch]
        metadatas  = [
            {
                "source":      chunk.source,
                "topic":       chunk.topic,
                "strategy":    chunk.strategy,
                "chunk_index": chunk.chunk_index,
                "intent":      chunk.metadata.get("intent", ""),
            }
            for chunk in batch
        ]

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        total_embedded += len(batch)
        print(f"  Embedded {total_embedded}/{len(chunks)} chunks...")

    print(f"\n✓ Done. {total_embedded} chunks stored in ChromaDB.")
    print(f"  Collection size: {collection.count()} documents")


def get_collection_stats() -> dict:
    """Print stats about what's currently in the collection."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    count = collection.count()

    # Sample a few items to show what's stored
    if count > 0:
        sample = collection.peek(limit=2)
        return {
            "total_chunks": count,
            "sample_ids": sample["ids"],
            "sample_topics": [m.get("topic") for m in sample["metadatas"]]
        }
    return {"total_chunks": 0}