# src/retrieval/dense.py

"""
DENSE RETRIEVAL
---------------
Embed the user query → find chunks with closest vectors in ChromaDB.

"Dense" = everything is represented as a dense vector (all 384 numbers
have values). Contrast with sparse where most values are 0.

Strength : catches semantic similarity — "refund" matches "money back"
Weakness : can miss exact keyword matches for rare/specific terms
"""

import chromadb
from sentence_transformers import SentenceTransformer
from src.ingestion.embedder import (
    get_chroma_client,
    get_or_create_collection,
    EMBEDDING_MODEL
)

# Module-level model cache — load once, reuse across calls
_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# src/retrieval/dense.py
# Replace the dense_retrieve function with this version

def dense_retrieve(query: str, top_k: int = 20) -> list[dict]:
    model = _get_model()

    # Create a fresh client each call to avoid stale connections
    import chromadb
    from src.ingestion.embedder import CHROMA_DIR, COLLECTION_NAME
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        chunks.append({
            "id":     doc_id,
            "text":   doc,
            "topic":  meta.get("topic", ""),
            "source": meta.get("source", ""),
            "score":  round(1 - dist, 4),
            "rank":   i + 1
        })

    return chunks