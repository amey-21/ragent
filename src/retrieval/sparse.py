# src/retrieval/sparse.py

"""
SPARSE RETRIEVAL — BM25
------------------------
BM25 (Best Match 25) is the algorithm behind classic search engines
like Elasticsearch. It scores chunks by keyword overlap with the query.

"Sparse" = the representation is a vector where MOST values are 0.
Only dimensions corresponding to words that appear in the text are non-zero.

Example:
  query: "cancel order"
  chunk A: "Q: how do I cancel my order? A: To cancel..."  ← high score
  chunk B: "Q: how do I get a refund? A: We process..."    ← low score (no "cancel")

Strength : exact keyword matches, rare terms, product names, order IDs
Weakness : "cancel" won't match "cancellation" unless you add stemming
           misses semantic similarity entirely

BM25 formula key idea:
  score = Σ IDF(word) × TF(word in chunk) / (TF + k1*(1 - b + b*len/avglen))
  - IDF: rare words score higher than common words
  - TF: more occurrences = higher score, but with diminishing returns
  - k1, b: tuning parameters (we use defaults: k1=1.5, b=0.75)
"""

import os
import pickle
import re
from rank_bm25 import BM25Okapi
from src.ingestion.chunker import chunk_all_files, Chunk

# Where we cache the BM25 index so we don't rebuild every time
BM25_CACHE_PATH = "data/bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase + split on non-alphanumeric characters.
    "How do I cancel my ORDER?" → ["how", "do", "i", "cancel", "my", "order"]

    In production you'd add stemming (cancel → cancel, cancellation → cancel)
    but this is sufficient for our demo.
    """
    return re.findall(r'\b\w+\b', text.lower())


def build_bm25_index(chunks: list[Chunk] | None = None) -> tuple[BM25Okapi, list[Chunk]]:
    """
    Build a BM25 index from all chunks and cache it to disk.

    We cache because rebuilding from 810 chunks takes ~1 second —
    fine for a one-time build, but annoying if done on every query.
    """
    if chunks is None:
        print("Loading chunks for BM25 index...")
        chunks = chunk_all_files("data/raw", strategy="qa_aware")

    print(f"Building BM25 index over {len(chunks)} chunks...")

    # Tokenize every chunk
    tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]

    # Build the index
    bm25 = BM25Okapi(tokenized_corpus)

    # Cache to disk
    os.makedirs("data", exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump((bm25, chunks), f)

    print(f"  ✓ BM25 index built and cached to {BM25_CACHE_PATH}")
    return bm25, chunks


def load_bm25_index() -> tuple[BM25Okapi, list[Chunk]]:
    """Load BM25 index from cache, rebuilding if cache doesn't exist."""
    if os.path.exists(BM25_CACHE_PATH):
        with open(BM25_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return build_bm25_index()


def sparse_retrieve(query: str, top_k: int = 20) -> list[dict]:
    """
    Score all chunks with BM25 and return top_k.
    Returns same dict format as dense_retrieve for easy combination.
    """
    bm25, chunks = load_bm25_index()

    # Tokenize the query the same way we tokenized chunks
    query_tokens = _tokenize(query)

    # BM25 scores all 810 chunks in one call (very fast — pure numpy)
    scores = bm25.get_scores(query_tokens)

    # Get top_k indices sorted by score descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        chunk = chunks[idx]
        results.append({
            "id":     chunk.chunk_id,
            "text":   chunk.text,
            "topic":  chunk.topic,
            "source": chunk.source,
            "score":  round(float(scores[idx]), 4),
            "rank":   rank + 1
        })

    return results