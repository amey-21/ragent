# src/retrieval/reranker.py

"""
RERANKING WITH CROSSENCODER
----------------------------
Two types of models in retrieval:

BI-ENCODER (what we used for embeddings):
  - Encodes query and chunk SEPARATELY into vectors
  - Fast: embed once, compare with dot product
  - Used for: initial retrieval over large corpus

CROSSENCODER (reranker):
  - Encodes query and chunk TOGETHER in one forward pass
  - Sees full interaction between query tokens and chunk tokens
  - Slow: must run once per (query, chunk) pair
  - Used for: reranking a small set of candidates (top 20)

This two-stage design is standard in production RAG:
  Stage 1: Bi-encoder retrieves top 20 (fast, approximate)
  Stage 2: CrossEncoder reranks top 20 → returns top 5 (slow, precise)

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking dataset
  - 22MB, runs on CPU in ~100ms for 20 passages
  - Outputs a raw logit score (higher = more relevant, no fixed range)
"""

from sentence_transformers import CrossEncoder
from src.retrieval.hybrid import hybrid_retrieve

# Module-level cache
_reranker: CrossEncoder | None = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        print("  ✓ Reranker loaded")
    return _reranker


def rerank(query: str, top_k: int = 5) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Hybrid retrieve top 20 candidates
      2. CrossEncoder reranks them
      3. Return top_k

    Args:
        query: user's question
        top_k: final number of chunks to return

    Returns:
        list of chunks sorted by reranker score, with rerank_score field added
    """
    reranker = _get_reranker()

    # Stage 1: get top 20 candidates from hybrid
    candidates = hybrid_retrieve(query, top_k=20)

    if not candidates:
        return []

    # Stage 2: score each (query, chunk) pair with CrossEncoder
    # Input format: list of [query, chunk_text] pairs
    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = reranker.predict(pairs)     # returns numpy array of floats

    # Attach reranker scores to candidates
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 4)

    # Sort by reranker score (descending) and return top_k
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    top = reranked[:top_k]
    for i, chunk in enumerate(top):
        chunk["final_rank"] = i + 1

    return top