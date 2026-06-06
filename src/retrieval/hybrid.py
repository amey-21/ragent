# src/retrieval/hybrid.py

"""
HYBRID RETRIEVAL — Reciprocal Rank Fusion (RRF)
-------------------------------------------------
We have two ranked lists (dense + sparse). We need to merge them
into one final ranking.

NAIVE APPROACH (wrong): average the scores.
  Problem: dense scores are cosine similarities (0.0 to 1.0)
           BM25 scores are term frequencies (0 to ~15)
  They're on completely different scales — can't average them directly.

RRF APPROACH (correct):
  For each chunk, its RRF score = Σ 1 / (k + rank_in_list)
  where k=60 is a smoothing constant.

  Example:
    chunk "get_refund_qa_3":
      rank 2 in dense  → 1/(60+2)  = 0.0161
      rank 5 in sparse → 1/(60+5)  = 0.0154
      RRF score        = 0.0315

    chunk "get_refund_qa_7":
      rank 1 in dense  → 1/(60+1)  = 0.0164
      not in sparse    → 0
      RRF score        = 0.0164

  chunk "get_refund_qa_3" wins because it appeared in BOTH lists.
  A chunk that ranks well in both signals beats one that's #1 in only one.

WHY k=60?
  Lower k = recent ranks matter more (winner-takes-all)
  Higher k = more uniform, older ranks still count
  k=60 is the empirically validated default from the original RRF paper.
"""

from src.retrieval.dense import dense_retrieve
from src.retrieval.sparse import sparse_retrieve


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def hybrid_retrieve(query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
    """
    Run dense + sparse retrieval, merge with RRF, return top_k.

    Args:
        query:  user's question
        top_k:  final number of chunks to return (default 5)
        rrf_k:  RRF smoothing constant (default 60)

    Returns:
        top_k chunks sorted by RRF score, each with rrf_score field added
    """
    # Get top 20 from each retriever
    dense_results  = dense_retrieve(query,  top_k=20)
    sparse_results = sparse_retrieve(query, top_k=20)

    # Build a map: chunk_id → accumulated RRF score + chunk data
    rrf_map: dict[str, dict] = {}

    for result in dense_results:
        cid = result["id"]
        if cid not in rrf_map:
            rrf_map[cid] = {**result, "rrf_score": 0.0, "in_dense": False, "in_sparse": False}
        rrf_map[cid]["rrf_score"] += rrf_score(result["rank"], rrf_k)
        rrf_map[cid]["in_dense"] = True

    for result in sparse_results:
        cid = result["id"]
        if cid not in rrf_map:
            rrf_map[cid] = {**result, "rrf_score": 0.0, "in_dense": False, "in_sparse": False}
        rrf_map[cid]["rrf_score"] += rrf_score(result["rank"], rrf_k)
        rrf_map[cid]["in_sparse"] = True

    # Sort by RRF score descending
    ranked = sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Return top_k with final rank assigned
    top = ranked[:top_k]
    for i, chunk in enumerate(top):
        chunk["final_rank"] = i + 1

    return top