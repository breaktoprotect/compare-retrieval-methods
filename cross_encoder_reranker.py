from typing import List, Tuple

from app.models.results_model import SearchResult

from sentence_transformers import CrossEncoder


# Small, fast, well-known reranker
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None


def get_reranker() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        # CPU fallback is automatic if no CUDA
        _cross_encoder = CrossEncoder(MODEL_NAME, device="cpu")
    return _cross_encoder


def rerank(
    query: str,
    hits: List[SearchResult],
) -> List[SearchResult]:
    model = get_reranker()

    # Build (query, doc) pairs
    pairs = [(query, hit.doc) for hit in hits]

    ce_scores = model.predict(pairs)

    # Attach CE scores to hits (temporarily)
    enriched = []
    for hit, ce_score in zip(hits, ce_scores):
        enriched.append(
            (
                hit,
                float(ce_score),
            )
        )

    # Sort by cross-encoder score (desc)
    enriched.sort(key=lambda x: x[1], reverse=True)

    # Rebuild SearchResult list with updated ranks and CE score
    reranked_hits: List[SearchResult] = []
    for rank, (hit, ce_score) in enumerate(enriched, start=1):
        reranked_hits.append(
            SearchResult(
                rank=rank,
                doc=hit.doc,
                score=ce_score,  # <-- now printing CE score
            )
        )

    return reranked_hits
