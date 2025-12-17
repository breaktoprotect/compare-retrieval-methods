from typing import List, Optional
from app.models.results_model import SearchResult


def score_to_indicator(pct: float) -> str:
    if pct >= 80.0:
        return "🟩"
    if pct >= 60.0:
        return "🟧"
    if pct >= 40.0:
        return "🟥"
    return "⚫"


def print_search_results(
    query: str,
    hits: List[SearchResult],
    title: Optional[str] = None,
) -> None:
    print("\n" + "=" * 80)
    print(f"Query   : {query}")

    if title:
        print(f"Mode    : {title}")

    print(f"Results : Top {len(hits)}")
    print("-" * 80)

    max_score = max((h.score for h in hits), default=0.0)

    for hit in hits:
        pct = (hit.score / max_score * 100.0) if max_score > 0 else 0.0
        indicator = score_to_indicator(pct)

        print(f"[{hit.rank:02d}]  {indicator}  score={hit.score:.4f}  ({pct:6.2f}%)")
        print(f"      └─ {hit.doc}")

    print("=" * 80 + "\n")
