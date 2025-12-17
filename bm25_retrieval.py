import bm25s  # On windows expect "resource module not available on Windows"
from typing import List
from corpus_data import corpus

from app.models.results_model import SearchResult


def ingest_corpus() -> bm25s.BM25:
    # 1) Tokenize
    corpus_tokens = bm25s.tokenize(corpus)

    # 2) Build retriever + index
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)

    return retriever


def search_corpus(
    retriever: bm25s.BM25,
    query: str,
    top_k: int = 5,
) -> List[SearchResult]:
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, k=top_k)

    docs = results[0]
    scs = scores[0]

    hits: List[SearchResult] = []
    for rank, (doc, score) in enumerate(zip(docs, scs), start=1):
        hits.append(
            SearchResult(
                rank=rank,
                doc=doc,
                score=float(score),
            )
        )

    return hits
