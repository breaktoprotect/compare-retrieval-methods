from typing import List

from bm25_retrieval import ingest_corpus, search_corpus
from cross_encoder_reranker import rerank
from output_formatter import print_search_results
from app.models.results_model import SearchResult


def main_bm25_only():
    retriever = ingest_corpus()
    results: List[SearchResult] = search_corpus(retriever, "windows audit policy")
    print_search_results("windows audit policy", results, title="BM25 Only")

    results: List[SearchResult] = search_corpus(retriever, "SBERT semantic search")
    print_search_results("SBERT semantic search", results, title="BM25 Only")


def main_bm25_with_cross_encoder_rerank():
    retriever = ingest_corpus()
    results: List[SearchResult] = search_corpus(retriever, "windows audit policy")
    reranked_results: List[SearchResult] = rerank("windows audit policy", results)

    results: List[SearchResult] = search_corpus(retriever, "SBERT semantic search")
    reranked_results: List[SearchResult] = rerank(
        "SBERT semantic search",
        results,
    )

    print_search_results(
        "windows audit policy", reranked_results, title="BM25 + Cross-Encoder Rerank"
    )
    print_search_results(
        "SBERT semantic search", reranked_results, title="BM25 + Cross-Encoder Rerank"
    )


if __name__ == "__main__":
    import sys

    sys.exit()

    main_bm25_only()
    main_bm25_with_cross_encoder_rerank()
