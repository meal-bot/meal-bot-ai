"""
Hybrid Search 모듈.
Semantic Search (Chroma) + BM25 Keyword Search를 RRF로 결합한다.
"""

from __future__ import annotations

from core.bm25_index import bm25_search
from core.retrieval import Hit, search

_RRF_K = 60
_CANDIDATE_K = 50


def hybrid_search(query: str, top_k: int = 10) -> list[Hit]:
    """
    Semantic + BM25 결과를 RRF로 결합해 상위 top_k개 반환.
    score 필드에 RRF 점수 저장.
    """
    semantic_hits = search(query, top_k=_CANDIDATE_K)
    bm25_hits = bm25_search(query, top_k=_CANDIDATE_K)

    # recipe_id → rank (1-indexed)
    semantic_ranks: dict[int, int] = {
        h.recipe_id: rank for rank, h in enumerate(semantic_hits, start=1)
    }
    bm25_ranks: dict[int, int] = {
        h.recipe_id: rank for rank, h in enumerate(bm25_hits, start=1)
    }

    # 전체 후보 doc_id 합집합
    all_ids = set(semantic_ranks) | set(bm25_ranks)

    rrf_scores: dict[int, float] = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in semantic_ranks:
            score += 1 / (_RRF_K + semantic_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += 1 / (_RRF_K + bm25_ranks[doc_id])
        rrf_scores[doc_id] = score

    top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    # Hit 객체 구성: semantic_hits 우선, 없으면 bm25_hits에서 가져옴
    hit_map: dict[int, Hit] = {h.recipe_id: h for h in bm25_hits}
    hit_map.update({h.recipe_id: h for h in semantic_hits})

    return [
        Hit(
            recipe_id=doc_id,
            name=hit_map[doc_id].name,
            score=round(rrf_scores[doc_id], 6),
            distance=hit_map[doc_id].distance,
            metadata=hit_map[doc_id].metadata,
            document=hit_map[doc_id].document,
        )
        for doc_id in top_ids
        if doc_id in hit_map
    ]
