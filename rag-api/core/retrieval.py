"""
벡터 검색 모듈.
사용자 쿼리 → 임베딩 → Chroma top-k 검색 → Hit 리스트 반환.

v1~v4 전 단계에서 공통으로 사용되는 핵심 부품.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import DEFAULT_TOP_K
from core.db import get_chroma_collection
from core.embedding import encode


@dataclass
class Hit:
    """검색 결과 한 건."""
    recipe_id: int       # rcp_seq
    name: str            # 레시피명
    score: float         # 유사도 (1 - distance), 클수록 유사
    distance: float      # cosine distance (raw), 작을수록 유사
    metadata: dict       # category, cooking_way, calories, sodium, name 등
    document: str        # 임베딩 텍스트 원본


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[Hit]:
    """
    쿼리와 유사한 레시피 top_k건 반환.

    Args:
        query: 사용자 쿼리 문자열
        top_k: 반환할 결과 개수 (기본 5)

    Returns:
        score 내림차순으로 정렬된 Hit 리스트
    """
    # 1. 쿼리 임베딩
    query_embedding = encode([query])[0].tolist()

    # 2. Chroma top-k 검색
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # 3. Hit 리스트로 변환
    # Chroma는 배치 쿼리 지원 탓에 이중 리스트 반환 → [0]으로 평탄화
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    hits = []
    for rid, dist, meta, doc in zip(ids, distances, metadatas, documents):
        hits.append(
            Hit(
                recipe_id=int(rid),
                name=meta["name"],
                score=1.0 - dist,
                distance=dist,
                metadata=meta,
                document=doc,
            )
        )

    return hits
