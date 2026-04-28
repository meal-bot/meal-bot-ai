"""
BM25 키워드 검색 모듈.
MySQL 전체 레시피를 로드해 BM25Okapi 인덱스를 빌드하고 검색한다.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pymysql
from rank_bm25 import BM25Okapi

from core.db import mysql_connection
from core.retrieval import Hit


def _tokenize(text: str) -> list[str]:
    text = text.replace("&", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return [t.lower() for t in text.split() if t]


@lru_cache(maxsize=1)
def get_bm25_index() -> tuple[BM25Okapi, list[int], list[dict]]:
    """
    MySQL 전체 레시피 로드 → BM25 인덱스 빌드.
    반환: (bm25, recipe_ids, recipe_metas)
    """
    with mysql_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT rcp_seq, name, category, cooking_way, ingredients, hash_tag "
                "FROM recipe ORDER BY rcp_seq"
            )
            rows = cur.fetchall()

    corpus: list[list[str]] = []
    recipe_ids: list[int] = []
    recipe_metas: list[dict] = []

    for row in rows:
        name = row["name"] or ""
        category = row["category"] or ""
        cooking_way = row["cooking_way"] or ""
        ingredients = row["ingredients"] or ""
        hash_tag = row["hash_tag"] or ""

        text = f"{name} {category} {cooking_way} {ingredients} {hash_tag}"
        corpus.append(_tokenize(text))

        recipe_ids.append(row["rcp_seq"])
        recipe_metas.append({
            "name":        name,
            "category":    category,
            "cooking_way": cooking_way,
            "document":    text,
        })

    bm25 = BM25Okapi(corpus)
    return bm25, recipe_ids, recipe_metas


def bm25_search(query: str, top_k: int = 10) -> list[Hit]:
    """BM25 키워드 검색. 상위 top_k개 Hit 반환."""
    bm25, recipe_ids, recipe_metas = get_bm25_index()

    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    hits = []
    for idx in top_indices:
        meta = recipe_metas[idx]
        hits.append(Hit(
            recipe_id=recipe_ids[idx],
            name=meta["name"],
            score=float(scores[idx]),
            distance=0.0,
            metadata={
                "name":        meta["name"],
                "category":    meta["category"],
                "cooking_way": meta["cooking_way"],
            },
            document=meta["document"],
        ))
    return hits
