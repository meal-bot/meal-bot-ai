"""
외부 저장소 연결.
MySQL(레시피 원본) + Chroma(벡터 DB) 연결 관리.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

import chromadb
import pymysql
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from core.config import CHROMA_PATH, COLLECTION_NAME, MYSQL_CONFIG


# ─── MySQL ───────────────────────────────────────
@contextmanager
def mysql_connection() -> Iterator[pymysql.connections.Connection]:
    """
    MySQL 연결 context manager.

    사용:
        with mysql_connection() as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute("SELECT ...")
    """
    conn = pymysql.connect(**MYSQL_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


# ─── Chroma ──────────────────────────────────────
@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Chroma PersistentClient 싱글톤."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=1)
def get_chroma_collection() -> Collection:
    """
    recipes_v1 collection 싱글톤.
    Collection이 없으면 에러 발생 (build_vector_db.py 먼저 실행 필요).
    """
    client = get_chroma_client()
    return client.get_collection(name=COLLECTION_NAME)
