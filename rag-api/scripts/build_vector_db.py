"""
MySQL recipe → Chroma 벡터 DB 적재.
core/ 모듈을 재사용하며, 검증은 core.retrieval.search() 사용.

실행:
    cd rag-api
    source venv/bin/activate
    python scripts/build_vector_db.py

멱등성: 기존 collection 삭제 후 재생성.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# core/ import를 위해 rag-api 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from _embedding_text import build_embedding_text
from core.config import COLLECTION_NAME, EMBEDDING_MODEL_NAME
from core.db import get_chroma_client, mysql_connection
from core.embedding import get_model
from core.retrieval import search

BATCH_SIZE = 32
ADD_BATCH_SIZE = 500

TEST_QUERIES = [
    "얼큰한 국물 요리",
    "다이어트 샐러드",
    "아이 간식",
    "매운 반찬",
    "고단백 요리",
]


def fetch_recipes() -> list[dict]:
    import pymysql
    with mysql_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT rcp_seq, name, category, cooking_way,
                       ingredients, hash_tag, calories, sodium
                FROM recipe
            """)
            return cur.fetchall()


def build_metadata(recipe: dict) -> dict:
    """Chroma는 None 허용 안 함 → 빈 값 정규화."""
    return {
        "rcp_seq":     int(recipe["rcp_seq"]),
        "name":        recipe["name"],
        "category":    recipe.get("category") or "",
        "cooking_way": recipe.get("cooking_way") or "",
        "calories":    int(recipe["calories"]) if recipe.get("calories") is not None else 0,
        "sodium":      int(recipe["sodium"]) if recipe.get("sodium") is not None else 0,
    }


def main():
    print(f"[1/5] MySQL 레시피 로드")
    recipes = fetch_recipes()
    print(f"  {len(recipes)}건 로드 완료")

    print(f"\n[2/5] 임베딩 텍스트 생성")
    texts = [build_embedding_text(r) for r in recipes]
    ids = [str(r["rcp_seq"]) for r in recipes]
    metadatas = [build_metadata(r) for r in recipes]
    avg_len = sum(len(t) for t in texts) / len(texts)
    max_len = max(len(t) for t in texts)
    print(f"  평균 길이: {avg_len:.0f}자 / 최대: {max_len}자")

    print(f"\n[3/5] 임베딩 모델 로드: {EMBEDDING_MODEL_NAME}")
    t0 = time.time()
    model = get_model()
    print(f"  완료 ({time.time() - t0:.1f}초)")

    print(f"\n[4/5] 벡터 생성 (batch={BATCH_SIZE})")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  완료 ({time.time() - t0:.1f}초) / shape={embeddings.shape}")

    print(f"\n[5/5] Chroma 적재: collection={COLLECTION_NAME}")
    client = get_chroma_client()

    # 멱등성: 기존 삭제 (없으면 skip)
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  기존 collection 삭제: {COLLECTION_NAME}")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 배치 add
    for i in tqdm(range(0, len(ids), ADD_BATCH_SIZE), desc="  Chroma add", unit="batch"):
        j = min(i + ADD_BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:j],
            embeddings=embeddings[i:j].tolist(),
            documents=texts[i:j],
            metadatas=metadatas[i:j],
        )

    print(f"  적재 완료: {collection.count()}건")

    # 검증 (core.retrieval.search 재사용)
    print(f"\n=== 검색 검증 (core.retrieval.search) ===")
    for q in TEST_QUERIES:
        print(f"\n  Q: {q}")
        hits = search(q, top_k=5)
        for h in hits:
            print(f"    {h.score:.3f}  {h.name}")


if __name__ == "__main__":
    main()
