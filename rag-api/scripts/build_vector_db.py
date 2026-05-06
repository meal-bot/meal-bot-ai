"""
build_vector_db.py

MySQL recipes 테이블 → ChromaDB 적재 스크립트 (v1 기준선)

- 임베딩 모델: jhgan/ko-sbert-nli
- 컬렉션명: meal_bot_recipes_v1
- 저장 경로: chroma/recipes_v1
- 청킹: 1 레시피 = 1 document
- document id: recipe_{rcp_seq}
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import chromadb
import pymysql
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"
COLLECTION_NAME = "meal_bot_recipes_v1"
PERSIST_DIR = "chroma/recipes_v1"
REPORT_PATH = "artifacts/build_vector_db_report.json"
ID_PREFIX = "recipe_"

ARRAY_FIELDS = [
    "recommended_situations",
    "meal_type_tags",
    "taste_tags",
    "texture_tags",
    "main_ingredients",
]

META_SCALAR_FIELDS = [
    "recipe_id",
    "name",
    "category",
    "cooking_way",
    "cooking_time",
    "difficulty",
    "spicy_level",
    "calories",
    "protein",
    "fat",
    "sodium",
    "carbs",
]

SELECT_COLUMNS = [
    "recipe_id",
    "name",
    "category",
    "cooking_way",
    "summary",
    "cooking_time",
    "difficulty",
    "spicy_level",
    "calories",
    "protein",
    "fat",
    "sodium",
    "carbs",
    "meal_type_tags",
    "recommended_situations",
    "taste_tags",
    "texture_tags",
    "main_ingredients",
    "hypothetical_questions",
]

# 실제 schema 컬럼명과의 매핑 (스크립트 내부 이름 → SQL 표현)
COLUMN_ALIASES = {
    "recipe_id": "rcp_seq AS recipe_id",
    "cooking_way": "cooking_method AS cooking_way",
}


# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
def get_mysql_connection():
    load_dotenv()
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3308")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_recipes(limit: int | None = None) -> list[dict]:
    """MySQL recipes 테이블에서 레시피 조회."""
    cols = ", ".join(COLUMN_ALIASES.get(c, c) for c in SELECT_COLUMNS)
    sql = f"SELECT {cols} FROM recipes ORDER BY CAST(rcp_seq AS UNSIGNED) ASC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()
    return rows


# -----------------------------------------------------------------------------
# embedding_text 빌더
# -----------------------------------------------------------------------------
def _safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _parse_array_field(v) -> list[str]:
    """MySQL JSON 컬럼은 pymysql이 list로 반환하기도, str로 반환하기도 함."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
        except json.JSONDecodeError:
            pass
        return [s]
    return []


def build_embedding_text(row: dict) -> str:
    name = _safe_str(row.get("name"))
    category = _safe_str(row.get("category"))
    summary = _safe_str(row.get("summary"))

    situations = ", ".join(_parse_array_field(row.get("recommended_situations")))
    meal_types = ", ".join(_parse_array_field(row.get("meal_type_tags")))
    tastes = ", ".join(_parse_array_field(row.get("taste_tags")))
    textures = ", ".join(_parse_array_field(row.get("texture_tags")))
    mains = ", ".join(_parse_array_field(row.get("main_ingredients")))

    cooking_time = row.get("cooking_time")
    cooking_time_str = f"{cooking_time}" if cooking_time is not None else "-"
    difficulty = _safe_str(row.get("difficulty")) or "-"
    spicy = row.get("spicy_level")
    spicy_str = f"{spicy}" if spicy is not None else "-"

    lines = [
        f"[이름] {name}",
        f"[카테고리] {category}",
        f"[요약] {summary}",
        f"[상황] {situations}",
        f"[식사유형] {meal_types}",
        f"[맛] {tastes}",
        f"[식감] {textures}",
        f"[주재료] {mains}",
        f"[조리시간] {cooking_time_str}분 / [난이도] {difficulty} / [매운맛] {spicy_str}/5",
    ]
    hyp_qs = _parse_array_field(row.get("hypothetical_questions"))
    if hyp_qs:
        lines.append(f"[예상질문] {' / '.join(hyp_qs)}")
    return "\n".join(lines)


def build_metadata(row: dict) -> dict:
    """scalar 필드만 metadata로. None은 키 자체 제외."""
    meta = {}
    for f in META_SCALAR_FIELDS:
        if f not in row:
            continue
        v = row[f]
        if v is None:
            continue
        if isinstance(v, (int, float, bool, str)):
            meta[f] = v
        else:
            meta[f] = str(v)
    return meta


# -----------------------------------------------------------------------------
# Chroma
# -----------------------------------------------------------------------------
def get_chroma_client():
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=PERSIST_DIR)


def get_or_create_collection(client, reset: bool):
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[info] 기존 컬렉션 삭제: {COLLECTION_NAME}")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_in_batches(collection, ids, docs, metas, batch_size: int = 100) -> int:
    total = 0
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=docs[i : i + batch_size],
            metadatas=metas[i : i + batch_size],
        )
        total += len(ids[i : i + batch_size])
        print(f"  upsert {total}/{len(ids)}", end="\r")
    print()
    return total


# -----------------------------------------------------------------------------
# 검증 / 샘플 검색
# -----------------------------------------------------------------------------
def print_sample_documents(collection, n: int = 3):
    print("\n[샘플 document 확인]")
    res = collection.peek(limit=n)
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    for i in range(len(ids)):
        print(f"--- id={ids[i]} ---")
        text = docs[i] or ""
        print(text[:600] + ("..." if len(text) > 600 else ""))
        meta = metas[i] or {}
        keys = ["recipe_id", "name", "category", "cooking_way", "cooking_time", "calories"]
        meta_preview = {k: meta.get(k) for k in keys if k in meta}
        print(f"meta(preview): {meta_preview}")


def run_sample_search(collection):
    queries = ["매콤한 국물", "다이어트 도시락", "아이 반찬"]
    print("\n[샘플 검색 결과 (top-5)]")
    for q in queries:
        print(f"\n>>> query: {q}")
        res = collection.query(query_texts=[q], n_results=5)
        ids = res["ids"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for rank, (rid, meta, dist) in enumerate(zip(ids, metas, dists), 1):
            name = meta.get("name", "?")
            cat = meta.get("category", "?")
            print(f"  {rank}. [{cat}] {name}  (id={rid}, dist={dist:.4f})")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
def write_report(report: dict):
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[report] {REPORT_PATH}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="기존 컬렉션 삭제 후 재생성")
    parser.add_argument("--limit", type=int, default=None, help="테스트용 일부만 적재")
    parser.add_argument("--sample-search", action="store_true", help="적재 후 샘플 검색 실행")
    args = parser.parse_args()

    print(f"[1/4] MySQL 조회")
    rows = fetch_recipes(limit=args.limit)
    fetched_count = len(rows)
    print(f"  fetched: {fetched_count}")
    if fetched_count == 0:
        print("[error] 조회된 레시피가 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"[2/4] embedding_text / metadata 빌드")
    ids, docs, metas = [], [], []
    text_lengths = []
    hyp_present = 0
    hyp_missing_ids: list[str] = []
    for r in rows:
        rid = r.get("recipe_id")
        if rid is None:
            continue
        text = build_embedding_text(r)
        meta = build_metadata(r)
        ids.append(f"{ID_PREFIX}{rid}")
        docs.append(text)
        metas.append(meta)
        text_lengths.append(len(text))
        if _parse_array_field(r.get("hypothetical_questions")):
            hyp_present += 1
        else:
            hyp_missing_ids.append(str(rid))
    print(f"  built: {len(ids)}")
    print(f"  hypothetical_questions coverage: {hyp_present}/{len(ids)}")

    print(f"[3/4] Chroma 적재 (model={EMBEDDING_MODEL}, collection={COLLECTION_NAME})")
    client = get_chroma_client()
    collection = get_or_create_collection(client, reset=args.reset)
    upserted_count = upsert_in_batches(collection, ids, docs, metas, batch_size=100)
    collection_count = collection.count()
    print(f"  upserted: {upserted_count}")
    print(f"  collection.count(): {collection_count}")

    print(f"[4/4] 검증")
    print(f"  MySQL fetched: {fetched_count}")
    print(f"  Chroma upserted: {upserted_count}")
    print(f"  collection.count(): {collection_count}")

    # 전체 적재 시 count 일치 검증
    if args.limit is None:
        if fetched_count != upserted_count or upserted_count != collection_count:
            print(
                f"[error] count mismatch: fetched={fetched_count}, "
                f"upserted={upserted_count}, collection={collection_count}",
                file=sys.stderr,
            )
            sys.exit(1)

    print_sample_documents(collection, n=3)

    if args.sample_search:
        run_sample_search(collection)

    avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    report = {
        "collection_name": COLLECTION_NAME,
        "persist_directory": PERSIST_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "id_prefix": ID_PREFIX,
        "source": "mysql.recipes",
        "fetched_count": fetched_count,
        "upserted_count": upserted_count,
        "collection_count": collection_count,
        "avg_embedding_text_length": round(avg_len, 2),
        "min_embedding_text_length": min(text_lengths) if text_lengths else 0,
        "max_embedding_text_length": max(text_lengths) if text_lengths else 0,
        "hypothetical_questions_coverage": f"{hyp_present}/{len(ids)}",
        "hypothetical_questions_missing": hyp_missing_ids[:20],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_report(report)


if __name__ == "__main__":
    main()
