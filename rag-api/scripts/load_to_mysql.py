"""
rag-api/data/processed/recipes.json → MySQL recipe 테이블 적재

실행 방법:
    cd /path/to/meal-bot/rag-api
    source venv/bin/activate
    python scripts/load_to_mysql.py

주의:
    - Docker가 실행 중이어야 함 (docker compose up -d)
    - 실행 시 recipe 테이블 TRUNCATE 후 전체 재적재 (멱등)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymysql
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.db import mysql_connection

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "recipes.json"

INSERT_SQL = """
INSERT INTO recipe (
    rcp_seq, name, category, cooking_way, ingredients,
    hash_tag, img_main, img_thumb,
    calories, carbs, protein, fat, sodium,
    manuals
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s
)
"""

BATCH_SIZE = 100


# ── 변환 ──────────────────────────────────────────────────────────────────────

def to_int(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value))


def to_nullable_str(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    return value.strip()


def build_row(r: dict) -> tuple:
    nut = r.get("nutrition", {})
    return (
        int(r["rcp_seq"]),
        r["name"],
        to_nullable_str(r.get("category")),
        to_nullable_str(r.get("cooking_way")),
        r.get("ingredients"),
        to_nullable_str(r.get("hash_tag")),
        r.get("img_main"),
        r.get("img_thumb"),
        to_int(nut.get("energy_kcal")),
        to_int(nut.get("carbs_g")),
        to_int(nut.get("protein_g")),
        to_int(nut.get("fat_g")),
        to_int(nut.get("sodium_mg")),
        json.dumps(r.get("manuals", []), ensure_ascii=False),
    )


# ── 통계 ──────────────────────────────────────────────────────────────────────

def print_stats(conn: pymysql.connections.Connection):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM recipe")
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT category, COUNT(*) FROM recipe GROUP BY category ORDER BY COUNT(*) DESC"
        )
        categories = cur.fetchall()

        cur.execute(
            "SELECT MIN(calories), MAX(calories), ROUND(AVG(calories)) FROM recipe WHERE calories IS NOT NULL"
        )
        cal_min, cal_max, cal_avg = cur.fetchone()

    print("\n" + "=" * 50)
    print("[ 적재 통계 ]")
    print(f"  총 적재: {total}개")
    print(f"\n  칼로리  min={cal_min}  max={cal_max}  avg={cal_avg}")
    print(f"\n  카테고리 분포:")
    for cat, cnt in categories:
        label = cat or "미분류"
        print(f"    {label}: {cnt}개")
    print("=" * 50)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    print(f"데이터 로드: {DATA_PATH}")
    with open(DATA_PATH, encoding="utf-8") as f:
        recipes = json.load(f)
    print(f"  {len(recipes)}개 로드 완료")

    rows = []
    errors = []
    for r in recipes:
        try:
            rows.append(build_row(r))
        except Exception as e:
            errors.append((r.get("rcp_seq"), str(e)))

    if errors:
        print(f"\n변환 실패 {len(errors)}건:")
        for seq, msg in errors:
            print(f"  rcp_seq={seq}: {msg}")

    with mysql_connection() as conn:
        print(f"\nMySQL 연결 완료")
        try:
            with conn.cursor() as cur:
                print("TRUNCATE recipe ...")
                cur.execute("TRUNCATE TABLE recipe")

            success = 0
            for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="INSERT", unit="batch"):
                batch = rows[i : i + BATCH_SIZE]
                with conn.cursor() as cur:
                    cur.executemany(INSERT_SQL, batch)
                conn.commit()
                success += len(batch)

            print(f"\n적재 완료: {success}개 성공 / {len(errors)}개 실패")
            print_stats(conn)

        except Exception as e:
            conn.rollback()
            print(f"\n오류 발생, 롤백: {e}")
            raise


if __name__ == "__main__":
    main()
