# scripts/load_to_mysql.py
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import pymysql
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _enrichment_common import build_manuals_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH  = os.path.join(BASE_DIR, "data", "recipes_cleaned.json")
ENRICHED_PATH = os.path.join(BASE_DIR, "data", "recipes_enriched.json")

BATCH_SIZE = 100


def to_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def to_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def to_json_str(v) -> Optional[str]:
    if v is None:
        return None
    return json.dumps(v, ensure_ascii=False)


def build_row(c: dict, e: dict, now: str) -> dict:
    enriched     = e.get("enriched", {})
    flags_e      = enriched.get("_flags", {})
    flags_c      = c.get("enrichment_flags", {})
    nutrition    = c.get("nutrition", {})
    merged_flags = {**flags_c, **flags_e}

    return {
        "rcp_seq":         c["rcp_seq"],
        "name":            c["name"],
        "category":        enriched.get("category"),
        "cooking_method":  enriched.get("cooking_method"),
        "calories":        to_float(nutrition.get("energy_kcal")),
        "protein":         to_float(nutrition.get("protein_g")),
        "fat":             to_float(nutrition.get("fat_g")),
        "sodium":          to_float(nutrition.get("sodium_mg")),
        "carbs":           to_float(nutrition.get("carbs_g")),
        "cooking_time":    to_int(enriched.get("cooking_time")),
        "difficulty":      to_int(enriched.get("difficulty")),
        "spicy_level":     to_int(enriched.get("spicy_level")),
        "ingredient_count": to_int(c.get("ingredient_count_total")),
        "meal_type_tags":           to_json_str(enriched.get("meal_type_tags")),
        "summary":                  enriched.get("summary"),
        "taste_tags":               to_json_str(enriched.get("taste_tags")),
        "texture_tags":             to_json_str(enriched.get("texture_tags")),
        "recommended_situations":   to_json_str(enriched.get("recommended_situations")),
        "main_ingredients":         to_json_str(enriched.get("main_ingredients")),
        "enrichment_flags":         to_json_str(merged_flags),
        "ingredients":   c.get("ingredients"),
        "manuals_text":  build_manuals_text(c.get("manuals", [])),
        "img_main":      c.get("img_main"),
        "img_thumb":     c.get("img_thumb"),
        "enrichment_model": flags_e.get("enrichment_model"),
        "enriched_at":      now,
    }


INSERT_SQL = """
INSERT INTO recipes (
    rcp_seq, name, category, cooking_method,
    calories, protein, fat, sodium, carbs,
    cooking_time, difficulty, spicy_level, ingredient_count,
    meal_type_tags, summary, taste_tags, texture_tags,
    recommended_situations, main_ingredients, enrichment_flags,
    ingredients, manuals_text, img_main, img_thumb,
    enrichment_model, enriched_at
) VALUES (
    %(rcp_seq)s, %(name)s, %(category)s, %(cooking_method)s,
    %(calories)s, %(protein)s, %(fat)s, %(sodium)s, %(carbs)s,
    %(cooking_time)s, %(difficulty)s, %(spicy_level)s, %(ingredient_count)s,
    %(meal_type_tags)s, %(summary)s, %(taste_tags)s, %(texture_tags)s,
    %(recommended_situations)s, %(main_ingredients)s, %(enrichment_flags)s,
    %(ingredients)s, %(manuals_text)s, %(img_main)s, %(img_thumb)s,
    %(enrichment_model)s, %(enriched_at)s
)
"""

VERIFY_SQL = [
    ("총 레코드 수",           "SELECT COUNT(*) FROM recipes"),
    ("category NULL 수",       "SELECT COUNT(*) FROM recipes WHERE category IS NULL"),
    ("summary NULL/빈값 수",   "SELECT COUNT(*) FROM recipes WHERE summary IS NULL OR summary = ''"),
    ("cooking_time NULL 수",   "SELECT COUNT(*) FROM recipes WHERE cooking_time IS NULL"),
    ("meal_type_tags 빈배열",  "SELECT COUNT(*) FROM recipes WHERE JSON_LENGTH(meal_type_tags) = 0"),
]

def run_verify(cur):
    print("\n─── 검증 결과 ───")
    for label, sql in VERIFY_SQL:
        cur.execute(sql)
        print(f"  {label}: {cur.fetchone()[0]}")

    print("\n─── category 분포 ───")
    cur.execute("SELECT category, COUNT(*) FROM recipes GROUP BY category ORDER BY COUNT(*) DESC")
    for row in cur.fetchall():
        print(f"  {row}")

    print("\n─── 랜덤 샘플 3건 ───")
    cur.execute("""
        SELECT rcp_seq, name, category, cooking_method, calories, summary
        FROM recipes ORDER BY RAND() LIMIT 3
    """)
    for row in cur.fetchall():
        print(f"  {row}")

    print("\n─── rcp_seq=28 전체 컬럼 ───")
    cur.execute("SELECT * FROM recipes WHERE rcp_seq = '28' LIMIT 1")
    print(cur.fetchone())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="DB 미접근, 첫 3건 매핑 출력")
    parser.add_argument("--truncate", action="store_true", help="INSERT 전 테이블 TRUNCATE")
    args = parser.parse_args()

    with open(CLEANED_PATH,  encoding="utf-8") as f:
        cleaned_list = json.load(f)
    with open(ENRICHED_PATH, encoding="utf-8") as f:
        enriched_list = json.load(f)

    assert len(cleaned_list)  == 1146, f"cleaned 수 이상: {len(cleaned_list)}"
    assert len(enriched_list) == 1146, f"enriched 수 이상: {len(enriched_list)}"

    enriched_map = {r["rcp_seq"]: r for r in enriched_list}
    missing = [c["rcp_seq"] for c in cleaned_list if c["rcp_seq"] not in enriched_map]
    assert not missing, f"enriched 누락 rcp_seq: {missing}"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [build_row(c, enriched_map[c["rcp_seq"]], now) for c in cleaned_list]

    if args.dry_run:
        print("=== DRY RUN: 첫 3건 매핑 결과 ===")
        for row in rows[:3]:
            print(json.dumps(row, ensure_ascii=False, indent=2))
        print(f"\n총 {len(rows)}건 매핑 완료 (DB 미접근)")
        return

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    conn = pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ.get("MYSQL_PORT", 3308)),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        charset="utf8mb4",
        autocommit=False,
    )

    try:
        with conn.cursor() as cur:
            if args.truncate:
                cur.execute("TRUNCATE TABLE recipes")
                print("TRUNCATE 완료")

            total = len(rows)
            for i in range(0, total, BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                cur.executemany(INSERT_SQL, batch)
                print(f"  INSERT {i + len(batch)}/{total}")

            conn.commit()
            print(f"\n✅ {total}건 적재 완료")

            run_verify(cur)

    except Exception as ex:
        conn.rollback()
        print(f"\n❌ 오류 발생 → 롤백: {ex}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
