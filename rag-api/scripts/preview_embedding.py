"""
임베딩용 텍스트 변환 미리보기
MySQL에서 랜덤 20건 뽑아 콘솔 출력.

실행 방법:
    cd /path/to/meal-bot/rag-api
    source venv/bin/activate
    python scripts/preview_embedding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.db import mysql_connection
from _embedding_text import clean_ingredients, build_embedding_text


def main():
    with mysql_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT rcp_seq, name, category, cooking_way, ingredients, hash_tag
                FROM recipe
                ORDER BY RAND()
                LIMIT 20
            """)
            rows = cur.fetchall()

    for row in rows:
        recipe = {
            "name":        row["name"],
            "category":    row["category"],
            "cooking_way": row["cooking_way"],
            "ingredients": row["ingredients"],
            "hash_tag":    row["hash_tag"],
        }
        original = row["ingredients"] or ""
        cleaned  = clean_ingredients(original)
        final    = build_embedding_text(recipe)

        print(f"[{row['rcp_seq']}] {row['name']}")
        print(f"ORIGINAL: {original}")
        print(f"CLEANED : {cleaned}")
        print(f"FINAL   : {final}")
        print(f"LENGTH  : {len(final)}")
        print("---")


if __name__ == "__main__":
    main()
