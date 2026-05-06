"""
update_hyp_questions_to_mysql.py

data/recipes_hyp_questions.json → recipes.hypothetical_questions UPDATE
- rcp_seq 기준 매칭
- JSON.dumps(questions, ensure_ascii=False)
- 100건 단위 commit
- 컬럼 존재 여부 확인 후 진행
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "recipes_hyp_questions.json"

UPDATE_SQL = "UPDATE recipes SET hypothetical_questions = %s WHERE rcp_seq = %s"
BATCH_COMMIT = 100


def get_conn():
    load_dotenv(BASE_DIR / ".env")
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ.get("MYSQL_PORT", 3308)),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        charset="utf8mb4",
        autocommit=False,
    )


def column_exists(cur) -> bool:
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'recipes'
          AND column_name = 'hypothetical_questions'
    """)
    return cur.fetchone()[0] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="실제 UPDATE 없이 SQL 미리보기")
    ap.add_argument("--rcp-seq", type=str, default=None, help="특정 1건만 업데이트 (테스트용)")
    ap.add_argument("--yes", action="store_true", help="확인 프롬프트 스킵")
    args = ap.parse_args()

    if not INPUT_PATH.exists():
        print(f"[error] 입력 파일 없음: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"입력 {len(data)}건  ({INPUT_PATH})")

    if args.rcp_seq is not None:
        data = [d for d in data if d["rcp_seq"] == args.rcp_seq]
        if not data:
            print(f"[error] rcp_seq={args.rcp_seq} 없음", file=sys.stderr)
            sys.exit(1)
        print(f"필터: rcp_seq={args.rcp_seq} → {len(data)}건")

    rows = [
        (json.dumps(d["hypothetical_questions"], ensure_ascii=False), d["rcp_seq"])
        for d in data
    ]

    if args.dry_run:
        print("\n--- DRY RUN: 첫 3건 ---")
        for q_json, rcp_seq in rows[:3]:
            print(f"UPDATE recipes SET hypothetical_questions = '{q_json}' "
                  f"WHERE rcp_seq = '{rcp_seq}';")
        print(f"... 총 {len(rows)}건 (실제 UPDATE 실행 안 함)")
        return

    if not args.yes:
        print(f"\n총 {len(rows)}건 UPDATE 예정. 진행하려면 Enter, 취소는 Ctrl+C.")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("취소")
            return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if not column_exists(cur):
                print("[error] recipes.hypothetical_questions 컬럼이 없습니다. "
                      "ALTER TABLE을 먼저 실행하세요.", file=sys.stderr)
                sys.exit(1)

            matched_total = 0
            for i, params in enumerate(rows, 1):
                cur.execute(UPDATE_SQL, params)
                matched_total += cur.rowcount
                if i % BATCH_COMMIT == 0:
                    conn.commit()
                    print(f"  commit {i}/{len(rows)}  rowcount누적={matched_total}")
            conn.commit()

            print(f"\n=== UPDATE 완료 ===")
            print(f"실행 건수:           {len(rows)}")
            print(f"rowcount 누적:       {matched_total}  (값이 같으면 매칭은 됐지만 변경 0일 수 있음)")

            cur.execute("SELECT COUNT(*) FROM recipes WHERE hypothetical_questions IS NOT NULL")
            not_null = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM recipes")
            total = cur.fetchone()[0]
            print(f"NOT NULL row 수:     {not_null}")
            print(f"recipes 전체 row 수: {total}")
    except Exception as e:
        conn.rollback()
        print(f"\n[error] 롤백: {e!r}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
