"""
golden_set_v1_final.csv (50행) + v3_new_candidates_labeled_draft.csv (25행)을
합쳐 golden_set_expanded_v3.csv (75행)를 생성한다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
V1_GOLDEN_PATH  = ARTIFACTS / "golden_set_v1_final.csv"
V3_DRAFT_PATH   = ARTIFACTS / "v3_new_candidates_labeled_draft.csv"
OUTPUT_PATH     = ARTIFACTS / "golden_set_expanded_v3.csv"

OUTPUT_FIELDNAMES = [
    "query_id", "query", "retrieved_doc_id", "doc_title",
    "llm_judgment", "llm_confidence", "llm_reason",
    "needs_review", "opus_judgment", "opus_reason",
    "human_judgment", "human_override_reason", "final_label",
]

# human override 대상
OVERRIDE_KEY    = ("Q1", "맑은육개장")
OVERRIDE_VALUES = {
    "human_judgment":        "0",
    "human_override_reason": "맑은탕 계열 — 기준 일관성",
    "final_label":           "0",
}


def load_v1(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_v3(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_v3_row(row: dict) -> dict:
    """v3 후보 행을 golden set 스키마로 변환한다."""
    llm_j = row["llm_judgment"]
    human_j = llm_j
    override_reason = ""
    final = human_j

    key = (row["query_id"], row["name"])
    if key == OVERRIDE_KEY:
        human_j        = OVERRIDE_VALUES["human_judgment"]
        override_reason = OVERRIDE_VALUES["human_override_reason"]
        final           = OVERRIDE_VALUES["final_label"]

    return {
        "query_id":             row["query_id"],
        "query":                row["query"],
        "retrieved_doc_id":     row["recipe_id"],
        "doc_title":            row["name"],
        "llm_judgment":         llm_j,
        "llm_confidence":       row["llm_confidence"],
        "llm_reason":           row["llm_reason"],
        "needs_review":         row["needs_review"],
        "opus_judgment":        "",
        "opus_reason":          "",
        "human_judgment":       human_j,
        "human_override_reason": override_reason,
        "final_label":          final,
    }


def main() -> None:
    v1_rows = load_v1(V1_GOLDEN_PATH)
    v3_rows = [convert_v3_row(r) for r in load_v3(V3_DRAFT_PATH)]

    all_rows = v1_rows + v3_rows

    # 중복 확인
    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple] = []
    for r in all_rows:
        key = (str(r["query_id"]), str(r["retrieved_doc_id"]))
        if key in seen:
            duplicates.append(key)
        seen.add(key)

    if duplicates:
        print(f"[ERROR] 중복 (query_id, retrieved_doc_id) 발견: {duplicates}")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # ─── 보고 ────────────────────────────────────
    print(f"총 행 수: {len(all_rows)}행")
    print()

    from collections import Counter, defaultdict
    qid_counts = Counter(r["query_id"] for r in all_rows)
    print("query_id별 행 수:")
    for qid in sorted(qid_counts):
        print(f"  {qid}: {qid_counts[qid]}행")

    total_label1 = sum(1 for r in all_rows if str(r["final_label"]) == "1")
    print(f"\nfinal_label=1 총 개수: {total_label1}건")

    qid_label1 = Counter(r["query_id"] for r in all_rows if str(r["final_label"]) == "1")
    print("query_id별 final_label=1:")
    for qid in sorted(qid_label1):
        print(f"  {qid}: {qid_label1[qid]}건")

    v3_label1 = sum(1 for r in v3_rows if str(r["final_label"]) == "1")
    print(f"\nv3 신규 25건 중 final_label=1: {v3_label1}건")

    overrides = [r for r in v3_rows if r["human_override_reason"]]
    print(f"human_override 적용 건수: {len(overrides)}건")
    for r in overrides:
        print(f"  [{r['query_id']}] {r['doc_title']} → human_judgment={r['human_judgment']} ({r['human_override_reason']})")

    print(f"\n중복 (query_id, retrieved_doc_id): 없음")
    print(f"\n저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
