"""
v3 top-10 결과에서 golden_set_v1_final.csv에 없는 신규 후보를 추출한다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
V3_RESULTS_PATH  = ARTIFACTS / "eval_v3_results.json"
GOLDEN_SET_PATH  = ARTIFACTS / "golden_set_v1_final.csv"
OUTPUT_PATH      = ARTIFACTS / "v3_new_candidates.csv"

OUTPUT_FIELDNAMES = ["query_id", "query", "recipe_id", "name", "rank", "score", "source", "v1_overlap"]


def load_golden_keys(path: Path) -> set[tuple[str, str]]:
    """golden set의 (query_id, recipe_id) 집합 반환."""
    keys: set[tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((row["query_id"], str(row["retrieved_doc_id"])))
    return keys


def main() -> None:
    golden_keys = load_golden_keys(GOLDEN_SET_PATH)

    with open(V3_RESULTS_PATH, encoding="utf-8") as f:
        v3 = json.load(f)

    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for pq in v3["per_query"]:
        qid   = pq["query_id"]
        query = pq["query"]
        for result in pq["results"]:
            key = (qid, str(result["recipe_id"]))
            if key in golden_keys:
                continue
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "query_id":  qid,
                "query":     query,
                "recipe_id": result["recipe_id"],
                "name":      result["name"],
                "rank":      result["rank"],
                "score":     result["score"],
                "source":    "v3_hybrid",
                "v1_overlap": False,
            })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(candidates)

    print(f"총 후보: {len(candidates)}건")
    print()

    from collections import Counter
    counts = Counter(c["query_id"] for c in candidates)
    for qid in sorted(counts):
        print(f"  {qid}: {counts[qid]}건")

    print(f"\n저장: {OUTPUT_PATH}")
    print("\n상위 10개 샘플:")
    print(f"{'query_id':<8} {'rank':>4}  {'recipe_id':>9}  {'score':>8}  name")
    print("-" * 60)
    for c in candidates[:10]:
        print(f"{c['query_id']:<8} {c['rank']:>4}  {c['recipe_id']:>9}  {c['score']:>8}  {c['name']}")


if __name__ == "__main__":
    main()
