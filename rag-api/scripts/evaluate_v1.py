"""
v1 베이스라인 평가 스크립트.
golden_set_v1_final.csv 기반으로 Precision@K, NDCG@K, MRR을 계산한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval import search

# ─── 상수 ────────────────────────────────────────
TOP_K = 10
K_VALUES = [5, 10]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=Path(__file__).parent.parent / "artifacts" / "golden_set_v1_final.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "artifacts" / "eval_v1_results.json",
    )
    return parser.parse_args()


# ─── 지표 계산 ───────────────────────────────────

def precision_at_k(labels: list[int], k: int) -> float:
    top = labels[:k]
    return sum(top) / k if k > 0 else 0.0


def dcg_at_k(labels: list[int], k: int) -> float:
    return sum(
        (2 ** rel - 1) / math.log2(rank + 2)
        for rank, rel in enumerate(labels[:k])
    )


def idcg_at_k(num_relevant: int, k: int) -> float:
    ideal = min(num_relevant, k)
    return sum(1.0 / math.log2(rank + 2) for rank in range(ideal))


def ndcg_at_k(labels: list[int], num_relevant: int, k: int) -> float:
    idcg = idcg_at_k(num_relevant, k)
    return dcg_at_k(labels, k) / idcg if idcg > 0 else 0.0


def mrr(labels: list[int]) -> float:
    for rank, rel in enumerate(labels, start=1):
        if rel == 1:
            return 1.0 / rank
    return 0.0


# ─── 골든셋 로드 ─────────────────────────────────

def load_golden_set(path: Path) -> dict[str, dict]:
    """
    반환: {query_id: {"query": str, "labels": {str(doc_id): int}}}
    """
    golden: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            if qid not in golden:
                golden[qid] = {"query": row["query"], "labels": {}}
            golden[qid]["labels"][str(row["retrieved_doc_id"])] = int(row["final_label"])
    return golden


# ─── 메인 ────────────────────────────────────────

def main() -> None:
    args = parse_args()
    GOLDEN_SET_PATH = args.golden_set
    OUTPUT_PATH = args.output

    golden = load_golden_set(GOLDEN_SET_PATH)

    per_query = []
    avg_accum: dict[str, float] = {
        "precision_at_5": 0.0,
        "precision_at_10": 0.0,
        "ndcg_at_5": 0.0,
        "ndcg_at_10": 0.0,
        "mrr": 0.0,
    }

    for qid, data in golden.items():
        query = data["query"]
        label_map = data["labels"]
        num_relevant = sum(label_map.values())

        try:
            hits = search(query, top_k=TOP_K)
        except Exception as e:
            print(f"[ERROR] query_id={qid} 검색 실패: {e}")
            raise

        labels = [label_map.get(str(h.recipe_id), 0) for h in hits]

        metrics = {
            "precision_at_5":  precision_at_k(labels, 5),
            "precision_at_10": precision_at_k(labels, 10),
            "ndcg_at_5":       ndcg_at_k(labels, num_relevant, 5),
            "ndcg_at_10":      ndcg_at_k(labels, num_relevant, 10),
            "mrr":             mrr(labels),
        }

        results = [
            {
                "rank":        rank,
                "recipe_id":   str(h.recipe_id),
                "name":        h.name,
                "score":       round(h.score, 4),
                "final_label": labels[rank - 1],
            }
            for rank, h in enumerate(hits, start=1)
        ]

        per_query.append({
            "query_id": qid,
            "query":    query,
            "metrics":  {k: round(v, 4) for k, v in metrics.items()},
            "results":  results,
        })

        for key in avg_accum:
            avg_accum[key] += metrics[key]

        print(
            f"[{qid}] {query}\n"
            f"  P@5={metrics['precision_at_5']:.3f}  P@10={metrics['precision_at_10']:.3f}"
            f"  NDCG@5={metrics['ndcg_at_5']:.3f}  NDCG@10={metrics['ndcg_at_10']:.3f}"
            f"  MRR={metrics['mrr']:.3f}"
        )

    n = len(per_query)
    average = {k: round(v / n, 4) for k, v in avg_accum.items()}

    print("\n─── 전체 평균 ───")
    for key, val in average.items():
        print(f"  {key}: {val:.4f}")

    output = {
        "version": "v1",
        "top_k":   TOP_K,
        "per_query": per_query,
        "average": average,
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
