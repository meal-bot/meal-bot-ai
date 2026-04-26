"""
v1 샘플 쿼리 top-10 검색 결과 덤프.
Golden set 설계 단계에서 관련성 판정 기준 논의용.

실행:
    cd rag-api
    source venv/bin/activate
    python scripts/dump_v1_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval import search

TEST_QUERIES = [
    "얼큰한 국물 요리",
    "다이어트 샐러드",
    "아이 간식",
    "매운 반찬",
    "고단백 요리",
]

TOP_K = 10


def main():
    for q in TEST_QUERIES:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print('=' * 60)
        hits = search(q, top_k=TOP_K)
        for rank, h in enumerate(hits, 1):
            cat = h.metadata.get("category") or "-"
            cw = h.metadata.get("cooking_way") or "-"
            print(f"  [{rank:2d}] {h.score:.3f} | {h.name}")
            print(f"       category={cat} / cooking_way={cw}")
            print(f"       doc={h.document}")


if __name__ == "__main__":
    main()

    