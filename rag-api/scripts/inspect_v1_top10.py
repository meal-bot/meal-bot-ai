"""
v1 베이스라인 Top-10 검색 결과 검사.
각 쿼리의 상위 10건을 마크다운 표로 터미널 출력 + artifacts/v1_top10.md 저장.
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

HEADER = "| 순위 | score | distance | recipe_id | name | category | cooking_way |"
SEPARATOR = "|---:|---:|---:|---:|---|---|---|"


def build_table(query: str) -> str:
    hits = search(query, top_k=10)
    lines = [f"## {query}", "", HEADER, SEPARATOR]
    for rank, hit in enumerate(hits, start=1):
        category = hit.metadata.get("category") or ""
        cooking_way = hit.metadata.get("cooking_way") or ""
        lines.append(
            f"| {rank} "
            f"| {hit.score:.4f} "
            f"| {hit.distance:.4f} "
            f"| {hit.recipe_id} "
            f"| {hit.name} "
            f"| {category} "
            f"| {cooking_way} |"
        )
    return "\n".join(lines)


def main() -> None:
    sections = [build_table(q) for q in TEST_QUERIES]
    md = "\n\n".join(sections) + "\n"

    # 터미널 출력
    print(md)

    # artifacts/ 저장
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    output_path = artifacts_dir / "v1_top10.md"
    output_path.write_text(md, encoding="utf-8")
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
