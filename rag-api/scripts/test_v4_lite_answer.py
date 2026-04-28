"""
v4-lite 전체 흐름 테스트.
검색 → LLM rerank → 자연어 답변 생성을 한 번에 확인한다.

실행:
python scripts/test_v4_lite_answer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.answer_generator import generate_answer
from core.reranker import rerank
from core.retrieval import search

TEST_QUERY = "운동 후 저녁으로 먹기 좋은 단백질 위주의 메뉴 추천해줘"
RETRIEVAL_TOP_K = 10
ANSWER_TOP_K = 5


def print_top5(hits) -> None:
    print("\n[top-5 추천 레시피]")
    print("| rank | recipe_id | name | category | cooking_way |")
    print("|---:|---:|---|---|---|")

    for rank, hit in enumerate(hits[:ANSWER_TOP_K], start=1):
        category = hit.metadata.get("category") or ""
        cooking_way = hit.metadata.get("cooking_way") or ""
        print(
            f"| {rank} "
            f"| {hit.recipe_id} "
            f"| {hit.name} "
            f"| {category} "
            f"| {cooking_way} |"
        )


def main() -> None:
    query = TEST_QUERY

    print(f"query: {query}")

    hits = search(query, top_k=RETRIEVAL_TOP_K)
    reranked_hits = rerank(query, hits)
    answer = generate_answer(query, reranked_hits, top_k=ANSWER_TOP_K)

    print_top5(reranked_hits)

    print("\n[최종 answer]")
    print(answer)


if __name__ == "__main__":
    main()
