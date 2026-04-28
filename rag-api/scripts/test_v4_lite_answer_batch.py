"""
v4-lite 배치 답변 테스트.
대표 사용자 쿼리 10개에 대해 검색 → LLM rerank → 자연어 답변 생성을 확인한다.

실행:
python scripts/test_v4_lite_answer_batch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.answer_generator import generate_answer
from core.reranker import rerank
from core.retrieval import search

TEST_QUERIES = [
    "운동 후 저녁으로 먹기 좋은 단백질 위주의 메뉴 추천해줘",
    "가볍게 먹을 수 있는 다이어트 저녁 메뉴 추천해줘",
    "아이가 먹기 좋은 자극적이지 않은 간식 추천해줘",
    "비 오는 날 먹기 좋은 따뜻한 국물 요리 추천해줘",
    "매운 음식이 먹고 싶은데 밥이랑 같이 먹을 반찬 추천해줘",
    "부모님께 드리기 좋은 자극적이지 않은 한식 메뉴 추천해줘",
    "집에 손님 왔을 때 내기 좋은 보기 좋은 요리 추천해줘",
    "아침에 부담 없이 먹을 수 있는 간단한 메뉴 추천해줘",
    "채소를 많이 먹고 싶은데 맛있게 먹을 수 있는 메뉴 추천해줘",
    "짜지 않고 건강하게 먹을 수 있는 반찬 추천해줘",
]

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


def run_query(index: int, query: str) -> None:
    print("=" * 80)
    print(f"[{index}] query: {query}")

    hits = search(query, top_k=RETRIEVAL_TOP_K)
    reranked_hits = rerank(query, hits)
    answer = generate_answer(query, reranked_hits, top_k=ANSWER_TOP_K)

    print_top5(reranked_hits)

    print("\n[최종 answer]")
    print(answer)
    print()


def main() -> None:
    for index, query in enumerate(TEST_QUERIES, start=1):
        run_query(index, query)


if __name__ == "__main__":
    main()
