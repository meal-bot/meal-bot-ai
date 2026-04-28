"""
v2 쿼리 재작성 모듈.
OpenAI gpt-4o-mini를 사용해 한국 레시피 검색에 최적화된 쿼리로 변환한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from core.config import OPENAI_API_KEY

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


_SYSTEM_PROMPT = """\
당신은 한국 레시피 벡터 검색 시스템을 위한 쿼리 보강 전문가입니다.
사용자의 자연어 쿼리를 분석하여 관련 후보를 넓게 회수할 수 있는 검색 쿼리로 재작성합니다.

데이터베이스는 1,146개의 한국 레시피를 포함하며, 각 레시피는 이름·카테고리·조리방식·주재료·특징으로 구성됩니다.
카테고리: 반찬, 국&찌개, 일품, 후식, 기타
조리방식: 끓이기, 굽기, 볶기, 찌기, 튀기기, 기타

[rewritten_query 작성 규칙]
- rewritten_query는 "정답 후보를 좁히는 쿼리"가 아니라 "관련 후보를 넓게 회수하기 위한 검색 쿼리"다.
- 원본 표현을 반드시 그대로 포함하고, 동의어·관련어·상위 카테고리를 2~5개 정도만 보강하라.
- 사용자 입력에 없는 특정 재료나 조리법을 과도하게 추가하지 말라.
- 하나의 대표 음식으로 치환하지 말라 (예: "얼큰한 국물 요리"를 "김치찌개"로 대체하는 것 금지).
- 맛/의도/카테고리 표현은 유지하되, 특정 레시피 하나로 수렴하지 않게 하라.
- 조건이 명시되지 않은 경우 새로운 제약을 임의로 만들지 말라.
- 문장 형식이 아닌 키워드 나열 형식으로 작성하라.

[Few-shot 예시]
원본: 얼큰한 국물 요리
rewritten_query: 얼큰한 국물 요리 매운 국 찌개 탕 국물요리

원본: 다이어트 샐러드
rewritten_query: 다이어트 샐러드 저칼로리 채소 샐러드 가벼운 샐러드

원본: 아이 간식
rewritten_query: 아이 간식 어린이 간식 후식 디저트 음료 빵 떡

원본: 매운 반찬
rewritten_query: 매운 반찬 매콤한 반찬 고추 양념 무침 볶음 구이

원본: 고단백 요리
rewritten_query: 고단백 요리 단백질 많은 음식 두부 생선 계란 고기 콩 요리

반드시 아래 JSON 형식으로만 응답하세요:
{
  "original_query": "...",
  "intent": "...",
  "conditions": ["...", "..."],
  "keywords": ["...", "..."],
  "rewritten_query": "..."
}
"""


def rewrite(query: str) -> dict:
    """
    사용자 쿼리를 레시피 검색에 최적화된 쿼리로 재작성한다.

    Args:
        query: 원본 사용자 쿼리

    Returns:
        {original_query, intent, conditions, keywords, rewritten_query} 딕셔너리.
        API 실패 시 rewritten_query는 원본 쿼리로 fallback.
    """
    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        result = json.loads(response.choices[0].message.content)
        result["original_query"] = query
        return result
    except Exception as e:
        print(f"[WARN] rewrite 실패 (fallback): {e}")
        return {
            "original_query": query,
            "intent": "",
            "conditions": [],
            "keywords": [],
            "rewritten_query": query,
        }


if __name__ == "__main__":
    test_queries = [
        "얼큰한 국물 요리",
        "다이어트 샐러드",
        "아이 간식",
        "매운 반찬",
        "고단백 요리",
    ]
    for q in test_queries:
        result = rewrite(q)
        print(f"\n원본: {result['original_query']}")
        print(f"의도: {result['intent']}")
        print(f"조건: {result['conditions']}")
        print(f"키워드: {result['keywords']}")
        print(f"재작성: {result['rewritten_query']}")
