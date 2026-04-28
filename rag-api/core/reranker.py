"""
LLM 기반 재정렬 모듈.
v1 semantic 검색 결과를 golden_set_criteria.md 기준으로 재판정 후 재정렬한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from core.config import OPENAI_API_KEY
from core.retrieval import Hit

CRITERIA_PATH = Path(__file__).parent.parent / "docs" / "golden_set_criteria.md"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _load_criteria() -> str:
    return CRITERIA_PATH.read_text(encoding="utf-8")


def _build_system_prompt(criteria: str) -> str:
    return (
        "당신은 한국 레시피 검색 시스템의 재판정 전문가입니다.\n"
        "아래 기준 문서에 따라 각 후보가 쿼리에 relevant한지 판정하십시오.\n"
        "상식이 아니라 반드시 아래 기준 문서를 따르십시오.\n\n"
        f"{criteria}\n\n"
        "반드시 아래 JSON object 형식만 반환하십시오. 다른 텍스트는 포함하지 마십시오.\n"
        "{\n"
        '  "results": [\n'
        '    {"id": 1, "relevant": 1},\n'
        '    {"id": 2, "relevant": 0}\n'
        "  ]\n"
        "}"
    )


def _build_user_prompt(query: str, hits: list[Hit]) -> str:
    lines = [f"쿼리: {query}\n", "아래 레시피 후보들이 위 쿼리에 relevant한지 판정하라.\n"]
    for i, h in enumerate(hits, start=1):
        category    = h.metadata.get("category", "") or ""
        cooking_way = h.metadata.get("cooking_way", "") or ""
        lines.append(f"{i}. {h.name} | {category} | {cooking_way} | {h.document}")
    return "\n".join(lines)


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    """
    hits를 LLM 기준 문서 판정으로 재정렬한다.
    relevant=1 → 상위, relevant=0 → 하위.
    동점 내에서는 원래 semantic score 순서 유지.
    실패 시 원본 순서 그대로 반환.
    """
    if not hits:
        return hits

    try:
        criteria = _load_criteria()
        system   = _build_system_prompt(criteria)
        user     = _build_user_prompt(query, hits)

        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )

        raw = json.loads(response.choices[0].message.content)

        judgments = raw.get("results")
        if not isinstance(judgments, list):
            raise ValueError("LLM 응답에 results 배열이 없습니다.")

        relevant_map: dict[int, int] = {
            item["id"]: int(item["relevant"])
            for item in judgments
            if "id" in item and "relevant" in item
        }

    except Exception as e:
        print(f"[WARN] rerank 실패 (fallback): {e}")
        return hits

    # relevant=1 → 0, relevant=0 → 1 로 정렬 키 (오름차순이므로 relevant 먼저)
    # 동점은 원래 인덱스(enumerate 순서) 유지
    indexed = list(enumerate(hits, start=1))
    indexed.sort(key=lambda x: (1 - relevant_map.get(x[0], 0), x[0]))

    return [h for _, h in indexed]
