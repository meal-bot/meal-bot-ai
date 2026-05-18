"""의도 분류기 v0.3. gpt-5-mini, JSON 강제, lazy OpenAI 클라이언트.

docs/prompts/intent-v0.3.md 명세 구현.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from api.errors import IntentClassifyError
from api.prompts.intent_prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    format_history,
)


logger = logging.getLogger(__name__)


_INTENT_VALUES = {"recommend", "slot_fill", "refine", "ask", "out_of_scope"}


class IntentResult(BaseModel):
    """의도 분류 결과. orchestrator가 분기 처리에 사용."""

    intent: Literal["recommend", "slot_fill", "refine", "ask", "out_of_scope"]
    reason: str = Field(min_length=1, max_length=200)


# ── OpenAI client lazy init (rag/reranker.py, rag/qa.py 패턴) ─────────────

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise IntentClassifyError("OPENAI_API_KEY not set")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


# ── 분류 함수 ─────────────────────────────────────────────────────────────


async def classify_intent(
    message: str,
    history: list,
    slots,
    has_last_recs: bool,
    previous_assistant_question: str | None,
) -> IntentResult:
    """사용자 발화를 5라벨 중 하나로 분류한다.

    Parameters
    ----------
    message : str
        이번 턴 사용자 발화
    history : list
        list[ChatMessage] 또는 list[dict], 최근 6개 메시지
    slots
        Slots Pydantic 객체 (meal_times, purpose, free_text 필드 보유)
    has_last_recs : bool
        직전 턴 추천 결과 존재 여부
    previous_assistant_question : str | None
        직전 assistant 발화가 슬롯 질문이면 그 문자열, 아니면 None
        판단 책임은 orchestrator에 있음

    Returns
    -------
    IntentResult
        intent + reason

    Raises
    ------
    IntentClassifyError
        JSON 파싱 실패, enum 외 값, 타임아웃, API 호출 실패 등
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        history_formatted=format_history(history),
        meal_times=slots.meal_times if slots.meal_times else "null",
        purpose=slots.purpose if slots.purpose else "null",
        free_text=slots.free_text if slots.free_text else "null",
        has_last_recs=str(has_last_recs).lower(),
        previous_assistant_question=previous_assistant_question or "null",
        message=message,
    )

    client = _get_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-5-mini",
            max_completion_tokens=2000,
            reasoning_effort="minimal",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            timeout=5.0,
        )
    except Exception as e:
        logger.warning("intent.classify_intent: LLM call failed: %s", e)
        raise IntentClassifyError(f"LLM call failed: {e}") from e

    raw = response.choices[0].message.content
    if not raw:
        raise IntentClassifyError("LLM returned empty content")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise IntentClassifyError(f"JSON parse failed: {e}, raw={raw[:200]}") from e

    if not isinstance(parsed, dict):
        raise IntentClassifyError(f"Expected dict, got {type(parsed).__name__}")

    intent_value = parsed.get("intent")
    if intent_value not in _INTENT_VALUES:
        raise IntentClassifyError(f"Invalid intent value: {intent_value!r}")

    try:
        result = IntentResult(intent=intent_value, reason=parsed.get("reason", ""))
    except ValidationError as e:
        raise IntentClassifyError(f"Schema validation failed: {e}") from e

    # reason 길이 위반은 경고만, 결과는 사용
    reason_len = len(result.reason)
    if reason_len < 20 or reason_len > 80:
        logger.warning(
            "intent.classify_intent: reason length out of expected range: %d chars (expected 20~80)",
            reason_len,
        )

    return result