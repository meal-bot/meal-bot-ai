"""슬롯 추출기 v0.3. gpt-5-mini, JSON 강제, delta 반환.

docs/prompts/slot-v0.3.md 명세 구현.
빈 결과(모든 필드 None)는 정상이며, 누적/merge는 orchestrator의 책임이다.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from api.errors import SlotExtractError
from api.prompts.intent_prompt import format_history
from api.prompts.slot_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


logger = logging.getLogger(__name__)


_MEAL_TIME_VALUES = {"아침", "점심", "저녁", "간식", "야식"}
_PURPOSE_VALUES = {"light", "protein", "hearty", "tasty"}


class SlotDelta(BaseModel):
    """이번 턴에 추출된 슬롯 delta.

    None = 이번 턴에 추출되지 않음 (실패 아님).
    orchestrator가 기존 slots와 merge.
    """

    meal_times: list[Literal["아침", "점심", "저녁", "간식", "야식"]] | None = None
    purpose: Literal["light", "protein", "hearty", "tasty"] | None = None
    free_text_delta: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _normalize_empty_meal_times(self):
        """빈 배열은 None으로 정규화 (orchestrator merge 로직 단순화)."""
        if self.meal_times is not None and len(self.meal_times) == 0:
            self.meal_times = None
        return self


# ── OpenAI client lazy init (rag/ 패턴 동일) ──────────────────────────────

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SlotExtractError("OPENAI_API_KEY not set")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


# ── 추출 함수 ─────────────────────────────────────────────────────────────


async def extract_slots(message: str, history: list) -> SlotDelta:
    """이번 턴 사용자 발화에서 슬롯 delta를 추출한다.

    Parameters
    ----------
    message : str
        이번 턴 사용자 발화
    history : list
        list[ChatMessage] 또는 list[dict], 최근 6개 메시지 (맥락 참고용)

    Returns
    -------
    SlotDelta
        delta 형태. 모든 필드 None일 수 있음 (정상).

    Raises
    ------
    SlotExtractError
        JSON 파싱 실패, 필수 키 누락, enum 외 값, 타입 오류,
        타임아웃, API 호출 실패 등 (빈 결과는 실패 아님)
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        history_formatted=format_history(history),
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
        logger.warning("slot.extract_slots: LLM call failed: %s", e)
        raise SlotExtractError(f"LLM call failed: {e}") from e

    raw = response.choices[0].message.content
    if not raw:
        raise SlotExtractError("LLM returned empty content")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SlotExtractError(f"JSON parse failed: {e}, raw={raw[:200]}") from e

    if not isinstance(parsed, dict):
        raise SlotExtractError(f"Expected dict, got {type(parsed).__name__}")

    # 필수 키 누락 검증
    for key in ("meal_times", "purpose", "free_text_delta"):
        if key not in parsed:
            raise SlotExtractError(f"Missing required key: {key}")

    # meal_times 검증
    meal_times = parsed["meal_times"]
    if meal_times is not None:
        if not isinstance(meal_times, list):
            raise SlotExtractError(
                f"meal_times must be list or null, got {type(meal_times).__name__}"
            )
        for v in meal_times:
            if v not in _MEAL_TIME_VALUES:
                raise SlotExtractError(f"Invalid meal_times value: {v!r}")

    # purpose 검증
    purpose = parsed["purpose"]
    if purpose is not None and purpose not in _PURPOSE_VALUES:
        raise SlotExtractError(f"Invalid purpose value: {purpose!r}")

    # free_text_delta 검증
    free_text_delta = parsed["free_text_delta"]
    if free_text_delta is not None and not isinstance(free_text_delta, str):
        raise SlotExtractError(
            f"free_text_delta must be string or null, got {type(free_text_delta).__name__}"
        )

    try:
        result = SlotDelta(
            meal_times=meal_times,
            purpose=purpose,
            free_text_delta=free_text_delta,
        )
    except ValidationError as e:
        raise SlotExtractError(f"Schema validation failed: {e}") from e

    return result