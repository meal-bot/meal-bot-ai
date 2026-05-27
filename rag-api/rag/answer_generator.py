"""recommend/refine intent의 answer 본문을 LLM으로 생성한다.

환각 통제는 prompt + structured output + 호출부 검증의 3중 방어로 구성된다.
이 모듈은 단일 LLM 호출 + 응답 검증 + JSONL 로깅까지 담당하고,
실패 시 도메인 예외를 raise한다. fallback 텍스트 결정은 호출부 책임.

rag/qa.py의 호출 패턴(lazy AsyncOpenAI client, beta.parse Structured Output,
JSONL 로깅)을 미러하되 retry loop는 두지 않는다. answer 본문은 추천 결과 위에
얹는 도입부 한두 문장이라 사용자 체감 latency 비용이 크고, 환각을 retry로 잡는
대신 호출부의 결정론적 fallback(이전 템플릿)으로 강등하는 편이 안전하다고 판단.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI

from api.schemas import Recommendation, Slots
from rag.answer_prompt import SYSTEM_PROMPT, AnswerResponse, build_user_prompt
from rag.config import (
    ANSWER_LOG_FILE_PATTERN,
    ANSWER_MAX_COMPLETION_TOKENS,
    ANSWER_MODEL,
    ANSWER_REASONING_EFFORT,
    ANSWER_TIMEOUT_SECONDS,
    LOG_DIR,
    OPENAI_API_KEY,
)


logger = logging.getLogger(__name__)


# ── 도메인 예외 ──────────────────────────────────────────────────────────────

class AnswerGeneratorError(Exception):
    """answer generator 일반 실패. LLM 호출 자체가 깨졌거나 parsed=None."""


class AnswerTimeoutError(AnswerGeneratorError):
    """ANSWER_TIMEOUT_SECONDS 초과."""


class AnswerValidationError(AnswerGeneratorError):
    """응답 검증 실패. used_recipe_ids 환각 또는 answer 빈 문자열."""


# ── OpenAI 클라이언트 (lazy, rag/qa.py 패턴 미러) ────────────────────────────

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """AsyncOpenAI 클라이언트 lazy 초기화. API key 미설정 시 RuntimeError."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY 미설정. .env 또는 환경변수에서 설정 필요."
            )
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


# ── 로깅 헬퍼 (rag/qa.py 패턴 미러) ──────────────────────────────────────────

def _build_log_path() -> Path:
    """logs/answer_YYYYMMDD.jsonl 경로 반환. 디렉터리 보장."""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = ANSWER_LOG_FILE_PATTERN.format(
        date=datetime.now().strftime("%Y%m%d")
    )
    return log_dir / filename


def _log_answer_jsonl(log_entry: dict) -> None:
    """답변 생성 로그 한 줄 append. IO 실패는 warn만 (응답 흐름 보호)."""
    try:
        path = _build_log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("answer JSONL 로그 기록 실패: %s", e)


def _build_base_log_entry(
    *,
    intent: str,
    user_query: str,
    slots: Slots,
    free_text_delta: str | None,
    recommendations: list[Recommendation],
    previously_recommended_names: list[str] | None,
    elapsed_seconds: float,
) -> dict:
    """성공/실패 양쪽에서 공통 사용하는 로그 베이스."""
    return {
        "timestamp": datetime.now().isoformat(),
        "intent": intent,
        "user_query": user_query,
        "slots_snapshot": slots.model_dump(),
        "free_text_delta": free_text_delta,
        "recommendation_ids": [rec.recipe_id for rec in recommendations],
        "previously_recommended_names": previously_recommended_names,
        "elapsed_seconds": elapsed_seconds,
    }


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

async def _call_llm(system_prompt: str, user_prompt: str) -> AnswerResponse:
    """OpenAI Structured Output 호출. 실패 시 예외 전파."""
    client = _get_client()
    response = await client.beta.chat.completions.parse(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format=AnswerResponse,
        max_completion_tokens=ANSWER_MAX_COMPLETION_TOKENS,
        reasoning_effort=ANSWER_REASONING_EFFORT,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise AnswerGeneratorError("LLM 응답 parsed=None")
    return parsed


# ── 메인 엔트리 ──────────────────────────────────────────────────────────────

async def generate_answer(
    *,
    intent: Literal["recommend", "refine"],
    user_query: str,
    slots: Slots,
    free_text_delta: str | None,
    recommendations: list[Recommendation],
    recipe_details: list[dict],
    previously_recommended_names: list[str] | None = None,
) -> AnswerResponse:
    """answer 본문 생성. 실패 시 도메인 예외 raise (호출부에서 fallback 결정).

    Args:
        intent: "recommend" 또는 "refine".
        user_query: 이번 턴 사용자 원문 발화.
        slots: 누적 슬롯 스냅샷.
        free_text_delta: 이번 턴 새로 추출된 자유 텍스트. 없으면 None.
        recommendations: 카드로 노출될 추천 결과. 1~2개.
        recipe_details: recommendations와 같은 순서·같은 길이의 RecipeStore
            lookup 결과 원본 dict 리스트.
        previously_recommended_names: refine intent일 때 직전 턴 추천 메뉴명.
            recommend intent면 None 또는 빈 리스트.

    Returns:
        AnswerResponse (answer, used_recipe_ids).

    Raises:
        ValueError: 입력 invariant 위반 (recommendations 길이 1~2 이탈,
            recommendations와 recipe_details 길이 불일치).
        AnswerTimeoutError: ANSWER_TIMEOUT_SECONDS 초과.
        AnswerValidationError: used_recipe_ids가 recommendations 밖의 id를
            포함하는 경우, 또는 answer가 빈 문자열인 경우.
        AnswerGeneratorError: 그 외 LLM 호출 실패.
    """
    # 1. 입력 검증
    rec_count = len(recommendations)
    if rec_count < 1 or rec_count > 2:
        raise ValueError(
            f"recommendations 길이는 1~2여야 합니다. got={rec_count}"
        )
    if len(recipe_details) != rec_count:
        raise ValueError(
            "recommendations와 recipe_details 길이가 일치해야 합니다. "
            f"recommendations={rec_count}, recipe_details={len(recipe_details)}"
        )
    if intent == "refine" and not previously_recommended_names:
        logger.warning(
            "answer_generator: intent=refine인데 previously_recommended_names가 "
            "비어 있음. 호출부 실수 가능성 있음."
        )

    # 2. 프롬프트 조립
    user_prompt = build_user_prompt(
        intent=intent,
        user_query=user_query,
        slots=slots,
        free_text_delta=free_text_delta,
        recommendations=recommendations,
        recipe_details=recipe_details,
        previously_recommended_names=previously_recommended_names,
    )

    # 3. LLM 호출 (retry 없음, timeout 래핑)
    start_ts = time.perf_counter()
    try:
        parsed = await asyncio.wait_for(
            _call_llm(SYSTEM_PROMPT, user_prompt),
            timeout=ANSWER_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as e:
        elapsed = time.perf_counter() - start_ts
        _log_answer_jsonl({
            **_build_base_log_entry(
                intent=intent,
                user_query=user_query,
                slots=slots,
                free_text_delta=free_text_delta,
                recommendations=recommendations,
                previously_recommended_names=previously_recommended_names,
                elapsed_seconds=elapsed,
            ),
            "answer": None,
            "used_recipe_ids": None,
            "status": "error",
            "skip_reason": f"timeout after {ANSWER_TIMEOUT_SECONDS}s",
        })
        logger.warning(
            "answer_generator: timeout after %.1fs", ANSWER_TIMEOUT_SECONDS,
        )
        raise AnswerTimeoutError(
            f"timeout after {ANSWER_TIMEOUT_SECONDS}s"
        ) from e
    except AnswerGeneratorError as e:
        elapsed = time.perf_counter() - start_ts
        _log_answer_jsonl({
            **_build_base_log_entry(
                intent=intent,
                user_query=user_query,
                slots=slots,
                free_text_delta=free_text_delta,
                recommendations=recommendations,
                previously_recommended_names=previously_recommended_names,
                elapsed_seconds=elapsed,
            ),
            "answer": None,
            "used_recipe_ids": None,
            "status": "error",
            "skip_reason": f"llm_error: {e}",
        })
        logger.warning("answer_generator: LLM 호출 실패: %s", e)
        raise
    except Exception as e:
        elapsed = time.perf_counter() - start_ts
        _log_answer_jsonl({
            **_build_base_log_entry(
                intent=intent,
                user_query=user_query,
                slots=slots,
                free_text_delta=free_text_delta,
                recommendations=recommendations,
                previously_recommended_names=previously_recommended_names,
                elapsed_seconds=elapsed,
            ),
            "answer": None,
            "used_recipe_ids": None,
            "status": "error",
            "skip_reason": f"llm_error: {e}",
        })
        logger.warning("answer_generator: LLM 호출 실패: %s", e)
        raise AnswerGeneratorError(f"LLM call failed: {e}") from e

    elapsed = time.perf_counter() - start_ts

    # 4. 응답 검증 (Pydantic Field 검증 외 추가 검증)
    allowed_ids = {rec.recipe_id for rec in recommendations}
    invalid_ids = [
        rid for rid in parsed.used_recipe_ids if rid not in allowed_ids
    ]
    if invalid_ids:
        _log_answer_jsonl({
            **_build_base_log_entry(
                intent=intent,
                user_query=user_query,
                slots=slots,
                free_text_delta=free_text_delta,
                recommendations=recommendations,
                previously_recommended_names=previously_recommended_names,
                elapsed_seconds=elapsed,
            ),
            "answer": parsed.answer,
            "used_recipe_ids": list(parsed.used_recipe_ids),
            "status": "error",
            "skip_reason": f"invalid_used_recipe_ids: {invalid_ids}",
        })
        raise AnswerValidationError(
            f"used_recipe_ids에 recommendations 밖의 id가 포함됨: {invalid_ids}"
        )

    if not parsed.answer.strip():
        _log_answer_jsonl({
            **_build_base_log_entry(
                intent=intent,
                user_query=user_query,
                slots=slots,
                free_text_delta=free_text_delta,
                recommendations=recommendations,
                previously_recommended_names=previously_recommended_names,
                elapsed_seconds=elapsed,
            ),
            "answer": parsed.answer,
            "used_recipe_ids": list(parsed.used_recipe_ids),
            "status": "error",
            "skip_reason": "empty_answer_after_strip",
        })
        raise AnswerValidationError("answer가 빈 문자열 (strip 후)")

    # 5. 정상 로깅 + 반환
    _log_answer_jsonl({
        **_build_base_log_entry(
            intent=intent,
            user_query=user_query,
            slots=slots,
            free_text_delta=free_text_delta,
            recommendations=recommendations,
            previously_recommended_names=previously_recommended_names,
            elapsed_seconds=elapsed,
        ),
        "answer": parsed.answer,
        "used_recipe_ids": list(parsed.used_recipe_ids),
        "status": "ok",
        "skip_reason": None,
    })

    return parsed


__all__ = [
    "generate_answer",
    "AnswerGeneratorError",
    "AnswerTimeoutError",
    "AnswerValidationError",
]
