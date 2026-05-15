"""LLM 후속 QA 모듈.

추천 결과를 받은 사용자가 이어서 던지는 후속 질문(재료/조리법/영양 등)에
대해 GPT 모델로 답변을 생성한다.
validation + 재시도 1회 + fallback + JSONL 로깅을 모두 처리한다.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from rag.config import (
    OPENAI_API_KEY,
    RAG_QA_MODEL,
    RAG_QA_MAX_TOKENS,
    RAG_QA_REASONING_EFFORT,
    RAG_QA_RETRY_LIMIT,
    QA_MAX_DOCS,
    LOG_DIR,
    QA_LOG_FILE_PATTERN,
)
from rag.qa_prompt import (
    SYSTEM_PROMPT,
    QA_DOC_FIELDS,
    build_qa_user_prompt,
    QAResponse,
)


# ── 모듈 상수 ─────────────────────────────────────────────────────────────────

_EMPTY_DOCS_ANSWER = (
    "아직 추천된 레시피가 없어 답변드릴 자료가 없습니다."
)
_FALLBACK_ANSWER = (
    "죄송합니다. 일시적인 문제로 답변을 생성하지 못했습니다. "
    "잠시 후 다시 시도해 주세요."
)


# ── 로거 ──────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── OpenAI 클라이언트 (lazy) ──────────────────────────────────────────────────

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


# ── Helper: 문서 ID 추출 ─────────────────────────────────────────────────────

def _doc_id(doc: dict) -> str:
    """문서에서 recipe_id를 가능한 한 안정적으로 뽑아낸다."""
    rid = doc.get("recipe_id")
    if rid is None:
        rid = doc.get("rcp_seq")
    return str(rid) if rid is not None else ""


# ── Helper: fallback 응답 ────────────────────────────────────────────────────

def _build_fallback_response() -> QAResponse:
    """LLM 호출/검증 실패 시 사용할 표준 fallback 응답."""
    return QAResponse(
        answer=_FALLBACK_ANSWER,
        used_fields=[],
        refused=False,
        out_of_scope=False,
        qa_failed=True,
        is_fallback=True,
    )


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_response(response: QAResponse | None) -> tuple[bool, list[str]]:
    """LLM 응답을 strict 검증. 실패 시 (False, errors)."""
    errors: list[str] = []

    if response is None:
        errors.append("parsed=None")
        return False, errors

    if not isinstance(response.answer, str) or not response.answer.strip():
        errors.append("answer 비어 있음")

    if not isinstance(response.used_fields, list) or not all(
        isinstance(x, str) for x in response.used_fields
    ):
        errors.append("used_fields가 문자열 리스트가 아님")

    if not isinstance(response.refused, bool):
        errors.append("refused가 bool이 아님")

    if not isinstance(response.out_of_scope, bool):
        errors.append("out_of_scope가 bool이 아님")

    return (len(errors) == 0, errors)


# ── 로깅 ──────────────────────────────────────────────────────────────────────

def _log_to_jsonl(log_entry: dict) -> None:
    """logs/qa_YYYYMMDD.jsonl 에 한 줄 append. IO 실패는 warn만."""
    try:
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        filename = QA_LOG_FILE_PATTERN.format(
            date=datetime.now().strftime("%Y%m%d")
        )
        path = log_dir / filename

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"qa JSONL 로그 기록 실패: {e}")


def _log_summary(
    query: str,
    doc_count: int,
    refused: bool,
    qa_failed: bool,
    latency_ms: int,
) -> None:
    """한 줄 요약을 logger.info로 출력."""
    q = query if query is not None else ""
    q_display = (q[:30] + "...") if len(q) > 30 else q
    logger.info(
        f"qa | query='{q_display}' | docs={doc_count} "
        f"| refused={refused} | failed={qa_failed} | latency={latency_ms}ms"
    )


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

async def _call_llm(system_prompt: str, user_prompt: str) -> QAResponse:
    """OpenAI Structured Output 호출. 실패 시 예외 전파."""
    client = _get_client()
    response = await client.beta.chat.completions.parse(
        model=RAG_QA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format=QAResponse,
        max_completion_tokens=RAG_QA_MAX_TOKENS,
        reasoning_effort=RAG_QA_REASONING_EFFORT,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM 응답 parsed=None")
    return parsed


# ── 메인 엔트리 ───────────────────────────────────────────────────────────────

async def answer(
    query: str,
    retrieved_docs: list[dict],
    chat_history: list[dict],
) -> QAResponse:
    """QA 메인. 빈 문서 / 호출 실패 / validation 실패 시 fallback 처리."""
    start_ts = time.perf_counter()

    safe_query   = query          if query          is not None else ""
    safe_docs    = retrieved_docs if retrieved_docs is not None else []
    safe_history = chat_history   if chat_history   is not None else []

    # 1. 상위 QA_MAX_DOCS 슬라이싱
    docs = safe_docs[:QA_MAX_DOCS]
    doc_ids = [_doc_id(d) for d in docs]

    retry_count       = 0
    qa_failed         = False
    validation_errors: list[str] = []
    error_detail:     str | None = None

    # 2. 빈 문서 → LLM 스킵, 안내 응답 즉시 반환
    if not docs:
        final = QAResponse(
            answer=_EMPTY_DOCS_ANSWER,
            used_fields=[],
            refused=False,
            out_of_scope=False,
            qa_failed=False,
            is_fallback=False,
        )
        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        _log_to_jsonl({
            "timestamp":           datetime.now().isoformat(),
            "query":               safe_query,
            "retrieved_doc_ids":   doc_ids,
            "chat_history_length": len(safe_history),
            "model":               RAG_QA_MODEL,
            "latency_ms":          latency_ms,
            "retry_count":         0,
            "answer":              final.answer,
            "used_fields":         final.used_fields,
            "refused":             final.refused,
            "out_of_scope":        final.out_of_scope,
            "qa_failed":           final.qa_failed,
            "is_fallback":         final.is_fallback,
            "skip_reason":         "empty_docs",
        })
        _log_summary(safe_query, 0, False, False, latency_ms)
        return final

    # 3. 유저 프롬프트 빌드
    user_prompt_base = build_qa_user_prompt(safe_query, docs, safe_history)

    # 4. LLM 호출 (1차 + 최대 RAG_QA_RETRY_LIMIT 회 재시도)
    parsed: QAResponse | None = None
    last_errors: list[str] = []
    user_prompt = user_prompt_base

    max_attempts = RAG_QA_RETRY_LIMIT + 1
    for attempt in range(max_attempts):
        try:
            candidate = await _call_llm(SYSTEM_PROMPT, user_prompt)
            ok, errors = _validate_response(candidate)
            if ok:
                parsed = candidate
                last_errors = []
                break

            last_errors = errors
            logger.warning(
                f"qa validation 실패 (attempt={attempt + 1}): {errors}"
            )
        except Exception as e:
            last_errors = [f"LLM 호출 예외: {e}"]
            error_detail = str(e)
            logger.warning(
                f"qa LLM 호출 실패 (attempt={attempt + 1}): {e}"
            )

        # 재시도 여지가 남았으면 프롬프트 보강
        if attempt < max_attempts - 1:
            retry_count += 1
            error_block = "\n".join(f"- {m}" for m in last_errors)
            user_prompt = (
                f"{user_prompt_base}\n\n"
                f"[이전 응답 오류]\n{error_block}\n"
                f"수정해서 다시 응답하시오."
            )

    # 5. 최종 처리
    if parsed is None:
        qa_failed = True
        validation_errors = last_errors
        final = _build_fallback_response()
    else:
        # LLM이 임의로 채웠을 가능성 차단 — 시스템 필드는 강제 덮어쓰기
        final = QAResponse(
            answer=parsed.answer,
            used_fields=list(parsed.used_fields),
            refused=bool(parsed.refused),
            out_of_scope=bool(parsed.out_of_scope),
            qa_failed=False,
            is_fallback=False,
        )

    latency_ms = int((time.perf_counter() - start_ts) * 1000)

    # 6. 로깅
    log_entry: dict = {
        "timestamp":           datetime.now().isoformat(),
        "query":               safe_query,
        "retrieved_doc_ids":   doc_ids,
        "chat_history_length": len(safe_history),
        "model":               RAG_QA_MODEL,
        "latency_ms":          latency_ms,
        "retry_count":         retry_count,
        "answer":              final.answer,
        "used_fields":         final.used_fields,
        "refused":             final.refused,
        "out_of_scope":        final.out_of_scope,
        "qa_failed":           final.qa_failed,
        "is_fallback":         final.is_fallback,
    }
    if validation_errors:
        log_entry["validation_errors"] = validation_errors
    if error_detail:
        log_entry["error"] = error_detail

    _log_to_jsonl(log_entry)
    _log_summary(safe_query, len(docs), final.refused, final.qa_failed, latency_ms)

    return final


# ── 셀프 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dummy_doc = {
        "recipe_id":      "104",
        "name":           "북엇국",
        "category":       "국",
        "summary":        "맑고 시원한 해장 국물",
        "manuals_text":   "1) 북어를 물에 불린다. 2) 무를 채 썬다. 3) 끓는 물에 함께 넣는다.",
        "cooking_time":   25,
        "difficulty":     "쉬움",
        "cooking_method": "끓이기",
        "main_ingredients": ["북어", "무", "달걀"],
        "ingredients":    "북어 50g, 무 100g, 달걀 1개, 소금 약간",
        "nutrition":      {"calories": 180, "protein": 14, "fat": 4, "carb": 12},
        "taste_tags":     ["담백한", "시원한"],
        "texture_tags":   ["부드러운"],
        "spicy_level":    1,
        "meal_time":      ["아침"],
        "purpose":        ["light"],
        "recommended_situations": ["해장"],
        "dish_type_tags": ["국물요리"],
    }

    dummy_history = [
        {"role": "user",      "content": "저녁에 먹을 안 매운 국물 요리 추천해줘"},
        {"role": "assistant", "content": "북엇국, 콩나물국, 미역국 등을 추천드립니다."},
    ]

    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    def _print_case(title: str, response: QAResponse) -> None:
        print("=" * 60)
        print(title)
        print("=" * 60)
        print(response.model_dump_json(indent=2))
        print()

    async def main():
        # 케이스 2: 빈 retrieved_docs → LLM 호출 없음, 무조건 실행
        result_empty = await answer(
            "그거 어떻게 만들어?",
            [],
            dummy_history,
        )
        _print_case("[Case 2] 빈 retrieved_docs", result_empty)

        # 케이스 1: 정상
        if has_api_key:
            result_ok = await answer(
                "이거 재료 뭐야?",
                [dummy_doc],
                dummy_history,
            )
            _print_case("[Case 1] 정상 케이스 (LLM 호출)", result_ok)
        else:
            print("=" * 60)
            print("[Case 1] 정상 케이스 — SKIP (OPENAI_API_KEY 미설정)")
            print("=" * 60)
            print()

        # 케이스 3: 거부 카테고리
        if has_api_key:
            result_refused = await answer(
                "이거 다이어트에 좋아?",
                [dummy_doc],
                dummy_history,
            )
            _print_case("[Case 3] 거부 카테고리 (LLM 호출)", result_refused)
        else:
            print("=" * 60)
            print("[Case 3] 거부 카테고리 — SKIP (OPENAI_API_KEY 미설정)")
            print("=" * 60)
            print()

    asyncio.run(main())