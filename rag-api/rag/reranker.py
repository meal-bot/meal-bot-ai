"""LLM rerank 모듈.

Hybrid top-30 후보를 받아 GPT-5-mini로 top-5를 선정한다.
validation + 재시도 1회 + fallback + JSONL 로깅까지 모두 처리한다.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from rag.config import (
    OPENAI_API_KEY,
    RERANK_MODEL,
    RERANK_REASONING_EFFORT,
    RERANK_MAX_COMPLETION_TOKENS,
    RERANK_TOP_K_OUTPUT,
    RERANK_MIN_CANDIDATES,
    RERANK_RETRY_LIMIT,
    LOG_DIR,
    RERANK_LOG_FILE_PATTERN,
)
from rag.rerank_prompt import (
    SYSTEM_PROMPT,
    ALLOWED_INTENTS,
    build_user_prompt,
    RerankResponse,
    RerankItem,
)


# ── 모듈 상수 ─────────────────────────────────────────────────────────────────

SOFT_REASON_MIN_LEN = 40
SOFT_REASON_MAX_LEN = 130

_ALLOWED_INTENT_SET = set(ALLOWED_INTENTS)

# 후보 정제 시 사용할 필드 화이트리스트
_PROMPT_FIELDS_LIST: tuple[str, ...] = (
    "main_ingredients", "meal_time", "purpose",
    "taste_tags", "texture_tags",
    "recommended_situations", "dish_type_tags",
)
_PROMPT_FIELDS_STR: tuple[str, ...] = (
    "name", "category", "cooking_method", "summary", "difficulty",
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


# ── Helper: 후보 정제 ─────────────────────────────────────────────────────────

def _candidate_to_prompt_dict(candidate: dict) -> dict:
    """hybrid 후보 dict에서 LLM 프롬프트에 넣을 18개 필드만 추출."""
    # recipe_id / rcp_seq 정규화
    recipe_id = candidate.get("recipe_id")
    if recipe_id is None:
        recipe_id = candidate.get("rcp_seq")
    recipe_id = str(recipe_id) if recipe_id is not None else ""

    out: dict = {"recipe_id": recipe_id}

    for key in _PROMPT_FIELDS_STR:
        out[key] = candidate.get(key) or ""

    for key in _PROMPT_FIELDS_LIST:
        value = candidate.get(key)
        out[key] = value if isinstance(value, list) else []

    out["cooking_time"] = candidate.get("cooking_time")
    out["spicy_level"]  = candidate.get("spicy_level")
    out["dense_rank"]   = candidate.get("dense_rank")
    out["bm25_rank"]    = candidate.get("bm25_rank")
    out["rrf_score"]    = candidate.get("rrf_score")

    return out


# ── Helper: fallback 응답 ─────────────────────────────────────────────────────

_FALLBACK_REASON = "유사한 후보로 추천되었습니다."
_FALLBACK_INTENTS = ["재료_일치"]


def _build_fallback_response(candidates: list[dict], top_k: int) -> RerankResponse:
    """LLM 호출이 불가/완전 실패한 경우 hybrid 순위 그대로 응답 생성."""
    picked = candidates[:top_k]

    items: list[RerankItem] = []
    for i, c in enumerate(picked):
        rid = c.get("recipe_id") or c.get("rcp_seq")
        items.append(RerankItem(
            rank=i + 1,
            recipe_id=str(rid) if rid is not None else "",
            reason=_FALLBACK_REASON,
            matched_intents=list(_FALLBACK_INTENTS),
        ))

    return RerankResponse(
        recommendations=items,
        insufficient_matches=len(items) < top_k,
    )


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_response(
    response: RerankResponse,
    candidate_ids: set[str],
    top_k: int = RERANK_TOP_K_OUTPUT,
) -> tuple[bool, list[str]]:
    """LLM 응답을 strict 검증. 실패 시 (False, errors)."""
    errors: list[str] = []
    items = response.recommendations
    n = len(items)

    # 1. 후보 밖 recipe_id (환각)
    unknown_ids = [it.recipe_id for it in items if it.recipe_id not in candidate_ids]
    if unknown_ids:
        errors.append(f"후보 목록 밖 recipe_id: {unknown_ids}")

    # 2. recipe_id 중복
    seen_ids = {it.recipe_id for it in items}
    if len(seen_ids) != n:
        errors.append("recipe_id 중복 존재")

    # 3. rank 1..N 정확히 1번씩
    ranks_sorted = sorted(it.rank for it in items)
    if ranks_sorted != list(range(1, n + 1)):
        errors.append(f"rank 시퀀스 비정상: {ranks_sorted} (기대: 1~{n})")

    # 4. insufficient_matches=False인데 부족
    if not response.insufficient_matches and n != top_k:
        errors.append(
            f"insufficient_matches=False지만 추천 {n}개 (기대: {top_k}개)"
        )

    # 5. 빈 reason
    if any(not it.reason or not it.reason.strip() for it in items):
        errors.append("빈 reason 존재")

    # 6. matched_intents 길이 (Pydantic이 이미 1~4 강제하지만 안전망)
    for it in items:
        if not (1 <= len(it.matched_intents) <= 4):
            errors.append(f"matched_intents 길이 비정상 (rank={it.rank})")

    return (len(errors) == 0, errors)


def _soft_validate(response: RerankResponse) -> list[str]:
    """Soft validation. 경고만 모아 반환 (실패 처리 X)."""
    warnings_: list[str] = []

    for it in response.recommendations:
        length = len(it.reason or "")
        if length < SOFT_REASON_MIN_LEN or length > SOFT_REASON_MAX_LEN:
            warnings_.append(
                f"rank={it.rank} reason 길이 {length}자 "
                f"(권장 {SOFT_REASON_MIN_LEN}~{SOFT_REASON_MAX_LEN})"
            )

        if it.matched_intents:
            unknown = [x for x in it.matched_intents if x not in _ALLOWED_INTENT_SET]
            ratio = len(unknown) / len(it.matched_intents)
            if ratio > 0.5:
                warnings_.append(
                    f"rank={it.rank} matched_intents 자유태그 비율 "
                    f"{ratio:.0%}: {unknown}"
                )

    return warnings_


def _fill_insufficient(
    response: RerankResponse,
    candidates: list[dict],
    top_k: int,
) -> RerankResponse:
    """insufficient_matches=False인데 부족한 비정상 케이스에 hybrid 순위로 보충."""
    if response.insufficient_matches:
        return response
    if len(response.recommendations) >= top_k:
        return response

    used_ids = {it.recipe_id for it in response.recommendations}
    items = list(response.recommendations)
    next_rank = len(items) + 1

    for c in candidates:
        if len(items) >= top_k:
            break
        rid = c.get("recipe_id") or c.get("rcp_seq")
        rid_str = str(rid) if rid is not None else ""
        if not rid_str or rid_str in used_ids:
            continue
        items.append(RerankItem(
            rank=next_rank,
            recipe_id=rid_str,
            reason=_FALLBACK_REASON,
            matched_intents=list(_FALLBACK_INTENTS),
        ))
        used_ids.add(rid_str)
        next_rank += 1

    return RerankResponse(
        recommendations=items,
        insufficient_matches=len(items) < top_k,
    )


# ── 로깅 ──────────────────────────────────────────────────────────────────────

def _log_to_jsonl(log_entry: dict) -> None:
    """logs/rerank_YYYYMMDD.jsonl 에 한 줄 append. IO 실패는 warn만."""
    try:
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        filename = RERANK_LOG_FILE_PATTERN.format(
            date=datetime.now().strftime("%Y%m%d")
        )
        path = log_dir / filename

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"rerank JSONL 로그 기록 실패: {e}")


def _log_summary(query: str, latency_ms: int, retry_count: int, rerank_failed: bool) -> None:
    """한 줄 요약을 logger.info로 출력."""
    q = query if query is not None else ""
    if len(q) > 30:
        q_display = q[:30] + "..."
    else:
        q_display = q
    logger.info(
        f"rerank | query='{q_display}' | latency={latency_ms}ms "
        f"| retries={retry_count} | failed={rerank_failed}"
    )


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

async def _call_llm(system_prompt: str, user_prompt: str) -> RerankResponse:
    """OpenAI gpt-5-mini Structured Output 호출. 실패 시 예외 전파."""
    client = _get_client()
    response = await client.beta.chat.completions.parse(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format=RerankResponse,
        max_completion_tokens=RERANK_MAX_COMPLETION_TOKENS,
        reasoning_effort=RERANK_REASONING_EFFORT,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM 응답 parsed=None")
    return parsed


# ── 메인 엔트리 ───────────────────────────────────────────────────────────────

async def rerank(
    query:             str,
    candidates:        list[dict],
    structured_inputs: dict | None = None,
    top_k:             int         = RERANK_TOP_K_OUTPUT,
) -> RerankResponse:
    """LLM rerank 메인. 후보 부족 / 호출 실패 / validation 실패 시 fallback 처리.

    structured_inputs는 user prompt 상단 [사용자 제약] 블록으로만 노출되며,
    rerank 로직 / validation / fallback 정책에는 영향을 주지 않는다.
    """
    start_ts = time.perf_counter()

    retry_count       = 0
    rerank_failed     = False
    validation_errors: list[str] = []
    soft_warnings:     list[str] = []
    llm_raw_response: dict | None = None

    # 1. 빈 후보 → 즉시 빈 응답
    if not candidates:
        final = RerankResponse(recommendations=[], insufficient_matches=True)
        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        _log_to_jsonl({
            "ts":                 datetime.now().isoformat(),
            "query":              query,
            "structured_inputs":  structured_inputs,
            "hybrid_top30_ids":   [],
            "llm_raw_response":   None,
            "validation_errors":  [],
            "soft_warnings":      [],
            "retry_count":        0,
            "rerank_failed":      False,
            "latency_ms":         latency_ms,
            "final_response":     final.model_dump(),
            "skip_reason":        "empty_candidates",
        })
        _log_summary(query, latency_ms, 0, False)
        return final

    # 2. 후보 정제 (LLM 호출 여부와 무관하게 한 번 정제)
    prompt_candidates = [_candidate_to_prompt_dict(c) for c in candidates]
    candidate_ids = {c["recipe_id"] for c in prompt_candidates if c["recipe_id"]}
    hybrid_ids = [c["recipe_id"] for c in prompt_candidates]

    # 3. 후보 부족 → LLM 스킵, fallback
    if len(candidates) < RERANK_MIN_CANDIDATES:
        final = _build_fallback_response(prompt_candidates, top_k)
        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        _log_to_jsonl({
            "ts":                 datetime.now().isoformat(),
            "query":              query,
            "structured_inputs":  structured_inputs,
            "hybrid_top30_ids":   hybrid_ids,
            "llm_raw_response":   None,
            "validation_errors":  [],
            "soft_warnings":      [],
            "retry_count":        0,
            "rerank_failed":      False,
            "latency_ms":         latency_ms,
            "final_response":     final.model_dump(),
            "skip_reason":        f"candidates<{RERANK_MIN_CANDIDATES}",
        })
        _log_summary(query, latency_ms, 0, False)
        return final

    # 4. 유저 프롬프트 빌드 (structured_inputs는 [사용자 제약] 블록으로만 노출)
    user_prompt_base = build_user_prompt(
        query, prompt_candidates, structured_inputs,
    )

    # 5. LLM 호출 (1차 + 최대 RERANK_RETRY_LIMIT 회 재시도)
    parsed: RerankResponse | None = None
    last_errors: list[str] = []
    user_prompt = user_prompt_base

    max_attempts = RERANK_RETRY_LIMIT + 1
    for attempt in range(max_attempts):
        try:
            candidate_response = await _call_llm(SYSTEM_PROMPT, user_prompt)
            llm_raw_response = candidate_response.model_dump()

            ok, errors = _validate_response(
                candidate_response, candidate_ids, top_k,
            )
            if ok:
                parsed = candidate_response
                last_errors = []
                break

            last_errors = errors
            logger.warning(
                f"rerank validation 실패 (attempt={attempt + 1}): {errors}"
            )
        except Exception as e:
            last_errors = [f"LLM 호출 예외: {e}"]
            logger.warning(
                f"rerank LLM 호출 실패 (attempt={attempt + 1}): {e}"
            )

        # 재시도 여지가 남았으면 프롬프트 보강 후 retry
        if attempt < max_attempts - 1:
            retry_count += 1
            error_block = "\n".join(f"- {m}" for m in last_errors)
            user_prompt = (
                f"{user_prompt_base}\n\n"
                f"[이전 응답 오류]\n{error_block}\n"
                f"수정해서 다시 응답하시오."
            )

    # 6. 최종 처리
    if parsed is None:
        rerank_failed = True
        validation_errors = last_errors
        final = _build_fallback_response(prompt_candidates, top_k)
    else:
        validation_errors = []
        final = _fill_insufficient(parsed, prompt_candidates, top_k)
        soft_warnings = _soft_validate(final)

    latency_ms = int((time.perf_counter() - start_ts) * 1000)

    # 7. 로깅
    _log_to_jsonl({
        "ts":                 datetime.now().isoformat(),
        "query":              query,
        "structured_inputs":  structured_inputs,
        "hybrid_top30_ids":   hybrid_ids,
        "llm_raw_response":   llm_raw_response,
        "validation_errors":  validation_errors,
        "soft_warnings":      soft_warnings,
        "retry_count":        retry_count,
        "rerank_failed":      rerank_failed,
        "latency_ms":         latency_ms,
        "final_response":     final.model_dump(),
    })
    _log_summary(query, latency_ms, retry_count, rerank_failed)

    return final


# ── 셀프 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dummy_candidates = [
        {
            "recipe_id": "28",
            "name": "새우 두부 계란찜",
            "category": "반찬",
            "cooking_method": "찌기",
            "summary": "부드럽고 담백한 한 끼",
            "main_ingredients": ["연두부", "새우", "달걀"],
            "meal_time": ["아침", "저녁"],
            "purpose": ["light", "protein"],
            "cooking_time": 20,
            "spicy_level": 1,
            "difficulty": "쉬움",
            "taste_tags": ["담백한"],
            "texture_tags": ["부드러운"],
            "recommended_situations": ["혼밥"],
            "dish_type_tags": ["찜"],
            "dense_rank": 3,
            "bm25_rank": 7,
            "rrf_score": 0.0321,
        },
        {
            "recipe_id": "104",
            "name": "북엇국",
            "category": "국",
            "cooking_method": "끓이기",
            "summary": "맑고 시원한 해장 국물",
            "main_ingredients": ["북어", "무", "달걀"],
            "meal_time": ["아침"],
            "purpose": ["light"],
            "cooking_time": 25,
            "spicy_level": 1,
            "difficulty": "쉬움",
            "taste_tags": ["담백한", "시원한"],
            "texture_tags": ["부드러운"],
            "recommended_situations": ["해장"],
            "dish_type_tags": ["국물요리"],
            "dense_rank": 1,
            "bm25_rank": 2,
            "rrf_score": 0.0492,
        },
    ]

    async def main():
        result = await rerank("안 매운 국물 요리", dummy_candidates)
        print("=" * 60)
        print("Rerank 결과 (후보 2건 → LLM 스킵 → fallback 경로)")
        print("=" * 60)
        print(result.model_dump_json(indent=2))

    asyncio.run(main())