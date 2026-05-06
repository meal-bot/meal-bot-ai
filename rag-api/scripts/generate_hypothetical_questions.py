"""
generate_hypothetical_questions.py

레시피 1146건에 대해 GPT-5-mini로 가상 사용자 질문 3개씩 생성.
data/recipes_hyp_questions.json 에 저장 (100건 단위 partial flush, --resume 지원).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _enrichment_common import PRICING

BASE_DIR = Path(__file__).resolve().parent.parent
ENRICHED_PATH = BASE_DIR / "data" / "recipes_enriched.json"
CLEANED_PATH = BASE_DIR / "data" / "recipes_cleaned.json"
OUTPUT_PATH = BASE_DIR / "data" / "recipes_hyp_questions.json"
FAILED_PATH = BASE_DIR / "data" / "hyp_failed.json"

MODEL = "gpt-5-mini"
FLUSH_EVERY = 100
MAX_QUESTION_LEN = 50
MAX_CONCURRENCY = 10
EST_INPUT_TOKENS_PER_CALL = 1100   # 실측 기반 (2026-05 limit 10 호출 평균)
EST_OUTPUT_TOKENS_PER_CALL = 1850  # reasoning_effort=default 기준. minimal로 낮추면 줄어듦

PERSONAS_TEXT = """P1. 1인 가구: 자취, 피곤, 퇴근, 간단, 야식
P2. 다이어트: 저칼로리, 체중, 가벼운 한식
P3. 운동: 고단백, 단백질, 운동 후
P4. 가정요리: 워킹맘, 가족식사, 아이도 먹는 반찬
P5. 요리 초보: 쉬운 요리, 실패 확률 낮음, 재료 적음
P6. 건강관리: 저염, 속 편함, 자극적이지 않음, 시니어/부모님
P7. 술안주: 맥주, 소주, 간단한 안주
P8. 손님 초대: 집들이, 손님상, 있어 보이는 메인요리"""

SYSTEM_PROMPT = f"""당신은 한국어 레시피 검색 시스템에 사용할 가상 사용자 질문을 생성하는 전문가입니다.
사용자가 실제 검색창/챗봇에 입력할 법한 짧고 자연스러운 질문 3개를 만들어야 합니다.

# 작성 규칙
1. 정확히 3개를 생성합니다.

2. 길이: **10~30자 권장**, 최대 50자. 짧고 압축된 자연어를 우선합니다.

3. 질문 형식 우선순위
   - 우선(기본 형식): **추천 요청형** — "...추천해줘", "...메뉴", "...레시피" 등.
   - 허용(3개 중 **최대 1개**까지만): 확인형 — "...있어?", "...괜찮을까?" 등.
   - 금지: **모든 질문이 확인형**인 경우.
   - 지양 예시: "이 오이무침 칼로리 낮아?", "샐러드 괜찮을까?"
   - 권장 예시: "다이어트 중 먹기 좋은 오이무침 추천해줘",
                "손님상에 올릴 상큼한 샐러드 추천해줘"

4. 너무 일반적인 질문 금지 (예: "맛있는 음식 추천", "건강한 메뉴").

5. 레시피의 주재료/맛/식감/조리방식/상황 중 일부가 자연스럽게 드러나야 합니다.

6. 페르소나 매칭
   - 페르소나 8개(P1~P8) 중 이 레시피에 **자연스럽게** 어울리는 것만 반영.
   - 1~3개 중 자연스러운 만큼만 사용. **3개 채우려고 억지로 늘리지 마세요.**
   - 1개 또는 2개로 충분하면 그만큼만 사용.
   - personas_hinted에는 실제 반영한 것만 표시.

7. **건강 키워드 수치 조건 (엄격)**
   다음 키워드는 입력된 영양 수치가 조건을 충족할 때만 사용 가능. 조건 불충족 시 사용 금지.
   - "저염" → sodium ≤ 500mg
   - "고단백" / "단백질 많은" → protein ≥ 15g
   - "저칼로리" / "다이어트" → calories ≤ 400kcal

8. 출력은 반드시 JSON 형식:
   {{"questions": ["...", "...", "..."], "personas_hinted": ["P3", ...]}}
   비어있는 personas_hinted는 [] 로.

# 페르소나
{PERSONAS_TEXT}
"""


def build_user_prompt(rcp_seq: str, name: str, enriched: dict, nutrition: dict) -> str:
    parts = [
        f"## 레시피 정보 (rcp_seq={rcp_seq})",
        f"- 이름: {name}",
        f"- 카테고리: {enriched.get('category')}",
        f"- 조리법: {enriched.get('cooking_method')}",
        f"- 요약: {enriched.get('summary')}",
        f"- 맛: {enriched.get('taste_tags')}",
        f"- 식감: {enriched.get('texture_tags')}",
        f"- 주재료: {enriched.get('main_ingredients')}",
        f"- 추천 상황: {enriched.get('recommended_situations')}",
        f"- 식사유형: {enriched.get('meal_type_tags')}",
        f"- 조리시간: {enriched.get('cooking_time')}분",
        f"- 난이도: {enriched.get('difficulty')}/3",
        f"- 매운맛: {enriched.get('spicy_level')}/5",
        f"- 칼로리: {nutrition.get('energy_kcal')}kcal",
        f"- 단백질: {nutrition.get('protein_g')}g",
        f"- 나트륨: {nutrition.get('sodium_mg')}mg",
        "",
        "위 레시피에 어울리는 가상 사용자 질문 3개를 JSON으로 생성해주세요.",
    ]
    return "\n".join(parts)


# --- 비즈니스 룰 검증 (사후 점검; retry 트리거 아님) -------------------------
RECOMMEND_KEYWORDS = ("추천", "메뉴", "레시피", "알려")
CONFIRM_ENDINGS = ("있어?", "있을까?", "괜찮을까?", "괜찮나?", "어때?", "어떨까?",
                   "좋을까?", "좋아?", "할까?", "될까?", "되나?", "맞을까?",
                   "낮아?", "높아?", "맞아?", "있나?")


def is_confirmation_question(q: str) -> bool:
    """추천 키워드 없고 ?로 끝나거나 확인형 어미면 확인형으로 판정."""
    s = q.strip()
    if any(kw in s for kw in RECOMMEND_KEYWORDS):
        return False
    if s.endswith(CONFIRM_ENDINGS):
        return True
    return s.endswith("?")


def check_health_keyword_violations(q: str, nutrition: dict) -> list[str]:
    issues: list[str] = []
    cals = nutrition.get("energy_kcal")
    prot = nutrition.get("protein_g")
    sod = nutrition.get("sodium_mg")
    if "저염" in q and sod is not None and sod > 500:
        issues.append(f"'저염' but sodium={sod}>500")
    if ("고단백" in q or "단백질 많" in q) and prot is not None and prot < 15:
        issues.append(f"'고단백' but protein={prot}<15")
    if ("저칼로리" in q or "다이어트" in q) and cals is not None and cals > 400:
        issues.append(f"'저칼로리/다이어트' but calories={cals}>400")
    return issues


def validate_business_rules(questions: list[str], nutrition: dict) -> list[str]:
    """길이/확인형/건강키워드 룰 위반 사항 리스트로 반환. 빈 리스트면 OK."""
    issues: list[str] = []
    for i, q in enumerate(questions, 1):
        if len(q) > MAX_QUESTION_LEN:
            issues.append(f"q{i} length={len(q)}>{MAX_QUESTION_LEN}")
        for h in check_health_keyword_violations(q, nutrition):
            issues.append(f"q{i} {h}")
    confirm_count = sum(1 for q in questions if is_confirmation_question(q))
    if confirm_count >= 2:
        issues.append(f"confirmation_count={confirm_count}>=2")
    return issues


def validate_response(parsed: dict) -> tuple[list[str], list[str]]:
    if not isinstance(parsed, dict):
        raise ValueError("response not a dict")
    qs = parsed.get("questions")
    if not isinstance(qs, list) or len(qs) != 3:
        raise ValueError(f"questions count != 3: got {qs!r}")
    cleaned = []
    for q in qs:
        if not isinstance(q, str):
            raise ValueError(f"non-string question: {q!r}")
        s = q.strip()
        if not s:
            raise ValueError("empty question")
        if len(s) > MAX_QUESTION_LEN:
            raise ValueError(f"too long ({len(s)}>{MAX_QUESTION_LEN}): {s}")
        cleaned.append(s)
    personas = parsed.get("personas_hinted", [])
    if not isinstance(personas, list):
        personas = []
    personas = [str(p) for p in personas if isinstance(p, str)]
    return cleaned, personas


async def call_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    rcp_seq: str,
    name: str,
    enriched: dict,
    nutrition: dict,
    reasoning_effort: str | None,
) -> tuple[dict, int, int, int]:
    """returns (row, prompt_tokens, completion_tokens, reasoning_tokens)"""
    user_prompt = build_user_prompt(rcp_seq, name, enriched, nutrition)
    last_err: Exception | None = None
    extra: dict = {}
    if reasoning_effort and reasoning_effort != "default":
        extra["reasoning_effort"] = reasoning_effort
    async with semaphore:
        for attempt in range(2):  # 1 try + 1 retry
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    **extra,
                )
                content = resp.choices[0].message.content or "{}"
                parsed = json.loads(content)
                qs, personas = validate_response(parsed)
                business_issues = validate_business_rules(qs, nutrition)
                usage = resp.usage
                reasoning_t = 0
                details = getattr(usage, "completion_tokens_details", None)
                if details is not None:
                    reasoning_t = getattr(details, "reasoning_tokens", 0) or 0
                row = {
                    "rcp_seq": rcp_seq,
                    "name": name,
                    "hypothetical_questions": qs,
                    "_meta": {
                        "model": MODEL,
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "personas_hinted": personas,
                        "reasoning_effort": reasoning_effort or "default",
                        "validation_issues": business_issues,
                    },
                }
                return row, usage.prompt_tokens, usage.completion_tokens, reasoning_t
            except Exception as e:
                last_err = e
                if attempt == 0:
                    await asyncio.sleep(0.7)
                    continue
                raise last_err  # type: ignore[misc]
    raise RuntimeError("unreachable")


def atomic_write(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def estimate_cost(in_tokens: int, out_tokens: int) -> float:
    p = PRICING[MODEL]
    return in_tokens / 1_000_000 * p["input"] + out_tokens / 1_000_000 * p["output"]


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY 없음", file=sys.stderr)
        sys.exit(1)

    with open(ENRICHED_PATH, encoding="utf-8") as f:
        enriched_list = json.load(f)
    with open(CLEANED_PATH, encoding="utf-8") as f:
        cleaned_list = json.load(f)
    cleaned_map = {c["rcp_seq"]: c for c in cleaned_list}

    targets = enriched_list
    if args.limit is not None:
        targets = targets[: args.limit]

    existing: list[dict] = []
    done_seqs: set[str] = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        done_seqs = {x["rcp_seq"] for x in existing}
        print(f"[resume] 기존 {len(done_seqs)}건 skip")

    todo = [r for r in targets if r["rcp_seq"] not in done_seqs]
    print(f"처리 대상: {len(todo)}건 (전체 입력 {len(targets)})")
    if not todo:
        print("처리할 건 없음. 종료.")
        return

    if args.dry_run:
        sample = todo[0]
        print("\n=== DRY RUN: 첫 1건 prompt ===")
        print("--- SYSTEM ---")
        print(SYSTEM_PROMPT)
        print("--- USER ---")
        print(build_user_prompt(
            sample["rcp_seq"], sample["name"],
            sample.get("enriched", {}),
            cleaned_map.get(sample["rcp_seq"], {}).get("nutrition", {}),
        ))
        print("\n[dry-run] API 호출 없이 종료")
        return

    if not args.yes:
        est_cost = estimate_cost(
            len(todo) * EST_INPUT_TOKENS_PER_CALL,
            len(todo) * EST_OUTPUT_TOKENS_PER_CALL,
        )
        print(f"\n예상 비용 약 ${est_cost:.2f} (모델={MODEL}, 건수={len(todo)}).")
        print("진행하려면 Enter, 취소는 Ctrl+C.")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("취소")
            return

    concurrency = max(1, min(args.concurrency, MAX_CONCURRENCY))
    print(f"동시성: {concurrency}  reasoning_effort={args.reasoning_effort}")
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    results = list(existing)
    failed: list[dict] = []
    total_in = 0
    total_out = 0
    total_reasoning = 0
    started = time.time()

    async def task(recipe: dict):
        rcp_seq = recipe["rcp_seq"]
        try:
            row, in_t, out_t, reason_t = await call_one(
                client, semaphore, rcp_seq, recipe["name"],
                recipe.get("enriched", {}),
                cleaned_map.get(rcp_seq, {}).get("nutrition", {}),
                args.reasoning_effort,
            )
            return ("ok", row, in_t, out_t, reason_t)
        except Exception as e:
            return ("err", {"rcp_seq": rcp_seq, "name": recipe.get("name"), "error": repr(e)}, 0, 0, 0)

    coros = [task(r) for r in todo]
    done = 0
    new_count = 0
    business_violations = 0
    for fut in asyncio.as_completed(coros):
        status, payload, in_t, out_t, reason_t = await fut
        done += 1
        total_in += in_t
        total_out += out_t
        total_reasoning += reason_t
        if status == "ok":
            results.append(payload)
            new_count += 1
            issues = payload.get("_meta", {}).get("validation_issues", [])
            if issues:
                business_violations += 1
                print(f"  [validation] {payload['rcp_seq']} {payload['name']}: {issues}")
        else:
            failed.append(payload)
        if done % 50 == 0 or done == len(todo):
            cost = estimate_cost(total_in, total_out)
            print(f"  진행 {done}/{len(todo)}  ok={new_count}  err={len(failed)}  "
                  f"in={total_in}  out={total_out}  reason={total_reasoning}  ${cost:.4f}")
        if done % FLUSH_EVERY == 0:
            atomic_write(OUTPUT_PATH, results)

    atomic_write(OUTPUT_PATH, results)
    if failed:
        atomic_write(FAILED_PATH, failed)

    elapsed = time.time() - started
    cost = estimate_cost(total_in, total_out)
    reasoning_ratio = (total_reasoning / total_out) if total_out > 0 else 0.0
    print("\n=== 완료 ===")
    print(f"성공(이번 실행):     {new_count}건")
    print(f"실패:                {len(failed)}건  → {FAILED_PATH if failed else '-'}")
    print(f"검증 룰 위반:        {business_violations}건 (validation_issues 비어있지 않음)")
    print(f"reasoning_effort:    {args.reasoning_effort}")
    print(f"누적 token:          input={total_in}  output={total_out}")
    print(f"  └ reasoning_tokens: {total_reasoning}  (output 중 {reasoning_ratio:.1%})")
    print(f"비용 추정:           ${cost:.4f}  (단가 input ${PRICING[MODEL]['input']}/M, output ${PRICING[MODEL]['output']}/M)")
    print(f"실행 시간:           {elapsed:.1f}s")
    print(f"출력 파일:           {OUTPUT_PATH} (총 {len(results)}건)")

    new_only = [r for r in results if r["rcp_seq"] not in done_seqs][:5]
    if new_only:
        print("\n=== 샘플 5건 ===")
        for r in new_only:
            issues = r["_meta"].get("validation_issues", [])
            print(f"\n[{r['rcp_seq']}] {r['name']}  personas={r['_meta']['personas_hinted']}"
                  + (f"  ⚠ {issues}" if issues else ""))
            for q in r["hypothetical_questions"]:
                print(f"  - {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="일부만 처리")
    ap.add_argument("--resume", action="store_true", help="기존 결과 skip")
    ap.add_argument("--concurrency", type=int, default=5, help=f"동시 호출 수 (1~{MAX_CONCURRENCY})")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 prompt만 출력")
    ap.add_argument("--yes", action="store_true", help="비용 안내 Enter 대기 스킵")
    ap.add_argument(
        "--reasoning-effort",
        choices=["default", "minimal", "low", "medium", "high"],
        default="default",
        help="gpt-5-mini reasoning_effort. 'default'면 파라미터 미전달(SDK 기본 동작).",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
