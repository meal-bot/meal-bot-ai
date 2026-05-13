"""
v2 보강 실행 스크립트.
data/recipes_enriched.json을 입력으로 받아 GPT-5-mini로 신규 5개 필드를 보강.

사용법:
  python scripts/enrich_recipes_v2.py --limit 10      # 10건 테스트
  python scripts/enrich_recipes_v2.py                  # 전체 (resume 기본값)
  python scripts/enrich_recipes_v2.py --no-resume      # 처음부터 재실행
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIStatusError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _enrichment_v2_schema import SCHEMA_V2                          # noqa: E402
from _enrichment_v2_prompts import SYSTEM_PROMPT_V2, build_user_prompt_v2  # noqa: E402

load_dotenv()

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODEL               = "gpt-5-mini"
DEFAULT_CONCURRENCY = 20          # 기존 enrich_recipes.py와 동일
SAVE_INTERVAL       = 20
MAX_RETRIES         = 3
BACKOFF_DELAYS      = [2, 4]      # 재시도 1차 2s, 2차 4s

INPUT_PATH   = Path("data/recipes_enriched.json")
OUTPUT_PATH  = Path("data/recipes_enriched_v2.json")
PROGRESS_DIR = Path("artifacts/enrichment_v2_progress")
PARTIAL_PATH = PROGRESS_DIR / "partial_results_v2.json"   # dict[rcp_seq → enriched]
IDS_PATH     = PROGRESS_DIR / "processed_ids_v2.json"
FAILED_PATH  = PROGRESS_DIR / "failed_v2.json"


# ── 파일 저장 ─────────────────────────────────────────────────────────────────

def save_atomic(data: object, path: Path) -> None:
    """JSON을 임시 파일로 쓴 뒤 원자적으로 교체."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def save_partial(
    partial_results: dict,
    processed_ids: set,
    failed: list,
) -> None:
    processed_str = {str(pid) for pid in processed_ids}
    cleaned_failed = [
        f for f in failed
        if str(f.get("rcp_seq")) not in processed_str
    ]
    save_atomic(partial_results, PARTIAL_PATH)
    save_atomic(sorted(processed_ids), IDS_PATH)
    save_atomic(cleaned_failed, FAILED_PATH)


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="recipes_enriched.json → GPT-5-mini v2 보강 → recipes_enriched_v2.json"
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="처리할 최대 건수 (기본값: 전체)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="이미 처리된 건 건너뛰기 (기본값)",
    )
    resume_group.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="처음부터 다시 처리",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, metavar="N",
        help=f"동시 호출 수 (기본값: {DEFAULT_CONCURRENCY})",
    )
    return parser.parse_args()


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

async def enrich_one_v2(
    client: AsyncOpenAI,
    recipe: dict,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
    processed_ids: set,
    token_counter: dict,
) -> dict:
    """레시피 1건을 v2 LLM 호출로 보강. 항상 {ok, ...} dict 반환."""
    rcp_seq = recipe.get("rcp_seq")
    name    = recipe.get("name", "(unknown)")

    async with semaphore:
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_V2},
                        {"role": "user",   "content": build_user_prompt_v2(recipe)},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": SCHEMA_V2,
                    },
                    max_completion_tokens=3000,
                    reasoning_effort="low",
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response content")

                enriched = json.loads(content)

                if response.usage:
                    token_counter["prompt"]     += response.usage.prompt_tokens
                    token_counter["completion"] += response.usage.completion_tokens

                async with lock:
                    processed_ids.add(rcp_seq)

                return {"ok": True, "rcp_seq": rcp_seq, "enriched": enriched}

            except APIStatusError as e:
                if e.status_code != 429:
                    return {"ok": False, "rcp_seq": rcp_seq, "name": name, "error": str(e)}
                last_error = e
            except Exception as e:
                last_error = e

            if attempt < len(BACKOFF_DELAYS):
                await asyncio.sleep(BACKOFF_DELAYS[attempt])

        return {
            "ok": False,
            "rcp_seq": rcp_seq,
            "name": name,
            "error": f"Max retries exceeded. Last: {last_error}",
        }


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    if not INPUT_PATH.exists():
        print(f"[에러] 입력 파일 없음: {INPUT_PATH}")
        print("먼저 enrich_recipes.py를 실행해 recipes_enriched.json을 생성하세요.")
        sys.exit(1)

    with INPUT_PATH.open(encoding="utf-8") as f:
        all_recipes: list[dict] = json.load(f)

    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    partial_results: dict[str, dict] = {}
    processed_ids:   set[str]        = set()
    failed:          list[dict]       = []

    if args.resume and IDS_PATH.exists() and PARTIAL_PATH.exists():
        with IDS_PATH.open(encoding="utf-8") as f:
            processed_ids = set(json.load(f))
        with PARTIAL_PATH.open(encoding="utf-8") as f:
            partial_results = json.load(f)
        if FAILED_PATH.exists():
            with FAILED_PATH.open(encoding="utf-8") as f:
                failed = json.load(f)
        tqdm.write(f"Resume: 기존 처리 {len(processed_ids)}건 로드. 미처리 건만 실행합니다.")
    elif not args.resume:
        tqdm.write("--no-resume: 기존 progress 무시. 처음부터 시작합니다.")

    to_process = [r for r in all_recipes if r.get("rcp_seq") not in processed_ids]
    if args.limit is not None:
        to_process = to_process[: args.limit]

    tqdm.write(
        f"\n입력: {len(all_recipes)}건 | 미처리: {len(to_process)}건 | "
        f"동시성: {args.concurrency} | 모델: {MODEL}"
    )

    if to_process:
        client        = AsyncOpenAI()
        semaphore     = asyncio.Semaphore(args.concurrency)
        lock          = asyncio.Lock()
        token_counter = {"prompt": 0, "completion": 0}

        tasks     = [
            enrich_one_v2(client, r, semaphore, lock, processed_ids, token_counter)
            for r in to_process
        ]
        completed = 0
        start_ts  = time.perf_counter()

        try:
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="v2 보강 중",
                unit="건",
            ):
                result = await coro
                completed += 1

                if result["ok"]:
                    partial_results[result["rcp_seq"]] = result["enriched"]
                else:
                    failed.append(result)
                    tqdm.write(
                        f"rcp_seq={result.get('rcp_seq')} 실패: {result.get('error')}"
                    )

                if completed % SAVE_INTERVAL == 0:
                    save_partial(partial_results, processed_ids, failed)
                    tqdm.write(f"중간 저장 완료 ({completed}/{len(tasks)}건)")

        except KeyboardInterrupt:
            tqdm.write("\n중단됨. 진행분 저장 중...")
            save_partial(partial_results, processed_ids, failed)
            tqdm.write(f"저장 완료: {PARTIAL_PATH}")
            sys.exit(0)

        elapsed = time.perf_counter() - start_ts
        save_partial(partial_results, processed_ids, failed)

        active_failed = [
            f for f in failed
            if str(f.get("rcp_seq")) not in {str(pid) for pid in processed_ids}
        ]
        tqdm.write(f"\n{'='*50}")
        tqdm.write(
            f"완료: {completed}건 처리 | 성공: {len(partial_results)}건 | 실패: {len(active_failed)}건"
        )
        tqdm.write(f"소요 시간: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
        tqdm.write(
            f"총 token usage: prompt={token_counter['prompt']:,}, "
            f"completion={token_counter['completion']:,}"
        )
        total_tokens = token_counter["prompt"] + token_counter["completion"]
        if total_tokens > 0:
            estimated_cost = (
                token_counter["prompt"]     / 1_000_000 * 0.15
                + token_counter["completion"] / 1_000_000 * 0.60
            )
            tqdm.write(f"예상 비용: ${estimated_cost:.4f}  (OpenAI 대시보드에서 확인)")
    else:
        tqdm.write("처리할 항목 없음. 최종 파일 병합만 수행합니다.")

    # ── 최종 병합: 원본 레코드 + v2 신규 필드 ────────────────────────────────
    merged = []
    for recipe in all_recipes:
        seq = recipe.get("rcp_seq")
        if seq and seq in partial_results:
            merged.append({**recipe, **partial_results[seq]})
        else:
            merged.append(recipe)   # 실패 건은 신규 필드 없이 그대로 포함

    save_atomic(merged, OUTPUT_PATH)
    tqdm.write(f"결과 저장: {OUTPUT_PATH}  ({len(merged)}건)")
    if failed:
        tqdm.write(f"실패 목록: {FAILED_PATH}  ({len(failed)}건, --no-resume 재실행으로 복구 가능)")

    if not partial_results:
        tqdm.write("[경고] 성공한 v2 보강 결과가 없습니다.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())