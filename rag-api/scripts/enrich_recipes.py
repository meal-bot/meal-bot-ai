"""
LLM 보강 실행 스크립트.
data/recipes_cleaned.json을 입력으로 받아 GPT-5-mini로 11개 필드를 보강.

사용법:
  python scripts/enrich_recipes.py --limit 1       # 1건 smoke test
  python scripts/enrich_recipes.py --limit 20      # 20건
  python scripts/enrich_recipes.py --all           # 전체 1,146건
  python scripts/enrich_recipes.py --limit 3 --dry-run
  python scripts/enrich_recipes.py --all --restart
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

# scripts/ 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _enrichment_schema import RECIPE_ENRICHMENT_SCHEMA  # noqa: E402
from _enrichment_prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

load_dotenv()

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODEL               = "gpt-5-mini"
DEFAULT_CONCURRENCY = 20
SAVE_INTERVAL       = 10
MAX_RETRIES         = 3

INPUT_PATH   = Path("data/recipes_cleaned.json")
OUTPUT_PATH  = Path("data/recipes_enriched.json")
PROGRESS_DIR = Path("artifacts/enrichment_progress")
PARTIAL_PATH = PROGRESS_DIR / "partial_results.json"
IDS_PATH     = PROGRESS_DIR / "processed_ids.json"
FAILED_PATH  = PROGRESS_DIR / "failed.json"


# ── 파일 저장 ─────────────────────────────────────────────────────────────────

def save_atomic(data: object, path: Path) -> None:
    """JSON을 임시 파일로 쓴 뒤 원자적으로 교체."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def save_partial(results: list, processed_ids: set, failed: list) -> None:
    """중간 진행 상태 저장 (partial_results / processed_ids / failed)."""
    save_atomic(results, PARTIAL_PATH)
    save_atomic(sorted(processed_ids), IDS_PATH)
    save_atomic(failed, FAILED_PATH)


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="recipes_cleaned.json → GPT-5-mini 보강 → recipes_enriched.json"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--limit", type=int, metavar="N",
        help="처리할 건수 (기본값: 1)",
    )
    mode_group.add_argument(
        "--all", action="store_true",
        help="전체 레시피 처리",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="기존 progress 삭제 후 처음부터 실행",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, metavar="N",
        help=f"비동기 동시성 (기본값: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="API 호출 없이 user prompt만 출력하고 종료",
    )
    return parser.parse_args()


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

async def enrich_one(
    client: AsyncOpenAI,
    recipe: dict,
    semaphore: asyncio.Semaphore,
    token_counter: dict,
) -> dict:
    """레시피 1건을 LLM 호출로 보강. 항상 {ok, ...} dict 반환."""
    rcp_seq = recipe.get("rcp_seq")
    name    = recipe.get("name", "(unknown)")

    async with semaphore:
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(recipe)},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": RECIPE_ENRICHMENT_SCHEMA,
                    },
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response content")

                enriched = json.loads(content)

                if response.usage:
                    token_counter["prompt"]     += response.usage.prompt_tokens
                    token_counter["completion"] += response.usage.completion_tokens

                return {"ok": True, "recipe": {**recipe, **enriched}}

            except APIStatusError as e:
                # 429(rate limit)만 재시도, 나머지 4xx는 즉시 실패
                if e.status_code != 429:
                    return {
                        "ok": False,
                        "rcp_seq": rcp_seq,
                        "name": name,
                        "error": str(e),
                    }
                last_error = e
            except Exception as e:
                last_error = e

            await asyncio.sleep(2 ** attempt)  # 1s → 2s → 4s 지수 백오프

        return {
            "ok": False,
            "rcp_seq": rcp_seq,
            "name": name,
            "error": f"Max retries exceeded. Last: {last_error}",
        }


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    # 입력 로드
    with INPUT_PATH.open(encoding="utf-8") as f:
        original_recipes: list[dict] = json.load(f)
    original_order = {
        r["rcp_seq"]: i
        for i, r in enumerate(original_recipes)
        if r.get("rcp_seq")
    }

    # 처리 범위 결정
    if args.all:
        target = original_recipes
    elif args.limit is not None:
        target = original_recipes[: args.limit]
    else:
        target = original_recipes[:1]  # 기본값 1건

    # rcp_seq 없는 레시피를 즉시 실패 처리
    valid_recipes: list[dict] = []
    pre_failed:    list[dict] = []
    for r in target:
        if not r.get("rcp_seq"):
            pre_failed.append({
                "ok": False, "rcp_seq": None,
                "name": r.get("name", "(unknown)"),
                "error": "rcp_seq missing",
            })
        else:
            valid_recipes.append(r)

    # ── dry-run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        for r in valid_recipes:
            sep = "=" * 10
            print(f"\n{sep} rcp_seq={r['rcp_seq']} {r.get('name', '')} {sep}")
            print(build_user_prompt(r))
        if pre_failed:
            print(f"\n[경고] rcp_seq 없어서 스킵: {len(pre_failed)}건")
        return

    # ── progress 디렉토리 ────────────────────────────────────────────────────
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    # ── resume / restart ─────────────────────────────────────────────────────
    results:       list[dict] = []
    processed_ids: set[str]   = set()
    failed:        list[dict] = list(pre_failed)

    if args.restart:
        for p in (PARTIAL_PATH, IDS_PATH, FAILED_PATH):
            p.unlink(missing_ok=True)
        tqdm.write("기존 progress 삭제. 처음부터 시작합니다.")
    elif IDS_PATH.exists() and PARTIAL_PATH.exists():
        with IDS_PATH.open(encoding="utf-8") as f:
            processed_ids = set(json.load(f))
        with PARTIAL_PATH.open(encoding="utf-8") as f:
            results = json.load(f)
        if FAILED_PATH.exists():
            with FAILED_PATH.open(encoding="utf-8") as f:
                failed = json.load(f)
        tqdm.write(
            f"Resume: 기존 처리 {len(processed_ids)}건 로드. 미처리 건만 실행합니다."
        )

    # 미처리 필터
    to_process = [r for r in valid_recipes if r["rcp_seq"] not in processed_ids]

    tqdm.write(
        f"\n대상: {len(valid_recipes)}건 | 미처리: {len(to_process)}건 | "
        f"동시성: {args.concurrency} | 모델: {MODEL}"
    )
    if not to_process:
        tqdm.write("처리할 항목 없음. 종료.")
        return

    client        = AsyncOpenAI()
    semaphore     = asyncio.Semaphore(args.concurrency)
    token_counter = {"prompt": 0, "completion": 0}

    tasks     = [enrich_one(client, r, semaphore, token_counter) for r in to_process]
    completed = 0
    start_ts  = time.perf_counter()

    # ── 비동기 처리 루프 ─────────────────────────────────────────────────────
    try:
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="보강 중",
            unit="건",
        ):
            result = await coro
            completed += 1

            if result["ok"]:
                results.append(result["recipe"])
                processed_ids.add(result["recipe"]["rcp_seq"])
            else:
                failed.append(result)
                tqdm.write(
                    f"rcp_seq={result.get('rcp_seq')} 실패: {result.get('error')}"
                )

            if completed % SAVE_INTERVAL == 0:
                save_partial(results, processed_ids, failed)
                tqdm.write(
                    f"중간 저장 완료: {PARTIAL_PATH}  ({completed}/{len(tasks)}건)"
                )

    except KeyboardInterrupt:
        tqdm.write("\n중단됨. 진행분 저장 중...")
        save_partial(results, processed_ids, failed)
        tqdm.write(f"저장 완료: {PARTIAL_PATH}")
        sys.exit(0)

    # ── 최종 저장 ────────────────────────────────────────────────────────────
    save_partial(results, processed_ids, failed)

    sorted_results = sorted(
        results,
        key=lambda r: original_order.get(r.get("rcp_seq", ""), float("inf")),
    )
    save_atomic(sorted_results, OUTPUT_PATH)

    elapsed = time.perf_counter() - start_ts
    tqdm.write(f"\n{'='*50}")
    tqdm.write(
        f"완료: {completed}건 처리 | 성공: {len(results)}건 | 실패: {len(failed)}건"
    )
    tqdm.write(f"소요 시간: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
    tqdm.write(
        f"총 token usage: prompt={token_counter['prompt']:,}, "
        f"completion={token_counter['completion']:,}"
    )
    tqdm.write("  (정확한 비용은 OpenAI 대시보드에서 확인)")
    tqdm.write(f"결과 저장: {OUTPUT_PATH}")
    if failed:
        tqdm.write(f"실패 목록: {FAILED_PATH}  ({len(failed)}건)")

    if not results:
        tqdm.write("[에러] 성공한 결과가 없습니다.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
