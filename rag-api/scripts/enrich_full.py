"""1,146개 풀 enrichment — OpenAI Batch API 사용.

모드:
  --dry-run         JSONL만 생성, batch는 만들지 않음
  --submit          batch_input.jsonl 업로드 + batch 생성 + state 저장
  --poll            batch 상태 확인/갱신
  --process-results 완료된 batch output 다운로드 + 후처리 + 리포트
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _enrichment_common import (  # noqa: E402
    ENRICH_SCHEMA,
    PRICING,
    SYSTEM_PROMPT,
    build_user_prompt,
    enrich_one,
    post_process,
    validate_enriched,
)


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "recipes_cleaned.json"
BATCH_INPUT_PATH = ROOT / "data" / "batch_input.jsonl"
BATCH_PREVIEW_PATH = ROOT / "data" / "batch_preview.json"
BATCH_STATE_PATH = ROOT / "data" / "batch_state.json"
BATCH_OUTPUT_PATH = ROOT / "data" / "batch_output.jsonl"
ENRICHED_OUT_PATH = ROOT / "data" / "recipes_enriched.json"
REPORT_OUT_PATH = ROOT / "data" / "enrichment_report.json"
FAILURES_OUT_PATH = ROOT / "data" / "enrichment_failures.json"

load_dotenv(ROOT / ".env")

MODEL = "gpt-5-mini"
ENDPOINT = "/v1/responses"

# enrich_sample.py 실측 평균 (20건 기반)
SAMPLE_AVG_INPUT_TOKENS = 2225
SAMPLE_AVG_OUTPUT_TOKENS = 1617


# =========================================================
# Batch 요청 빌더
# =========================================================

def build_batch_request(recipe: dict) -> dict:
    """단일 레시피를 OpenAI Batch JSONL 라인 dict로 변환."""
    user_prompt, _, _ = build_user_prompt(recipe)
    return {
        "custom_id": f"rcp_seq_{recipe['rcp_seq']}",
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": MODEL,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": ENRICH_SCHEMA["name"],
                    "strict": True,
                    "schema": ENRICH_SCHEMA["schema"],
                },
            },
        },
    }


# =========================================================
# --dry-run
# =========================================================

def cmd_dry_run() -> None:
    recipes = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    BATCH_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BATCH_INPUT_PATH.open("w", encoding="utf-8") as f:
        for r in recipes:
            req = build_batch_request(r)
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    line_count = sum(1 for _ in BATCH_INPUT_PATH.open("r", encoding="utf-8"))
    with BATCH_INPUT_PATH.open("r", encoding="utf-8") as f:
        first_req = json.loads(f.readline())

    total_in = SAMPLE_AVG_INPUT_TOKENS * len(recipes)
    total_out = SAMPLE_AVG_OUTPUT_TOKENS * len(recipes)
    p = PRICING[MODEL]
    cost_std = (total_in * p["input"] + total_out * p["output"]) / 1_000_000
    cost_batch = cost_std * 0.5

    preview = {
        "total_recipes": len(recipes),
        "batch_input_lines": line_count,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "first_custom_id": first_req["custom_id"],
        "first_request": first_req,
        "estimated_cost": {
            "basis": "enrich_sample.py 실측 20건 평균",
            "avg_input_tokens": SAMPLE_AVG_INPUT_TOKENS,
            "avg_output_tokens": SAMPLE_AVG_OUTPUT_TOKENS,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "standard_usd": round(cost_std, 4),
            "batch_50pct_usd": round(cost_batch, 4),
        },
    }
    BATCH_PREVIEW_PATH.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    body = first_req["body"]
    sys_msg = body["input"][0]["content"]
    user_msg = body["input"][1]["content"]

    print("=== dry-run 완료 ===")
    print(f"total recipes: {len(recipes)}")
    print(f"batch_input.jsonl 라인 수: {line_count}")
    print(f"batch_input.jsonl: {BATCH_INPUT_PATH.relative_to(ROOT)}")
    print(f"batch_preview.json: {BATCH_PREVIEW_PATH.relative_to(ROOT)}")
    print(f"endpoint: {ENDPOINT}")
    print(f"model: {MODEL}")
    print()
    print("=== 첫 요청 preview ===")
    print(f"custom_id: {first_req['custom_id']}")
    print(f"method/url: {first_req['method']} {first_req['url']}")
    print(f"body.model: {body['model']}")
    print(f"body.input messages: {len(body['input'])}개")
    print(f"  [0] role=system content (앞 80자): {sys_msg[:80]}...")
    print(f"  [1] role=user   content (앞 200자): {user_msg[:200]}...")
    fmt = body["text"]["format"]
    print(f"body.text.format: type={fmt['type']} name={fmt['name']} strict={fmt['strict']}")
    print(f"body.text.format.schema.required: {fmt['schema']['required']}")
    print()
    print("=== 예상 비용 ===")
    print(f"sample(20건) 평균 input/output: {SAMPLE_AVG_INPUT_TOKENS} / {SAMPLE_AVG_OUTPUT_TOKENS}")
    print(f"총 input tokens 추정: {total_in:,}")
    print(f"총 output tokens 추정: {total_out:,}")
    print(f"표준 단가 비용: ${cost_std:.4f}")
    print(f"배치 50% 할인 비용: ${cost_batch:.4f}")
    print()
    print("※ 실제 OpenAI Batch는 아직 생성되지 않았습니다.")
    print("  승인 후 'python3 scripts/enrich_full.py --submit'으로 진행하세요.")


# =========================================================
# --submit
# =========================================================

def cmd_submit() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    if BATCH_STATE_PATH.exists():
        state = json.loads(BATCH_STATE_PATH.read_text(encoding="utf-8"))
        print(f"이미 batch_state.json이 있습니다.")
        print(f"  기존 batch_id: {state.get('batch_id')}")
        print(f"  기존 status:   {state.get('status')}")
        print(f"새 batch를 만들지 않습니다.")
        print(f"  - 진행 확인: python3 scripts/enrich_full.py --poll")
        print(f"  - 결과 처리: python3 scripts/enrich_full.py --process-results")
        print(f"  - 강제 재시작 시: data/batch_state.json 을 직접 삭제하세요.")
        return

    if not BATCH_INPUT_PATH.exists():
        raise RuntimeError(
            f"{BATCH_INPUT_PATH} 없음 — 먼저 --dry-run으로 JSONL을 생성하세요"
        )

    client = OpenAI()
    print(f"파일 업로드 중: {BATCH_INPUT_PATH}")
    with BATCH_INPUT_PATH.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  file_id={file_obj.id}")

    print("Batch 생성 중...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=ENDPOINT,
        completion_window="24h",
        metadata={"description": "meal-bot RAG v2 enrichment full run"},
    )
    state = {
        "batch_id": batch.id,
        "input_file_id": file_obj.id,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "status": batch.status,
        "created_at": batch.created_at,
    }
    BATCH_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Batch 생성됨: {batch.id} (status={batch.status})")
    print(f"상태 저장: {BATCH_STATE_PATH.relative_to(ROOT)}")


# =========================================================
# --poll
# =========================================================

def cmd_poll() -> None:
    if not BATCH_STATE_PATH.exists():
        raise RuntimeError(f"{BATCH_STATE_PATH} 없음 — 먼저 --submit으로 batch를 만드세요")
    state = json.loads(BATCH_STATE_PATH.read_text(encoding="utf-8"))

    client = OpenAI()
    batch = client.batches.retrieve(state["batch_id"])
    state["status"] = batch.status
    if batch.request_counts is not None:
        state["request_counts"] = {
            "total":     batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed":    batch.request_counts.failed,
        }
    if batch.output_file_id:
        state["output_file_id"] = batch.output_file_id
    if batch.error_file_id:
        state["error_file_id"] = batch.error_file_id
    BATCH_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Batch {state['batch_id']}")
    print(f"  status: {batch.status}")
    if batch.request_counts is not None:
        rc = state["request_counts"]
        print(f"  request_counts: total={rc['total']} completed={rc['completed']} failed={rc['failed']}")
    if batch.output_file_id:
        print(f"  output_file_id: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"  error_file_id: {batch.error_file_id}")


# =========================================================
# --process-results
# =========================================================

def parse_response_body(resp_body: dict) -> str | None:
    """Responses API body에서 JSON 본문 텍스트 추출."""
    # 우선순위 1: output[].content[].text (Responses API 표준)
    for output_item in resp_body.get("output", []) or []:
        for c in output_item.get("content", []) or []:
            t = c.get("type")
            if t in ("output_text", "text"):
                txt = c.get("text")
                if txt:
                    return txt
    # 우선순위 2: output_text 헬퍼 필드
    if isinstance(resp_body.get("output_text"), str):
        return resp_body["output_text"]
    return None


def cmd_process_results() -> None:
    if not BATCH_STATE_PATH.exists():
        raise RuntimeError(f"{BATCH_STATE_PATH} 없음")
    state = json.loads(BATCH_STATE_PATH.read_text(encoding="utf-8"))
    if not state.get("output_file_id"):
        raise RuntimeError(
            "output_file_id 가 state에 없음 — 먼저 --poll 로 상태를 갱신하세요"
        )

    client = OpenAI()
    print(f"output 파일 다운로드 중: {state['output_file_id']}")
    content = client.files.content(state["output_file_id"]).text
    BATCH_OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"  저장: {BATCH_OUTPUT_PATH.relative_to(ROOT)}")

    recipes = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    by_seq = {r["rcp_seq"]: r for r in recipes}

    results: list[dict] = []
    failures: list[dict] = []

    for line in BATCH_OUTPUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cid = item.get("custom_id", "")
        seq = cid.replace("rcp_seq_", "") if cid.startswith("rcp_seq_") else None
        recipe = by_seq.get(seq) if seq else None

        if recipe is None:
            failures.append({"custom_id": cid, "error": "recipe not found in cleaned"})
            continue

        if item.get("error"):
            failures.append({
                "custom_id": cid, "rcp_seq": seq, "name": recipe.get("name"),
                "error": item["error"],
            })
            continue

        try:
            resp_body = item["response"]["body"]
            text = parse_response_body(resp_body)
            if text is None:
                raise ValueError("response body에서 text 추출 실패")
            enriched_raw = json.loads(text)
        except Exception as ex:
            failures.append({
                "custom_id": cid, "rcp_seq": seq, "name": recipe.get("name"),
                "error": f"parse error: {type(ex).__name__}: {ex}",
            })
            continue

        _, needs_cat, needs_cm = build_user_prompt(recipe)
        enriched = post_process(enriched_raw, recipe, needs_cat, needs_cm, MODEL)
        usage = resp_body.get("usage", {}) or {}
        enriched["_meta"] = {
            "input_tokens":  usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "elapsed_sec":   None,
        }
        issues = validate_enriched(enriched)
        results.append(_build_result_record(recipe, enriched, issues))

    # 실패 케이스 1회 재시도 (일반 호출)
    if failures:
        print(f"\n실패 {len(failures)}건 → 일반 호출로 1회 재시도")
        retry_failed: list[dict] = []
        for fail in failures:
            seq = fail.get("rcp_seq")
            recipe = by_seq.get(seq) if seq else None
            if recipe is None:
                retry_failed.append(fail)
                continue
            try:
                enriched = enrich_one(client, recipe, MODEL)
                issues = validate_enriched(enriched)
                results.append(_build_result_record(recipe, enriched, issues))
                print(f"  재시도 성공: [{seq}] {recipe.get('name')}")
            except Exception as ex:
                retry_failed.append({**fail, "retry_error": f"{type(ex).__name__}: {ex}"})
                print(f"  재시도 실패: [{seq}] {recipe.get('name')} — {ex}")
        failures = retry_failed

    # 저장
    ENRICHED_OUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n저장: {ENRICHED_OUT_PATH.relative_to(ROOT)} ({len(results)}건)")

    if failures:
        FAILURES_OUT_PATH.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"저장: {FAILURES_OUT_PATH.relative_to(ROOT)} ({len(failures)}건)")
    elif FAILURES_OUT_PATH.exists():
        FAILURES_OUT_PATH.unlink()

    report = build_full_report(results, failures, by_seq)
    REPORT_OUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"저장: {REPORT_OUT_PATH.relative_to(ROOT)}")

    print(f"\n=== 풀 enrichment 결과 요약 ===")
    print(f"성공: {report['success']} / 실패: {report['failed']}")
    print(f"validation issues: {report['validation']['issues_found']}건")
    print(f"사후 점검 카테고리별:")
    for k, v in report["post_review"]["counts"].items():
        print(f"  {k}: {v}건")
    print(f"비용: ${report['cost']['total_cost_usd']}")


def _build_result_record(recipe: dict, enriched: dict, issues: list[str]) -> dict:
    return {
        "rcp_seq": recipe["rcp_seq"],
        "name": recipe.get("name"),
        "input_summary": {
            "category": recipe.get("category"),
            "cooking_way": recipe.get("cooking_way"),
            "ingredient_count_total": recipe.get("ingredient_count_total"),
            "parser_flags": recipe.get("enrichment_flags", {}),
        },
        "enriched": enriched,
        "validation_issues": issues,
    }


# =========================================================
# 리포트
# =========================================================

REVIEW_NAME_KEYWORDS_FOR_BLENDING = ["주스", "음료", "드레싱", "소스"]


def build_full_report(
    results: list[dict], failures: list[dict], by_seq: dict[str, dict]
) -> dict:
    success = results
    total = len(success) + len(failures)

    total_in = sum(r["enriched"]["_meta"]["input_tokens"] for r in success)
    total_out = sum(r["enriched"]["_meta"]["output_tokens"] for r in success)
    p = PRICING[MODEL]
    cost = (total_in * p["input"] + total_out * p["output"]) / 1_000_000

    cases_with_issues = [
        {"rcp_seq": r["rcp_seq"], "name": r["name"], "issues": r["validation_issues"]}
        for r in success if r["validation_issues"]
    ]

    cat_filled = sum(1 for r in success if r["enriched"]["_flags"]["category_ai_filled"])
    cm_filled = sum(1 for r in success if r["enriched"]["_flags"]["cooking_method_ai_filled"])

    spicy_dist = Counter(str(r["enriched"]["spicy_level"]) for r in success)
    diff_dist = Counter(str(r["enriched"]["difficulty"]) for r in success)

    meal_dist: Counter = Counter()
    sit_dist: Counter = Counter()
    for r in success:
        for tag in r["enriched"].get("meal_type_tags", []):
            meal_dist[tag] += 1
        for tag in r["enriched"].get("recommended_situations", []):
            sit_dist[tag] += 1

    summary_lens = [len(r["enriched"].get("summary") or "") for r in success]
    summary_stats = {
        "avg": round(sum(summary_lens) / len(summary_lens), 1) if summary_lens else 0,
        "min": min(summary_lens) if summary_lens else 0,
        "max": max(summary_lens) if summary_lens else 0,
    }

    cts = [r["enriched"]["cooking_time"] for r in success
           if r["enriched"].get("cooking_time") is not None]
    cooking_time_stats = {
        "avg": round(sum(cts) / len(cts), 1) if cts else None,
        "min": min(cts) if cts else None,
        "max": max(cts) if cts else None,
    }

    # 사후 점검 8가지
    post_review = _build_post_review(success, failures, by_seq)

    return {
        "total": total,
        "success": len(success),
        "failed": len(failures),
        "enrichment_failed_count": len(failures),
        "validation": {
            "issues_found": len(cases_with_issues),
            "cases_with_issues": cases_with_issues,
        },
        "enrichment_stats": {
            "category_ai_filled_count": cat_filled,
            "cooking_method_ai_filled_count": cm_filled,
            "spicy_level_distribution": dict(sorted(spicy_dist.items())),
            "difficulty_distribution": dict(sorted(diff_dist.items())),
            "meal_type_tags_distribution": dict(meal_dist.most_common()),
            "recommended_situations_distribution": dict(sit_dist.most_common()),
            "summary_length": summary_stats,
            "cooking_time": cooking_time_stats,
        },
        "post_review": post_review,
        "cost": {
            "model": MODEL,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_usd": round(cost, 4),
        },
    }


def _build_post_review(
    success: list[dict], failures: list[dict], by_seq: dict[str, dict]
) -> dict:
    bins: dict[str, list] = {
        "건강식_고나트륨": [],          # 1
        "다이어트_고칼로리": [],         # 2
        "조리기_블렌딩의심": [],         # 3
        "ingredients_missing_main": [],  # 4
        "spicy_5": [],                  # 5
        "difficulty_3": [],             # 6
        "validation_issues": [],         # 7
        "enrichment_failed": [],         # 8
    }

    for r in success:
        e = r["enriched"]
        seq = r["rcp_seq"]
        name = r["name"]
        cleaned = by_seq.get(seq, {})
        nut = cleaned.get("nutrition") or {}
        kcal = nut.get("energy_kcal")
        sodium = nut.get("sodium_mg")
        sits = e.get("recommended_situations") or []

        if "건강식" in sits and sodium is not None and sodium > 800:
            bins["건강식_고나트륨"].append({
                "rcp_seq": seq, "name": name, "sodium_mg": sodium,
            })
        if "다이어트" in sits and kcal is not None and kcal > 500:
            bins["다이어트_고칼로리"].append({
                "rcp_seq": seq, "name": name, "energy_kcal": kcal,
            })
        if e.get("cooking_method") == "조리기" and any(
            kw in (name or "") for kw in REVIEW_NAME_KEYWORDS_FOR_BLENDING
        ):
            bins["조리기_블렌딩의심"].append({
                "rcp_seq": seq, "name": name, "cooking_method": "조리기",
            })

        parser_flags = (cleaned.get("enrichment_flags") or {})
        if parser_flags.get("ingredients_missing"):
            bins["ingredients_missing_main"].append({
                "rcp_seq": seq, "name": name,
                "main_ingredients": e.get("main_ingredients", []),
            })

        if e.get("spicy_level") == 5:
            bins["spicy_5"].append({"rcp_seq": seq, "name": name})
        if e.get("difficulty") == 3:
            bins["difficulty_3"].append({"rcp_seq": seq, "name": name})
        if r["validation_issues"]:
            bins["validation_issues"].append({
                "rcp_seq": seq, "name": name, "issues": r["validation_issues"],
            })

    for f in failures:
        bins["enrichment_failed"].append({
            "rcp_seq": f.get("rcp_seq"), "name": f.get("name"),
            "error": f.get("error"), "retry_error": f.get("retry_error"),
        })

    return {
        "counts": {k: len(v) for k, v in bins.items()},
        "cases": bins,
    }


# =========================================================
# CLI
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", dest="dry_run",
                       help="JSONL 생성만, batch는 만들지 않음")
    group.add_argument("--submit", action="store_true",
                       help="batch_input.jsonl 업로드 + batch 생성")
    group.add_argument("--poll", action="store_true",
                       help="batch 상태 확인/갱신")
    group.add_argument("--process-results", action="store_true", dest="process_results",
                       help="완료된 batch output 다운로드 + 후처리 + 리포트")
    args = parser.parse_args()

    if args.dry_run:
        cmd_dry_run()
    elif args.submit:
        cmd_submit()
    elif args.poll:
        cmd_poll()
    elif args.process_results:
        cmd_process_results()


if __name__ == "__main__":
    main()
