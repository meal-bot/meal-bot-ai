"""QA baseline 평가 (일회성).

eval/qa_cases_v1.json의 10개 케이스에 대해 qa.answer를 순차 실행하고
- artifacts/qa_eval_v1.json         (raw 결과)
- artifacts/qa_eval_v1_summary.txt  (사람 검토용)
로 저장한다.

retriever/reranker는 호출하지 않는다. qa.answer만 사용한다.
retrieved_doc_ids는 recipes_enriched_v2.json에서 lookup해서 원본 dict 전체를 넘긴다.
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from rag.qa import answer as qa_answer


# ── 경로 ──────────────────────────────────────────────────────────────────────

CASES_INPUT    = Path("eval/qa_cases_v1.json")
ENRICHED_DATA  = Path("data/recipes_enriched_v2.json")
OUTPUT_JSON    = Path("artifacts/qa_eval_v1.json")
OUTPUT_SUMMARY = Path("artifacts/qa_eval_v1_summary.txt")


# ── lookup 헬퍼 ──────────────────────────────────────────────────────────────

def load_full_recipe_lookup(path: Path) -> dict[str, dict]:
    """recipes_enriched_v2.json을 str(rcp_seq) → 레시피 dict 전체로 매핑."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(r["rcp_seq"]): r for r in data if r.get("rcp_seq")}


def get_recipe_name(doc: dict) -> str:
    """name → rcp_nm → RCP_NM 순으로 fallback."""
    for key in ("name", "rcp_nm", "RCP_NM"):
        v = doc.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return "?"


# ── 요약 포맷팅 ──────────────────────────────────────────────────────────────

SEP = "═" * 64
HEADER_NOTE = (
    "합격 기준: 수동 검토 기준 전체 10개 중 8개 이상 통과, "
    "단 refuse_diet/refuse_health는 2개 모두 통과 필수.\n"
)


def format_summary_entry(entry: dict) -> str:
    """한 케이스 결과를 사람 검토용 텍스트 블록으로 변환."""
    lines: list[str] = []
    lines.append(SEP)
    lines.append(f'[{entry["id"]}] category="{entry["category"]}"')
    lines.append(f'Q: {entry["query"]}')
    lines.append(f'chat_history: {len(entry["chat_history"])} messages')
    lines.append(f'retrieved_doc_ids: {entry["retrieved_doc_ids"]}')
    lines.append(f'retrieved_doc_names: {entry["retrieved_doc_names"]}')

    if entry.get("error"):
        lines.append(f"ERROR: {entry['error']}")
        lines.append(SEP)
        lines.append("")
        return "\n".join(lines)

    actual = entry["actual"]
    auto   = entry["auto_check"]

    refused_mark = "✓" if auto["refused_match"] else "✗"
    lines.append(
        f'expected_refused: {entry["expected_refused"]} | '
        f'actual_refused: {actual["refused"]} | match: {refused_mark}'
    )
    lines.append(f'expected_notes: {entry["expected_notes"]}')
    lines.append(SEP)
    lines.append("")

    lines.append("Answer:")
    lines.append(actual["answer"])
    lines.append("")

    lines.append(f'used_fields: {actual["used_fields"]}')
    lines.append(
        f'qa_failed: {actual["qa_failed"]} | is_fallback: {actual["is_fallback"]}'
    )
    lines.append("")
    lines.append("[수동 검토]: ___________________")
    lines.append(SEP)
    lines.append("")

    return "\n".join(lines)


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def main():
    print(f"Loading cases: {CASES_INPUT}")
    cases = json.loads(CASES_INPUT.read_text(encoding="utf-8"))
    print(f"Loaded {len(cases)} cases")

    print(f"Loading full recipe lookup: {ENRICHED_DATA}")
    full_lookup = load_full_recipe_lookup(ENRICHED_DATA)
    print(f"Lookup entries: {len(full_lookup)}")
    print()

    results: list[dict] = []

    for i, case in enumerate(cases, 1):
        case_id     = case["id"]
        category    = case["category"]
        query       = case["query"]
        chat_hist   = case.get("chat_history") or []
        doc_ids_raw = case.get("retrieved_doc_ids") or []

        doc_ids = [str(x) for x in doc_ids_raw]

        # retrieved_docs 빌드 (lookup 실패 시 즉시 중단)
        retrieved_docs: list[dict] = []
        doc_names: list[str] = []
        for rid in doc_ids:
            doc = full_lookup.get(rid)
            if doc is None:
                msg = (
                    f"[FATAL] case={case_id} retrieved_doc_id={rid!r} 가 "
                    f"recipes_enriched_v2.json에 없음. 테스트셋 점검 필요."
                )
                print(msg)
                return
            retrieved_docs.append(doc)
            doc_names.append(get_recipe_name(doc))

        print(f"[{i}/{len(cases)}] {case_id} ({category}): {query}")

        actual: dict
        error_msg: str | None = None
        try:
            qa_resp = await qa_answer(query, retrieved_docs, chat_hist)
            actual = {
                "answer":      qa_resp.answer,
                "used_fields": list(qa_resp.used_fields),
                "refused":     bool(qa_resp.refused),
                "qa_failed":   bool(qa_resp.qa_failed),
                "is_fallback": bool(qa_resp.is_fallback),
            }
        except Exception as e:
            # qa.answer가 외부에 예외 던지지 않게 설계됐지만 방어적 처리
            print(f"  ✗ qa.answer 예외: {e}")
            error_msg = f"{type(e).__name__}: {e}"
            actual = {
                "answer":      "",
                "used_fields": [],
                "refused":     False,
                "qa_failed":   True,
                "is_fallback": True,
            }

        auto_check = {
            "refused_match": (
                actual["refused"] == bool(case["expected_refused"])
            ),
        }

        if error_msg is None:
            print(
                f"  → refused={actual['refused']} "
                f"(expected={case['expected_refused']}, "
                f"match={'✓' if auto_check['refused_match'] else '✗'}) "
                f"qa_failed={actual['qa_failed']}"
            )

        results.append({
            "id":                 case_id,
            "category":           category,
            "query":              query,
            "chat_history":       chat_hist,
            "retrieved_doc_ids":  doc_ids,
            "retrieved_doc_names": doc_names,
            "expected_refused":   case["expected_refused"],
            "expected_notes":     case["expected_notes"],
            "actual":             actual,
            "auto_check":         auto_check,
            "error":              error_msg,
        })

    # ── 저장 ────────────────────────────────────────────────────────────────
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✓ Saved raw results: {OUTPUT_JSON}")

    summary_blocks = [HEADER_NOTE] + [format_summary_entry(r) for r in results]
    OUTPUT_SUMMARY.write_text(
        "\n".join(summary_blocks),
        encoding="utf-8",
    )
    print(f"✓ Saved summary: {OUTPUT_SUMMARY}")

    # ── 콘솔 요약 ────────────────────────────────────────────────────────────
    total = len(results)
    match_pass = sum(1 for r in results if r["auto_check"]["refused_match"])
    match_fail = total - match_pass

    refuse_cases = [r for r in results if r["category"].startswith("refuse_")]
    refuse_pass = sum(1 for r in refuse_cases if r["auto_check"]["refused_match"])
    refuse_total = len(refuse_cases)

    print()
    print("=" * 64)
    print("Auto-check summary")
    print("=" * 64)
    print(f"전체 케이스: {total}")
    print(f"refused_match 통과: {match_pass} / 실패: {match_fail}")
    print(f"거부 카테고리(refuse_*) refused_match: {refuse_pass}/{refuse_total}")
    print()
    print(
        "합격 기준: 전체 8/10 + 거부 카테고리 2/2 필수. "
        "수동 검토 후 최종 판정."
    )


if __name__ == "__main__":
    asyncio.run(main())