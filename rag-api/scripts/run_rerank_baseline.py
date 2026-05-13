"""LLM rerank baseline 실행 스크립트 (일회성 평가용).

artifacts/hybrid_baseline_top30.json의 10개 쿼리에 대해 rerank를 순차 실행하고,
- artifacts/rerank_baseline_v1.json         (raw JSON)
- artifacts/rerank_baseline_v1_summary.txt  (사람 검토용)
로 저장한다.

누락된 4개 필드(spicy_level, main_ingredients, meal_time, purpose)는
recipes_enriched_v2.json에서 lookup해서 보강한다.
보강은 이 스크립트 안에서만 처리. reranker.py는 수정하지 않는다.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.reranker import rerank


# ── 경로 ──────────────────────────────────────────────────────────────────────

BASELINE_INPUT = Path("artifacts/hybrid_baseline_top30.json")
ENRICHED_DATA  = Path("data/recipes_enriched_v2.json")
OUTPUT_JSON    = Path("artifacts/rerank_baseline_v1.json")
OUTPUT_SUMMARY = Path("artifacts/rerank_baseline_v1_summary.txt")


# ── 보강 ─────────────────────────────────────────────────────────────────────

_AUGMENT_FIELDS_DEFAULTS = {
    "spicy_level":      None,
    "main_ingredients": [],
    "meal_time":        [],
    "purpose":          [],
}


def load_enriched_lookup(path: Path) -> dict[str, dict]:
    """recipes_enriched_v2.json을 rcp_seq → 4개 보강 필드 dict로 변환."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    lookup: dict[str, dict] = {}
    for r in data:
        rid = str(r.get("rcp_seq", ""))
        if not rid:
            continue
        lookup[rid] = {
            "spicy_level":      r.get("spicy_level"),
            "main_ingredients": r.get("main_ingredients") or [],
            "meal_time":        r.get("meal_time")        or [],
            "purpose":          r.get("purpose")          or [],
        }
    return lookup


def enrich_candidate(candidate: dict, lookup: dict[str, dict]) -> dict:
    """baseline 후보에 4개 누락 필드 보강. 원본 미수정."""
    enriched = dict(candidate)
    rid = str(candidate.get("recipe_id", ""))
    extra = lookup.get(rid, {})

    for field, default in _AUGMENT_FIELDS_DEFAULTS.items():
        if field in enriched and enriched[field] not in (None, [], ""):
            # baseline에 값이 이미 있으면 우선 유지
            continue
        enriched[field] = extra.get(field, default)

    return enriched


# ── 요약 ─────────────────────────────────────────────────────────────────────

SEP = "=" * 64


def format_summary_entry(
    query_id:   str,
    query_text: str,
    category:   str,
    response:   dict,
    name_map:   dict[str, str],
) -> str:
    """한 쿼리에 대한 사람 검토용 텍스트 블록."""
    lines: list[str] = []
    lines.append(SEP)
    lines.append(f'[{query_id}] category="{category}"')
    lines.append(f"Q: {query_text}")
    lines.append(
        f"insufficient_matches: {response.get('insufficient_matches', False)}"
    )
    if response.get("error"):
        lines.append(f"ERROR: {response['error']}")
    lines.append(SEP)
    lines.append("")

    recs = response.get("recommendations") or []
    if not recs:
        lines.append("(no recommendations)")
        lines.append("")
    else:
        for r in recs:
            rid = r.get("recipe_id", "?")
            name = name_map.get(rid, "이름은 별도 조회 필요")
            intents = ", ".join(r.get("matched_intents") or [])
            lines.append(f"{r.get('rank', '?')}. recipe_id={rid} | {name}")
            lines.append(f"   reason: {r.get('reason', '')}")
            lines.append(f"   intents: [{intents}]")
            lines.append("")

    return "\n".join(lines)


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def main():
    print(f"Loading baseline: {BASELINE_INPUT}")
    baseline = json.loads(BASELINE_INPUT.read_text(encoding="utf-8"))
    print(f"Loaded {len(baseline)} queries")

    print(f"Loading enriched data for field augmentation: {ENRICHED_DATA}")
    lookup = load_enriched_lookup(ENRICHED_DATA)
    print(f"Lookup table: {len(lookup)} recipes")

    results:   list[dict] = []
    summaries: list[str]  = []

    for i, item in enumerate(baseline, 1):
        query_id       = item["id"]
        query_text     = item["query"]
        category       = item.get("category", "")
        candidates_raw = item["results"]

        print(f"\n[{i}/{len(baseline)}] {query_id}: {query_text}")

        name_map = {c["recipe_id"]: c.get("name", "?") for c in candidates_raw}

        enriched_candidates = [enrich_candidate(c, lookup) for c in candidates_raw]

        try:
            response = await rerank(query_text, enriched_candidates)
            response_dict = response.model_dump()
            print(
                f"  → top-{len(response.recommendations)} returned, "
                f"insufficient={response.insufficient_matches}"
            )
        except Exception as e:
            print(f"  ✗ rerank failed: {e}")
            response_dict = {
                "recommendations": [],
                "insufficient_matches": True,
                "error": str(e),
            }

        result_entry = {
            "id":               query_id,
            "category":         category,
            "query":            query_text,
            "hybrid_top30_ids": [c["recipe_id"] for c in candidates_raw],
            "rerank_response":  response_dict,
            "name_map":         name_map,
        }
        results.append(result_entry)

        summaries.append(format_summary_entry(
            query_id, query_text, category, response_dict, name_map,
        ))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✓ Saved raw results: {OUTPUT_JSON}")

    OUTPUT_SUMMARY.write_text(
        "\n".join(summaries),
        encoding="utf-8",
    )
    print(f"✓ Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    asyncio.run(main())