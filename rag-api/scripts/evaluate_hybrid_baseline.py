"""Hybrid RRF 검색 baseline 품질 점검 스크립트.

대표 쿼리 10개에 대해 top-30 결과를 JSON으로 저장하고,
top-5 요약을 텍스트와 콘솔로 출력해서 사람이 수동 점검할 수 있게 한다.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import DenseRetriever, BM25Retriever, HybridRetriever


QUERIES = [
    {"id": "q01", "category": "맛-단순",      "query": "매콤한 국물 요리"},
    {"id": "q02", "category": "맛-조합",      "query": "달콤하고 부드러운 디저트"},
    {"id": "q03", "category": "재료",         "query": "두부 들어간 반찬"},
    {"id": "q04", "category": "시간",         "query": "10분 안에 만드는 간단한 한 끼"},
    {"id": "q05", "category": "상황-다이어트", "query": "다이어트에 좋은 저칼로리 메뉴"},
    {"id": "q06", "category": "상황-야식",    "query": "혼자 먹기 좋은 야식"},
    {"id": "q07", "category": "대상-아이",    "query": "아이가 먹기 좋은 부드러운 요리"},
    {"id": "q08", "category": "영양",         "query": "고단백 닭가슴살 요리"},
    {"id": "q09", "category": "부정조건",     "query": "안 매운 국물 요리"},
    {"id": "q10", "category": "키워드정확",   "query": "김치찌개 레시피"},
]


def safe_parse_list(value):
    """metadata의 list 계열 필드를 안전하게 list로 복원."""
    if isinstance(value, list):
        return value

    if value is None:
        return []

    if isinstance(value, tuple):
        return list(value)

    if not isinstance(value, str):
        return [value] if value else []

    text = value.strip()
    if not text:
        return []

    # JSON array string
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    # comma-separated string fallback
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]

    return [text]


def hit_to_dict(hit):
    meta = hit.metadata or {}

    return {
        "rank": hit.rank,
        "recipe_id": hit.recipe_id,
        "name": hit.name,
        "rrf_score": round(float(hit.score), 6),
        "dense_rank": hit.dense_rank,
        "bm25_rank": hit.bm25_rank,
        "category": meta.get("category"),
        "cooking_method": meta.get("cooking_method") or meta.get("cooking_way"),
        "cooking_time": meta.get("cooking_time"),
        "difficulty": meta.get("difficulty"),
        "dish_type_tags":         safe_parse_list(meta.get("dish_type_tags")),
        "taste_tags":             safe_parse_list(meta.get("taste_tags")),
        "texture_tags":           safe_parse_list(meta.get("texture_tags")),
        "recommended_situations": safe_parse_list(meta.get("recommended_situations")),
        "summary": meta.get("summary"),
    }


def format_top5(query_result):
    lines = []
    lines.append("═" * 60)
    lines.append(
        f"[{query_result['id']} / {query_result['category']}] "
        f"{query_result['query']}"
    )
    lines.append("═" * 60)

    for r in query_result["results"][:5]:
        d = r["dense_rank"] if r["dense_rank"] is not None else "-"
        b = r["bm25_rank"]  if r["bm25_rank"]  is not None else "-"

        cooking_time = r["cooking_time"]
        cooking_time_text = f"{cooking_time}분" if cooking_time is not None else "-"

        lines.append(f"Rank {r['rank']} | recipe_id={r['recipe_id']} | {r['name']}")
        lines.append(f"       rrf={r['rrf_score']:.4f} | D={d} | B={b}")
        lines.append(
            f"       category: {r['category']} | "
            f"method: {r['cooking_method']} | "
            f"cooking_time: {cooking_time_text} | "
            f"difficulty: {r['difficulty']}"
        )
        lines.append(f"       dish_type_tags:         {r['dish_type_tags']}")
        lines.append(f"       taste_tags:             {r['taste_tags']}")
        lines.append(f"       texture_tags:           {r['texture_tags']}")
        lines.append(f"       recommended_situations: {r['recommended_situations']}")
        if r.get("summary"):
            lines.append(f"       summary: {r['summary']}")
        lines.append("")

    return "\n".join(lines)


def main():
    print("Loading retrievers...")
    dense  = DenseRetriever()
    bm25   = BM25Retriever()
    hybrid = HybridRetriever(dense, bm25)

    all_results = []
    summary_text = []

    for q in QUERIES:
        print(f"  Searching: {q['id']} - {q['query']}")
        hits = hybrid.search(q["query"], top_k=30)

        result = {
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "results": [hit_to_dict(h) for h in hits],
        }

        all_results.append(result)
        summary_text.append(format_top5(result))

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    json_path = artifacts / "hybrid_baseline_top30.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    summary_path = artifacts / "hybrid_baseline_top5_summary.txt"
    summary_str = "\n".join(summary_text)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_str)

    print()
    print(summary_str)
    print()
    print("✅ 저장 완료:")
    print(f"   - {json_path}")
    print(f"   - {summary_path}")


if __name__ == "__main__":
    main()