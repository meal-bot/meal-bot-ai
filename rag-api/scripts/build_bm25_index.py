"""
BM25 인덱스 빌드 스크립트.
data/recipes_enriched_v2.json → bm25/recipes_v2/{bm25.pkl, recipe_ids.pkl, tokenized_corpus.pkl}
"""

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.tokenizer import KiwiTokenizer
from rag.config import BM25_INDEX_PATH


# ── 라벨 매핑 ─────────────────────────────────────────────────────────────────
PURPOSE_LABELS = {
    "light":   "가볍게 가벼운",
    "protein": "단백질 고단백",
    "hearty":  "든든하게 든든한",
    "tasty":   "맛있게",
}

SPICY_LABELS_BM25 = {
    1: "안매운 안매움",
    2: "약간매운 살짝매운",
    3: "보통매운 적당히매운",
    4: "매운",
    5: "매우매운 아주매운 엄청매운",
}


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

def join_list(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    if value is None:
        return ""
    return str(value)


def build_time_text(cooking_time) -> str:
    """
    누적 라벨링: 짧은 시간 버킷일수록 더 많은 버킷에 매칭됨.
    예) 9분 요리 → "9분 약9분 10분 초간단 빠른요리 20분 간단요리 30분 보통요리"
    """
    if cooking_time is None:
        return ""
    try:
        n = int(cooking_time)
    except (ValueError, TypeError):
        return ""

    parts = [f"{n}분", f"약{n}분"]
    if n <= 10:
        parts.append("10분 초간단 빠른요리")
    if n <= 20:
        parts.append("20분 간단요리")
    if n <= 30:
        parts.append("30분 보통요리")
    if n > 30:
        parts.append("오래걸리는 시간있는요리")
    return " ".join(parts)


def build_bm25_text(recipe: dict) -> str:
    parts = []

    # 기본 필드
    parts.append(str(recipe.get("name", "")))
    parts.append(str(recipe.get("summary", "")))
    parts.append(str(recipe.get("category", "")))
    parts.append(str(recipe.get("cooking_method", "")))
    parts.append(str(recipe.get("difficulty", "")))

    # list 필드
    parts.append(join_list(recipe.get("main_ingredients")))
    parts.append(join_list(recipe.get("meal_time")))
    parts.append(join_list(recipe.get("taste_tags")))
    parts.append(join_list(recipe.get("texture_tags")))
    parts.append(join_list(recipe.get("recommended_situations")))
    parts.append(join_list(recipe.get("dish_type_tags")))

    # purpose 라벨 변환
    purpose = recipe.get("purpose") or []
    purpose_labels = " ".join(PURPOSE_LABELS.get(p, p) for p in purpose)
    parts.append(purpose_labels)

    # spicy_level 라벨 변환
    spicy = recipe.get("spicy_level")
    if spicy is not None:
        parts.append(SPICY_LABELS_BM25.get(int(spicy), ""))

    # cooking_time 시간 버킷
    parts.append(build_time_text(recipe.get("cooking_time")))

    # 공백으로 join, 연속 공백 정리
    text = " ".join(p for p in parts if p)
    return " ".join(text.split())


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    data_path = Path("data/recipes_enriched_v2.json")
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = KiwiTokenizer()

    tokenized_corpus = []
    recipe_ids = []
    raw_texts = []

    for recipe in data:
        text = build_bm25_text(recipe)
        tokens = tokenizer.tokenize(text)
        tokenized_corpus.append(tokens)
        recipe_ids.append(str(recipe["rcp_seq"]))
        raw_texts.append(text)

    bm25 = BM25Okapi(tokenized_corpus)

    out_dir = Path(BM25_INDEX_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(out_dir / "recipe_ids.pkl", "wb") as f:
        pickle.dump(recipe_ids, f)
    with open(out_dir / "tokenized_corpus.pkl", "wb") as f:
        pickle.dump(tokenized_corpus, f)

    # 통계
    lens = [len(tc) for tc in tokenized_corpus]
    vocab: set[str] = set()
    for tc in tokenized_corpus:
        vocab.update(tc)

    print(f"총 문서 수: {len(tokenized_corpus)}")
    print(f"평균 토큰 수: {sum(lens) / len(lens):.1f}")
    print(f"최소 토큰 수: {min(lens)}")
    print(f"최대 토큰 수: {max(lens)}")
    print(f"어휘 사전 크기: {len(vocab)}")

    # 샘플 1건
    print(f"\n--- 샘플 1 ---")
    print(f"recipe_id: {recipe_ids[0]}")
    print(f"문서: {raw_texts[0]}")
    print(f"토큰 ({len(tokenized_corpus[0])}개): {tokenized_corpus[0]}")

    # 가장 긴 문서
    max_idx = lens.index(max(lens))
    print(f"\n--- 샘플 2 (가장 긴 문서) ---")
    print(f"recipe_id: {recipe_ids[max_idx]}")
    print(f"문서: {raw_texts[max_idx]}")
    print(f"토큰 ({len(tokenized_corpus[max_idx])}개): {tokenized_corpus[max_idx]}")


if __name__ == "__main__":
    main()