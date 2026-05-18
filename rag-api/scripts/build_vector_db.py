"""
LLM 보강 완료된 레시피를 ChromaDB에 적재.

사용법:
  python scripts/build_vector_db.py
"""

import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── 경로 ──────────────────────────────────────────────────────────────────────
INPUT_PATH      = Path("data/recipes_enriched_v2.json")
CHROMA_PATH     = Path("chroma/recipes_v2")
COLLECTION_NAME = "meal_bot_recipes_v2"

# ── 모델·배치 ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE      = 32

# ── 라벨 매핑 ─────────────────────────────────────────────────────────────────
PURPOSE_LABELS: dict[str, str] = {
    "light":   "가볍게",
    "protein": "단백질",
    "hearty":  "든든하게",
    "tasty":   "맛있게",
}
SPICY_LABELS: dict[int, str] = {
    1: "안 매움",
    2: "약간 매움",
    3: "매움",
    4: "아주 매움",
}


# ── 텍스트 빌더 ───────────────────────────────────────────────────────────────

def build_embedding_text(recipe: dict) -> str:
    """임베딩 대상 텍스트 생성."""
    name            = recipe.get("name", "")
    category        = recipe.get("category", "")
    cooking_method  = recipe.get("cooking_method", "")
    summary         = recipe.get("summary", "")

    main_ingredients = recipe.get("main_ingredients") or []
    meal_time        = recipe.get("meal_time") or []
    purpose          = recipe.get("purpose") or []
    spicy_level      = recipe.get("spicy_level", 1)
    cooking_time     = recipe.get("cooking_time")

    taste_tags             = recipe.get("taste_tags") or []
    texture_tags           = recipe.get("texture_tags") or []
    recommended_situations = recipe.get("recommended_situations") or []
    dish_type_tags         = recipe.get("dish_type_tags") or []
    difficulty             = recipe.get("difficulty", "")

    purpose_str = ", ".join(PURPOSE_LABELS.get(p, p) for p in purpose)
    spicy_label = SPICY_LABELS.get(spicy_level, "안 매움")
    time_line   = f"조리시간: 약 {cooking_time}분" if cooking_time is not None else "조리시간: 미상"

    return (
        f"{name} | {category} | {cooking_method}\n"
        f"{summary}\n"
        f"주재료: {', '.join(main_ingredients)}\n"
        f"시간대: {', '.join(meal_time)}\n"
        f"목적: {purpose_str}\n"
        f"매운맛: {spicy_label}\n"
        f"{time_line}\n"
        f"음식유형: {', '.join(dish_type_tags)}\n"
        f"맛: {', '.join(taste_tags)}\n"
        f"식감: {', '.join(texture_tags)}\n"
        f"추천상황: {', '.join(recommended_situations)}\n"
        f"난이도: {difficulty}"
    )


# ── 메타데이터 빌더 ───────────────────────────────────────────────────────────

def build_metadata(recipe: dict) -> dict:
    """ChromaDB 메타데이터 생성 (스칼라 전용; list는 JSON 직렬화)."""
    # 영양정보: nutrition이 dict가 아니거나 energy_kcal이 0·None이면 -1.0
    nutrition = recipe.get("nutrition")
    if isinstance(nutrition, dict) and nutrition.get("energy_kcal"):
        kcal     = float(nutrition.get("energy_kcal")    or -1.0)
        carbs    = float(nutrition.get("carbs_g")         or -1.0)
        protein  = float(nutrition.get("protein_g")      or -1.0)
        fat      = float(nutrition.get("fat_g")          or -1.0)
        sodium   = float(nutrition.get("sodium_mg")      or -1.0)
        if kcal == 0.0:
            kcal = carbs = protein = fat = sodium = -1.0
    else:
        kcal = carbs = protein = fat = sodium = -1.0

    # 조리 단계: manuals[].desc 줄바꿈 연결
    manuals = recipe.get("manuals") or []
    steps = [
        step["desc"].strip()
        for step in manuals
        if isinstance(step, dict) and step.get("desc", "").strip()
    ]
    cooking_steps_text = "\n".join(steps)

    return {
        "recipe_id":          str(recipe.get("rcp_seq", "")),
        "name":               str(recipe.get("name", "")),
        "category":           str(recipe.get("category", "")),
        "cooking_method":     str(recipe.get("cooking_method", "")),
        "meal_time":          json.dumps(recipe.get("meal_time") or [],          ensure_ascii=False),
        "purpose":            json.dumps(recipe.get("purpose") or [],            ensure_ascii=False),
        "spicy_level":        int(recipe.get("spicy_level", 1)),
        "summary":            str(recipe.get("summary", "")),
        "question_1":         str(recipe.get("question_1", "")),
        "question_2":         str(recipe.get("question_2", "")),
        "question_3":         str(recipe.get("question_3", "")),
        "main_ingredients":   json.dumps(recipe.get("main_ingredients") or [],   ensure_ascii=False),
        "cooking_time":       int(recipe.get("cooking_time", 0)),
        "image_url":          str(recipe.get("img_main", "") or ""),
        "thumbnail_url":      str(recipe.get("img_thumb", "") or ""),
        "cooking_steps_text":    cooking_steps_text,
        "taste_tags":            json.dumps(recipe.get("taste_tags") or [],             ensure_ascii=False),
        "texture_tags":          json.dumps(recipe.get("texture_tags") or [],           ensure_ascii=False),
        "recommended_situations":json.dumps(recipe.get("recommended_situations") or [], ensure_ascii=False),
        "dish_type_tags":        json.dumps(recipe.get("dish_type_tags") or [],         ensure_ascii=False),
        "difficulty":            str(recipe.get("difficulty") or ""),
        "kcal":                  kcal,
        "carbs_g":            carbs,
        "protein_g":          protein,
        "fat_g":              fat,
        "sodium_mg":          sodium,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. 입력 파일 확인
    if not INPUT_PATH.exists():
        print(f"[에러] 입력 파일 없음: {INPUT_PATH}")
        print("먼저 scripts/enrich_recipes.py를 실행해 recipes_enriched.json을 생성하세요.")
        return

    # 2. 데이터 로드
    print(f"데이터 로드 중: {INPUT_PATH}")
    with INPUT_PATH.open(encoding="utf-8") as f:
        recipes: list[dict] = json.load(f)
    print(f"  {len(recipes)}건 로드 완료")

    # 3. ChromaDB 초기화 (기존 컬렉션 삭제 후 재생성)
    print(f"\nChromaDB 초기화: {CHROMA_PATH}")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  기존 컬렉션 '{COLLECTION_NAME}' 삭제")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  컬렉션 '{COLLECTION_NAME}' 생성 완료")

    # 4. 임베딩 모델 로드
    print(f"\n임베딩 모델 로드 중: {EMBEDDING_MODEL}")
    print("  (첫 실행 시 ~2.2GB 다운로드 필요)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  모델 로드 완료")

    # 5. 배치 처리
    total    = len(recipes)
    start_ts = time.perf_counter()
    print(f"\n적재 시작 (총 {total}건, 배치 크기: {BATCH_SIZE})")

    for batch_start in tqdm(
        range(0, total, BATCH_SIZE),
        desc="적재 중",
        unit="배치",
    ):
        batch     = recipes[batch_start: batch_start + BATCH_SIZE]
        ids       = [f"recipe_{r['rcp_seq']}" for r in batch]
        documents = [build_embedding_text(r) for r in batch]
        metadatas = [build_metadata(r) for r in batch]

        try:
            embeddings = model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        except Exception as e:
            names = [r.get("name") or r.get("rcp_seq") for r in batch]
            print(f"\n[에러] 임베딩 실패 (배치 offset={batch_start}): {e}")
            print(f"  레시피: {names}")
            raise

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    elapsed = time.perf_counter() - start_ts

    # 6. 검증
    count = collection.count()
    print(f"\n총 적재: {count}건 (소요 시간: {elapsed:.1f}초)")

    if count != total:
        print(f"[경고] 예상 {total}건 vs 실제 {count}건 — 불일치 확인 필요")

    # 샘플 검색 (BGE-M3로 쿼리 임베딩 후 query_embeddings 사용)
    sample_query = "비 오는 날 따뜻한 국물이 먹고 싶을 때"
    print(f"\n샘플 검색: '{sample_query}'")

    query_embedding = model.encode(
        [sample_query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n[{i+1}] {meta['name']} (distance={dist:.3f})")
        print(f"    category: {meta['category']} | meal_time: {meta['meal_time']}")
        print(f"    summary: {meta['summary'][:60]}...")

    print(f"\n저장 경로: {CHROMA_PATH}")


if __name__ == "__main__":
    main()