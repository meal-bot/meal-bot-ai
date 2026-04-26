"""
식약처 COOKRCP01 API → data/raw/recipes_raw.json (원본)
                      → data/processed/recipes.json (정제본)

실행 방법:
    cd /path/to/meal-bot/rag-api
    source venv/bin/activate
    python scripts/collect_recipes.py
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("FOOD_API_KEY")
BASE_URL = "http://openapi.foodsafetykorea.go.kr/api/{key}/COOKRCP01/json/{start}/{end}"

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "recipes_raw.json"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "recipes.json"

MANUAL_COUNT = 20

GARBAGE_PATTERN = re.compile(r"^[a-zA-Z가-힣]{1}$|^\s*$")
TRAILING_GARBAGE = re.compile(r"\.[a-zA-Z]+$")


# ── 수집 ──────────────────────────────────────────────────────────────────────

def fetch_batch(start: int, end: int) -> dict:
    """
    1회 배치 응답을 dict째 반환 (호출부에서 total_count, row 꺼내 씀).
    """
    url = BASE_URL.format(key=API_KEY, start=start, end=end)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result_code = data.get("COOKRCP01", {}).get("RESULT", {}).get("CODE", "")
    if result_code != "INFO-000":
        msg = data.get("COOKRCP01", {}).get("RESULT", {}).get("MSG", "unknown")
        raise RuntimeError(f"API 오류: {result_code} — {msg}")

    return data["COOKRCP01"]


def fetch_total_count() -> int:
    """전체 레시피 개수 조회 (1건 요청)."""
    body = fetch_batch(1, 1)
    return int(body["total_count"])


def fetch_all(total: int) -> list[dict]:
    rows = []
    batch_size = 1000

    for start in range(1, total + 1, batch_size):
        end = min(start + batch_size - 1, total)
        print(f"  요청: {start}~{end} ...", end=" ", flush=True)
        batch = fetch_batch(start, end).get("row", [])
        rows.extend(batch)
        print(f"{len(batch)}개 수신")
        if end < total:
            time.sleep(0.5)

    return rows


# ── 정제 ──────────────────────────────────────────────────────────────────────

def clean_str(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    if GARBAGE_PATTERN.match(v):
        return None
    return v or None


def clean_numeric(value: str | None) -> float | None:
    if not value:
        return None
    v = value.strip()
    if not v or GARBAGE_PATTERN.match(v):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_manuals(row: dict) -> list[dict]:
    steps = []
    for i in range(1, MANUAL_COUNT + 1):
        idx = str(i).zfill(2)
        text = clean_str(row.get(f"MANUAL{idx}"))
        img = clean_str(row.get(f"MANUAL_IMG{idx}"))
        if text:
            desc = TRAILING_GARBAGE.sub(".", text)
            steps.append({"step": i, "desc": desc, "img": img})
    return steps


def process_row(row: dict) -> dict | None:
    name = clean_str(row.get("RCP_NM"))
    if not name:
        return None

    return {
        "rcp_seq":      clean_str(row.get("RCP_SEQ")),
        "name":         name,
        "cooking_way":  clean_str(row.get("RCP_WAY2")),
        "category":     clean_str(row.get("RCP_PAT2")),
        "ingredients":  clean_str(row.get("RCP_PARTS_DTLS")),
        "hash_tag":     clean_str(row.get("HASH_TAG")),
        "img_main":     clean_str(row.get("ATT_FILE_NO_MAIN")),
        "img_thumb":    clean_str(row.get("ATT_FILE_NO_MK")),
        "nutrition": {
            "energy_kcal": clean_numeric(row.get("INFO_ENG")),
            "carbs_g":     clean_numeric(row.get("INFO_CAR")),
            "protein_g":   clean_numeric(row.get("INFO_PRO")),
            "fat_g":       clean_numeric(row.get("INFO_FAT")),
            "sodium_mg":   clean_numeric(row.get("INFO_NA")),
        },
        "manuals": parse_manuals(row),
    }


def process_all(raw: list[dict]) -> tuple[list[dict], int]:
    processed, skipped = [], 0
    for row in raw:
        result = process_row(row)
        if result is None:
            skipped += 1
        else:
            processed.append(result)
    return processed, skipped


# ── 통계 ──────────────────────────────────────────────────────────────────────

def print_stats(raw: list[dict], processed: list[dict], skipped: int):
    kept = len(processed)
    null_energy = sum(1 for r in processed if r["nutrition"]["energy_kcal"] is None)
    null_protein = sum(1 for r in processed if r["nutrition"]["protein_g"] is None)
    no_manuals  = sum(1 for r in processed if not r["manuals"])
    no_img      = sum(1 for r in processed if not r["img_main"])

    categories: dict[str, int] = {}
    for r in processed:
        cat = r["category"] or "미분류"
        categories[cat] = categories.get(cat, 0) + 1

    print("\n" + "=" * 50)
    print("[ 정제 통계 ]")
    print(f"  원본:          {len(raw)}개")
    print(f"  정제 후:       {kept}개  (제거: {skipped}개)")
    print(f"  칼로리 없음:   {null_energy}개 ({null_energy/kept*100:.1f}%)")
    print(f"  단백질 없음:   {null_protein}개 ({null_protein/kept*100:.1f}%)")
    print(f"  조리순서 없음: {no_manuals}개")
    print(f"  이미지 없음:   {no_img}개")
    print(f"\n  카테고리 분포:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {cnt}개")
    print("=" * 50)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== 1단계: API 수집 ===")
    total = fetch_total_count()
    print(f"  전체 레시피 수: {total}")
    raw = fetch_all(total=total)

    print(f"\n원본 저장 중 → {RAW_PATH}")
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"완료: {len(raw)}개")

    print("\n=== 2단계: 정제 ===")
    processed, skipped = process_all(raw)

    print(f"\n정제본 저장 중 → {PROCESSED_PATH}")
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print_stats(raw, processed, skipped)


if __name__ == "__main__":
    main()
