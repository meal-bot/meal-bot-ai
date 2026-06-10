"""RAG 시연 시나리오 발굴 배치 v2 (읽기 전용 호출, 코드 수정 없음).

v1과의 차이: message에 끼니를 담은 자연 발화 + slots에도 동일 끼니 주입(이중 안전).
저녁/점심 중심, 빈약 조합 제외. ambient 오염을 회피한 상태에서 추천 품질·안정성 발굴.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from pathlib import Path

import httpx

TARGET = "http://localhost:8000/chat"
REPEAT = 5
REQUEST_TIMEOUT = 60.0
SLEEP_BETWEEN = 0.5
RESULT_PATH = Path(__file__).parent / "discovery_v2_result.json"

PROTEIN_ING = ["닭", "소고기", "돼지", "두부", "새우", "달걀", "계란"]
HEAVY_KW = ["튀김", "강정", "탕수", "까스", "프라이"]

COMBINATIONS = [
    {"label": "저녁_protein_닭가슴살", "message": "저녁에 단백질 위주로 먹고싶어", "meal_times": ["저녁"], "purpose": "protein", "free_text": "닭가슴살 들어간거"},
    {"label": "저녁_protein_소고기", "message": "저녁에 단백질 챙길 수 있는거 추천해줘", "meal_times": ["저녁"], "purpose": "protein", "free_text": "소고기 들어간거"},
    {"label": "저녁_protein_돼지고기", "message": "저녁으로 단백질 많은거 먹고싶어", "meal_times": ["저녁"], "purpose": "protein", "free_text": "돼지고기 요리"},
    {"label": "저녁_protein_두부", "message": "저녁에 단백질 위주로 가볍게", "meal_times": ["저녁"], "purpose": "protein", "free_text": "두부 같은 식물성 단백질"},
    {"label": "저녁_protein_달걀", "message": "저녁에 단백질 보충할 메뉴 추천", "meal_times": ["저녁"], "purpose": "protein", "free_text": "달걀 들어간거"},
    {"label": "저녁_light_두부", "message": "저녁에 가볍게 먹고싶어", "meal_times": ["저녁"], "purpose": "light", "free_text": "두부로 담백하게"},
    {"label": "저녁_light_채소", "message": "저녁으로 가벼운 채소 요리 추천해줘", "meal_times": ["저녁"], "purpose": "light", "free_text": "채소 위주로"},
    {"label": "저녁_light_토마토", "message": "저녁에 산뜻하게 먹을거", "meal_times": ["저녁"], "purpose": "light", "free_text": "토마토 들어간 가벼운거"},
    {"label": "점심_protein_닭가슴살", "message": "점심에 단백질 위주로 먹고싶어", "meal_times": ["점심"], "purpose": "protein", "free_text": "닭가슴살로"},
    {"label": "점심_protein_소고기", "message": "점심으로 단백질 챙길거 추천해줘", "meal_times": ["점심"], "purpose": "protein", "free_text": "소고기 들어간거"},
    {"label": "점심_light_토마토", "message": "점심에 가볍게 먹고싶어", "meal_times": ["점심"], "purpose": "light", "free_text": "토마토 넣은 가벼운거"},
    {"label": "점심_light_파프리카", "message": "점심으로 산뜻한 채소 요리", "meal_times": ["점심"], "purpose": "light", "free_text": "파프리카 같은 채소"},
    {"label": "점심_light_샐러드", "message": "점심에 샐러드처럼 가벼운거 추천", "meal_times": ["점심"], "purpose": "light", "free_text": "샐러드 느낌으로"},
    {"label": "저녁_hearty_감자", "message": "저녁에 든든하게 먹고싶어", "meal_times": ["저녁"], "purpose": "hearty", "free_text": "감자 들어간 든든한거"},
    {"label": "저녁_hearty_단호박", "message": "저녁으로 든든한 한끼 추천해줘", "meal_times": ["저녁"], "purpose": "hearty", "free_text": "단호박으로 든든하게"},
    {"label": "점심_hearty_고기", "message": "점심에 든든하게 먹을거", "meal_times": ["점심"], "purpose": "hearty", "free_text": "고기 들어간 든든한거"},
]


def call_once(combo: dict) -> dict:
    payload = {
        "session_id": uuid.uuid4().hex[:12],
        "turn_id": uuid.uuid4().hex[:12],
        "message": combo["message"],
        "history": [],
        "slots": {
            "meal_times": combo["meal_times"],
            "purpose": combo["purpose"],
            "free_text": combo["free_text"],
        },
        "last_recommendations": [],
    }
    try:
        resp = httpx.post(TARGET, json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        return {"error": f"network: {type(e).__name__}: {e}"}
    if resp.status_code >= 400:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
    try:
        body = resp.json()
    except Exception as e:
        return {"error": f"parse: {e}"}
    su = body.get("slots_updated") or {}
    return {
        "intent": body.get("intent"),
        "slots_updated_meal_times": su.get("meal_times"),
        "slots_updated_purpose": su.get("purpose"),
        "answer": body.get("answer"),
        "recommendations": [
            {"recipe_id": r.get("recipe_id"), "name": r.get("name"), "reason": r.get("reason")}
            for r in (body.get("recommendations") or [])
        ],
    }


def main() -> None:
    raw: dict[str, list[dict]] = {}
    for combo in COMBINATIONS:
        label = combo["label"]
        raw[label] = []
        for i in range(REPEAT):
            r = call_once(combo)
            raw[label].append(r)
            tag = "ERR" if "error" in r else f"{r.get('intent')}/{r.get('slots_updated_meal_times')}"
            print(f"  [{label}] {i+1}/{REPEAT} -> {tag}")
            time.sleep(SLEEP_BETWEEN)

    RESULT_PATH.write_text(
        json.dumps({"target": TARGET, "combinations": COMBINATIONS, "results": raw},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── 요약 표 ────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"{'조합':<24}{'rec율':<7}{'끼니일치':<9}{'고유메뉴':<9}대표메뉴쌍")
    print("-" * 110)
    for combo in COMBINATIONS:
        label = combo["label"]
        inj = combo["meal_times"]
        runs = raw[label]
        ok = [r for r in runs if "error" not in r]
        n_rec = sum(1 for r in ok if r.get("intent") == "recommend")
        n_match = sum(1 for r in ok if r.get("slots_updated_meal_times") == inj)
        menu_counter = Counter()
        pairs = []
        for r in ok:
            names = tuple(sorted(rec.get("name") for rec in r.get("recommendations", [])))
            if names:
                pairs.append(names)
            for rec in r.get("recommendations", []):
                menu_counter[rec.get("name")] += 1
        rep_pair = pairs[0] if pairs else ()
        print(f"{label:<24}{n_rec}/{REPEAT}    {n_match}/{REPEAT}      {len(menu_counter):<9}{rep_pair}")
    print("=" * 110)

    # ── 메뉴 빈도 상세 + 대표 reason ──────────────────────────────────
    print("\n[조합별 고유메뉴 빈도 + 대표 reason]")
    for combo in COMBINATIONS:
        label = combo["label"]
        ok = [r for r in raw[label] if "error" not in r]
        menu_counter = Counter()
        rep_reason = ""
        for r in ok:
            for rec in r.get("recommendations", []):
                menu_counter[rec.get("name")] += 1
                if not rep_reason:
                    rep_reason = rec.get("reason") or ""
        freq = ", ".join(f"{m}×{c}" for m, c in menu_counter.most_common())
        print(f"\n  [{label}]")
        print(f"    메뉴빈도: {freq}")
        print(f"    대표reason: {rep_reason}")

    # ── 정합성 1차 체크 ───────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("[정합성 위반 목록]")
    violations = []
    for combo in COMBINATIONS:
        label = combo["label"]
        is_protein = "protein" in label
        is_light = "light" in label
        is_hearty = "hearty" in label
        for i, r in enumerate(raw[label]):
            if "error" in r:
                continue
            recs = r.get("recommendations", [])
            if is_hearty and r.get("intent") == "recommend" and not recs:
                violations.append((label, i + 1, "(빈손)", "hearty인데 추천 0건"))
            for rec in recs:
                name = rec.get("name") or ""
                reason = rec.get("reason") or ""
                hay = name + " " + reason
                if is_protein and not any(k in hay for k in PROTEIN_ING):
                    violations.append((label, i + 1, name, "protein인데 단백질 주재료 키워드 없음"))
                if is_light and any(k in name for k in HEAVY_KW):
                    violations.append((label, i + 1, name, "light인데 무거운 조리법"))
    if not violations:
        print("  위반 없음 (키워드 기준)")
    else:
        for label, turn, name, why in violations:
            print(f"  [{label}] {turn}회차: {name} — {why}")
    print(f"\n  총 위반 {len(violations)}건 (거친 키워드 매칭, 사람 최종판단)")
    print(f"\nraw 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
