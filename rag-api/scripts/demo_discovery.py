"""RAG 시연 시나리오 발굴용 배치 테스트 스크립트.

slots를 직접 주입해 history 없이 추천 턴을 재현한다. 각 조합을 3회씩 호출해
(1) recommend 안정성 (2) 추천 메뉴 일관성 (3) 빈손/에러 여부를 본다.

기존 코드는 수정하지 않으며, 읽기 전용으로 /chat 엔드포인트만 호출한다.
실행: python scripts/demo_discovery.py  (서버가 localhost:8000에 떠 있어야 함)
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

TARGET = "http://localhost:8000/chat"
REPEAT = 3
REQUEST_TIMEOUT = 60.0
SLEEP_BETWEEN = 0.5
RESULT_PATH = Path(__file__).parent / "discovery_result.json"

COMBINATIONS = [
    # 저녁 + protein (후보 473, 단백질 재료)
    {"label": "저녁_protein_닭가슴살", "meal_times": ["저녁"], "purpose": "protein", "free_text": "닭가슴살 위주로 먹고싶어"},
    {"label": "저녁_protein_소고기", "meal_times": ["저녁"], "purpose": "protein", "free_text": "소고기 들어간거"},
    {"label": "저녁_protein_돼지고기", "meal_times": ["저녁"], "purpose": "protein", "free_text": "돼지고기로 단백질"},
    {"label": "저녁_protein_두부", "meal_times": ["저녁"], "purpose": "protein", "free_text": "두부 같은 식물성 단백질"},
    {"label": "저녁_protein_새우", "meal_times": ["저녁"], "purpose": "protein", "free_text": "새우 들어간 단백질 메뉴"},
    {"label": "저녁_protein_달걀", "meal_times": ["저녁"], "purpose": "protein", "free_text": "달걀 위주 단백질"},
    # 저녁 + light (후보 596, 최다)
    {"label": "저녁_light_두부", "meal_times": ["저녁"], "purpose": "light", "free_text": "두부로 가볍게"},
    {"label": "저녁_light_토마토", "meal_times": ["저녁"], "purpose": "light", "free_text": "토마토 들어간 가벼운거"},
    {"label": "저녁_light_채소", "meal_times": ["저녁"], "purpose": "light", "free_text": "채소 위주로 담백하게"},
    {"label": "저녁_light_오이", "meal_times": ["저녁"], "purpose": "light", "free_text": "오이 같은 상큼한 재료"},
    # 점심 + protein (후보 440)
    {"label": "점심_protein_닭가슴살", "meal_times": ["점심"], "purpose": "protein", "free_text": "닭가슴살로 단백질 채우기"},
    {"label": "점심_protein_소고기", "meal_times": ["점심"], "purpose": "protein", "free_text": "소고기 단백질 메뉴"},
    # 점심 + light (후보 534)
    {"label": "점심_light_토마토", "meal_times": ["점심"], "purpose": "light", "free_text": "토마토 넣은 가벼운 점심"},
    {"label": "점심_light_파프리카", "meal_times": ["점심"], "purpose": "light", "free_text": "파프리카 같은 채소로 가볍게"},
    {"label": "점심_light_샐러드", "meal_times": ["점심"], "purpose": "light", "free_text": "샐러드처럼 가벼운거"},
    # 저녁/점심 + hearty (후보 157/158, hearty는 이쪽만)
    {"label": "저녁_hearty_감자", "meal_times": ["저녁"], "purpose": "hearty", "free_text": "감자 들어간 든든한거"},
    {"label": "저녁_hearty_단호박", "meal_times": ["저녁"], "purpose": "hearty", "free_text": "단호박으로 든든하게"},
    {"label": "점심_hearty_고기", "meal_times": ["점심"], "purpose": "hearty", "free_text": "고기 들어간 든든한 점심"},
    # 간식 + tasty (후보 185, 간식 유일 풍부)
    {"label": "간식_tasty_달콤", "meal_times": ["간식"], "purpose": "tasty", "free_text": "달콤한 간식거리"},
    {"label": "간식_tasty_과일", "meal_times": ["간식"], "purpose": "tasty", "free_text": "과일 들어간 맛있는 간식"},
]


def call_once(combo: dict) -> dict:
    """단일 호출. 성공 시 추출 필드, 실패 시 error 키를 담은 dict 반환."""
    payload = {
        "session_id": uuid.uuid4().hex[:12],
        "turn_id": uuid.uuid4().hex[:12],
        "message": "추천해줘",
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

    recs = body.get("recommendations", []) or []
    return {
        "intent": body.get("intent"),
        "answer": body.get("answer"),
        "slots_updated": body.get("slots_updated"),
        "recommendations": [
            {"recipe_id": r.get("recipe_id"), "name": r.get("name"), "reason": r.get("reason")}
            for r in recs
        ],
    }


def main() -> None:
    raw: dict[str, list[dict]] = {}

    for combo in COMBINATIONS:
        label = combo["label"]
        raw[label] = []
        for i in range(REPEAT):
            result = call_once(combo)
            raw[label].append(result)
            tag = "ERR" if "error" in result else result.get("intent")
            print(f"  [{label}] {i + 1}/{REPEAT} -> {tag}")
            time.sleep(SLEEP_BETWEEN)

    RESULT_PATH.write_text(
        json.dumps({"target": TARGET, "results": raw}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── 터미널 요약 표 ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'조합':<24} {'성공':<6} {'recommend율':<12} {'고유메뉴(3회)':<10} 대표 reason")
    print("-" * 100)
    for combo in COMBINATIONS:
        label = combo["label"]
        runs = raw[label]
        ok = [r for r in runs if "error" not in r]
        n_ok = len(ok)
        n_rec = sum(1 for r in ok if r.get("intent") == "recommend")
        # 3회 동안 나온 고유 메뉴 set
        menus = set()
        for r in ok:
            for rec in r.get("recommendations", []):
                menus.add(rec.get("name"))
        # 대표 reason 1개
        rep_reason = ""
        for r in ok:
            recs = r.get("recommendations", [])
            if recs:
                rep_reason = (recs[0].get("reason") or "")[:40]
                break
        consistency = f"{len(menus)}종"
        rec_rate = f"{n_rec}/{REPEAT}"
        print(f"{label:<24} {n_ok}/{REPEAT}   {rec_rate:<12} {consistency:<12} {rep_reason}")

    print("=" * 100)

    # ── 분류 요약 ─────────────────────────────────────────────────────
    stable, inconsistent, problem = [], [], []
    for combo in COMBINATIONS:
        label = combo["label"]
        runs = raw[label]
        ok = [r for r in runs if "error" not in r]
        n_rec = sum(1 for r in ok if r.get("intent") == "recommend")
        menus = set()
        for r in ok:
            for rec in r.get("recommendations", []):
                menus.add(rec.get("name"))
        if len(ok) < REPEAT or n_rec < REPEAT or not menus:
            problem.append(label)
        elif len(menus) == 2:  # 3회 모두 같은 2메뉴
            stable.append(label)
        else:
            inconsistent.append(label)

    print(f"\n[안정적 recommend + 메뉴 일관(2종)] {len(stable)}개: {stable}")
    print(f"[recommend지만 메뉴 변동] {len(inconsistent)}개: {inconsistent}")
    print(f"[문제(빈손/에러/recommend탈락)] {len(problem)}개: {problem}")
    print(f"\nraw 결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
