"""ambient 억제 가드 검증 배치 (읽기 전용 호출, 코드 수정 없음).

그룹1: 중립 발화 + 끼니 주입 → 주입값 보존돼야 함 (가드 성공)
그룹2: 끼니 명시/정정 발화 → 발화 끼니로 반영, 가드가 막으면 안 됨 (회귀 없음)
그룹3: 상대시간 발화 → ambient 정상 작동 (과수정 아님)
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

TARGET = "http://localhost:8000/chat"
REPEAT = 5
REQUEST_TIMEOUT = 60.0
SLEEP_BETWEEN = 0.4
RESULT_PATH = Path(__file__).parent / "verify_ambient_result.json"

CASES = [
    # 그룹1: 중립 발화 + 끼니 주입 (버그 재현 조건)
    {"group": 1, "label": "중립_점심주입", "message": "추천해줘", "inject_meal_times": ["점심"], "purpose": "protein", "free_text": "닭가슴살 위주로"},
    {"group": 1, "label": "중립_아침주입", "message": "추천해줘", "inject_meal_times": ["아침"], "purpose": "light", "free_text": "두부로 가볍게"},
    {"group": 1, "label": "중립_저녁주입", "message": "메뉴 골라줘", "inject_meal_times": ["저녁"], "purpose": "protein", "free_text": "소고기 들어간거"},
    {"group": 1, "label": "중립_점심light주입", "message": "추천 부탁해", "inject_meal_times": ["점심"], "purpose": "light", "free_text": "토마토 넣은 가벼운거"},
    # 그룹2: 끼니 명시/정정 발화
    {"group": 2, "label": "명시_저녁protein", "message": "저녁에 단백질 위주로 추천해줘", "inject_meal_times": [], "purpose": None, "free_text": "단백질 위주로 먹고싶어"},
    {"group": 2, "label": "명시_점심light", "message": "점심으로 가볍게 먹을거 추천해줘", "inject_meal_times": [], "purpose": None, "free_text": "가볍게 먹고싶어"},
    {"group": 2, "label": "명시_정정발화", "message": "점심 말고 저녁으로 추천해줘", "inject_meal_times": ["점심"], "purpose": "protein", "free_text": "고기 들어간거"},
    # 그룹3: 상대시간 발화 (ambient 정상 작동해야)
    {"group": 3, "label": "상대시간_지금", "message": "지금 추천해줘", "inject_meal_times": [], "purpose": "light", "free_text": "가벼운거 먹고싶어"},
]


def build_slots(case: dict) -> dict:
    """ChatRequest.Slots 구성. 빈 배열 meal_times / None purpose는 그대로 전달."""
    slots: dict = {"free_text": case["free_text"]}
    slots["meal_times"] = case["inject_meal_times"] if case["inject_meal_times"] else None
    slots["purpose"] = case["purpose"]  # None이면 null로 전달
    return slots


def call_once(case: dict) -> dict:
    payload = {
        "session_id": uuid.uuid4().hex[:12],
        "turn_id": uuid.uuid4().hex[:12],
        "message": case["message"],
        "history": [],
        "slots": build_slots(case),
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
        "menus": [r.get("name") for r in (body.get("recommendations") or [])],
    }


def main() -> None:
    raw: dict[str, list[dict]] = {}
    for case in CASES:
        label = case["label"]
        raw[label] = []
        for i in range(REPEAT):
            r = call_once(case)
            raw[label].append(r)
            tag = "ERR" if "error" in r else f"{r.get('intent')}/{r.get('slots_updated_meal_times')}"
            print(f"  [G{case['group']}][{label}] {i+1}/{REPEAT} -> {tag}")
            time.sleep(SLEEP_BETWEEN)

    RESULT_PATH.write_text(
        json.dumps({"target": TARGET, "cases": CASES, "results": raw}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── 판정 표 ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'G':<3}{'label':<20}{'주입끼니':<10}{'5회 slots_updated.meal_times':<40}{'판정'}")
    print("-" * 100)
    for case in CASES:
        label = case["label"]
        runs = raw[label]
        ok = [r for r in runs if "error" not in r]
        inj = case["inject_meal_times"] or None
        got = [r.get("slots_updated_meal_times") for r in ok]
        got_str = " ".join(str(g) for g in got)

        if case["group"] == 1:
            n_keep = sum(1 for g in got if g == inj)
            verdict = f"보존 {n_keep}/{len(ok)}" + (" ✅" if n_keep == len(ok) and len(ok) == REPEAT else " ❌")
        elif case["group"] == 2:
            n_rec = sum(1 for r in ok if r.get("intent") == "recommend")
            uniq = set(tuple(g) if g else () for g in got)
            verdict = f"intent recommend {n_rec}/{len(ok)}, 끼니패턴={uniq}"
        else:  # group 3
            n_filled = sum(1 for g in got if g)  # 비어있지 않으면 ambient 작동
            verdict = f"ambient채움 {n_filled}/{len(ok)}"
        print(f"{case['group']:<3}{label:<20}{str(inj):<10}{got_str:<40}{verdict}")
    print("=" * 100)
    print(f"\nraw 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
