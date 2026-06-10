"""멀티턴 풀 플로우 최종 검증 v2 (안전망 포함 4턴). 읽기 전용 호출, 코드 수정 없음.

상태전이 로직(apply_response_to_next_request)은 v1과 동일 — Spring ChatService 규약.
FLOWS만 교체: 닭가슴살 4턴(slot_fill→recommend→refine→ask) + 샐러드 3턴 재확인.
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
HISTORY_LIMIT = 50
RESULT_PATH = Path(__file__).parent / "flow_result_v2.json"

recipes = json.loads((Path(__file__).parent.parent / "data/recipes_enriched_v2.json").read_text(encoding="utf-8"))
BY_ID = {r["rcp_seq"]: r for r in recipes}

FLOWS = [
    {"label": "닭가슴살_안전망_4턴", "turns": [
        "저녁에 단백질 위주로 먹고싶어",
        "닭가슴살로 부탁해",
        "다른거 추천해줘",
        "1번 메뉴 칼로리랑 단백질 알려줘",
    ]},
    {"label": "샐러드_3턴_재확인", "turns": [
        "점심에 샐러드처럼 가볍게 먹을거 추천해줘",
        "다른 샐러드로 추천해줘",
        "1번 메뉴 칼로리 알려줘",
    ]},
]


def empty_state(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "turn_counter": 0,
        "slots": {"meal_times": None, "purpose": None, "free_text": None},
        "history": [],
        "last_recommendations": [],
    }


def build_payload(state: dict, message: str) -> dict:
    return {
        "session_id": state["session_id"],
        "turn_id": f"t{state['turn_counter']}",
        "message": message,
        "history": state["history"][-HISTORY_LIMIT:],
        "slots": state["slots"],
        "last_recommendations": state["last_recommendations"],
    }


def apply_response_to_next_request(state: dict, message: str, resp: dict) -> dict:
    """Spring ChatService 규약대로 응답을 다음 요청 상태로 전이 (v1과 동일)."""
    intent = resp.get("intent", "?")
    su = resp.get("slots_updated") or {}
    free_text_delta = resp.get("free_text_delta")
    flags = resp.get("flags") or {}
    refused = flags.get("refused", False)

    new_meal_times = su.get("meal_times")
    new_purpose = su.get("purpose")

    prev_ft = state["slots"].get("free_text")
    if free_text_delta is None or str(free_text_delta).strip() == "" or refused:
        new_ft = prev_ft
    elif intent in ("recommend", "slot_fill", "refine"):
        new_ft = free_text_delta if not prev_ft else f"{prev_ft} {free_text_delta}"
    elif intent == "ask":
        new_ft = prev_ft
    else:
        new_ft = prev_ft

    state["slots"] = {"meal_times": new_meal_times, "purpose": new_purpose, "free_text": new_ft}
    state["history"].append({"role": "user", "content": message})
    state["history"].append({"role": "assistant", "content": resp.get("answer", "")})

    recs = resp.get("recommendations") or []
    if intent in ("recommend", "refine") and recs:
        state["last_recommendations"] = [
            {"recipe_id": r.get("recipe_id", ""), "name": r.get("name", "")} for r in recs
        ]
    state["turn_counter"] += 1
    return state


def call(payload: dict) -> dict:
    try:
        r = httpx.post(TARGET, json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        return {"error": f"network: {type(e).__name__}: {e}"}
    if r.status_code >= 400:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    try:
        return r.json()
    except Exception as e:
        return {"error": f"parse: {e}"}


def run_flow(flow: dict) -> list[dict]:
    state = empty_state(uuid.uuid4().hex[:12])
    turn_records = []
    for ti, message in enumerate(flow["turns"]):
        payload = build_payload(state, message)
        sent_snapshot = {
            "slots": json.loads(json.dumps(payload["slots"])),
            "last_recommendations": json.loads(json.dumps(payload["last_recommendations"])),
            "history_len": len(payload["history"]),
        }
        resp = call(payload)
        rec = {"turn_index": ti, "message": message, "sent": sent_snapshot}
        if "error" in resp:
            rec["error"] = resp["error"]
            turn_records.append(rec)
            time.sleep(SLEEP_BETWEEN)
            break
        su = resp.get("slots_updated") or {}
        rec.update({
            "turn_id": resp.get("turn_id"),
            "intent": resp.get("intent"),
            "slots_updated": {"meal_times": su.get("meal_times"), "purpose": su.get("purpose"), "free_text": su.get("free_text")},
            "free_text_delta": resp.get("free_text_delta"),
            "flags": resp.get("flags"),
            "answer": resp.get("answer"),
            "recommendations": [
                {"recipe_id": x.get("recipe_id"), "name": x.get("name"), "reason": x.get("reason")}
                for x in (resp.get("recommendations") or [])
            ],
        })
        turn_records.append(rec)
        state = apply_response_to_next_request(state, message, resp)
        time.sleep(SLEEP_BETWEEN)
    return turn_records


def _meal_of(turn: dict) -> list | None:
    return (turn.get("slots_updated") or {}).get("meal_times")


def main() -> None:
    raw: dict[str, list[list[dict]]] = {}
    for flow in FLOWS:
        label = flow["label"]
        raw[label] = []
        for run_i in range(REPEAT):
            records = run_flow(flow)
            raw[label].append(records)
            seq = "->".join(r.get("intent", "ERR") for r in records)
            print(f"  [{label}] run {run_i+1}/{REPEAT}: {seq}")
    RESULT_PATH.write_text(json.dumps({"target": TARGET, "flows": FLOWS, "results": raw}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    for flow in FLOWS:
        label = flow["label"]
        runs = raw[label]
        nturns = len(flow["turns"])
        print(f"\n########## {label} ({len(runs)}회, {nturns}턴) ##########")
        # 시퀀스
        seqs = Counter("->".join(rr.get("intent", "ERR") for rr in r) for r in runs)
        print(f"  [흐름 시퀀스] {dict(seqs)}")

        # 턴별 intent 분포
        for ti in range(nturns):
            intents = Counter(r[ti].get("intent", "ERR") for r in runs if len(r) > ti)
            print(f"  T{ti} intent 분포: {dict(intents)}")

        # 마지막 턴 ask 정확도 + 슬롯 보존
        ask_ti = nturns - 1
        prev_ti = ask_ti - 1
        ask_ok = num_ok = num_chk = slot_keep = present = 0
        for r in runs:
            if len(r) <= ask_ti or "error" in r[ask_ti]:
                continue
            present += 1
            t_ask = r[ask_ti]; t_prev = r[prev_ti]
            recs_prev = t_prev.get("recommendations", [])
            ans = t_ask.get("answer", "") or ""
            if recs_prev:
                first = recs_prev[0]
                data = BY_ID.get(str(first.get("recipe_id")), {})
                nut = data.get("nutrition", {}) or {}
                kcal = nut.get("energy_kcal"); prot = nut.get("protein_g")
                name_hit = first.get("name", "") and (first["name"] in ans or first["name"][:4] in ans)
                num_hit = False
                if kcal is not None:
                    num_chk += 1
                    if str(round(kcal)) in ans or str(int(kcal)) in ans:
                        num_hit = True; num_ok += 1
                if name_hit or num_hit:
                    ask_ok += 1
            # 슬롯 보존: ask 턴 meal/purpose == 직전 턴
            if _meal_of(t_ask) == _meal_of(t_prev) and (t_ask.get("slots_updated") or {}).get("purpose") == (t_prev.get("slots_updated") or {}).get("purpose") and _meal_of(t_ask):
                slot_keep += 1
        print(f"  [T{ask_ti} ask] 1번특정 OK {ask_ok}/{present}, 수치일치 {num_ok}/{num_chk}, ★슬롯보존 {slot_keep}/{present}")

        # refine 턴 탐지(라벨에 4턴이면 T2가 refine 기대)
        if "안전망" in label:
            # T1 recommend, T2 refine exclude 체크
            t1_rec = sum(1 for r in runs if len(r) > 1 and r[1].get("intent") == "recommend")
            t2_refine = t2_overlap = t2_present = 0
            for r in runs:
                if len(r) > 2 and "error" not in r[2]:
                    t2_present += 1
                    if r[2].get("intent") == "refine":
                        t2_refine += 1
                    t1_ids = {x["recipe_id"] for x in r[1].get("recommendations", [])} if len(r) > 1 else set()
                    t2_ids = {x["recipe_id"] for x in r[2].get("recommendations", [])}
                    if t1_ids & t2_ids:
                        t2_overlap += 1
            print(f"  [T1 recommend] {t1_rec}/{len(runs)} | [T2 refine] {t2_refine}/{t2_present}, T1메뉴재출현 {t2_overlap}(0이어야)")
            # T1후 누적 free_text (T2 sent에 실린 값)
            ft = [r[2]["sent"]["slots"].get("free_text") for r in runs if len(r) > 2 and "sent" in r[2]]
            print(f"  [T1후 누적 free_text] {ft}")

    print("\n" + "=" * 100)
    print(f"raw 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
