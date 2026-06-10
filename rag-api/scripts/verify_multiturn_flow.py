"""멀티턴 풀 플로우 검증 (추천→refine→ask). 읽기 전용 호출, 코드 수정 없음.

Spring ChatService 실동작 규약대로 턴간 상태를 전이한다(apply_response_to_next_request).
검증: refine 안정성 / ask 정확도(1번 특정 + 수치 환각) / ask 턴 슬롯 보존 / 흐름 일관성.
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
RESULT_PATH = Path(__file__).parent / "flow_result.json"

recipes = json.loads((Path(__file__).parent.parent / "data/recipes_enriched_v2.json").read_text(encoding="utf-8"))
BY_ID = {r["rcp_seq"]: r for r in recipes}

FLOWS = [
    {"label": "샐러드_풀플로우", "turns": [
        "점심에 샐러드처럼 가볍게 먹을거 추천해줘",
        "다른 샐러드로 추천해줘",
        "1번 메뉴 칼로리 알려줘",
    ]},
    {"label": "닭가슴살_풀플로우", "turns": [
        "저녁에 단백질 위주로 먹고싶어",
        "다른거 추천해줘",
        "1번 메뉴 칼로리랑 단백질 알려줘",
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
    """Spring ChatService 규약대로 응답을 다음 요청 상태로 전이."""
    intent = resp.get("intent", "?")
    su = resp.get("slots_updated") or {}
    free_text_delta = resp.get("free_text_delta")
    flags = resp.get("flags") or {}
    refused = flags.get("refused", False)

    # meal_times/purpose: 직전 응답 slots_updated로 매 턴 통째 overwrite (intent 무관, ask도)
    new_meal_times = su.get("meal_times")
    new_purpose = su.get("purpose")

    # free_text: slots_updated.free_text(echo) 무시, free_text_delta를 intent별 누적
    prev_ft = state["slots"].get("free_text")
    if free_text_delta is None or str(free_text_delta).strip() == "" or refused:
        new_ft = prev_ft  # null/blank/refused → 변경 없음
    elif intent in ("recommend", "slot_fill", "refine"):
        # append(교체 아님). refine도 기존 free_text 뒤에 " "로 append
        new_ft = free_text_delta if not prev_ft else f"{prev_ft} {free_text_delta}"
    elif intent == "ask":
        new_ft = prev_ft  # ask는 변경 없음
    else:
        new_ft = prev_ft

    state["slots"] = {"meal_times": new_meal_times, "purpose": new_purpose, "free_text": new_ft}

    # history: 직전 턴까지 누적. assistant content=answer만 (추천목록 미포함)
    state["history"].append({"role": "user", "content": message})
    state["history"].append({"role": "assistant", "content": resp.get("answer", "")})

    # last_recommendations: recommend/refine + 비어있지 않으면 {recipe_id,name} 순서보존 갱신
    recs = resp.get("recommendations") or []
    if intent in ("recommend", "refine") and recs:
        state["last_recommendations"] = [
            {"recipe_id": r.get("recipe_id", ""), "name": r.get("name", "")} for r in recs
        ]
    # ask/slot_fill: 직전 값 유지

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
    """한 세션(3턴) 실행. 턴별 요청/응답/전이상태 기록."""
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
            break  # 플로우 중단(후속 턴 의미 없음)
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

    # ── 판정 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    for flow in FLOWS:
        label = flow["label"]
        runs = raw[label]
        print(f"\n########## {label} ({len(runs)}회) ##########")

        # T0
        t0_recommend = sum(1 for r in runs if len(r) > 0 and r[0].get("intent") == "recommend")
        t0_meal_match = 0
        inj_meal = ["점심"] if "점심" in flow["turns"][0] else (["저녁"] if "저녁" in flow["turns"][0] else [])
        for r in runs:
            if len(r) > 0 and (r[0].get("slots_updated") or {}).get("meal_times") == inj_meal:
                t0_meal_match += 1
        print(f"  [T0 recommend] intent=recommend {t0_recommend}/{len(runs)}, 끼니={inj_meal} 일치 {t0_meal_match}/{len(runs)}")

        # T1 refine
        t1_refine = 0; t1_overlap = 0; t1_reclassified = 0; t1_present = 0
        for r in runs:
            if len(r) < 2 or "error" in r[1]:
                continue
            t1_present += 1
            it = r[1].get("intent")
            if it == "refine":
                t1_refine += 1
            elif it == "recommend":
                t1_reclassified += 1
            t0_ids = {x["recipe_id"] for x in r[0].get("recommendations", [])} if len(r) > 0 else set()
            t1_ids = {x["recipe_id"] for x in r[1].get("recommendations", [])}
            if t0_ids & t1_ids:
                t1_overlap += 1
        print(f"  [T1 refine] intent=refine {t1_refine}/{t1_present}, recommend재분류 {t1_reclassified}, T0메뉴재출현 {t1_overlap} (둘다 0이어야 정상)")
        # 누적 free_text 확인 (T2 요청에 실린 free_text = T1까지 누적)
        ft_samples = []
        for r in runs:
            if len(r) >= 3 and "sent" in r[2]:
                ft_samples.append(r[2]["sent"]["slots"].get("free_text"))
        print(f"  [T1후 누적 free_text 샘플] {ft_samples[:2]}")

        # T2 ask
        t2_ask = 0; t2_present = 0; t2_target_ok = 0; t2_num_ok = 0; t2_num_checks = 0
        slot_keep = 0
        for r in runs:
            if len(r) < 3 or "error" in r[2]:
                continue
            t2_present += 1
            if r[2].get("intent") == "ask":
                t2_ask += 1
            # "1번" = T1 추천 1번 메뉴
            t1_recs = r[1].get("recommendations", []) if len(r) > 1 else []
            answer = r[2].get("answer", "") or ""
            if t1_recs:
                first = t1_recs[0]
                fid = first.get("recipe_id"); fname = first.get("name", "")
                # answer가 1번 메뉴명을 포함하거나, 수치가 그 레시피 nutrition과 일치하면 OK
                data = BY_ID.get(str(fid), {})
                nut = data.get("nutrition", {}) or {}
                kcal = nut.get("energy_kcal"); prot = nut.get("protein_g")
                name_hit = fname and (fname in answer or fname[:4] in answer)
                num_in_answer = False
                if kcal is not None and (str(int(kcal)) in answer or str(round(kcal)) in answer):
                    num_in_answer = True; t2_num_ok += 1; t2_num_checks += 1
                elif kcal is not None:
                    t2_num_checks += 1
                if name_hit or num_in_answer:
                    t2_target_ok += 1
            # 슬롯 보존: T2 응답 slots_updated meal/purpose == T1까지 누적값
            t1_slots = r[1].get("slots_updated", {}) if len(r) > 1 else {}
            t2_slots = r[2].get("slots_updated", {})
            if t2_slots.get("meal_times") == t1_slots.get("meal_times") and t2_slots.get("purpose") == t1_slots.get("purpose") and t2_slots.get("meal_times"):
                slot_keep += 1
        print(f"  [T2 ask] intent=ask {t2_ask}/{t2_present}, 1번특정 OK {t2_target_ok}/{t2_present}, 수치일치 {t2_num_ok}/{t2_num_checks}")
        print(f"  [★슬롯보존] T2 meal_times/purpose가 T1과 동일유지 {slot_keep}/{t2_present}")

        # 흐름 시퀀스
        seqs = Counter("->".join(rr.get("intent", "ERR") for rr in r) for r in runs)
        print(f"  [흐름 시퀀스] {dict(seqs)}")

    print("\n" + "=" * 100)
    print(f"raw 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
