"""POST /chat 엔드포인트를 대화형으로 검증하는 단일 파일 CLI.

Spring이 담당해야 할 최소한의 상태 관리(session_id, turn_id, 6턴 히스토리,
last_recommendations 갱신)만 흉내낸다. slots는 백엔드 응답을 그대로 echo한다.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any

# httpx 우선, 없으면 requests 폴백
try:
    import httpx  # type: ignore
    _HTTP_LIB = "httpx"
except ImportError:
    try:
        import requests  # type: ignore
        _HTTP_LIB = "requests"
    except ImportError:
        print("Error: httpx 또는 requests가 필요합니다.", file=sys.stderr)
        sys.exit(1)

# 한글 입력 누락 회피용 prompt_toolkit. 미설치면 표준 input() 폴백.
try:
    from prompt_toolkit import prompt as _pt_prompt  # type: ignore
    _USE_PT = True
except ImportError:
    _USE_PT = False


DEFAULT_BASE_URL = "http://localhost:8000"
# 서버 HISTORY_MAX_MESSAGES와 동일하게 유지
HISTORY_SEND_LIMIT = 50   # API 요청 시 잘라 보내는 최근 메시지 개수
REQUEST_TIMEOUT = 60.0


class ChatClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id: str = uuid.uuid4().hex[:12]
        self.turn_counter: int = 0
        self.slots: dict[str, Any] = self._empty_slots()
        self.history: list[dict[str, str]] = []
        self.last_recommendations: list[dict[str, str]] = []
        self.debug_mode: bool = True

    @staticmethod
    def _empty_slots() -> dict[str, Any]:
        return {"meal_times": None, "purpose": None, "free_text": None}

    def reset(self) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.turn_counter = 0
        self.slots = self._empty_slots()
        self.history = []
        self.last_recommendations = []

    # ── HTTP ────────────────────────────────────────────────────────────

    def _post_chat(self, message: str) -> dict[str, Any] | None:
        """POST /chat. 실패 시 None 반환 + 화면에 사유 출력."""
        payload = {
            "session_id": self.session_id,
            "turn_id": f"t{self.turn_counter}",
            "message": message,
            "history": self.history[-HISTORY_SEND_LIMIT:],
            "slots": self.slots,
            "last_recommendations": self.last_recommendations,
        }
        url = f"{self.base_url}/chat"
        try:
            if _HTTP_LIB == "httpx":
                resp = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                status = resp.status_code
                text = resp.text
            else:
                resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                status = resp.status_code
                text = resp.text
        except Exception as e:
            print(f"[network error] {e}")
            return None

        if status >= 400:
            try:
                body = resp.json()
                detail = body.get("detail", body) if isinstance(body, dict) else body
                print(f"[HTTP {status}] {json.dumps(detail, ensure_ascii=False)}")
            except Exception:
                print(f"[HTTP {status}] {text[:500]}")
            return None

        try:
            return resp.json()
        except Exception as e:
            print(f"[parse error] {e} body={text[:200]}")
            return None

    # ── send + state update ─────────────────────────────────────────────

    # free_text 누적 규칙 (Spring 백엔드와 동일):
    # - recommend/slot_fill: 기존값 + " " + delta로 누적
    # - refine: delta로 덮어쓰기 (조건 좁히기)
    # - ask: 유지 (변경 없음)
    # - flags.refused=True 또는 delta=None: 유지
    def send(self, message: str) -> None:
        resp = self._post_chat(message)
        if resp is None:
            # 상태 변경 없이 루프 계속
            return

        # 1) 상태 업데이트 먼저 → debug 출력이 갱신된 상태를 보여줌
        intent = resp.get("intent", "?")
        slots_updated = resp.get("slots_updated") or {}
        free_text_delta = resp.get("free_text_delta")
        flags = resp.get("flags") or {}
        refused = flags.get("refused", False)

        # meal_times / purpose는 서버 응답 그대로 echo
        new_meal_times = slots_updated.get("meal_times")
        new_purpose = slots_updated.get("purpose")

        # free_text는 stateless 서버 대신 클라이언트가 intent별 규칙으로 누적/유지
        prev = self.slots.get("free_text")
        if free_text_delta is None or refused:
            new_free_text = prev
        elif intent in ("recommend", "slot_fill"):
            if prev is None or prev == "":
                new_free_text = free_text_delta
            else:
                new_free_text = f"{prev} {free_text_delta}"
        elif intent == "refine":
            new_free_text = free_text_delta
        elif intent == "ask":
            new_free_text = prev
        else:
            # 알 수 없는 intent는 안전하게 유지
            new_free_text = prev

        self.slots = {
            "meal_times": new_meal_times,
            "purpose": new_purpose,
            "free_text": new_free_text,
        }
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": resp.get("answer", "")})

        if intent in ("recommend", "refine"):
            self.last_recommendations = [
                {"recipe_id": r.get("recipe_id", ""), "name": r.get("name", "")}
                for r in (resp.get("recommendations") or [])
            ]
        # ask/slot_fill에서는 last_recommendations 유지

        self.turn_counter += 1

        # 2) 화면 출력
        self._print_response(resp)

    # ── 출력 ────────────────────────────────────────────────────────────

    def _print_response(self, resp: dict[str, Any]) -> None:
        intent = resp.get("intent", "?")
        answer = resp.get("answer", "")
        print(f"Bot [{intent}]: {answer}")

        recs = resp.get("recommendations") or []
        if intent in ("recommend", "refine") and recs:
            for i, r in enumerate(recs, 1):
                name = r.get("name", "")
                rid = r.get("recipe_id", "")
                print(f"  {i}. {name} (id: {rid})")
                reason = r.get("reason", "")
                if reason:
                    print(f"     {reason}")
                meta_parts: list[str] = []
                ing = r.get("main_ingredients") or []
                if ing:
                    meta_parts.append("주재료: " + ", ".join(ing))
                ct = r.get("cooking_time")
                if ct is not None:
                    meta_parts.append(f"조리시간: {ct}분")
                if meta_parts:
                    print("     " + " | ".join(meta_parts))

        if self.debug_mode:
            slots = resp.get("slots_updated", {})
            flags = resp.get("flags", {})
            free_text_delta = resp.get("free_text_delta")
            print()
            print("  [debug]")
            print(f"  slots: {json.dumps(slots, ensure_ascii=False)}")
            print(f"  free_text_accumulated: {json.dumps(self.slots.get('free_text'), ensure_ascii=False)}")
            print(f"  flags: {json.dumps(flags, ensure_ascii=False)}")
            print(f"  free_text_delta: {json.dumps(free_text_delta, ensure_ascii=False)}")
            print(
                f"  history_len: {len(self.history)}, "
                f"last_recs: {len(self.last_recommendations)}"
            )


# ── 명령어 처리 ─────────────────────────────────────────────────────────

HELP_TEXT = """Commands:
  /quit, /exit  - 종료
  /reset        - 세션 전체 초기화 (새 session_id)
  /debug        - debug 출력 토글
  /state        - 현재 상태 출력
  /clear_free_text - free_text 누적값만 초기화 (slots 다른 필드와 history는 유지)
  /help         - 이 도움말"""


def _handle_command(client: ChatClient, line: str) -> bool:
    """명령어면 처리하고 True, 종료 명령이면 SystemExit."""
    cmd = line.strip()
    if cmd in ("/quit", "/exit"):
        print("Bye.")
        raise SystemExit(0)
    if cmd == "/reset":
        client.reset()
        print(f"[reset. new session: {client.session_id}]")
        return True
    if cmd == "/debug":
        client.debug_mode = not client.debug_mode
        print(f"[debug: {'ON' if client.debug_mode else 'OFF'}]")
        return True
    if cmd == "/state":
        print(f"session: {client.session_id}, turn: {client.turn_counter}")
        print(f"slots: {json.dumps(client.slots, ensure_ascii=False)}")
        print(f"history: {len(client.history)} messages")
        print(f"last_recommendations: {len(client.last_recommendations)}")
        return True
    if cmd == "/clear_free_text":
        client.slots["free_text"] = None
        print("[free_text cleared]")
        return True
    if cmd == "/help":
        print(HELP_TEXT)
        return True
    print(f"[unknown command: {cmd}. /help for list]")
    return True


# ── main ────────────────────────────────────────────────────────────────


def main() -> None:
    base_url = os.getenv("MEALBOT_BASE_URL", DEFAULT_BASE_URL)
    client = ChatClient(base_url)

    print(f"[session: {client.session_id}] (debug: {'ON' if client.debug_mode else 'OFF'})")
    print(f"[target: {client.base_url}/chat | http lib: {_HTTP_LIB}]")
    print(f"[input: {'prompt_toolkit' if _USE_PT else 'stdlib input()'}]")
    print("Type /help for commands.")

    try:
        while True:
            try:
                if _USE_PT:
                    line = _pt_prompt("\nYou: ")
                else:
                    line = input("\nYou: ")
            except EOFError:
                print("\nBye.")
                return

            if not line.strip():
                continue

            if line.lstrip().startswith("/"):
                _handle_command(client, line)
                continue

            client.send(line)
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
