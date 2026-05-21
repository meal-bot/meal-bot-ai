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
HISTORY_SEND_LIMIT = 6   # API 요청 시 잘라 보내는 최근 메시지 개수
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

    def send(self, message: str) -> None:
        resp = self._post_chat(message)
        if resp is None:
            # 상태 변경 없이 루프 계속
            return

        # 1) 상태 업데이트 먼저 → debug 출력이 갱신된 상태를 보여줌
        intent = resp.get("intent", "?")
        self.slots = resp.get("slots_updated") or self.slots
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
