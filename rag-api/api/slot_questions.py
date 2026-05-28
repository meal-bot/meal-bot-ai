"""정형 슬롯 질문 풀과 검출 로직.

orchestrator가 슬롯 부족 시 사용자에게 보낼 질문 문구를 이 모듈에서 가져온다.
또한 history 마지막 assistant 메시지가 슬롯 질문이었는지 판단해
intent 분류기의 previous_assistant_question 입력으로 사용한다.

원칙: 슬롯 질문 텍스트는 절대 모듈 외부에서 하드코딩하지 않는다.
build_slot_question() 또는 SLOT_QUESTIONS 상수만 참조한다.

UX 노트:
- 두 슬롯 모두 비어있는 첫 질문에는 SLOT_INTRO 안내 멘트를 앞에 붙여
  사용자에게 슬롯 채우기 흐름을 알리고 시간대부터 하나씩 묻는다.
- 두 번째 질문(purpose)에는 SLOT_INTRO를 붙이지 않는다.
- detect_slot_question은 SLOT_INTRO 접두사를 떼고 비교하므로
  intent 분류기에는 항상 정규화된 질문 문구만 노출된다.
"""

from __future__ import annotations

from typing import Sequence

from api.schemas import ChatMessage, Slots


# 슬롯 질문 풀. 키는 의미적 분류, 값은 사용자에게 노출되는 정확한 문자열.
# detect_slot_question은 history의 assistant 메시지가 이 값들 중 하나와
# (SLOT_INTRO 접두사를 제거한 뒤) 완전 일치할 때 슬롯 질문으로 판정한다.
SLOT_QUESTIONS: dict[str, str] = {
    "missing_meal_times": "어떤 시간대에 드실 거예요? (아침 🌅 / 점심 ☀️ / 저녁 🌙 / 간식 🍪 / 야식 🌃)",
    "missing_purpose": "어떤 스타일을 원하세요? (가볍게 🥗 / 근육 💪 / 든든하게 🍚 / 맛있게 😋)",
    "missing_both": "어떤 시간대에 어떤 스타일로 드실 거예요?",
    "retrieval_zero": "조건에 맞는 메뉴를 못 찾았어요. 조건을 조금 풀어볼까요?",
    "lookup_insufficient": "조건에 맞는 메뉴를 충분히 못 찾았어요. 조건을 조금 풀어볼까요?",
}


# 첫 슬롯 질문(둘 다 비어있는 케이스)에 한 번만 붙이는 안내 멘트.
# detect_slot_question이 비교 전에 이 접두사를 제거하므로 intent 분류기는
# 항상 SLOT_QUESTIONS.values()와 동일한 정규화된 문구만 보게 된다.
SLOT_INTRO = "아래 질문에 답해주시면 더 좋은 식단을 추천해드릴 수 있어요!\n\n"


def build_slot_question(slots: Slots) -> str:
    """부족한 슬롯에 맞는 질문 문구를 반환한다.

    Parameters
    ----------
    slots : Slots
        현재 슬롯 상태. meal_times와 purpose 중 비어있는 항목 기준으로 분기.

    Returns
    -------
    str
        - 둘 다 비어있으면(첫 질문) SLOT_INTRO + missing_meal_times 를 반환해
          안내 멘트와 함께 시간대부터 묻는다.
        - 하나만 비면 해당 질문만 반환.
        - 둘 다 채워져 있으면 호출자(orchestrator) 측 실수이므로
          안전 fallback으로 missing_both를 반환한다.
    """
    missing_meal_times = not slots.meal_times
    missing_purpose = not slots.purpose

    if missing_meal_times and missing_purpose:
        # 첫 질문: 안내 멘트 + 시간대 질문부터 하나씩 진행.
        return SLOT_INTRO + SLOT_QUESTIONS["missing_meal_times"]
    if missing_meal_times:
        return SLOT_QUESTIONS["missing_meal_times"]
    if missing_purpose:
        return SLOT_QUESTIONS["missing_purpose"]
    # 둘 다 채워져 있을 때는 호출자 책임. 안전 fallback으로 missing_both 반환.
    return SLOT_QUESTIONS["missing_both"]


def detect_slot_question(history: Sequence[ChatMessage]) -> str | None:
    """history 마지막 assistant 메시지가 슬롯 질문 풀에 속하는지 검사.

    첫 슬롯 질문에는 SLOT_INTRO 접두사가 붙어 있을 수 있으므로
    비교 전에 접두사를 제거하고 SLOT_QUESTIONS.values()와 비교한다.

    Parameters
    ----------
    history : Sequence[ChatMessage]
        최근 메시지. 비어있으면 None.

    Returns
    -------
    str | None
        슬롯 질문이면 SLOT_INTRO 접두사를 제거한 정규화된 문자열,
        아니면 None. intent 분류기의 previous_assistant_question 입력으로 사용.
    """
    if not history:
        return None
    last = history[-1]
    if last.role != "assistant":
        return None

    # SLOT_INTRO 접두사가 붙은 첫 질문도 정상 매칭되도록 떼고 비교.
    content = last.content
    if content.startswith(SLOT_INTRO):
        content = content[len(SLOT_INTRO):]

    if content in SLOT_QUESTIONS.values():
        return content
    return None
