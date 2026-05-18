"""정형 슬롯 질문 풀과 검출 로직.

orchestrator가 슬롯 부족 시 사용자에게 보낼 질문 문구를 이 모듈에서 가져온다.
또한 history 마지막 assistant 메시지가 슬롯 질문이었는지 판단해
intent 분류기의 previous_assistant_question 입력으로 사용한다.

원칙: 슬롯 질문 텍스트는 절대 모듈 외부에서 하드코딩하지 않는다.
build_slot_question() 또는 SLOT_QUESTIONS 상수만 참조한다.
"""

from __future__ import annotations

from typing import Sequence

from api.schemas import ChatMessage, Slots


# 슬롯 질문 풀. 키는 의미적 분류, 값은 사용자에게 노출되는 정확한 문자열.
# detect_slot_question은 history의 assistant 메시지가 이 값들 중 하나와
# 완전 일치할 때 슬롯 질문으로 판정한다.
SLOT_QUESTIONS: dict[str, str] = {
    "missing_meal_times": "어떤 시간대에 드실 거예요? (아침/점심/저녁/간식/야식)",
    "missing_purpose": "어떤 스타일을 원하세요? (가볍게/단백질/든든하게/맛있게)",
    "missing_both": "어떤 시간대에 어떤 스타일로 드실 거예요?",
    "retrieval_zero": "조건에 맞는 메뉴를 못 찾았어요. 조건을 조금 풀어볼까요?",
    "lookup_insufficient": "조건에 맞는 메뉴를 충분히 못 찾았어요. 조건을 조금 풀어볼까요?",
}


def build_slot_question(slots: Slots) -> str:
    """부족한 슬롯에 맞는 질문 문구를 반환한다.

    Parameters
    ----------
    slots : Slots
        현재 슬롯 상태. meal_times와 purpose 중 비어있는 항목 기준으로 분기.

    Returns
    -------
    str
        SLOT_QUESTIONS 상수 중 하나. 둘 다 채워져 있으면 missing_both를 반환
        (이 경우는 호출자(orchestrator)가 잘못 호출한 상황이지만 안전한 fallback).
    """
    missing_meal_times = not slots.meal_times
    missing_purpose = not slots.purpose

    if missing_meal_times and missing_purpose:
        return SLOT_QUESTIONS["missing_both"]
    if missing_meal_times:
        return SLOT_QUESTIONS["missing_meal_times"]
    if missing_purpose:
        return SLOT_QUESTIONS["missing_purpose"]
    # 둘 다 채워져 있을 때는 호출자 책임. 안전 fallback으로 missing_both 반환.
    return SLOT_QUESTIONS["missing_both"]


def detect_slot_question(history: Sequence[ChatMessage]) -> str | None:
    """history 마지막 assistant 메시지가 슬롯 질문 풀에 속하는지 검사.

    Parameters
    ----------
    history : Sequence[ChatMessage]
        최근 메시지. 비어있으면 None.

    Returns
    -------
    str | None
        슬롯 질문이면 그 문자열 그대로, 아니면 None.
        intent 분류기의 previous_assistant_question 입력으로 사용.
    """
    if not history:
        return None
    last = history[-1]
    if last.role != "assistant":
        return None
    if last.content in SLOT_QUESTIONS.values():
        return last.content
    return None