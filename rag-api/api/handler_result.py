"""분기 핸들러 공용 반환 타입.

각 핸들러(recommend, refine, qa_handler)는 자기 책임 필드만 반환한다.
공통 필드(turn_id, slots_updated, flags 기본값)는 orchestrator가 조립한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from api.schemas import Recommendation


@dataclass
class HandlerResult:
    """핸들러 반환 타입.

    Fields
    ------
    intent : str
        이 핸들러가 결정한 최종 intent ("recommend" / "refine" / "ask" / "slot_fill")
        slot_fill은 핸들러가 내부 fallback으로 전환한 경우 사용
        (예: recommend 핸들러가 0건 retrieval로 slot_fill 폴백)
    answer : str
        사용자에게 보여줄 응답 문자열, 항상 비어있지 않음
    recommendations : list[Recommendation]
        0개 또는 2개. intent에 따라 결정됨.
    flags_override : dict[str, bool]
        핸들러가 결정하는 플래그만 dict 형태로 반환.
        키는 needs_more_slots / out_of_scope / is_fallback 중 일부.
        orchestrator가 기본값(False)에 override 적용.
    timings : dict[str, int]
        핸들러 내부 측정한 latency. 키 예: retrieval_ms, rerank_ms, qa_ms, query_rebuild_ms.
        orchestrator가 자기 단계 timings와 합쳐서 최종 로깅.
    """

    intent: Literal["recommend", "refine", "ask", "slot_fill"]
    answer: str
    recommendations: list[Recommendation] = field(default_factory=list)
    flags_override: dict[str, bool] = field(default_factory=dict)
    timings: dict[str, int] = field(default_factory=dict)