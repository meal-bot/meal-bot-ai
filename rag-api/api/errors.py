"""api 모듈 도메인 예외. orchestrator가 try/except로 분기 처리하는 데 사용."""

from __future__ import annotations


class IntentClassifyError(Exception):
    """의도 분류 실패. JSON 파싱 실패, enum 외 값, 타임아웃 등."""


class SlotExtractError(Exception):
    """슬롯 추출 실패. JSON 파싱 실패, 스키마 위반, 타임아웃 등."""


class QueryRebuildError(Exception):
    """refine query 재구성 실패. orchestrator가 결정론적 fallback concat query로 대체."""


class RerankError(Exception):
    """LLM rerank 실패. orchestrator가 retrieval score 순 top2로 fallback."""