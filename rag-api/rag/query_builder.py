"""정형 입력(시간대/목적) + 자유 입력을 단일 query string으로 합성.

HybridRetriever.search가 자유 입력 string 하나만 받기 때문에,
FastAPI 레이어에서 받은 정형 필드를 임베딩 텍스트 포맷과 동일한
한글 라벨로 평탄화하는 역할을 한다.
"""


# ── 라벨 매핑 (private) ──────────────────────────────────────────────────────

_PURPOSE_LABELS: dict[str, str] = {
    "light":   "가볍게",
    "protein": "단백질 챙기기",
    "hearty":  "든든하게",
    "tasty":   "맛있게",
}


# ── 공개 함수 ────────────────────────────────────────────────────────────────

def build_retrieval_query(
    meal_times: list[str],
    purpose:    str,
    free_text:  str | None = None,
) -> str:
    """정형 입력 + 자유 입력 → HybridRetriever용 단일 query string."""
    # 입력 검증
    if not meal_times:
        raise ValueError("meal_times가 비어 있음. 1개 이상 필요.")

    if purpose not in _PURPOSE_LABELS:
        raise ValueError(
            f"purpose={purpose!r} 가 정의되지 않음. "
            f"허용값: {sorted(_PURPOSE_LABELS.keys())}"
        )

    # 합성
    meal_times_str = " ".join(meal_times)
    purpose_label  = _PURPOSE_LABELS[purpose]

    parts = [
        f"시간대: {meal_times_str}.",
        f"목적: {purpose_label}.",
    ]

    if free_text is not None and free_text.strip():
        parts.append(free_text.strip())

    return " ".join(parts)