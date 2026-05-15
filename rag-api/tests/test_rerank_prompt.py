"""rag.rerank_prompt.build_user_prompt — [사용자 제약] 블록 단위 테스트.

structured_inputs 확장이 정확히 user prompt에 들어가는지만 검증.
LLM 호출은 검증 대상 아님.
"""

from rag.rerank_prompt import build_user_prompt


CONSTRAINT_HEADER = "[사용자 제약]"


# ── 빈/없는 structured_inputs ──────────────────────────────────────────────

def test_structured_inputs_none_omits_constraint_block():
    """structured_inputs=None이면 [사용자 제약] 블록이 프롬프트에 없어야 한다."""
    prompt = build_user_prompt(
        query="안 매운 국물",
        candidates=[],
        structured_inputs=None,
    )
    assert CONSTRAINT_HEADER not in prompt
    # 기존 블록은 그대로 있어야 한다
    assert "# 사용자 질의" in prompt
    assert "# 작업" in prompt


def test_structured_inputs_empty_dict_omits_constraint_block():
    """빈 dict({})도 표시할 라인이 없으므로 블록 자체 생략."""
    prompt = build_user_prompt(
        query="국물 요리",
        candidates=[],
        structured_inputs={},
    )
    assert CONSTRAINT_HEADER not in prompt


# ── 풀세트 ─────────────────────────────────────────────────────────────────

def test_full_structured_inputs_includes_all_four_lines():
    """모든 필드가 채워지면 4개 라인 전부 포함."""
    prompt = build_user_prompt(
        query="국물 요리",
        candidates=[],
        structured_inputs={
            "meal_times":    ["저녁", "야식"],
            "purpose":       "light",
            "purpose_label": "가볍게",
            "spicy_max":     2,
            "spicy_label":   "약간 매운 정도까지",
            "free_text":     "국물 요리 추천해줘",
        },
    )
    assert CONSTRAINT_HEADER in prompt
    assert "- 시간대: 저녁, 야식" in prompt
    assert "- 목적: 가볍게" in prompt
    assert "- 매운맛 허용도: 약간 매운 정도까지 (spicy_level 2 이하 선호)" in prompt
    assert "- 자유 요청: 국물 요리 추천해줘" in prompt

    # 제약 블록이 사용자 질의 블록보다 먼저 등장해야 한다
    assert prompt.index(CONSTRAINT_HEADER) < prompt.index("# 사용자 질의")


# ── 부분 입력 ──────────────────────────────────────────────────────────────

def test_only_meal_times_shows_only_meal_time_line():
    """meal_times만 있고 나머지가 None이면 시간대 라인만 표시."""
    prompt = build_user_prompt(
        query="추천",
        candidates=[],
        structured_inputs={
            "meal_times":    ["아침"],
            "purpose":       None,
            "purpose_label": None,
            "spicy_max":     None,
            "spicy_label":   None,
            "free_text":     None,
        },
    )
    assert CONSTRAINT_HEADER in prompt
    assert "- 시간대: 아침" in prompt
    assert "- 목적" not in prompt
    assert "- 매운맛 허용도" not in prompt
    assert "- 자유 요청" not in prompt


def test_free_text_whitespace_only_omits_line():
    """free_text가 공백만이면 자유 요청 라인을 표시하지 않는다."""
    prompt = build_user_prompt(
        query="추천",
        candidates=[],
        structured_inputs={
            "meal_times":    ["점심"],
            "free_text":     "   \t\n  ",
        },
    )
    assert CONSTRAINT_HEADER in prompt
    assert "- 시간대: 점심" in prompt
    assert "- 자유 요청" not in prompt


def test_spicy_max_without_label_uses_fallback_format():
    """spicy_label 없이 spicy_max=3만 있을 때 fallback 포맷."""
    prompt = build_user_prompt(
        query="추천",
        candidates=[],
        structured_inputs={
            "spicy_max": 3,
        },
    )
    assert CONSTRAINT_HEADER in prompt
    assert "- 매운맛 허용도: spicy_level 3 이하 선호" in prompt
    # 라벨 결합 형태는 들어가지 않아야 한다
    assert "(spicy_level 3 이하 선호)" not in prompt