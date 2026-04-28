"""
사용자용 추천 답변 생성 모듈.
이미 정렬된 레시피 hits를 받아 자연어 추천 답변만 생성한다.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from core.config import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if TYPE_CHECKING:
    from core.retrieval import Hit
else:
    Hit = Any

MODEL_NAME = "gpt-4o-mini"
ANSWER_TEMPERATURE = 0.3
DEFAULT_ANSWER_TOP_K = 5
FORBIDDEN_EXPRESSIONS = [
    "건강한",
    "건강하게",
    "건강에 도움이",
    "건강에 도움",
    "영양가",
    "풍부",
    "고소한",
    "상큼한",
    "맛있는",
    "맛있게",
    "좋은 맛",
    "시원한",
    "깔끔한",
    "부드러운 식감",
    "자극적이지 않은 맛",
    "아이가 좋아할",
    "보기 좋은",
    "고급스러운",
    "저칼로리",
    "고단백",
    "짜지 않은",
    "짜지 않고",
    "맛이 좋",
    "독특한 맛",
    "칼로리가 낮",
    "저염",
    "짜지 않",
    "영양소가 풍부",
    "영양이 풍부",
    "영양을 챙",
    "건강식",
    "비주얼",
    "보기에도 좋",
    "즐길 수 있는",
    "느낄 수 있는",
    "부드러운 맛",
    "신선한",
    "단백질 위주의 메뉴",
    "섬유소",
    "단백질과 섬유소",
    "다양한 야채",
    "다양한 채소",
]
DEFAULT_CLOSING = "각 레시피의 재료와 조리 방식을 참고해 선택해 보세요."
SAFE_REASON = "재료와 조리 방식을 기준으로 현재 조건에 맞는 후보로 고려할 수 있습니다."
UNKNOWN_INGREDIENTS = "제공 정보 확인 필요"
INGREDIENT_STOP_WORDS = {
    "재료",
    "주재료",
    "필수",
    "선택",
    "인분",
    "기준",
    "기준br",
    "br",
    "말린",
    "각종",
    "물",
    "요리",
    "방식",
    "기타",
}

_client: Any | None = None


class AnswerFallbackError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _get_client() -> Any:
    global _client
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다.")
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _format_optional_number(value: object, unit: str) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "정보 없음"

    if number <= 0:
        return "정보 없음"
    return f"{number}{unit}"


def _limit_hits(hits: list[Hit], top_k: int) -> list[Hit]:
    if top_k <= 0:
        return []
    return hits[:top_k]


def _build_recipe_context(hits: list[Hit]) -> str:
    lines: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        category = hit.metadata.get("category") or "정보 없음"
        cooking_way = hit.metadata.get("cooking_way") or "정보 없음"
        calories = _format_optional_number(hit.metadata.get("calories"), "kcal")
        sodium = _format_optional_number(hit.metadata.get("sodium"), "mg")

        lines.append(
            "\n".join(
                [
                    f"{rank}.",
                    f"- recipe_id: {hit.recipe_id}",
                    f"- name: {hit.name}",
                    f"- category: {category}",
                    f"- cooking_way: {cooking_way}",
                    f"- calories: {calories}",
                    f"- sodium: {sodium}",
                    f"- document: {hit.document}",
                ]
            )
        )
    return "\n\n".join(lines)


def _build_system_prompt() -> str:
    return (
        "당신은 한국어 레시피 추천 서비스의 추천 이유 작성자입니다.\n"
        "사용자 질문과 이미 정렬된 레시피 후보를 바탕으로 JSON만 작성하십시오.\n"
        "최종 사용자 답변 문장은 작성하지 마십시오.\n"
        "검색이나 재정렬을 하지 마십시오.\n"
        "후보는 이미 추천 우선순위대로 정렬되어 있으므로 순서를 절대 바꾸지 마십시오.\n"
        "후보에 없는 레시피, 재료, 조리법, 영양정보를 절대 지어내지 마십시오.\n"
        "reason은 후보 1개당 1문장만 작성하십시오.\n"
        "reason은 후보 정보에 있는 재료, category, cooking_way, calories, sodium만 근거로 쓰십시오.\n"
        "후보 정보에 없는 맛, 효능, 건강 효과, 비주얼 평가는 쓰지 마십시오.\n"
        '"건강한", "건강하게", "건강에 도움이", "영양가", "풍부", "고소한", '
        '"상큼한", "맛있는", "맛있게", "좋은 맛", "시원한", "깔끔한", '
        '"부드러운 식감", "자극적이지 않은 맛", "아이가 좋아할", "보기 좋은", '
        '"고급스러운", "저칼로리", "고단백", "짜지 않은", "짜지 않고" 같은 표현을 '
        "쓰지 마십시오.\n"
        '"즐길 수 있는", "느낄 수 있는", "부드러운 맛", "신선한", '
        '"단백질 위주의 메뉴" 같은 표현도 쓰지 마십시오.\n'
        '"섬유소", "단백질과 섬유소", "다양한 야채", "다양한 채소"처럼 '
        "영양 성분 또는 범위가 넓은 표현은 쓰지 마십시오.\n"
        "여러 재료가 있더라도 구체적인 재료명 2~3개만 언급하십시오.\n"
        "예: 호박잎, 적채, 파프리카가 포함되어 있어 반찬 후보로 고려할 수 있습니다.\n"
        '사용자 질문에 "맛있게", "건강하게", "짜지 않고", "자극적이지 않은"이 있어도 '
        "reason에서 이를 단정하지 마십시오.\n"
        "대신 재료와 조리 방식만 근거로 설명하십시오.\n"
        "예: 고춧가루가 포함되어 있어 매운 반찬 후보로 고려할 수 있습니다.\n"
        "예: 닭가슴살이 포함되어 있어 단백질 위주의 메뉴 후보로 고려할 수 있습니다.\n"
        "예: 국&찌개 카테고리이고 끓이기 방식이라 따뜻한 국물 요리 후보로 고려할 수 있습니다.\n"
        "수치가 있는 경우에만 calories, sodium을 언급하십시오.\n"
        "수치가 없으면 저칼로리, 저염, 고단백 같은 표현을 쓰지 마십시오.\n"
        "단백질 수치가 없으면 고단백이라고 단정하지 말고, 닭가슴살/쇠고기/두부/생선/계란 등이 "
        "포함되어 있다는 식으로만 말하십시오.\n"
        '"로 고려할 수 있습니다", "에 활용할 수 있습니다", "에 비교적 맞을 수 있습니다"처럼 '
        "완곡하게 쓰십시오.\n"
        "사용자 질문의 의도와 관련 있는 핵심 재료만 2~3개 정도 언급하십시오.\n"
        "재료 전체를 길게 나열하지 마십시오.\n"
        '반드시 {"items": [...], "closing": "..."} 형식의 JSON 객체만 반환하십시오.\n'
        "items의 각 원소는 rank, recipe_id, name, reason을 포함해야 합니다.\n"
        "closing은 한 문장으로 작성하십시오."
    )


def _build_user_prompt(query: str, hits: list[Hit]) -> str:
    return (
        f"사용자 질문: {query}\n\n"
        "추천 후보:\n"
        f"{_build_recipe_context(hits)}\n\n"
        "JSON 예시:\n"
        "{\n"
        '  "items": [\n'
        '    {"rank": 1, "recipe_id": 236, "name": "닭가슴살카나페", '
        '"reason": "닭가슴살과 고구마가 포함되어 있어 운동 후 저녁 메뉴로 고려할 수 있습니다."}\n'
        "  ],\n"
        '  "closing": "각 레시피의 재료와 조리 방식을 참고해 선택해 보세요."\n'
        "}\n\n"
        "위 후보 개수와 같은 개수의 items를 입력 순서대로 작성하십시오."
    )


def _has_forbidden_expression(text: str) -> bool:
    return any(expr in text for expr in FORBIDDEN_EXPRESSIONS)


def _safe_reason(reason: str) -> str:
    reason = reason.strip()
    if _has_forbidden_expression(reason):
        return SAFE_REASON
    return reason


def _parse_llm_json(content: str) -> dict:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AnswerFallbackError("json_parse_failed") from exc

    if not isinstance(raw, dict):
        raise AnswerFallbackError("json_parse_failed")
    return raw


def _validate_llm_result(result: dict, hits: list[Hit]) -> dict[int, str]:
    items = result.get("items")
    closing = result.get("closing")
    if not isinstance(items, list) or len(items) != len(hits):
        raise AnswerFallbackError("item_count_mismatch")
    if not isinstance(closing, str) or not closing.strip():
        raise AnswerFallbackError("empty_reason")

    expected_by_id = {hit.recipe_id: hit.name for hit in hits}
    reasons_by_id: dict[int, str] = {}

    for item in items:
        if not isinstance(item, dict):
            raise AnswerFallbackError("recipe_mismatch")

        recipe_id = item.get("recipe_id")
        name = item.get("name")
        reason = item.get("reason")
        rank = item.get("rank")

        if not isinstance(rank, int):
            raise AnswerFallbackError("recipe_mismatch")
        if not isinstance(recipe_id, int) or recipe_id not in expected_by_id:
            raise AnswerFallbackError("recipe_mismatch")
        if name != expected_by_id[recipe_id]:
            raise AnswerFallbackError("recipe_mismatch")
        if recipe_id in reasons_by_id:
            raise AnswerFallbackError("recipe_mismatch")
        if not isinstance(reason, str) or not reason.strip():
            raise AnswerFallbackError("empty_reason")
        reasons_by_id[recipe_id] = _safe_reason(reason)

    if set(reasons_by_id) != set(expected_by_id):
        raise AnswerFallbackError("item_count_mismatch")

    return reasons_by_id


def _safe_closing(closing: str) -> str:
    if not closing.strip() or _has_forbidden_expression(closing):
        return DEFAULT_CLOSING
    return closing.strip()


def _clean_ingredient_text(text: str) -> str:
    # 주재료 토큰 추출에만 사용하는 정리 함수다.
    # 최종 answer, 레시피명, reason 문장에는 적용하지 않는다.
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", " ")
    text = re.sub(r"[<>]", " ", text)
    text = re.sub(r"[^\w\s,()/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_valid_ingredient_token(token: str, seen: set[str]) -> bool:
    if not token or len(token) <= 1:
        return False
    if token in seen:
        return False
    if token in INGREDIENT_STOP_WORDS:
        return False
    if token.isdigit():
        return False
    return True


def _extract_ingredients(hit: Hit, max_items: int = 3) -> list[str]:
    document = hit.document or ""
    match = re.search(r"주재료\s+(.+?)(?:\.\s*특징|\.|$)", document)
    if not match:
        return []

    raw = _clean_ingredient_text(match.group(1))
    parts = re.split(r"[,/()\s]+", raw)

    ingredients: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = part.strip(" .:;|")
        if not _is_valid_ingredient_token(item, seen):
            continue
        seen.add(item)
        ingredients.append(item)
        if len(ingredients) >= max_items:
            break

    return ingredients


def _format_ingredients(hit: Hit) -> str:
    ingredients = _extract_ingredients(hit)
    if not ingredients:
        return UNKNOWN_INGREDIENTS
    return ", ".join(ingredients)


def _format_cooking_way(hit: Hit) -> str:
    return hit.metadata.get("cooking_way") or "제공 정보 확인 필요"


def _render_answer(query: str, hits: list[Hit], reasons_by_id: dict[int, str], closing: str) -> str:
    lines = ["요청하신 조건에 맞춰 아래 레시피를 고려해볼 수 있습니다.", ""]
    for rank, hit in enumerate(hits, start=1):
        lines.append(f"{rank}. {hit.name}")
        lines.append(f"- 주재료: {_format_ingredients(hit)}")
        lines.append(f"- 조리 방식: {_format_cooking_way(hit)}")
        lines.append(f"- 추천 포인트: {reasons_by_id[hit.recipe_id]}")
        lines.append("")
    lines.append(_safe_closing(closing))
    return "\n".join(lines)


def _fallback_answer(query: str, hits: list[Hit]) -> str:
    if not hits:
        return "조건에 맞는 추천 레시피를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요."

    lines = ["요청하신 조건에 맞춰 아래 레시피를 고려해볼 수 있습니다.", ""]
    for rank, hit in enumerate(hits, start=1):
        lines.append(f"{rank}. {hit.name}")
        lines.append(f"- 주재료: {_format_ingredients(hit)}")
        lines.append(f"- 조리 방식: {_format_cooking_way(hit)}")
        lines.append("- 추천 포인트: 제공된 후보 정보 기준으로 선택할 수 있는 메뉴입니다.")
        lines.append("")

    lines.append(DEFAULT_CLOSING)
    return "\n".join(lines)


def generate_answer(query: str, hits: list[Hit], top_k: int = DEFAULT_ANSWER_TOP_K) -> str:
    """
    이미 정렬된 hits를 받아 사용자용 자연어 추천 답변을 생성한다.
    LLM 호출 실패 시 원본 순서 기반 rule-based 답변을 반환한다.
    """
    selected_hits = _limit_hits(hits, top_k)
    if not selected_hits:
        return _fallback_answer(query, selected_hits)

    if not OPENAI_API_KEY:
        return _fallback_answer(query, selected_hits)

    try:
        try:
            response = _get_client().chat.completions.create(
                model=MODEL_NAME,
                temperature=ANSWER_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _build_system_prompt()},
                    {"role": "user", "content": _build_user_prompt(query, selected_hits)},
                ],
            )
        except Exception as exc:
            raise AnswerFallbackError("llm_call_failed") from exc

        content = response.choices[0].message.content
        if not content:
            raise AnswerFallbackError("empty_response")

        result = _parse_llm_json(content)
        reasons_by_id = _validate_llm_result(result, selected_hits)
        return _render_answer(query, selected_hits, reasons_by_id, result["closing"])

    except AnswerFallbackError as exc:
        print(f"[WARN] answer generation fallback: {exc.reason}")
        return _fallback_answer(query, selected_hits)
    except Exception:
        print("[WARN] answer generation fallback: unknown")
        return _fallback_answer(query, selected_hits)
