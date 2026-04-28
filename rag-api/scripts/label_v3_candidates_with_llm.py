"""
v3 신규 후보 25건 LLM 1차 판정 스크립트.
golden_set_criteria.md 기준을 그대로 적용한다.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pymysql.cursors

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai

from core.config import OPENAI_API_KEY, PROJECT_ROOT
from core.db import mysql_connection

# ─── 경로 ────────────────────────────────────────
ARTIFACTS        = PROJECT_ROOT / "artifacts"
INPUT_PATH       = ARTIFACTS / "v3_new_candidates.csv"
OUTPUT_PATH      = ARTIFACTS / "v3_new_candidates_labeled_draft.csv"

INPUT_FIELDNAMES = ["query_id", "query", "recipe_id", "name", "rank", "score", "source", "v1_overlap"]
OUTPUT_FIELDNAMES = INPUT_FIELDNAMES + ["llm_judgment", "llm_confidence", "llm_reason", "needs_review"]

# ─── OpenAI ──────────────────────────────────────
OPENAI_MODEL       = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0

_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ─── needs_review 트리거 표현 ────────────────────
_REVIEW_KEYWORDS = ["애매", "판단하기 어렵", "정보 부족", "명확하지 않", "추정", "가능성이 있다"]

# ─── 쿼리별 판정 기준 ────────────────────────────
QUERY_CRITERIA: dict[str, str] = {
    "Q1": """[쿼리 의도]
고추 계열 매운맛이 있는 국/찌개/탕류

[Relevant 조건 (모두 충족)]
- 카테고리: 국&찌개 또는 일품 중 국물요리
- 조리방식: 끓이기
- 매운맛 재료 포함: 고춧가루, 고추장, 김치, 청양고추, 매운 양념 중 1개 이상

[애매 케이스]
- 맑은탕(지리탕, 초계탕): not relevant
- 된장찌개 단독: not relevant
- 부대찌개, 김치찌개: relevant""",

    "Q2": """[쿼리 의도]
저칼로리 + 샐러드 형태

[Relevant 조건 (모두 충족)]
- 형태: "샐러드" 명칭 또는 채소 중심 차가운 요리
- 칼로리: 1인분 350 kcal 이하
- 드레싱: 드레싱의 주재료가 마요네즈 또는 크림이면 not relevant. 마요/크림이 부재료로 소량 포함된 경우는 relevant 유지.

[애매 케이스]
- 다이어트국수: not relevant (형태 불일치)
- 닭가슴살 구이 단독: not relevant (샐러드 아님)
- 닭가슴살 샐러드, 채소 샐러드: relevant""",

    "Q3": """[쿼리 의도]
초등 저학년(7~10세) 남아가 먹기 적합한 간식

[Relevant 조건 (모두 충족)]
- 카테고리: 후식, 빵, 떡, 음료
- 자극 재료 없음: 매운맛, 술, 커피/카페인
- 강한 향신료 없음: 생강, 계피, 한약재

[애매 케이스]
- 수정과 계열(계피/생강): not relevant
- 생강 크림 포함 간식: not relevant
- 바나나 쉐이크, 고구마 라떼, 누룽지 요거트: relevant""",

    "Q4": """[쿼리 의도]
매운맛이 있는 밥반찬

[Relevant 조건 (모두 충족)]
- 카테고리: 반찬 한정 (국/찌개/면/일품 제외)
- 매운맛 재료 포함: 고춧가루, 고추장, 청양고추 등

[애매 케이스]
- 매운 볶음면(일품): not relevant
- 매운탕(국&찌개): not relevant
- 매운닭날개구이, 김치볶음, 고추장떡: relevant""",

    "Q5": """[쿼리 의도]
단백질 함량이 높은 요리

[Relevant 조건 (정량 또는 정성 중 하나라도 충족하면 relevant)]
- 정량 조건: 1인분 단백질 20g 이상
- 정성 조건: 주재료가 고단백 식재료 (육류, 생선, 두부, 계란, 콩류)

[판정 규칙]
- 다음 중 하나라도 충족하면 relevant:
  - (정량) protein 값이 20 이상
  - (정성) 주재료가 육류, 생선, 두부, 계란, 콩류 중 하나
- protein 값이 NULL이어도 정성 조건만으로 relevant 가능
- 정량과 정성 둘 다 불충족일 때만 not relevant

[주재료 해석 가이드]
- 레시피명에 포함된 명사 중 육류/생선/두부/계란/콩류가 있으면 주재료로 간주한다
- ingredients 상위 2~3개 항목도 참고한다

[애매 케이스]
- 겉절이(채소 위주): not relevant
- 함박스테이크, 두부구이, 닭고기 요리: relevant
- protein NULL이지만 두부/계란/육류/생선/콩류 주재료: relevant""",
}

COMMON_RULES = """당신은 레시피 추천 시스템의 관련성 판정자입니다.
아래 판정 기준에 따라 엄격하게(strict) 판정하세요.

[공통 원칙]
- 이진 판정: 1(relevant) 또는 0(not relevant)
- 제공된 필드와 명시된 정보만 사용하여 판정한다.
- 일반적인 요리 상식, 통상적 조리법, 외부 지식으로 누락 정보를 보완하지 않는다.
- 명시되지 않은 조건은 충족한 것으로 간주하지 않는다.
- strict 판정에서는 조건 충족이 명시적으로 확인될 때만 1로 판정한다.
- 조건 충족 여부가 불명확하거나 추론이 필요하면 0으로 판정한다.
- category는 제공값을 우선 사용하고, NULL/공백일 때만 레시피명과 주재료로 보조 판단한다.
- 조리방식은 제공된 값만 사용하며 추정하지 않는다.
- 재료 조건은 재료 목록(ingredients)에 명시된 경우만 인정한다.
- protein이 NULL이 아니면 정량 값 우선, NULL일 때만 주재료로 정성 판단한다.

[판정 순서]
Relevant 조건을 나열된 순서대로 하나씩 확인한다.
하나라도 불충족 또는 불명확하면 즉시 0으로 판정한다.
모든 조건이 명시적으로 충족되면 1로 판정한다.

[데이터 주의]
- 식약처 원본 데이터 한계로 일부 영양 값이 비정상적으로 낮을 수 있음

[출력 형식]
JSON 객체만 출력. 다른 텍스트 금지.
{"relevant": 0 또는 1, "confidence": 0.0~1.0, "reason": "어떤 조건을 충족했고 어떤 조건을 불충족/불명확이었는지 구체적으로 한 문장."}"""

# ─── SQL ─────────────────────────────────────────
_SQL_FETCH = (
    "SELECT name, category, cooking_way, ingredients, hash_tag, "
    "calories, protein, sodium FROM recipe WHERE rcp_seq = %s"
)


# ─── 헬퍼 ────────────────────────────────────────
def _none_to_null(v) -> str:
    return str(v) if v is not None else "NULL"


def _normalize(s) -> str:
    if s is None:
        return "NULL"
    s = re.sub(r"\n+", ", ", s.strip())
    return re.sub(r",\s*,", ", ", s)


def _fmt_num(v, unit: str) -> str:
    return f"{v} {unit}" if v is not None else "NULL"


def fetch_recipe(recipe_id: int) -> dict:
    with mysql_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(_SQL_FETCH, (recipe_id,))
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"recipe_id={recipe_id} 없음")
    return row


def build_user_prompt(query: str, info: dict) -> str:
    return (
        f"쿼리: {query}\n\n"
        "레시피 정보:\n"
        f"- 이름: {_none_to_null(info['name'])}\n"
        f"- 카테고리: {_none_to_null(info['category'])}\n"
        f"- 조리방식: {_none_to_null(info['cooking_way'])}\n"
        f"- 재료: {_normalize(info['ingredients'])}\n"
        f"- 해시태그: {_normalize(info['hash_tag'])}\n"
        f"- 칼로리(1인분): {_fmt_num(info['calories'], 'kcal')}\n"
        f"- 단백질(1인분): {_fmt_num(info['protein'], 'g')}\n"
        f"- 나트륨(1인분): {_fmt_num(info['sodium'], 'mg')}\n\n"
        "판정해주세요."
    )


def call_llm(query_id: str, query: str, recipe_id: int) -> dict:
    info = fetch_recipe(recipe_id)
    system = COMMON_RULES + "\n\n" + QUERY_CRITERIA[query_id]
    user   = build_user_prompt(query, info)
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"relevant": -1, "confidence": 0.0, "reason": f"ERROR: {e}"}


def compute_needs_review(judgment: int, confidence: float, reason: str) -> bool:
    if judgment == -1:
        return True
    if confidence < 0.7:
        return True
    return any(kw in reason for kw in _REVIEW_KEYWORDS)


# ─── 메인 ────────────────────────────────────────
def main() -> None:
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))

    total = len(candidates)
    print(f"판정 대상: {total}건\n")

    file_exists = OUTPUT_PATH.exists()
    counts = {"relevant_1": 0, "relevant_0": 0, "error": 0, "needs_review": 0}

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for idx, row in enumerate(candidates, 1):
            qid       = row["query_id"]
            query     = row["query"]
            recipe_id = int(row["recipe_id"])

            result     = call_llm(qid, query, recipe_id)
            judgment   = result.get("relevant", -1)
            confidence = float(result.get("confidence", 0.0))
            reason     = result.get("reason", "")
            needs_rev  = compute_needs_review(judgment, confidence, reason)

            out_row = {**row,
                       "llm_judgment":    judgment,
                       "llm_confidence":  round(confidence, 2),
                       "llm_reason":      reason,
                       "needs_review":    needs_rev}
            writer.writerow(out_row)
            f.flush()

            if judgment == 1:
                counts["relevant_1"] += 1
            elif judgment == 0:
                counts["relevant_0"] += 1
            else:
                counts["error"] += 1
            if needs_rev:
                counts["needs_review"] += 1

            flag = "✓" if judgment == 1 else "✗" if judgment == 0 else "!"
            rev  = " [needs_review]" if needs_rev else ""
            print(f"[{idx:02}/{total}] {qid} {flag} conf={confidence:.2f}{rev}  {row['name']}")

    print(f"\n─── 완료 ───")
    print(f"  relevant=1 : {counts['relevant_1']}건")
    print(f"  relevant=0 : {counts['relevant_0']}건")
    print(f"  error(-1)  : {counts['error']}건")
    print(f"  needs_review: {counts['needs_review']}건")
    print(f"\n저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
