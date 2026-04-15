import json
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# ===== 입력/출력 스키마 =====

class MealRequest(BaseModel):
    question: str
    age: int
    gender: Literal["male", "female"]
    goal: Literal["diet", "muscle", "none"]
    meal_time: Literal["점심", "저녁"]


class Meal(BaseModel):
    rice: str | None = Field(description="밥/면/만두류 중 1개")
    soup: str | None = Field(description="국 또는 찌개 1개")
    main: str | None = Field(description="메인 반찬 1개")
    banchan: list[str] = Field(description="일반 반찬 2~3개")


class MealResponse(BaseModel):
    meal: Meal
    comment: str = Field(description="친근한 말투의 추천 이유 2~3문장")


# ===== 도메인 상수 =====

CATEGORIES: dict[str, list[str]] = {
    "rice":    ["밥류", "죽 및 스프류"],
    "soup_clear": ["국 및 탕류"],
    "soup_stew":  ["찌개 및 전골류"],
    "main":    ["구이류", "볶음류", "조림류", "튀김류", "찜류", "전·적 및 부침류"],
    "banchan": ["나물·숙채류", "생채·무침류", "김치류", "장아찌·절임류", "젓갈류"],
}

K_VALUES = {"rice": 3, "soup_clear": 2, "soup_stew": 2, "main": 5, "banchan": 5}

EXCLUDE_KEYWORDS = ["간편조리세트", "떡볶이", "물회", "회무침"]

SLOT_LABELS = {
    "rice":       "밥 후보",
    "soup_clear": "국/탕 후보 (맑은국, 탕류)",
    "soup_stew":  "찌개/전골 후보 (진한 국물)",
    "main":       "메인 반찬 후보",
    "banchan":    "반찬 후보",
}

GOAL_DESCRIPTIONS = {
    "diet":   "체중 감량 (저칼로리, 저지방 음식 우선)",
    "muscle": "근육 증가 (단백질이 풍부한 음식 우선)",
    "none":   "특별한 목표 없음 (균형 잡힌 식단)",
}

GENDER_KR = {"male": "남성", "female": "여성"}


# ===== 벡터스토어 (lazy singleton) =====

_vectorstore: Chroma | None = None

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory="vectorstore",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
    return _vectorstore


# ===== 검색 =====

def retrieve_by_slot(question: str, slot: str):
    vs = get_vectorstore()
    docs = vs.similarity_search(
        query=question,
        k=K_VALUES[slot] * 5,
        filter={"대분류": {"$in": CATEGORIES[slot]}},
    )
    docs = [d for d in docs if not any(kw in d.metadata["식품명"] for kw in EXCLUDE_KEYWORDS)]
    # 식품명 중복 제거 (DB에 동일 식품명이 여러 행 존재)
    seen: set[str] = set()
    deduped = []
    for d in docs:
        name = d.metadata["식품명"]
        if name not in seen:
            seen.add(name)
            deduped.append(d)
    return deduped[:K_VALUES[slot]]


def _fmt(val: float) -> str:
    return "N/A" if val < 0 else f"{val:.0f}"


def format_candidate(doc) -> str:
    m = doc.metadata
    name = m['식품명'].replace('_', ' ')
    return (
        f"- {name} | "
        f"{_fmt(m['에너지'])}kcal / 탄{_fmt(m['탄수화물'])} / 단{_fmt(m['단백질'])} / 지{_fmt(m['지방'])}"
    )


def build_candidates_block(question: str) -> str:
    sections = []
    for slot in ("rice", "soup_clear", "soup_stew", "main", "banchan"):
        docs = retrieve_by_slot(question, slot)
        lines = [f"[{SLOT_LABELS[slot]}]"]
        lines.extend(format_candidate(d) for d in docs)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


# ===== 프롬프트 =====

SYSTEM_PROMPT = """당신은 친근하고 따뜻한 말투로 식단을 추천해주는 AI 영양사입니다.
사용자가 편하게 식사할 수 있도록 한 끼 식단을 구성하고, 왜 이렇게 골랐는지 간단한 이유를 설명해주세요.

규칙:
- 반드시 아래 제공된 후보 음식 목록에서만 선택하세요. 목록에 없는 음식을 절대 만들어내지 마세요.
- 음식명만 선택하세요. 영양소 수치는 참고용이며 응답 필드에 포함하지 마세요.
- 각 슬롯은 반드시 지정된 섹션에서만 선택하세요: rice는 [밥 후보]에서, soup는 [국/탕 후보] 또는 [찌개/전골 후보]에서, main은 [메인 반찬 후보]에서, banchan은 [반찬 후보]에서만 선택하세요.
- soup은 [국/탕 후보]와 [찌개/전골 후보] 중 사용자 요청(담백함, 얼큰함 등)에 맞는 것을 선택하세요.
- rice, soup, main, banchan에 같은 재료(예: 열무, 닭, 계란, 돼지 등)가 중복되지 않도록 구성하세요.
- 사용자의 요청 의도(담백함, 매운 음식 기피, 가벼운 식사 등)를 반드시 반영하세요.
- comment는 친근한 한국어 말투로 2~3문장 정도로 작성하세요."""

USER_PROMPT = """사용자 정보:
- 나이: {age}세
- 성별: {gender}
- 건강 목표: {goal_desc}
- 식사 시간대: {meal_time}

사용자 요청:
{question}

후보 음식 목록:
{candidates}

위 후보 중에서 사용자에게 가장 적합한 한 끼 식단을 구성해주세요."""


# ===== 메인 진입점 =====

def recommend(req: MealRequest) -> MealResponse:
    retrieval_query = f"{req.question} {GOAL_DESCRIPTIONS[req.goal]}"
    candidates = build_candidates_block(retrieval_query)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = prompt | llm.with_structured_output(MealResponse)

    return chain.invoke({
        "age": req.age,
        "gender": GENDER_KR[req.gender],
        "goal_desc": GOAL_DESCRIPTIONS[req.goal],
        "meal_time": req.meal_time,
        "question": req.question,
        "candidates": candidates,
    })


# ===== 로컬 테스트 =====

if __name__ == "__main__":
    test_cases = [
        # goal=diet
        MealRequest(question="가볍게 먹고 싶어", age=28, gender="female", goal="diet", meal_time="점심"),
        MealRequest(question="다이어트 중인데 배는 부르게 먹고 싶어", age=22, gender="male", goal="diet", meal_time="저녁"),

        # goal=muscle
        MealRequest(question="운동 후 단백질 보충하고 싶어", age=25, gender="male", goal="muscle", meal_time="점심"),
        MealRequest(question="헬스하는데 든든하게 먹고 싶어", age=30, gender="female", goal="muscle", meal_time="저녁"),

        # goal=none
        MealRequest(question="오늘 뭐 먹을지 모르겠어", age=45, gender="male", goal="none", meal_time="점심"),
        MealRequest(question="따뜻하고 담백한 거 먹고 싶어", age=60, gender="female", goal="none", meal_time="저녁"),
    ]

    for i, req in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"[케이스 {i}] goal={req.goal}, gender={req.gender}, meal_time={req.meal_time}")
        print(f"질문: {req.question}")
        result = recommend(req)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

