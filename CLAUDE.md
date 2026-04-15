# meal-bot 프로젝트 지침

## 프로젝트 목적
자연어 처리 기반 식단 추천 앱의 AI 서버 파트 (RAG 파이프라인)

## 역할 분담
- 나: AI 서버 (데이터 정제 + RAG 파이프라인 + FastAPI)
- 파트너: 프론트(React) + 백엔드(Spring Boot)

## 기술 스택
- Python 3.11
- LangChain (langchain-chroma, langchain-openai)
- OpenAI GPT-4o-mini (LLM, temperature=0.7)
- OpenAI text-embedding-3-small (임베딩)
- ChromaDB (벡터DB, persist_directory="vectorstore")
- FastAPI (API 서버, 예정)
- Pydantic v2 (스키마)

## 개발 환경
- 경로: ~/projects/meal-bot
- venv 활성화 후 작업
- .env에 OPENAI_API_KEY 설정 완료

## 데이터
- 원본: data/food_nutrition.xlsx (식품영양DB, 19495행)
- 출처: 식품의약품안전처 (K-FCDB)
- 재임베딩 완료 (간편조리세트 제외 후)
- DB 대분류 총 25개 (ingest 시점 기준)

## 폴더 구조
```
meal-bot/
├── data/food_nutrition.xlsx
├── vectorstore/         # Chroma persist
├── ingest.py            # 완료
├── chain.py             # 1단계 MVP 완료
├── main.py              # FastAPI (예정)
└── .env
```

---

## 1단계 MVP 상태 (완료)
- 점심/저녁 식단 추천만 지원
- 아침은 미구현 (의도적 스코프 축소)
- 영양 계산 미구현 (의도적으로 2단계로 미룸)

## 입력/출력 스키마 (chain.py)

```python
class MealRequest(BaseModel):
    question: str
    age: int
    gender: Literal["male", "female"]
    goal: Literal["diet", "muscle", "none"]
    meal_time: Literal["점심", "저녁"]

class Meal(BaseModel):
    rice: str | None
    soup: str | None
    main: str | None
    banchan: list[str]

class MealResponse(BaseModel):
    meal: Meal
    comment: str
```

- Spring DTO 매핑 고려해 필드명은 영어, 의미는 한국어 도메인 유지 (banchan 등)
- 아침 추가 시에도 동일 스키마로 커버 예정 (rice/soup/main을 null로 두고 banchan에 담는 방식)
- 응답 생성: `llm.with_structured_output(MealResponse)` 사용 → JSON 파싱/스키마 자동 강제

## 카테고리 매핑 (확정)

```python
CATEGORIES = {
    "rice":    ["밥류", "죽 및 스프류"],
    "soup":    ["국 및 탕류", "찌개 및 전골류"],
    "main":    ["구이류", "볶음류", "조림류", "튀김류", "찜류", "전·적 및 부침류"],
    "banchan": ["나물·숙채류", "생채·무침류", "김치류", "장아찌·절임류", "젓갈류"],
}
K_VALUES = {"rice": 3, "soup": 3, "main": 5, "banchan": 5}
```

- `면 및 만두류`: 초기엔 rice에 포함했으나 실사용에서 라면/냉면이 어색해 제외. 나중에 "면 요리 모드" 별도로 분리 예정
- 아침 전용 카테고리(빵 및 과자류, 유제품류 및 빙과류, 과일류)는 DB에 존재하나 1단계에선 미사용

## 블랙리스트 (2중 필터)

- **Ingest 단계 (ingest.py의 `EXCLUDE_KEYWORDS`)**: `["간편조리세트"]`
  - 데이터 품질 문제 → 영구 제외
  - 수정 시 `python ingest.py` 재실행 필요 (재임베딩)
- **Query 단계 (chain.py의 `EXCLUDE_KEYWORDS`)**: `["간편조리세트", "떡볶이"]`
  - 분식류/유연한 제외용
  - 오버페치(k*3) 후 식품명 부분일치 제거 → k개로 자름

## 프롬프트 설계 (확정)

- **시스템 프롬프트**: 친근한 AI 영양사 톤 (C안 선택)
- **사용자 정보 블록**: 구조화된 리스트. `goal` 값은 영문 코드 + 의미 설명 병기
  - `diet` → "체중 감량 (저칼로리, 저지방 음식 우선)"
  - `muscle` → "근육 증가 (단백질이 풍부한 음식 우선)"
  - `none` → "특별한 목표 없음 (균형 잡힌 식단)"
- **후보 음식 블록**: 카테고리별 섹션([밥 후보], [국/찌개 후보], [메인 반찬 후보], [반찬 후보])
- **음식 표시 포맷**: `이름 | Nkcal / 탄N / 단N / 지N` (핵심 4영양소만)
- **식품명 정제**: retrieval 시 `_` → 공백 치환 (예: "비빔밥_간편조리세트_꼬막비빔밥")

---

## 개발 순서

1. ✅ 공공데이터 수집 및 정제 (food_nutrition.xlsx)
2. ✅ ingest.py (ChromaDB 임베딩, 간편조리세트 제외)
3. ✅ chain.py 1단계 MVP (점심/저녁 추천)
4. 👉 **다음**: 테스트 케이스 확대 → 프롬프트 튜닝
5. 영양 계산(KDRIs) 추가 및 `nutrition_summary` 필드
6. 아침 식단 지원
7. retrieval 병렬화 (asyncio)
8. main.py FastAPI 엔드포인트
9. 질병 5가지 연동 (추후)
10. 알레르기 안내 (추후)

## 설계 시 중요 판단 원칙 (이 프로젝트에서 적용 중)

- **돌아가는 MVP 먼저, 최적화/분리는 나중에**: 한 파일로 시작, 중복이 보이면 분리
- **범위 축소가 언제나 안전**: 아침/영양/알레르기는 의도적으로 미룸
- **실제 결과 보고 수정**: 설계 단계 판단을 실행 결과로 검증해서 바꾸는 걸 주저하지 않음 (면 및 만두류 제외가 대표 사례)
- **데이터 원본 변형 금지**: 엑셀 손대지 않고 ingest.py 로직으로만 정제

## 다음 대화 시작 시 체크포인트
- `chain.py` 실행 → 정상 작동 확인
- EXCLUDE_KEYWORDS에 추가할 항목 있는지 교수님께 확인
- 다음 작업: 다양한 테스트 케이스로 프롬프트 품질 검증부터

## 응답 형태 (파트너 연동용)
- Pydantic 모델 기반 구조화 응답 (LangChain `with_structured_output` 사용)
- 스키마:
  - `meal.rice`, `meal.soup`, `meal.main`: str | None
  - `meal.banchan`: list[str]
  - `comment`: str (친근한 말투의 추천 이유)
- 말투는 프롬프트로 조절, JSON 스키마는 Pydantic이 강제
