# meal-bot 프로젝트 지침

## 프로젝트 목적
자연어 처리 기반 식단 추천 앱의 AI 서버 파트 (RAG 파이프라인)

## 역할 분담
- 나: AI 서버 (데이터 정제 + RAG 파이프라인 + FastAPI)
- 파트너: 프론트 + 백엔드

## 기술 스택
- Python 3.11
- LangChain
- OpenAI GPT-4o-mini (LLM)
- OpenAI text-embedding-3-small (임베딩)
- ChromaDB (벡터DB)
- FastAPI (API 서버)

## 개발 환경
- 경로: ~/projects/meal-bot
- venv 활성화 후 작업
- .env에 OPENAI_API_KEY 설정 완료

## 데이터 계획
- 식단 데이터: 공공데이터포털 식품영양DB (음식 DB 사용)
- 질병 데이터: 공공데이터 (할루시네이션 방지)
- 건강식단 관련 흔한 질병 5가지 추려서 DB 저장

## 데이터 출처 저작권
- 한글: 식품영양성분 데이터베이스
- 영문: Korean Food Composition Database system(K-FCDB)
- 출처: 식품의약품안전처

## 응답 형태
- JSON (텍스트 필드 포함, 말투/형식 프롬프트로 조절)

## 폴더 구조
```
meal-bot/
├── data/
├── vectorstore/
├── ingest.py
├── chain.py
├── main.py
└── .env
```

## 개발 순서
1. ✅ 공공데이터 수집 및 정제 (data/food_nutrition.xlsx - 음식 DB, 19495행)
2. 질병 5가지 선정 및 데이터 정제 (나중에 추가 예정)
3. ✅ ingest.py - 데이터 → 임베딩 → ChromaDB (완료, vectorstore/ 생성됨)
4. 👉 chain.py - RAG 파이프라인 (다음 작업)
5. main.py - FastAPI 엔드포인트

## 사용 컬럼 (ingest.py)
식품명, 식품대분류명, 식품중분류명,
에너지(kcal), 탄수화물(g), 단백질(g), 지방(g),
나트륨(mg), 식이섬유(g), 당류(g),
콜레스테롤(mg), 포화지방산(g), 칼륨(mg), 칼슘(mg), 철(mg)
