# Meal-bot RAG API 코드 분석 리포트

분석 기준: 2026-05-21 현재 워킹트리. `mealbot-demo-ver0` 브랜치.
주의: 본 리포트의 모든 사실 진술은 실제 코드 기준이며, 추측은 명시.

---

## 0. 가장 먼저 짚어야 할 것 — v0.2 → v0.3 마이그레이션 상황

요청 명세는 v0.2 스타일(`/recommend`, `/ask`, `spicy_max` 등)로 작성되어 있으나
**실제 코드는 v0.3로 이미 이행**되었다. 차이를 먼저 정리한다.

| 영역            | v0.2 (명세)                                                  | v0.3 (실제 코드)                                                                            |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 엔드포인트      | `/recommend` + `/ask` 분리                                   | `/chat` 단일 + `/healthz`                                                                   |
| 라우팅          | FastAPI 라우터가 직접 분기                                   | `ChatOrchestrator`가 intent 분류 후 분기                                                    |
| 입력 슬롯       | `spicy_max` 있음                                             | `spicy_max` 폐기 → `free_text`로 흡수                                                       | /                                                  |
| 응답 필드       | `recommendations[5]`, `image_url`, `kcal`, `matched_intents` | `recommendations[0                                                                          | 2]`, `Recommendation` 슬림화 (image_url/kcal 없음) |
| 응답 의미       | `refused`/`out_of_scope`가 응답 필드                         | `Flags{needs_more_slots, out_of_scope, is_fallback}`로 통합 (`refused`는 응답에 노출 안 됨) |
| `/debug/search` | (요청에 언급됨)                                              | **존재하지 않음**                                                                           |

테스트 코드(`tests/`)도 v0.2 시대 산물이라 현재 코드에서 import error로 깨진다 (§9).

---

## 1. 전체 아키텍처 요약

### 현재 (v0.3) 요청 흐름

모든 클라이언트 요청은 `POST /chat` 하나로 들어와 [api/chat_orchestrator.py](api/chat_orchestrator.py)의
`ChatOrchestrator.handle()`이 흐름을 조정한다.

```
POST /chat
  → ChatOrchestrator.handle()
    1. 입력 정규화 (slots, history[-6:], last_recs)
    2. (LLM1) classify_intent → 5라벨 분류
    3. out_of_scope → 즉시 ask 응답
    4. refine + last_recs=[] → recommend로 강등
    5. (LLM2) extract_slots → SlotDelta
    6. _merge_slots (meal_times/purpose만 replace, free_text는 요청값 유지)
    7. intent별 분기
        - slot_fill: build_slot_question (확정적 풀)
        - recommend: handle_recommend (안전망 + handle_recommend)
        - refine:    handle_refine
        - ask:       handle_ask
    8. HandlerResult → ChatResponse 조립 (Flags 기본값 + override)
    9. 로깅
```

### 계층 역할

| 계층                                                                                                          | 책임                                                                                     |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| FastAPI 엔트리 ([api/main.py](api/main.py))                                                                   | lifespan에서 retriever/store/orchestrator 1회 DI, `/chat`은 `orchestrator.handle()` 위임 |
| Orchestrator ([api/chat_orchestrator.py](api/chat_orchestrator.py))                                           | intent/slot LLM 호출, 머지, 분기, 안전망, 응답 조립                                      |
| 핸들러 ([api/recommend.py](api/recommend.py), [refine.py](api/refine.py), [qa_handler.py](api/qa_handler.py)) | 도메인 분기별 처리. `HandlerResult` 반환                                                 |
| Intent/Slot LLM ([api/intent.py](api/intent.py), [api/slot.py](api/slot.py))                                  | gpt-5-mini, JSON 모드, 5초 타임아웃                                                      |
| Retrieval ([rag/retriever.py](rag/retriever.py))                                                              | DenseRetriever(BGE-M3+Chroma) + BM25Retriever(Kiwi+rank-bm25) + HybridRetriever(RRF)     |
| Rerank ([rag/reranker.py](rag/reranker.py), [rerank_prompt.py](rag/rerank_prompt.py))                         | gpt-5-mini Structured Output, validation+retry, JSONL 로깅                               |
| QA ([rag/qa.py](rag/qa.py), [qa_prompt.py](rag/qa_prompt.py))                                                 | gpt-5-mini Structured Output, refused/out_of_scope/qa_failed 산출                        |
| RecipeStore ([rag/recipe_store.py](rag/recipe_store.py))                                                      | recipes_enriched_v2.json 인메모리, `rcp_seq` 키 조회                                     |

---

## 2. 주요 파일별 역할

| 파일                                                         | 역할                                                    | 중요도    | 검토 필요                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------- | --------- | ----------------------------------------------------------- |
| [api/main.py](api/main.py)                                   | FastAPI 엔트리, lifespan DI, `/healthz`/`/chat`         | 높음      | 낮음 (얇음)                                                 |
| [api/chat_orchestrator.py](api/chat_orchestrator.py)         | 흐름 조정의 본체                                        | 매우 높음 | **높음** (안전망/print debug 잔존, 374→460라인 가까이 성장) |
| [api/recommend.py](api/recommend.py)                         | recommend 핸들러. retrieval → rerank → lookup           | 높음      | 보통                                                        |
| [api/refine.py](api/refine.py)                               | refine 핸들러. LLM query 재구성 → exclude → rerank      | 높음      | 보통                                                        |
| [api/qa_handler.py](api/qa_handler.py)                       | ask 핸들러. last_recs lookup → qa.answer 위임           | 높음      | 낮음                                                        |
| [api/intent.py](api/intent.py)                               | intent LLM 호출 + IntentResult 검증                     | 높음      | 낮음                                                        |
| [api/slot.py](api/slot.py)                                   | slot LLM 호출 + SlotDelta 검증                          | 높음      | 보통                                                        |
| [api/slot_questions.py](api/slot_questions.py)               | 슬롯 질문 풀(deterministic). 안전망과 별개              | 보통      | 낮음                                                        |
| [api/schemas.py](api/schemas.py)                             | ChatRequest/ChatResponse/Slots/Flags/Recommendation     | 매우 높음 | 낮음                                                        |
| [api/handler_result.py](api/handler_result.py)               | HandlerResult dataclass (핸들러↔orchestrator 계약)      | 보통      | 낮음                                                        |
| [api/errors.py](api/errors.py)                               | IntentClassifyError 등 도메인 예외                      | 낮음      | 낮음                                                        |
| [api/prompts/intent_prompt.py](api/prompts/intent_prompt.py) | 5라벨 정의 + 10 예시                                    | 매우 높음 | 보통                                                        |
| [api/prompts/slot_prompt.py](api/prompts/slot_prompt.py)     | 슬롯 추출 규칙 + 8 예시                                 | 매우 높음 | 보통                                                        |
| [api/prompts/refine_prompt.py](api/prompts/refine_prompt.py) | refine query 재구성 규칙 + 3 예시                       | 높음      | 낮음                                                        |
| [rag/retriever.py](rag/retriever.py)                         | Dense/BM25/Hybrid + RRF + exclude_ids                   | 매우 높음 | 낮음                                                        |
| [rag/query_builder.py](rag/query_builder.py)                 | slots → 단일 query 문자열 합성                          | 높음      | 낮음                                                        |
| [rag/reranker.py](rag/reranker.py)                           | LLM rerank, validation, retry, fallback, JSONL          | 매우 높음 | 보통                                                        |
| [rag/rerank_prompt.py](rag/rerank_prompt.py)                 | rerank SYSTEM_PROMPT + RerankResponse 스키마            | 매우 높음 | 보통                                                        |
| [rag/qa.py](rag/qa.py)                                       | QA LLM 호출 + validation + retry + fallback + JSONL     | 매우 높음 | 보통                                                        |
| [rag/qa_prompt.py](rag/qa_prompt.py)                         | QA SYSTEM_PROMPT + QAResponse 스키마 + user prompt 빌더 | 매우 높음 | **높음** (refused/out_of_scope 정책의 단일 소스)            |
| [rag/recipe_store.py](rag/recipe_store.py)                   | 인메모리 dict 조회                                      | 보통      | 낮음                                                        |
| [rag/config.py](rag/config.py)                               | 모델/경로/top_k/로그패턴 상수                           | 높음      | 낮음                                                        |
| [rag/tokenizer.py](rag/tokenizer.py)                         | KiwiTokenizer (BM25용). **본 분석에서 미열람**          | 보통      | 본 리포트에서 확인 안 됨                                    |
| [cli_chat.py](cli_chat.py)                                   | 로컬 검증용 대화형 CLI (운영 무관)                      | 낮음      | 낮음                                                        |
| [tests/\*](tests/)                                           | 6개 중 4개가 stale (v0.2 스키마/엔드포인트)             | 보통      | **높음** (§9)                                               |

---

## 3. /chat 동작 흐름 — recommend / refine 케이스

요청 명세의 "/recommend 동작 흐름"은 v0.2 명세이지만, 실제 처리는 `/chat` 안의
`effective_intent == "recommend"` 분기에서 일어난다.

### 3.1 recommend 분기 ([api/recommend.py:41-175](api/recommend.py#L41))

1. **query 빌드** — `build_retrieval_query(meal_times, purpose, free_text)` 호출.
   - `slots.free_text or free_text_delta`로 폴백 (Spring이 누적 안 했을 때 신호 살림).
   - 결과 예: `"시간대: 점심. 목적: 단백질 챙기기. 매운 거"`
2. **hybrid retrieval** — `HybridRetriever.search(query)`.
   - Dense(BGE-M3+Chroma, top_k=50) + BM25(Kiwi+rank-bm25, top_k=50) → RRF 병합(k=60, top_k=30).
3. **0건 → slot_fill 폴백** (intent="slot_fill", needs_more_slots+is_fallback).
4. **structured_inputs 구성**
   - `{meal_times, purpose, free_text=effective_free_text}` — rerank user prompt의 `[사용자 제약]` 블록에만 사용.
5. **LLM rerank** — `rerank(query, candidates, structured_inputs, top_k=2, previously_recommended=None)`.
   - gpt-5-mini Structured Output (`beta.chat.completions.parse`).
   - validation 실패 시 [이전 응답 오류] 블록 붙여 1회 재시도. 최종 실패 시 hybrid 순위 그대로 fallback.
6. **recipe lookup + 매핑** — `recipe_store.get_recipe_by_id`로 원본 dict 꺼내 `Recommendation`(name, summary, main_ingredients, cooking_time, reason)으로 매핑.
7. **2개 미만 → slot_fill 폴백** (응답 불변식 `len==0 or 2` 보호).

`effective_intent == "recommend"` 블록 진입 시 **free_text 안전망**이 한 번 끼어든다 ([api/chat_orchestrator.py:282-302](api/chat_orchestrator.py#L282)):

- 슬롯 충족 검사 통과 후, 직전 assistant 메시지가 안전망 질문이 아니고 `slots.free_text`+`free_text_delta` 둘 다 strip 3자 미만이면
  `FREETEXT_SAFETYNET_QUESTION`을 slot_fill로 반환.
- 직전이 안전망 질문이었으면 `_normalize_freetext_negative`로 NEGATIVE_ANSWERS 매칭 시 free_text를 null로 강제.

### 3.2 refine 분기 ([api/refine.py:190-325](api/refine.py#L190))

1. **LLM query 재구성** — gpt-5-mini JSON 모드. `{search_query: "..."}`. 5초 타임아웃.
   - 실패 시 `_fallback_concat_query`(slots+delta+message concat)로 폴백, `is_fallback=True`.
2. **exclude_ids 구성** — 직전 추천 recipe_id를 정규화해서 hybrid에 전달.
3. **hybrid retrieval with exclusion**.
4. **0건 → ask 폴백** (intent="ask", is_fallback). recommend와 달리 ask로 떨어짐에 유의.
5. **LLM rerank** — `previously_recommended=[lr.name for lr in last_recs]` 전달.
   - reranker가 user prompt 끝에 `[직전 추천 메뉴]` 블록을 append해서 차별화 유도.
6. **매핑 + 2개 미만이면 ask 폴백**.
7. **answer 생성** — `"{free_text_delta} 반영해서 다시 골라봤어요."` 또는 generic.

---

## 4. /chat 동작 흐름 — ask 케이스 (구 /ask)

요청 명세의 "/ask 동작 흐름"은 `effective_intent == "ask"` 분기에서 일어난다.

### 4.1 ask 분기 ([api/qa_handler.py:31-128](api/qa_handler.py#L31))

1. **last_recs 비어 있으면** orchestrator가 ANSWER_ASK_NO_LAST_REC 안내 메시지를 반환 (이 핸들러 호출 안 됨).
2. **last_recs lookup** — 모든 LastRecommendation을 `recipe_store.get_recipe_by_id`로 조회.
   - 다중 메뉴 컨텍스트(v0.3 변경점). 단일 recipe_id 단답 아님.
3. **전부 store에 없으면** is_fallback 안내 반환.
4. **history dict 변환** — ChatMessage 객체를 `{"role", "content"}` dict로.
5. **`rag.qa.answer(query, retrieved_docs, chat_history)` 호출**.
6. **QA 응답 → HandlerResult 매핑** ([api/qa_handler.py:114-127](api/qa_handler.py#L114))
   - `qa_failed or is_fallback` → `flags_override.is_fallback = True`
   - `out_of_scope` → `flags_override.is_fallback = True` (응답 스키마상 ask에서 out_of_scope 플래그 재마킹은 안 함. 1차 분류기에서만 set한다는 정책)
   - **`refused`는 응답에 노출되지 않음**. 거부 텍스트는 `answer`에 들어가지만 호출자(Spring/CLI)가 "거부였음"을 알 방법이 없음. § 6, § 7에서 다시 언급.

### 4.2 QA 본체 ([rag/qa.py:178-310](rag/qa.py#L178))

1. **빈 docs → LLM 스킵** + `_EMPTY_DOCS_ANSWER` 즉시 반환.
2. **상위 `QA_MAX_DOCS=3` 슬라이싱**.
3. **user prompt 빌드** — `[대화 맥락] + [검색된 레시피 문서 N건] + [현재 질문] + [작업]`.
4. **LLM 호출** — `beta.chat.completions.parse`로 `QAResponse` 직접 파싱. `RAG_QA_MODEL = RERANK_MODEL = "gpt-5-mini"`, max_completion_tokens=4000.
5. **validation** — answer/used_fields/refused/out_of_scope 타입 검증. 실패 시 [이전 응답 오류] 블록 붙여 1회 재시도.
6. **최종 처리** — LLM이 `qa_failed`/`is_fallback`을 임의로 set해도 시스템이 강제로 False로 덮어쓴다 (parsed가 있을 때).
7. **JSONL 로깅** — `logs/qa_YYYYMMDD.jsonl`.

---

## 5. 프롬프트 구조 분석

### 5.1 [rag/rerank_prompt.py](rag/rerank_prompt.py) — rerank SYSTEM_PROMPT

핵심 섹션과 영향:

| 섹션                   | 내용                                                                                                          | 답변 품질 영향              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------- |
| [역할]                 | 한국 가정식 reranker, 새 후보 생성 금지                                                                       | 환각 차단                   |
| [후보 선택 규칙]       | recipe_id는 후보 안에서만, rank 1..N 유니크, insufficient_matches 양방향                                      | **직접** (응답 정합성)      |
| [자유 요청 우선순위]   | meal_times/purpose보다 자유 요청을 더 구체적 신호로. 단 빈 결과 금지, 부분 매칭 허용, reason에 충족/부족 명시 | **직접** (자유 요청 반영도) |
| [특히 주의할 조건]     | "10분 안에"·"국물" 같은 표현은 여러 필드 함께 보기                                                            | **직접**                    |
| [reason 생성 규칙]     | 60~100자, 평서문, 후보 metadata만, 효능/주관표현 금지, 자유 요청과 구체적 연결 1개 이상                       | **직접** (reason 품질)      |
| [matched_intents 규칙] | 1~4개(권장 2~3), ALLOWED_INTENTS 우선, 자유 태그는 절반 이하                                                  | **직접** (intent 정합성)    |
| [ALLOWED_INTENTS]      | 26개 카테고리                                                                                                 | matched_intents 후보군      |

`build_user_prompt` 구조:
`[사용자 제약]` (선택) + `# 사용자 질의` + `# 후보 Hybrid Search 상위 N건` + `# 작업` (top_k 지시).
refine 분기에서는 reranker가 `[직전 추천 메뉴]` 블록을 끝에 append.

**중복/충돌 가능성**:

- `[자유 요청 우선순위]`에서 "빈 결과 반환 금지"와, `[후보 선택 규칙]`의 "부족하면 insufficient_matches=true" 가 충돌하지 않도록 톤이 절묘하게 맞춰져 있다. 사용자가 직전에 톤다운한 결과(부분 매칭 허용)가 반영됨.
- `[reason 생성 규칙]`에 "필드명 직접 노출 금지" 항목이 `- cooking_time → 조리 시간` 하나만 들어있어 보호 범위가 좁다. 다른 필드명(`dish_type_tags`, `meal_time`)이 새어나갈 위험 있음 (모델이 실수로).

### 5.2 [rag/qa_prompt.py](rag/qa_prompt.py) — QA SYSTEM_PROMPT

훨씬 더 두꺼움 (~260줄). 주요 섹션:

| 섹션                           | 핵심                                                                         |
| ------------------------------ | ---------------------------------------------------------------------------- |
| [답변 톤]                      | 존댓말, 과장 금지, 효능 표현 금지, 레시피명 1회 이내                         |
| [intent-answer 일관성]         | 질문이 묻는 것만 답변, 평가 금지, 문서에 없는 사실 단정 금지                 |
| [답변 소스 우선순위]           | 문서 → 대화 맥락 → 일반 상식                                                 |
| [후속 질문 처리]               | 지시대명사 해석(대화 맥락), 거부 카테고리 분리                               |
| [거부 카테고리]                | 질병/효능/임산부/다이어트/영양 평가 → refused=true                           |
| [out_of_scope 판단]            | 날씨/정치/AI 정체성 등 → out_of_scope=true. 음식 페어링은 out_of_scope=false |
| [허용 영역]                    | 조리법/재료/영양(평가 없이 수치만)/보관 등                                   |
| [nutrition 인용 규칙]          | "높다/낮다/충분/건강하다" 금지. 수치+단위만                                  |
| [정보 없음 안내 예시]          | 정형 톤                                                                      |
| [거부 응답 작성 규칙]          | refused=true면 사용자가 명시 요청 안 한 nutrition 추가 금지 (예시까지 정밀)  |
| [out_of_scope 응답 작성 규칙]  | 강한 거절 톤 회피, 도메인 안내 톤                                            |
| [used_fields 작성 규칙]        | 실제 인용한 필드만, LLM 일반 상식만 썼으면 빈 리스트                         |
| [qa_failed / is_fallback 규칙] | LLM은 항상 false로 (시스템 전용)                                             |

**답변 품질에 직접 영향**:

- nutrition 평가 금지 (가장 강력하게 보호됨, 좋은 답변/나쁜 답변 예시까지 정밀).
- 거부 응답에서 nutrition 덧붙이지 말라는 규칙 (실수하기 쉬운 지점 명시).
- out_of_scope에서 친절히 답변해 도메인이 모호해지지 않도록 톤 가이드.

**중복/충돌 가능성**:

- `[허용 영역]`에 "영양 수치 (평가 없이 수치만)"이 있고 `[nutrition 인용 규칙]`이 별도로 또 있다. 같은 얘기를 두 번 — 일관성에는 도움이지만 길이 증가.
- `[거부 카테고리]`와 `[out_of_scope 판단 기준]`이 "동시에 true 불가" 규칙으로 분리되어 있는데, 경계 케이스("이 김치찌개로 다이어트해도 돼?")에서 두 분류 모두 후보가 됨. 프롬프트는 다이어트 적합성을 refused로 정의해놓아 명확하지만 LLM이 헷갈릴 여지 있음.

**누락 가능성**:

- "임산부/소아/만성질환자"는 거부 카테고리에 있지만 알레르기 자체에 대한 "이 메뉴에 견과류 들어 있어?" 같은 사실 조회는 답변 가능 영역인지 명시 부족. 현재는 ingredients 인용으로 답변 가능하나 안전 규칙과 경계가 모호.

### 5.3 [api/prompts/intent_prompt.py](api/prompts/intent_prompt.py)

- 5라벨 정의 + 5단계 우선순위 + 10 예시.
- 핵심 규칙: "건강/다이어트 질문도 ask로 분류 (거절은 후속 QA가 담당)". 이 분업이 § 4의 refused 미노출 문제와 맞물려 의미를 가짐.

### 5.4 [api/prompts/slot_prompt.py](api/prompts/slot_prompt.py)

- meal_times/purpose 매핑 규칙 + free_text_delta 보존 규칙 + 8 예시.
- 원칙 3: "meal_times/purpose에 이미 반영된 정보는 free_text_delta에서 제외". 이 규칙이 안전망 발동 조건의 전제(단일 매핑 입력에서 delta=null)와 잘 맞물림.

### 5.5 [api/prompts/refine_prompt.py](api/prompts/refine_prompt.py)

- 30자 이내, 메타 발화 제거, 직전 메뉴명 query에 넣지 말기, 3 예시. 짧고 명확.

---

## 6. 스키마 및 응답 필드 분석

### 6.1 [ChatRequest](api/schemas.py) (요청)

```python
session_id: str (1-100)
turn_id: str (1-100)
message: str (1-500)
history: list[ChatMessage] = [] (max 6)
slots: Slots = Slots()      # meal_times/purpose/free_text
last_recommendations: list[LastRecommendation] = [] (max 5, 실제는 2)
```

- `slots.free_text`: **Spring 누적 책임**. FastAPI는 누적 안 함 (orchestrator `_merge_slots` 주석에 명시).
- `last_recommendations`: 직전 턴만. 그 이전 턴은 잊음.

### 6.2 [ChatResponse](api/schemas.py) (응답)

```python
turn_id: str
intent: Literal["recommend", "slot_fill", "refine", "ask"]   # out_of_scope 없음!
answer: str (1-2000)
slots_updated: Slots
recommendations: list[Recommendation] = [] (max 2)
flags: Flags   # {needs_more_slots, out_of_scope, is_fallback} 전부 required
```

`model_validator`:

- `intent in {slot_fill, ask}` → `len(recommendations) == 0`
- `intent in {recommend, refine}` → `len(recommendations) == 2`

### 6.3 Recommendation (v0.3 슬림)

```python
recipe_id, name, summary, main_ingredients[], cooking_time?, reason(10-200)
```

v0.2 대비 사라진 것: `image_url`, `kcal`, `matched_intents`, `rank`.
`matched_intents`는 reranker 내부 구조(RerankItem)에는 있지만 외부 응답에는 안 나옴.

### 6.4 Flags

| 필드               | 의미                                | 누가 set                                                                                        |
| ------------------ | ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| `needs_more_slots` | 슬롯/안전망 부족으로 추가 질문 필요 | orchestrator / handle_recommend                                                                 |
| `out_of_scope`     | intent 분류기가 out_of_scope로 판정 | orchestrator 3단계 (다른 분기에서는 set 안 함)                                                  |
| `is_fallback`      | 핸들러 내부에서 fallback 경로 사용  | recommend(rerank fail), refine(query rebuild fail), ask(qa fail/oos), orchestrator(intent fail) |

### 6.5 프론트/Spring 연동 시 중요한 필드

- `turn_id` echo (idempotency/매칭).
- `intent`: 4값(out_of_scope 없음). 분기 시 주의.
- `recommendations[]`: 0 or 2개 보장.
- `slots_updated`: Spring이 누적해서 다음 turn에 echo. `free_text`는 Spring 책임.
- `flags.out_of_scope`: out_of_scope 응답 식별. answer 텍스트만으로는 구분 어려움.
- `flags.is_fallback`: 품질 저하 신호. UI에 "추천 결과가 충분치 않을 수 있어요" 같은 표시 후보.

### 6.6 혼동 가능성

- **`intent`에 `out_of_scope`가 없다**. 응답 시 항상 `intent="ask"` + `flags.out_of_scope=true`로 표현됨. 클라이언트가 `intent`만 보고 분기하면 out_of_scope 케이스를 놓침. (응답 스키마 Literal과 처리 분기 사이 불일치)
- **`refused`가 응답 어디에도 노출 안 됨**. QA가 refused=true로 거부해도 호출자는 `answer` 텍스트로만 구분 가능. § 7에서 다시 언급.
- **`last_recommendations`는 max=5인데 실제는 2개만 들어옴** (recommendations.max=2). 5라는 숫자는 v0.2 잔재로 추정.
- **Spring이 free_text 누적을 안 하면 신호 손실 위험**. recommend 핸들러에는 `slots.free_text or free_text_delta` 폴백이 있지만, refine은 `slots.free_text`만 structured_inputs에 넣는다 ([api/refine.py:248-252](api/refine.py#L248)). refine 측에도 동일 폴백을 적용할지 검토 필요.

---

## 7. 답변 품질 관점의 잠재 문제

각 항목은 **코드 근거 + 가설** 형식으로 정리.

### 7.1 추천 이유가 너무 일반적인지

- 코드 근거: [rag/rerank_prompt.py](rag/rerank_prompt.py)의 `[reason 생성 규칙]`이 길이 60~100자 + "후보 속성 1~2개 + 사용자 요청과의 연결" 명시. 직전 작업에서 자유 요청 연결 의무 추가됨.
- 잠재 문제: fallback 경로(`_FALLBACK_REASON = "유사한 후보로 추천되었습니다."`)는 10자라 `Recommendation.reason` 하한(`min_length=10`)에 딱 맞춤. 즉 fallback 시 reason이 무내용. 사용자는 fallback 여부를 모르므로 빈약한 reason만 본다 — `flags.is_fallback=true`를 UI에서 표시해야 의미가 산다.

### 7.2 matched_intents가 실제 의도와 잘 연결되는지

- 코드 근거: ALLOWED_INTENTS 26개 + 자유 태그 50% 룰.
- 잠재 문제: **응답 스키마에서 외부 노출 안 됨**. RerankItem.matched_intents → Recommendation 매핑 시 버려짐 ([api/recommend.py:128-137](api/recommend.py#L128), [api/refine.py:277-287](api/refine.py#L277)). 내부 품질 시그널로만 쓰이고 클라이언트는 못 봄. 의도된 설계인지 누락인지 확인 필요.

### 7.3 후속 질문 답변이 문서 근거를 충분히 쓰는지

- 코드 근거: QA가 `used_fields`를 출력하고 시스템 강제 검증함. `QA_MAX_DOCS=3`으로 슬라이싱(`rag/qa.py:191`).
- 잠재 문제: ask 응답에 used*fields가 노출 안 됨 (Recommendation에도 ChatResponse에도 없음). 운영 시 "근거 필드 사용률" 같은 품질 지표는 JSONL 로그(`logs/qa*\*.jsonl`)에서만 사후 확인 가능.

### 7.4 refused와 out_of_scope 섞일 위험

- 코드 근거: qa_prompt에 "동시 true 금지" 규칙 + ask/refused/oos 케이스별 예시 정밀.
- **확정 문제**: QA가 `refused=true`로 거부해도 [api/qa_handler.py:114-120](api/qa_handler.py#L114-L120)에서 `refused`를 flags로 매핑 안 함. 거부 답변은 answer 텍스트로만 노출. 클라이언트는 refused 여부를 알 수 없음 → 응답 후처리/통계가 불가능. 의도된 정책일 수도 있지만 v0.2 응답에는 있던 필드라 회귀.
- 추가: orchestrator가 `out_of_scope`로 분류한 케이스(`flags.out_of_scope=true`)와, ask에 갔는데 QA가 `out_of_scope=true`로 판정한 케이스가 모두 결과적으로 `is_fallback=true`로만 표현됨 ([api/qa_handler.py:117](api/qa_handler.py#L117)). 두 시그널이 같은 플래그에 섞인다.

### 7.5 nutrition에서 평가 표현이 나올 위험

- 코드 근거: qa_prompt가 "높다/낮다/충분/건강하다" 등을 명시적으로 금지. 좋은 답변/나쁜 답변 예시 정밀.
- 잠재 문제: validation 단계에서 nutrition 표현을 검사하는 후처리 검증은 없음 — LLM이 가이드를 어기면 그대로 통과. 평가 트리거 단어 후처리 검출은 P1 후보.

### 7.6 recipe 문서에 없는 내용을 LLM이 추측할 위험

- 코드 근거: qa_prompt가 "문서에 없는 사실을 단정하지 않습니다", rerank_prompt가 "metadata에 명시된 사실만 근거". 강하게 보호됨.
- 잠재 문제: QA의 "답변 소스 우선순위 3. LLM의 일반 음식 상식"이 명시 허용이라 경계가 흐려질 수 있음. 일반 상식과 해당 레시피의 사실을 LLM이 헷갈리면 검출 어려움. 또한 QA에는 "필드명 노출 금지" 규칙이 없음 (rerank에는 있는데). `used_fields` 자체가 출력 필드라서 의도된 차이지만 reason 같은 본문에 필드명이 누설될 가능성은 미검증.

### 7.7 chat_history가 실제 답변에 의미 있게 쓰이는지

- 코드 근거: QA `build_qa_user_prompt`가 `# 대화 맥락`을 user prompt 최상단에 배치. 비었으면 "대화 맥락 없음".
- 흐름: `ChatRequest.history[-6:]` → orchestrator → handle_ask → `[{role, content}]` dict 변환 → qa.answer → user prompt.
- 잠재 문제: history는 `max=6`인데 v0.3 흐름에서 매 턴 user/assistant 한 쌍이 들어가므로 3턴 분량. ask에서 "두 턴 전에 말했던 첫 번째 추천 메뉴" 같은 anaphora는 6 메시지를 넘어가면 끊김. 다중 메뉴 ask의 경우 last_recs 자체로 보완되므로 큰 문제는 아님.

### 7.8 안전망 인터랙션의 회귀 위험

- 코드 근거: [api/chat_orchestrator.py:282-302](api/chat_orchestrator.py#L282)의 free_text 안전망.
- 임시 디버그 잔존: `print("[SAFETYNET_DEBUG] ...", flush=True)` ([api/chat_orchestrator.py:289-296](api/chat_orchestrator.py#L289))과 `logger.info("safetynet check: ...")` 두 줄이 운영 코드에 남아있음. 사용자가 명시적으로 검증 후 제거 예정이라 표시했으므로 정리 필요.

### 7.9 retrieval query 자체가 신호를 다 못 살리는지

- query_builder는 free_text를 마지막 절로 단순 append. dense는 BGE-M3로 자연어를 임베딩하므로 그럭저럭 작동하지만, BM25는 "시간대: 점심." 같은 라벨 토큰까지 그대로 토큰화 → 점수에 라벨 노이즈가 포함됨. (특히 한국어 BM25는 라벨이 강한 시그널이 되기 어려움)
- 검증된 문제는 아니지만 검색 품질에 영향 가능. dense/bm25 점수 비교 평가가 있다면 확인 가치.

---

## 8. 오버엔지니어링 여부 (졸업작품 MVP 기준)

### 8.1 유지 권장

- **orchestrator + handler 분리**: 5라벨 분기를 한 함수에 다 박으면 가독성이 무너졌을 것. 현재 구조가 명확.
- **Pydantic 스키마 + model_validator**: API 계약 안전망. 의존 0.
- **LLM Structured Output (`beta.parse`)**: JSON 파싱 실패율 낮추는 표준 패턴.
- **fallback chain**: rerank 실패 → hybrid 순위, refine 실패 → concat. 시연 도중 LLM이 죽어도 응답이 나가는 보장.
- **JSONL 로깅**: 디버깅에 필수. 운영 부담 거의 없음.
- **HybridRetriever + RRF**: dense+BM25 보완은 검색 품질 기본기.

### 8.2 과한 부분 (MVP 기준)

- **`RERANK_RETRY_LIMIT = 1`로 재시도 + [이전 응답 오류] 블록 보강**: gpt-5-mini Structured Output이 후보 밖 ID를 뱉는 일이 흔치 않고, 한 번 더 호출하면 latency가 2배. 졸업작품 시연에서 latency가 더 중요할 수도. (다만 정합성 깨지면 fallback이 무내용 reason이라 트레이드오프 있음)
- **`RAG_QA_RETRY_LIMIT = 1`도 동일**.
- **slot 추출과 intent 분류를 별도 LLM 호출 2회로 분리**: 같은 발화에 대해 LLM이 두 번 도는 셈. 둘이 본질적으로 같은 입력을 보는데 결과만 다르게 뽑는다. 하나로 합치면 비용/latency 절반. (다만 구조적 분리로 디버깅이 쉬워진 측면 있음)
- **`NEGATIVE_ANSWERS` 30개 사전**: 발화 다양성 흡수에는 좋지만, 사용자 실제 자연어를 모두 커버하긴 어렵고 유지 비용 있음. embedding 유사도 1줄로도 충분할 수 있음.
- **slot_questions의 deterministic 풀**: LLM이 생성하지 않아 일관성은 좋으나, 정답이 5개 문구뿐이라 `detect_slot_question`이 완전 일치만 검사함 → 사용자가 챗봇 답변을 인용하거나 글자 하나 다르면 인식 못 함. 안전망 시그니처처럼 부분 문자열 매칭으로 가도 됨.
- **임시 디버그 print + logger 중복**: 정리 필요 (§ 7.8).

### 8.3 부족한 부분 (MVP인데 빠진 것)

- 응답에 `refused` 노출 안 됨 (§ 6, § 7).
- 응답에 `matched_intents` 노출 안 됨 (§ 7.2).
- ChatResponse intent Literal에 out_of_scope 없음 (§ 6).
- 회귀 테스트 부재 (§ 9).

---

## 9. 테스트 현황

### 9.1 존재하는 테스트 — `tests/`

| 파일                                                       | 상태           | 비고                                                                                                                                                                                     |
| ---------------------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [tests/test_api_main.py](tests/test_api_main.py)           | **stale**      | `/recommend`, `/ask` 엔드포인트로 호출. `RecommendRequest{spicy_max, free_text, session_id}` 사용. 현재 main.py에는 `/chat`만 있고 `RecommendRequest`/`AskRequest`도 없음 → import error |
| [tests/test_schemas.py](tests/test_schemas.py)             | **stale**      | `RecommendRequest`/`RecommendResponse`/`AskRequest`/`AskResponse` import. 현재 schemas.py에 없는 클래스. 전체 깨짐                                                                       |
| [tests/test_query_builder.py](tests/test_query_builder.py) | **stale**      | `build_retrieval_query(..., spicy_max=1)` 호출. 실제 함수에 spicy_max 인자 없음 → TypeError                                                                                              |
| [tests/test_rerank_prompt.py](tests/test_rerank_prompt.py) | **부분 stale** | spicy_max/spicy_label 라인 검증. `_build_constraint_block`은 spicy 안 다룸. 스키마 sanity 테스트만 통과할 듯                                                                             |
| [tests/test_recipe_store.py](tests/test_recipe_store.py)   | **유효**       | `rcp_seq` 기반 인터페이스 그대로                                                                                                                                                         |
| [tests/test_reranker.py](tests/test_reranker.py)           | **유효**       | rerank() 시그니처 호환. is_fallback 분기 검증                                                                                                                                            |

### 9.2 추가 통합 테스트

- [scripts/test_qa_integration.py](scripts/test_qa_integration.py): 본 분석에서 미열람.

### 9.3 부족한 테스트 (현재 v0.3 코드 기준)

- `/chat` 엔드포인트 자체 (정상 → recommend, slot_fill 질문, refine, ask, out_of_scope 5가지 케이스)
- `ChatOrchestrator.handle()`의 분기/fall-through (slot_fill → recommend, refine → recommend 강등)
- `_is_slots_sufficient`, `_merge_slots`, `_was_freetext_safetynet_asked`, `_needs_freetext_safetynet`, `_normalize_freetext_negative` 단위 테스트
- `build_slot_question` / `detect_slot_question` 테스트
- `handle_refine` 0건 fallback, exclude 동작
- `handle_ask` last_recs lookup 실패 케이스
- `Slots` validator (빈 meal_times → None 정규화)

### 9.4 우선 추가 권장

| 우선순위 | 테스트                                                                             |
| -------- | ---------------------------------------------------------------------------------- |
| P0       | `/chat` 엔드포인트 5라벨 분기 happy path (TestClient + mock orchestrator)          |
| P0       | `_is_slots_sufficient` / `_needs_freetext_safetynet` 단위                          |
| P0       | 안전망 라운드트립 (slot_fill question → 부정응답 → free_text=null 정규화)          |
| P1       | `handle_recommend`의 effective_free_text 폴백 (slots.free_text 없을 때 delta 사용) |
| P1       | `handle_refine`의 exclude_ids 정규화                                               |
| P2       | 기존 stale 테스트 정리 (삭제 또는 v0.3로 재작성)                                   |

---

## 10. 다음 개선 후보 — P0/P1/P2

### P0 — 답변 품질/안정성 즉시 영향

1. **임시 디버그 잔존 정리** — [api/chat_orchestrator.py:289-304](api/chat_orchestrator.py#L289)의 print + logger.info 한 줄. 운영 stdout 오염.
2. **응답에 `refused` 노출** — 현재 QA가 거부해도 `flags`에 안 들어옴. UI 처리/통계 모두 불가. `Flags`에 추가하거나 별도 필드.
3. **`intent` Literal에 `out_of_scope` 추가** OR **`flags.out_of_scope`만으로 충분함을 문서화** — 현재 둘이 어긋남. 클라이언트가 `intent`만 보고 분기하면 oos를 놓침.
4. **stale 테스트 일괄 정리** — `tests/test_api_main.py`, `test_schemas.py`, `test_query_builder.py`, `test_rerank_prompt.py` 중 v0.2 부분. CI가 있다면 모두 깨질 것 (없을 가능성 큼). 최소한 v0.3 happy path 1개라도 작성.
5. **refine 분기에도 effective_free_text 폴백 적용** — recommend는 적용됨, refine은 `slots.free_text`만 structured_inputs에 들어감 ([api/refine.py:248-252](api/refine.py#L248)). Spring이 free_text 누적 안 하면 동일하게 신호 손실.

### P1 — 시연 완성도

6. **응답 `Recommendation`에 `matched_intents` 노출** — UI에서 "왜 골랐는지" 카드/태그로 보여주기 좋음. 내부에선 이미 생성 중인데 버려짐.
7. **rerank fallback reason 강화** — `_FALLBACK_REASON = "유사한 후보로 추천되었습니다."`는 무내용. fallback 시 hybrid 점수 기반으로 "주재료/카테고리 일치도가 높아 추천되었습니다." 같은 1줄로라도.
8. **slot_questions detect 매칭 완화** — `detect_slot_question`이 완전 일치만 검사 ([api/slot_questions.py:76](api/slot_questions.py#L76)). 안전망 시그니처처럼 부분 문자열 매칭 검토.
9. **nutrition 평가 표현 후처리 검증** — qa_prompt가 강하게 금지하지만 LLM이 어기면 그대로 통과. answer 텍스트 검출 함수 1개로 detect → 로그 경고 또는 재시도.
10. **`/debug/search` 엔드포인트 신설** — 요청 명세에 언급된 것. 검색 결과만 확인하는 가벼운 디버그용으로 시연 중 유용.

### P2 — 나중에 해도 되는 고도화

11. **intent + slot LLM 호출 통합** — 한 번 호출로 둘 다 뽑기. latency/비용 절반.
12. **chat 응답에 used_fields 노출 (ask 케이스)** — QA가 어떤 필드 인용했는지 디버그/평가에 유용.
13. **retrieval query 빌더 개선** — 라벨 prefix("시간대:", "목적:") 토큰이 BM25 점수에 라벨 노이즈로 들어감. dense/bm25 입력 분리 검토.
14. **NEGATIVE_ANSWERS 사전 → embedding 유사도 1줄** — 자연어 커버리지 향상.
15. **slot/intent 결과 JSONL 로깅 추가** — 현재 rerank/qa만 JSONL. slot/intent 분석으로 회귀 추적이 어려움.
16. **`RecipeStore` 캐싱/Hot-reload** — 현재 lifespan 1회 로드. 데이터 갱신 시 재기동 필요.
17. **`config.py`의 prefix 정리** — `RAG_QA_*` / `RERANK_*` 혼재 (코드에 NOTE로 직접 적혀 있음).

---

## 11. 추가 확인 필요 파일

본 분석에서 **열람하지 못한 파일** — 필요 시 별도 분석.

| 파일                                                                                                                                                                               | 추정 역할                                                     | 우선순위                             |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------ |
| [rag/tokenizer.py](rag/tokenizer.py)                                                                                                                                               | KiwiTokenizer 구현. BM25 토큰화 품질에 직접 영향              | 보통                                 |
| [scripts/test_qa_integration.py](scripts/test_qa_integration.py)                                                                                                                   | QA 통합 테스트 (스크립트 형태)                                | 보통                                 |
| [scripts/run_query.py](scripts/run_query.py)                                                                                                                                       | 검색 단독 실행 CLI일 가능성                                   | 낮음                                 |
| [scripts/run_rerank_baseline.py](scripts/run_rerank_baseline.py)                                                                                                                   | rerank 베이스라인 평가                                        | 낮음                                 |
| [scripts/evaluate_qa_baseline.py](scripts/evaluate_qa_baseline.py)                                                                                                                 | QA 평가                                                       | 낮음                                 |
| [scripts/evaluate_hybrid_baseline.py](scripts/evaluate_hybrid_baseline.py)                                                                                                         | retrieval 평가                                                | 낮음                                 |
| [docs/api.md](docs/api.md)                                                                                                                                                         | API 문서. v0.2 잔재인지 v0.3 갱신본인지 확인 필요             | **높음** (Spring 팀 참조 가능)       |
| [docs/prompts/intent-v0.3.md](docs/prompts/intent-v0.3.md)                                                                                                                         | intent 프롬프트 설계 문서                                     | 낮음 (실제 프롬프트와 동기화 여부만) |
| [docs/prompts/slot-v0.3.md](docs/prompts/slot-v0.3.md)                                                                                                                             | slot 프롬프트 설계 문서                                       | 낮음                                 |
| [docs/refine-v0.3.md](docs/refine-v0.3.md)                                                                                                                                         | refine 프롬프트 설계 문서                                     | 낮음                                 |
| [eval/qa_cases_v1.json](eval/qa_cases_v1.json)                                                                                                                                     | QA 평가 케이스                                                | 낮음                                 |
| [scripts/enrich_recipes_v2.py](scripts/enrich_recipes_v2.py), [scripts/build_vector_db.py](scripts/build_vector_db.py), [scripts/build_bm25_index.py](scripts/build_bm25_index.py) | 데이터 파이프라인                                             | 낮음 (런타임 영향 없음)              |
| `data/recipes_enriched_v2.json`                                                                                                                                                    | 실제 데이터 — 필드 구성을 직접 보지 않고 코드 기반으로만 추론 | 보통                                 |

---

## 부록 — 검증을 위해 실행할 만한 명령 (실행은 사용자 판단)

본 리포트는 코드 정적 분석 기반이며, 다음 명령들은 실행하지 않았다.
필요 시 사용자가 직접 실행:

```bash
# 1. 현재 테스트 상태 확인 (4개는 import error로 fail 예상)
pytest tests/ -v

# 2. 유효한 2개만 실행
pytest tests/test_recipe_store.py tests/test_reranker.py -v

# 3. import 가능 여부만 빠르게 확인
python -c "import api.main; import api.chat_orchestrator; import rag.reranker; import rag.qa"

# 4. 슬롯 안전망 헬퍼 동작 단위 확인 (LLM 호출 없음)
python -c "from api.chat_orchestrator import _needs_freetext_safetynet, _normalize_freetext_negative; from api.schemas import Slots; print(_needs_freetext_safetynet(Slots(), None))"

# 5. /chat 라운드트립 (서버 기동 + cli_chat)
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
python cli_chat.py
```
