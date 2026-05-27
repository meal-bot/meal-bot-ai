# rag-api Architecture Snapshot

- 캡처 시각: 2026-05-27
- 브랜치: `chore/rebuild-cd` (working dir 기준)
- 목적: AI 서버 아키텍처 스냅샷. 이후 작업의 참조 문서.

---

## 1. 디렉터리 트리 (2단계)

```
rag-api/
├── api/                        # FastAPI 진입점 + intent별 분기 핸들러
│   └── prompts/                # intent/slot/refine 프롬프트 (분류 단계 LLM용)
├── rag/                        # 검색·rerank·QA 핵심 모듈 (도메인 로직)
├── scripts/                    # 인덱스 빌드/평가/원본 enrich 일회성 스크립트
│   └── _enrichment_*           # 데이터 보강 단계용 프롬프트·스키마
├── tests/                      # pytest. 일부 v0.2 잔재로 깨짐 (10절 참조)
├── data/                       # recipes_cleaned.json, recipes_enriched(_v2).json
├── chroma/recipes_v2/          # ChromaDB persistent storage
├── bm25/recipes_v2/            # BM25 인덱스 pickle 3종
├── docs/                       # 설계 문서 (orchestrator-v0.3.md, refine-v0.3.md, api.md)
│   └── prompts/                # intent-v0.3.md, slot-v0.3.md (프롬프트 명세)
├── eval/                       # qa_cases_v1.json (QA baseline 케이스)
├── artifacts/                  # 일회성 스크립트 결과물 (enrich 진행/baseline 출력)
├── baseline/answer_tone_v0/    # 응답 톤 baseline 캡처 (백로그 #2)
└── logs/                       # rerank/qa JSONL 데일리 로그
```

루트 단일 파일:
- [cli_chat.py](../cli_chat.py): 로컬 CLI 챗 (개발용 진입점)
- [.env](../.env), [.env.example](../.env.example): 환경변수
- [requirements.txt](../requirements.txt) / [requirements-dev.txt](../requirements-dev.txt)

---

## 2. 엔트리포인트

### FastAPI app
- 정의: [api/main.py:67](../api/main.py#L67) `app = FastAPI(...)`
- lifespan: [api/main.py:32-61](../api/main.py#L32-L61) — DenseRetriever / BM25Retriever / HybridRetriever / RecipeStore / ChatOrchestrator를 1회 초기화하고 `app.state`에 보관.

### 라우트
| Path | Method | Handler |
| --- | --- | --- |
| `/healthz` | GET | [api/main.py:99](../api/main.py#L99) `healthz()` → `HealthResponse(status="ok")` |
| `/chat` | POST | [api/main.py:104](../api/main.py#L104) `chat(req, request)` → `orchestrator.handle(req)`로 위임 |

v0.2의 `/recommend`·`/ask`는 폐기. `/chat` 단일 엔드포인트로 통합됨 (api/main.py 모듈 docstring).

### 전역 예외 핸들러
- [api/main.py:78](../api/main.py#L78) `unhandled_exception_handler()` — HTTPException은 통과, 나머지는 500 마스킹.

---

## 3. ChatOrchestrator 흐름

### 클래스 위치
- [api/chat_orchestrator.py:129](../api/chat_orchestrator.py#L129) `class ChatOrchestrator`
- 핵심 메서드: [api/chat_orchestrator.py:145](../api/chat_orchestrator.py#L145) `async def handle(self, request: ChatRequest) -> ChatResponse`

### intent 분류기 호출 지점
- [api/chat_orchestrator.py:163](../api/chat_orchestrator.py#L163) `classify_intent(...)` (정의는 [api/intent.py:57](../api/intent.py#L57))
- 실패 시 `IntentClassifyError` → `ANSWER_INTENT_FALLBACK` 답변으로 즉시 반환.

### intent → 핸들러 1:1 매핑

| intent | 분기 지점 (line) | 핸들러 (파일:함수) | 최종 응답 생성자 |
| --- | --- | --- | --- |
| `out_of_scope` | [chat_orchestrator.py:192](../api/chat_orchestrator.py#L192) | (없음, orchestrator 내부 상수) | `ANSWER_OUT_OF_SCOPE` 상수 |
| `slot_fill` | [chat_orchestrator.py:255](../api/chat_orchestrator.py#L255) | (충족 시 recommend로 fall-through) [api/slot_questions.py:30](../api/slot_questions.py#L30) `build_slot_question()` | `SLOT_QUESTIONS` 풀 텍스트 |
| `recommend` | [chat_orchestrator.py:277](../api/chat_orchestrator.py#L277) | [api/recommend.py:41](../api/recommend.py#L41) `handle_recommend()` | "조건에 맞춰 2개 골라봤어요." (recommend.py:189) |
| `refine` | [chat_orchestrator.py:340](../api/chat_orchestrator.py#L340) | [api/refine.py:196](../api/refine.py#L196) `handle_refine()` | [refine.py:180](../api/refine.py#L180) `_build_refine_answer()` 템플릿 |
| `ask` | [chat_orchestrator.py:360](../api/chat_orchestrator.py#L360) | [api/qa_handler.py:31](../api/qa_handler.py#L31) `handle_ask()` → [rag/qa.py:178](../rag/qa.py#L178) `answer()` | LLM `QAResponse.answer` |

### 재분류 규칙
- `refine` + last_recs 비어있음 → `recommend`로 재분류 ([chat_orchestrator.py:210](../api/chat_orchestrator.py#L210))
- `slot_fill` + 슬롯 충족 → `recommend`로 fall-through ([chat_orchestrator.py:257](../api/chat_orchestrator.py#L257))
- `recommend` + 슬롯 미충족 → `slot_fill` 폴백 ([chat_orchestrator.py:279](../api/chat_orchestrator.py#L279))
- `recommend` + free_text 안전망 발동 → `slot_fill`로 1턴 질문 ([chat_orchestrator.py:303](../api/chat_orchestrator.py#L303))

### 응답 조립
- [chat_orchestrator.py:438](../api/chat_orchestrator.py#L438) `_build_response()`: HandlerResult + 공통 필드(flags 기본값, free_text_delta 마스킹, timings) → `ChatResponse`. 로깅 포함.
- free_text_delta 노출 화이트리스트: `recommend / refine / slot_fill`만 ([chat_orchestrator.py:436](../api/chat_orchestrator.py#L436) `_ALLOWED_DELTA_INTENTS`). ask/out_of_scope/refused는 null로 마스킹.

---

## 4. 각 intent 핸들러 내부 구조

### recommend ([api/recommend.py:41](../api/recommend.py#L41))
1. `slots.free_text or free_text_delta`로 effective_free_text 결정 (recommend.py:71)
2. [rag/query_builder.py:21](../rag/query_builder.py#L21) `build_retrieval_query()` — 정형 슬롯 + 자유텍스트 합성
3. `retriever.search(query)` → [rag/retriever.py:196](../rag/retriever.py#L196) `HybridRetriever.search()`
4. candidates 0건이면 → slot_fill 폴백 (`needs_more_slots`+`is_fallback`)
5. [rag/reranker.py:443](../rag/reranker.py#L443) `rerank(query, candidates, top_k=2, previously_recommended=None)`
6. `RecipeStore.get_recipe_by_id()`로 메타 lookup
7. 매핑된 추천이 2개 미만이면 다시 slot_fill 폴백 (응답 불변식 보호)

### refine ([api/refine.py:196](../api/refine.py#L196)) — 8단계
1. [refine.py:121](../api/refine.py#L121) `_rebuild_query_with_llm()` — gpt-5-mini로 검색용 query 한 줄 재구성. 실패 시 `_fallback_concat_query()` (refine.py:98) + `is_fallback=true`
2. `exclude_ids = [_normalize_recipe_id(lr.recipe_id) for lr in last_recommendations]` (refine.py:229) — **직전 턴 추천 id만**
3. `retriever.search(search_query, exclude_ids=...)` — exclusion 포함
4. 0건이면 즉시 `intent="ask"` fallback (refine.py:250)
5. `rerank(query, candidates, top_k=2, previously_recommended=[...])` — 직전 메뉴명 LLM에 전달해 차별화 유도
6. RecipeStore lookup
7. [refine.py:180](../api/refine.py#L180) `_build_refine_answer(free_text_delta, count)` — count=1이면 "조건이 좁아서 1개만…", free_text_delta 있으면 `{delta} 반영해서 다시 골라봤어요.` 템플릿
8. HandlerResult 반환
- **free_text_delta 처리 지점**: refine.py:266 `effective_free_text = slots.free_text or free_text_delta`. recommend와 동일 패턴.

### slot_fill ([chat_orchestrator.py:255](../api/chat_orchestrator.py#L255))
- 충족 검사: `_is_slots_sufficient()` (chat_orchestrator.py:430) — meal_times AND purpose
- 충족 시 → recommend로 fall-through
- 미충족 시 → [slot_questions.py:30](../api/slot_questions.py#L30) `build_slot_question(slots)`로 질문
- **안전망 분기** (recommend 분기 안에 있음): chat_orchestrator.py:296-320
  - `_was_freetext_safetynet_asked(history)` — 직전 assistant가 안전망 질문이었으면 `_normalize_freetext_negative()`로 부정 응답 처리
  - `_needs_freetext_safetynet(slots, delta)` — free_text·delta 둘 다 3자 미만이면 한 번만 질문

### ask ([api/qa_handler.py:31](../api/qa_handler.py#L31))
- `last_recommendations`의 각 메뉴 → `RecipeStore.get_recipe_by_id()`로 lookup (qa_handler.py:60-69) — 최대 5개
- 전부 missing이면 fallback 답변 (qa_handler.py:79)
- history를 `[{role, content}]` dict로 변환 (qa_handler.py:91)
- [rag/qa.py:178](../rag/qa.py#L178) `answer(query, retrieved_docs, chat_history)` 호출
- **슬롯/맥락 활용 지점**: chat_history는 LLM 프롬프트에 그대로 전달되어 지시대명사("그거", "1번") 해소에 쓰임. slots 자체는 ask에서 사용 안 함 (chat_orchestrator.py:228-232에서 ask는 merge_slots 화이트리스트 제외).
- `qa_resp.refused/qa_failed/out_of_scope/is_fallback`을 flags_override로 매핑 (qa_handler.py:114-124)

### out_of_scope
- LLM 호출 없이 [chat_orchestrator.py:192](../api/chat_orchestrator.py#L192)에서 즉시 반환
- 응답: `ANSWER_OUT_OF_SCOPE = "식사/레시피 관련 질문만 도와드릴 수 있어요."` + `flags.out_of_scope=true`
- slot 추출도 스킵

---

## 5. LLM 호출 지점 전수조사

| 호출자 | 함수 (파일:줄) | 모델 | 프롬프트 파일 | 호출 방식 |
| --- | --- | --- | --- | --- |
| 의도 분류 | [api/intent.py:103](../api/intent.py#L103) `classify_intent()` | `gpt-5-mini` | [api/prompts/intent_prompt.py](../api/prompts/intent_prompt.py) | `chat.completions.create` JSON mode |
| 슬롯 추출 | [api/slot.py:96](../api/slot.py#L96) `extract_slots()` | `gpt-5-mini` | [api/prompts/slot_prompt.py](../api/prompts/slot_prompt.py) | `chat.completions.create` JSON mode |
| refine query 재구성 | [api/refine.py:143](../api/refine.py#L143) `_rebuild_query_with_llm()` | `gpt-5-mini` | [api/prompts/refine_prompt.py](../api/prompts/refine_prompt.py) | `chat.completions.create` JSON mode |
| rerank (recommend & refine) | [rag/reranker.py:367](../rag/reranker.py#L367) `_call_llm()` | `gpt-5-mini` ([config.py:33](../rag/config.py#L33) `RERANK_MODEL`) | [rag/rerank_prompt.py](../rag/rerank_prompt.py) (`SYSTEM_PROMPT` + `build_user_prompt()`) | `beta.chat.completions.parse` Structured Output (`RerankResponse`) |
| ask QA 답변 | [rag/qa.py:157](../rag/qa.py#L157) `_call_llm()` | `gpt-5-mini` ([config.py:42](../rag/config.py#L42) `RAG_QA_MODEL` = `RERANK_MODEL`) | [rag/qa_prompt.py](../rag/qa_prompt.py) (`SYSTEM_PROMPT` + `build_qa_user_prompt()`) | `beta.chat.completions.parse` Structured Output (`QAResponse`) |

### 공통 설정 위치 — [rag/config.py](../rag/config.py)
- 모델명: `RERANK_MODEL = "gpt-5-mini"` (line 33), `RAG_QA_MODEL = RERANK_MODEL` (line 42)
- temperature 설정 없음 (gpt-5-mini는 reasoning model). 대신 `reasoning_effort`로 제어:
  - 의도 분류/슬롯 추출/refine query 재구성: `reasoning_effort="minimal"` (각 호출부에 하드코딩)
  - rerank/qa: `RERANK_REASONING_EFFORT = "low"` (config.py:34), `RAG_QA_REASONING_EFFORT = RERANK_REASONING_EFFORT` (config.py:44)
- timeout (config.py:11-13):
  - `RERANK_TIMEOUT_SECONDS = 30.0`
  - `INTENT_TIMEOUT_SECONDS = 5.0`
  - `SLOT_TIMEOUT_SECONDS = 8.0`
  - refine query 재구성은 SDK timeout만 사용, `timeout=5.0` 하드코딩 (refine.py:152)
  - QA는 별도 asyncio.wait_for 래핑 없음 (config에 QA_TIMEOUT 없음)
- max tokens:
  - intent/slot: `max_completion_tokens=2000` (각 호출부 하드코딩)
  - refine: `max_completion_tokens=1000` (refine.py:145)
  - rerank: `RERANK_MAX_COMPLETION_TOKENS = 3000` (config.py:35)
  - qa: `RAG_QA_MAX_TOKENS = 4000` (config.py:43)

### 재시도/타임아웃 정책
- rerank: `RERANK_RETRY_LIMIT = 1` (config.py:38). retry 루프 전체를 `asyncio.wait_for(RERANK_TIMEOUT_SECONDS)`로 감쌈 (reranker.py:534). timeout 시 rule-based `_build_timeout_fallback_response()`로 강등.
- qa: `RAG_QA_RETRY_LIMIT = 1` (config.py:45). asyncio.wait_for 래핑 없음. Validation 실패 시 retry 프롬프트에 `[이전 응답 오류]` 블록 append (qa.py:262).
- intent/slot/refine_query: 모두 `asyncio.wait_for`로 감싸고 재시도 없음. 실패 시 도메인 예외 발생.

---

## 6. 데이터 흐름

### 3개 retrieval 소스
1. **ChromaDB** ([rag/retriever.py:84](../rag/retriever.py#L84) `class DenseRetriever`)
   - 접근: `chromadb.PersistentClient(path=CHROMA_PATH).get_collection(name=COLLECTION_NAME)` (retriever.py:86-87)
   - 경로: `chroma/recipes_v2` ([config.py:15](../rag/config.py#L15)), 컬렉션 `meal_bot_recipes_v2`
   - 임베딩: BAAI/bge-m3 (SentenceTransformer, normalize=True)
   - top_k: `DENSE_TOP_K = 50`

2. **BM25** ([rag/retriever.py:126](../rag/retriever.py#L126) `class BM25Retriever`)
   - 접근: `bm25/recipes_v2/{bm25.pkl, recipe_ids.pkl, tokenized_corpus.pkl}` pickle 3종 (retriever.py:136-141)
   - 또한 `data/recipes_enriched_v2.json`을 **직접 로드해 `self.recipe_metadata` dict 보관** (retriever.py:145-149) — BM25Retriever 인스턴스가 자체적으로 recipe 메타데이터 들고 있음
   - 토크나이저: [rag/tokenizer.py](../rag/tokenizer.py) `KiwiTokenizer` (kiwipiepy)
   - top_k: `BM25_TOP_K = 50`

3. **recipes_enriched_v2.json 직접 조회** ([rag/recipe_store.py:12](../rag/recipe_store.py#L12) `class RecipeStore`)
   - lifespan에서 `RECIPE_JSON_PATH` env (기본 `data/recipes_enriched_v2.json`) 로드 (main.py:40-41)
   - `{rcp_seq: recipe_dict}` 매핑 (recipe_store.py:21-29)
   - 핸들러들이 `get_recipe_by_id()`로 rerank 결과 → 상세 dict 조회

### candidates 합성 흐름
```
HybridRetriever.search(query, exclude_ids?)        # rag/retriever.py:196
  ├─ DenseRetriever.search(top_k=50)               # dense rank/score 부여
  ├─ BM25Retriever.search(top_k=50)                # bm25 rank/score 부여
  └─ _rrf_merge(dense_hits, bm25_hits, RRF_TOP_K=30)
       # 1/(RRF_K=60 + rank) 누적 합산 → top-30
       → list[Hit] with {dense_rank, bm25_rank, rrf_score, metadata}

핸들러(recommend/refine):
  candidates = [{...hit.metadata, dense_rank, bm25_rank, rrf_score} for hit in hits]
  rerank(query, candidates, ...)                    # rag/reranker.py:443
    ├─ _candidate_to_prompt_dict(c)                 # 18개 필드만 추출
    ├─ LLM gpt-5-mini Structured Output → RerankResponse
    └─ recipe_id 환각/중복/rank 시퀀스 검증

handler:
  for item in rerank_resp.recommendations:
    recipe = recipe_store.get_recipe_by_id(item.recipe_id)
    → Recommendation(recipe_id, name, summary, main_ingredients, cooking_time, reason)
```

### 같은 JSON을 두 곳에서 따로 로드
- `BM25Retriever.__init__`이 `recipes_enriched_v2.json`을 직접 읽어 `recipe_metadata` 보관 (BM25 검색 결과의 `name`/메타 결합용).
- `RecipeStore`도 같은 파일을 lifespan에서 로드.
- **중복 로드**: 메모리에 두 벌. RecipeStore 쪽이 main lookup, BM25Retriever 쪽은 hit.metadata 채우기용. (10절 "구조적 빚" 참조)

---

## 7. 스키마 핵심 — [api/schemas.py](../api/schemas.py)

### 요청
- `ChatRequest` (schemas.py:74): `session_id`, `turn_id`, `message`(≤500자), `history`(≤6 ChatMessage), `slots`, `last_recommendations`(≤5)
- `ChatMessage` (schemas.py:20): `role` Literal["user","assistant"], `content`(1-2000자)
- `Slots` (schemas.py:27): `meal_times` Literal["아침","점심","저녁","간식","야식"] 리스트, `purpose` Literal["light","protein","hearty","tasty"], `free_text`(≤500자). 빈 배열 자동 None 정규화 (model_validator)
- `LastRecommendation` (schemas.py:42): `recipe_id`, `name`

### 응답
- `ChatResponse` (schemas.py:88): `turn_id`, `intent` Literal["recommend","slot_fill","refine","ask"], `answer`(1-2000자), `slots_updated`, `recommendations`(≤2), `flags`, `free_text_delta`(≤500자)
- `Recommendation` (schemas.py:49): `recipe_id`, `name`, `summary`, `main_ingredients`, `cooking_time` Optional[int], `reason`(10-200자)
- `Flags` (schemas.py:61): `needs_more_slots`, `out_of_scope`, `is_fallback`, `refused`(default False)

### 일관성 검증
- `ChatResponse._check_intent_recommendations_consistency` (schemas.py:107):
  - `slot_fill`/`ask`: recommendations 길이=0
  - `recommend`: 정확히 2
  - `refine`: 1~2

### 핸들러 공용 반환 타입 — [api/handler_result.py:16](../api/handler_result.py#L16)
`HandlerResult(intent, answer, recommendations, flags_override, timings)` — dataclass. orchestrator가 ChatResponse로 조립.

### 도메인 예외 — [api/errors.py](../api/errors.py)
- `IntentClassifyError`, `SlotExtractError`, `QueryRebuildError`, `RerankError`

### Slot 추출 결과 — [api/slot.py:31](../api/slot.py#L31)
`SlotDelta(meal_times, purpose, free_text_delta)` — 이번 턴 delta만. orchestrator가 누적 slots에 merge.

### LLM 응답 스키마
- `RerankItem` / `RerankResponse` ([rag/rerank_prompt.py:30-49](../rag/rerank_prompt.py#L30-L49)): rank, recipe_id, reason, matched_intents (min 1 max 4), is_fallback
- `QAResponse` ([rag/qa_prompt.py:42](../rag/qa_prompt.py#L42)): answer, used_fields, refused, out_of_scope, qa_failed, is_fallback
- `IntentResult` ([api/intent.py:32](../api/intent.py#L32)): intent Literal 5, reason (1-200자)

---

## 8. 설정/환경변수 — [rag/config.py](../rag/config.py)

### 설정 파일 위치
- 단일 모듈 `rag/config.py`에서 `load_dotenv()` 후 모듈 상수로 export.
- `.env`는 프로젝트 루트.

### 주요 상수
- ChromaDB: `CHROMA_PATH="chroma/recipes_v2"`, `COLLECTION_NAME="meal_bot_recipes_v2"`, `EMBEDDING_MODEL="BAAI/bge-m3"`
- BM25: `BM25_INDEX_PATH="bm25/recipes_v2"`, `BM25_TOP_K=50`
- Hybrid: `DENSE_TOP_K=50`, `RRF_TOP_K=30`, `RRF_K=60`
- Rerank: `RERANK_MODEL="gpt-5-mini"`, `RERANK_REASONING_EFFORT="low"`, `RERANK_MAX_COMPLETION_TOKENS=3000`, `RERANK_TOP_K_OUTPUT=5`, `RERANK_MIN_CANDIDATES=5`, `RERANK_RETRY_LIMIT=1`
- QA: `RAG_QA_MODEL=RERANK_MODEL`, `RAG_QA_MAX_TOKENS=4000`, `QA_MAX_DOCS=3`
- Timeout (env override 가능): `RERANK_TIMEOUT_SECONDS=30.0`, `INTENT_TIMEOUT_SECONDS=5.0`, `SLOT_TIMEOUT_SECONDS=8.0`
- Logging: `LOG_DIR="logs"`, `RERANK_LOG_FILE_PATTERN="rerank_{date}.jsonl"`, `QA_LOG_FILE_PATTERN="qa_{date}.jsonl"`

### .env 키 (값 마스킹)
```
OPENAI_API_KEY=<masked>
FOOD_API_KEY=<masked>                # 데이터 수집 단계에서만 사용 (런타임 미사용)
MYSQL_ROOT_PASSWORD=<masked>         # rag-api 런타임은 MySQL 사용 안 함 (Spring 쪽 용도로 추정)
MYSQL_USER=<masked>
MYSQL_PASSWORD=<masked>
MYSQL_HOST=<masked>
MYSQL_PORT=<masked>
MYSQL_DATABASE=<masked>
```
또한 main.py:40에서 `RECIPE_JSON_PATH` env를 읽음 (기본값 `data/recipes_enriched_v2.json`).

---

## 9. scripts/ 디렉터리 역할

| 스크립트 | 역할 | reindex 필요 여부 |
| --- | --- | --- |
| [build_vector_db.py](../scripts/build_vector_db.py) | `recipes_enriched_v2.json` → ChromaDB 적재 (BAAI/bge-m3 임베딩) | **이 스크립트 = reindex 본체** (ChromaDB 재생성). 데이터 변경 시 필수 실행 |
| [build_bm25_index.py](../scripts/build_bm25_index.py) | `recipes_enriched_v2.json` → BM25 pickle 3종 빌드 | **이 스크립트 = reindex 본체** (BM25 재생성). 데이터·토크나이저 변경 시 필수 |
| [enrich_recipes.py](../scripts/enrich_recipes.py) | `recipes_cleaned.json` → 11개 필드 LLM 보강 → `recipes_enriched.json` | 후속으로 enrich_v2 + 인덱스 재빌드 필요 |
| [enrich_recipes_v2.py](../scripts/enrich_recipes_v2.py) | `recipes_enriched.json` → 신규 5개 필드 추가 → `recipes_enriched_v2.json` | 후속으로 reindex (BM25 + Chroma) 필요 |
| [parse_ingredients.py](../scripts/parse_ingredients.py) | raw ingredients 텍스트 구조화 파서 | enrich 단계 의존성. 별도 reindex 필요 X (재호출은 데이터 재빌드 시) |
| [evaluate_hybrid_baseline.py](../scripts/evaluate_hybrid_baseline.py) | Hybrid RRF 검색 품질 점검 (대표 쿼리 10개) | 평가용. reindex 불필요 |
| [evaluate_qa_baseline.py](../scripts/evaluate_qa_baseline.py) | `eval/qa_cases_v1.json` 10케이스 qa.answer 평가 | 평가용. reindex 불필요 |
| [run_query.py](../scripts/run_query.py) | Dense / BM25 / Hybrid 3-way 비교 (수동 디버깅) | 평가용 |
| [run_rerank_baseline.py](../scripts/run_rerank_baseline.py) | hybrid top30 → rerank baseline 산출 | 평가용 |
| [test_qa_integration.py](../scripts/test_qa_integration.py) | Retriever → Rerank → QA 통합 동작 콘솔 출력 | 통합 점검용 |
| [_enrichment_prompts.py](../scripts/_enrichment_prompts.py) / [_enrichment_schema.py](../scripts/_enrichment_schema.py) | enrich_recipes.py 의존성 (프롬프트/스키마) | (직접 실행 X) |
| [_enrichment_v2_prompts.py](../scripts/_enrichment_v2_prompts.py) / [_enrichment_v2_schema.py](../scripts/_enrichment_v2_schema.py) | enrich_recipes_v2.py 의존성 | (직접 실행 X) |

**Reindex 워크플로우** ([deploy-ai-reindex.yml](../../.github/workflows/deploy-ai-reindex.yml))은 `build_bm25_index.py` → `build_vector_db.py` 순으로 실행.

---

## 10. tests/ 디렉터리 현황

### 테스트 파일 목록 + 상태

| 파일 | 테스트 수 (collect) | 상태 |
| --- | --- | --- |
| [test_query_builder.py](../tests/test_query_builder.py) | (collect 정상) | ✅ 정상 |
| [test_recipe_store.py](../tests/test_recipe_store.py) | 9 | ✅ 정상 |
| [test_rerank_prompt.py](../tests/test_rerank_prompt.py) | 7 | ✅ 정상 |
| [test_reranker.py](../tests/test_reranker.py) | 5 | ✅ 정상 |
| [test_reranker_prompt_dict.py](../tests/test_reranker_prompt_dict.py) | 3 | ✅ 정상 |
| [test_retriever_metadata_normalization.py](../tests/test_retriever_metadata_normalization.py) | 9 | ✅ 정상 |
| [test_api_main.py](../tests/test_api_main.py) | 14 (collect는 됨) | ❌ **stale**: v0.2의 `/recommend`·`/ask` 엔드포인트 가정. 실제 실행 시 깨짐 |
| [test_schemas.py](../tests/test_schemas.py) | (collect 단계 실패) | ❌ **stale**: `from api.schemas import AskRequest, RecommendRequest` ImportError (v0.2 스키마 잔재) |

```
.venv/bin/python -m pytest --collect-only -q
→ 62 tests collected, 1 error in 3.21s
   (test_schemas.py ImportError on AskRequest)
```

### Stale 테스트 상세 (4건 표현은 부정확 — 실제론 2개 파일 통째로)
- **test_schemas.py 전체**: `AskRequest`, `RecommendRequest` import 실패. v0.3에서 ChatRequest로 통합되며 두 클래스 삭제됨. 파일 collect 자체가 실패.
- **test_api_main.py 전체**: `test_recommend_*`, `test_ask_*` 14개 모두 v0.2 엔드포인트 호출 가정. collect는 되지만 실행하면 404·route mismatch.

---

## 구조적 빚 / 이상 동작 별도 섹션

조사 중 발견한 항목 (수정은 별도 작업 필요):

### A. recipes_enriched_v2.json 이중 로드
- `BM25Retriever.__init__` (retriever.py:145)이 자체적으로 JSON을 읽어 `recipe_metadata` 보관.
- `RecipeStore.__init__` (recipe_store.py:18)도 같은 파일 로드.
- 메모리 상 5MB짜리 dict가 두 벌 존재. RecipeStore를 BM25Retriever에 주입하는 리팩토링 여지 있음.

### B. stale tests 2개 파일이 CI 그물에서 빠져 있음
- test_schemas.py는 import 단계 실패 → pytest가 1 error로 중단.
- test_api_main.py 14건은 v0.2 잔재. 둘 다 v0.3 이후 정리되지 않음.
- 현재 진행 가능한 테스트는 62건이나, 워크플로우([deploy-ai.yml](../../.github/workflows/deploy-ai.yml))에는 pytest 단계가 없어 회귀가 잡히지 않는 상태.

### C. QA에 asyncio.wait_for 래핑 없음
- rerank는 `RERANK_TIMEOUT_SECONDS=30s`로 전체 래핑.
- qa.answer는 SDK timeout만 의존 (rag/qa.py:160-169). config에 `QA_TIMEOUT_SECONDS` 없음.
- ask 요청이 hang하면 orchestrator 전체 turn이 SDK 기본 timeout까지 매달림.

### D. recommend/refine answer가 정적 템플릿
- recommend: 항상 "조건에 맞춰 2개 골라봤어요." 하드코딩 (recommend.py:189)
- refine: `{free_text_delta} 반영해서 다시 골라봤어요.` 1형식 + 1개 추천 시 별도 문구 (refine.py:186-190)
- baseline/answer_tone_v0/ 캡처로 톤 단조로움 확인됨 (백로그 #2 본 작업 대상).

### E. refine의 free_text_delta 추출 일관성 부족
- baseline scenario_a T4 관찰: 사용자 "다른거 추천해줘"가 free_text_delta로 추출되어 응답 answer 템플릿에 "다른거 추천해줘 반영해서…"로 노출됨.
- slot_prompt가 메타 발화를 free_text_delta로 잘못 잡는 케이스로 보임. orchestrator-v0.3.md 정책에는 메타 발화 제외 명시되어 있으나 LLM이 지키지 못한 사례.

### F. refine query 재구성 timeout 하드코딩
- refine.py:152의 `timeout=5.0`은 config 상수가 아닌 매직 넘버. 다른 호출은 모두 config.py의 `*_TIMEOUT_SECONDS` 참조.

### G. main.py의 `__pycache__`가 워킹 트리에 남아있음
- 루트와 api/rag/scripts 모두 `__pycache__` 디렉터리 존재. .gitignore에는 들어가 있는 듯 (git status clean이라 추적은 안 됨).
