# refine 로직 v0.3

## 1. 목적

직전 턴 추천 결과에 대한 사용자의 재추천 요청을 처리한다.
chat_orchestrator의 `refine` 의도 분기에서 호출된다.

## 2. 모듈 역할

- 위치: `refine.py`
- 호출자: `chat_orchestrator`
- 호출 시점: `intent="refine"` 확정 + `last_recommendations` 존재
- `refine` + `last_recommendations=[]` 케이스는 orchestrator 3단계에서 `recommend`로 재분류되어 **도달하지 않는다**.

## 3. 전체 흐름

1. LLM query 재구성 (refine 전용 LLM 호출)
2. exclude id 구성 (직전 턴 추천 id만)
3. `hybrid_retrieve_with_exclusion` 실행
4. 검색 결과 0건이면 즉시 fallback 반환
5. LLM rerank (기존 모듈 재사용, `previously_recommended` 컨텍스트 추가)
6. top2 선정
7. answer 메시지 생성 (템플릿 + `free_text_delta` 변수)
8. 응답 반환

## 4. LLM query 재구성

### 목적
- "이거 말고", "다시" 같은 메타 발화 제거
- 누적된 조건(`slots`, `free_text`)과 이번 턴 발화를 검색용 자연어 query 한 줄로 압축

### 입력
- `slots`: `meal_times`, `purpose`, `free_text` (누적 전체)
- `free_text_delta`: 이번 턴 슬롯 추출 결과
- `message`: 이번 턴 사용자 발화
- `last_recommendations`: 직전 추천 메뉴 이름 리스트

### 출력
- `search_query`: string (한 줄 자연어 query)
- 예: `"매운 생선 요리 저녁 메뉴"`

### 모델
- `gpt-5-mini`
- `temperature`: 0
- `response_format`: JSON
- `max_tokens`: 80
- `timeout`: 5초

### 실패 처리
- JSON 파싱 실패, 타임아웃 등 → **결정론적 fallback으로 자동 전환**
- fallback query 구성:
  - `" ".join([meal_times..., purpose, free_text, free_text_delta, message])` 단순 concat
  - `free_text_delta`는 이번 턴 신규 조건이므로 fallback에서도 포함 (정보 손실 방지)
  - 일부 단어가 중복될 수 있으나, hybrid retrieval에서 자연스럽게 흡수됨
- fallback 사용 시 `flags.is_fallback=true` 마킹

### System Prompt (풀텍스트)

```text
당신은 한국어 음식 추천 챗봇의 검색 query 재구성기입니다.
사용자가 직전 추천 결과에 대해 재추천을 요청했습니다.
누적된 조건과 이번 턴 발화를 종합해 검색에 적합한 자연어 query 한 줄을 생성하세요.

[원칙]

"이거 말고", "다시", "다른 거", "추천해줘" 같은 메타 발화는 query에서 제외합니다.
누적된 slots(meal_times, purpose)와 free_text는 query에 반영합니다.
이번 턴에 새로 등장한 조건(free_text_delta, message에서 추출된 신규 조건)을 우선 반영합니다.
직전 추천 메뉴 이름은 query에 넣지 마세요. (exclude는 별도 처리)
query는 한 줄, 한국어 자연어, 30자 이내로 압축합니다.

[출력 형식]
{"search_query": "..."}

[예시]
입력: slots={meal_times:["저녁"], purpose:"hearty", free_text:"매운"}, free_text_delta:"생선", message:"생선으로 다시", last_recs:["돼지고기 김치찌개"]
출력: {"search_query": "매운 생선 저녁 든든한 요리"}

입력: slots={meal_times:["점심"], purpose:"light", free_text:""}, free_text_delta:null, message:"이거 말고 다른 거", last_recs:["샐러드"]
출력: {"search_query": "가벼운 점심 메뉴"}

입력: slots={meal_times:["저녁"], purpose:"protein", free_text:"닭고기"}, free_text_delta:"기름기 적은", message:"기름기 적은 걸로", last_recs:["치킨 데리야끼"]
출력: {"search_query": "기름기 적은 닭고기 저녁 단백질 요리"}
```

### User Prompt Template

```text
[slots]
meal_times: {slots.meal_times}
purpose: {slots.purpose}
free_text: {slots.free_text}

[free_text_delta]
{free_text_delta}

[message]
{message}

[last_recommendations]
{last_rec_names}
```

## 5. exclude id 구성

- `exclude_ids = [r.recipe_id for r in last_recommendations]`
- **직전 턴 추천만 exclude**
- 세션 전체 누적 exclude는 v0.3 범위 밖
- 사유:
  - `last_recommendations` 스키마 정의가 "직전 턴"
  - 세션 누적 exclude를 하려면 Spring이 추가 필드 보내야 함 (스키마 변경 필요)
  - 사용자가 "처음 거 좋았는데"로 되돌아갈 여지 보존

## 6. hybrid_retrieve_with_exclusion

- 기존 `hybrid_retrieve` 함수에 `exclude_ids` 파라미터 추가한 변형
- ChromaDB dense retrieval + BM25 + RRF merge
- `exclude_ids`에 포함된 `recipe_id`는 결과에서 제거
- 반환: 후보 리스트 (정렬된 상위 K개, K는 기존 정책 따름)

## 7. 0건 처리 (fallback)

검색 결과 0건이면 **재시도 없이 즉시** fallback 응답.

응답:
- `intent`: `"ask"`
- `answer`: `"조건이 너무 까다로워서 맞는 메뉴를 찾지 못했어요. 조건을 조금 풀어주시면 다시 찾아볼게요."`
- `recommendations`: `[]`
- `slots_updated`: (변경 없는 전체 스냅샷)
- `flags`:
  - `needs_more_slots`: false
  - `out_of_scope`: false
  - `is_fallback`: true

**재시도 안 하는 이유:**
- exclude를 풀면 사용자가 거부한 메뉴를 다시 보여주는 셈 (UX 나쁨)
- 조건 완화 retry는 어떤 조건을 풀지 휴리스틱이 복잡 (v0.4에서 검토)
- 0건은 정직하게 사용자에게 알리는 게 더 명확

## 8. LLM rerank (재사용)

- 기존 `llm_rerank_and_pick_top2` 모듈 재사용
- 입력에 `previously_recommended` 컨텍스트 추가
  - 직전 추천 메뉴 이름 리스트
  - rerank가 "이전과 다른 매력 포인트"를 `reason`에 강조하도록 유도
- `recommend` 호출 시에는 `previously_recommended=[]` 빈 값으로 전달
- `refine` 호출 시에는 `last_recommendations`의 메뉴명 전달

rerank 모듈의 프롬프트 변경 사항:
- `previously_recommended` 필드 **옵셔널로 추가**
- 값이 비어 있으면 일반 `reason` 생성
- 값이 있으면 **"이전 추천(X, Y)과 어떻게 다른지" 포함한 `reason` 생성**
- 단, `reason` 길이 60~120자 제약은 그대로

## 9. answer 메시지 생성 (템플릿 + 변수)

정상 응답 시 `answer` 문자열 생성 규칙:

- `free_text_delta`가 not null:
  - `answer = f"{free_text_delta} 반영해서 다시 골라봤어요."`
- `free_text_delta`가 null:
  - `answer = "조건 반영해서 다시 골라봤어요."`

예시:
- `free_text_delta="생선"` → `"생선 반영해서 다시 골라봤어요."`
- `free_text_delta="기름기 적은"` → `"기름기 적은 반영해서 다시 골라봤어요."`
- `free_text_delta=null` → `"조건 반영해서 다시 골라봤어요."`

LLM 호출 없이 **결정론적 처리**.

## 10. 정상 응답 스키마

- `intent`: `"refine"`
- `answer`: §9 템플릿
- `slots_updated`: orchestrator의 `merge_slots()` 결과 전체 스냅샷
- `recommendations`: top2 (`id`, `name`, `summary`, `main_ingredients`, `cooking_time`, `reason` 60~120자)
- `flags`:
  - `needs_more_slots`: false
  - `out_of_scope`: false
  - `is_fallback`: false (query 재구성 LLM 실패 시 true)

## 11. fallback 응답 종합 표

| 케이스 | intent | answer | recommendations | flags |
|---|---|---|---|---|
| query 재구성 LLM 실패 | refine | 정상 템플릿 | top2 (결정론 query로 검색한 결과) | `is_fallback=true` |
| 검색 0건 | ask | 조건 완화 안내 | `[]` | `is_fallback=true` |
| rerank 실패 | refine | 정상 템플릿 | top2 (retrieval score 순) | `is_fallback=true` |

## 12. 로깅 추가 키 (orchestrator 로깅 정책 위에 추가)

- `query_rebuild_ms`: query 재구성 LLM 소요 시간
- `search_query`: LLM이 생성한 자연어 query (또는 fallback query)
- `exclude_ids`: exclude 처리된 `recipe_id` 리스트
- `excluded_count`: exclude된 개수

## 13. 불변식

- `exclude_ids`는 항상 `last_recommendations`의 `recipe_id`로만 구성
- `search_query`는 항상 null이 아님 (LLM 실패해도 fallback query 존재)
- 0건 응답 시 `intent`는 반드시 `"ask"`
- 정상 응답 시 `recommendations.length=2`

---

## 부록. 의사코드

```text
def handle_refine(request, slots, free_text_delta, last_recs):
    t = now_ms()

    # 1. query 재구성 (LLM)
    try:
        search_query = llm_rebuild_query(
            slots=slots,
            free_text_delta=free_text_delta,
            message=request.message,
            last_recs=last_recs
        )
        query_fallback = False
    except QueryRebuildError:
        search_query = fallback_concat_query(slots, free_text_delta, request.message)
        query_fallback = True

    # 2. exclude 구성
    exclude_ids = [r.recipe_id for r in last_recs]

    # 3. hybrid retrieval with exclusion
    candidates = hybrid_retrieve_with_exclusion(search_query, exclude_ids)

    # 4. 0건 fallback
    if not candidates:
        return refine_zero_result_response(slots)

    # 5. rerank (previously_recommended 전달)
    try:
        top2 = llm_rerank_and_pick_top2(
            candidates,
            slots,
            previously_recommended=[r.name for r in last_recs]
        )
        rerank_fallback = False
    except RerankError:
        top2 = candidates[:2]
        rerank_fallback = True

    # 6. answer 생성
    answer = build_refine_answer(free_text_delta)

    # 7. 응답
    return ChatResponse(
        intent="refine",
        answer=answer,
        slots_updated=slots,
        recommendations=top2,
        flags={
            "needs_more_slots": False,
            "out_of_scope": False,
            "is_fallback": query_fallback or rerank_fallback
        }
    )
```
