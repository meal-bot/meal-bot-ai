# chat_orchestrator 흐름 v0.3

본 문서는 FastAPI 측 `chat_orchestrator`가 단일 턴을 처리하는 v0.3 흐름을 정의한다. 구현 코드는 포함하지 않으며, 흐름 이해용 의사코드는 맨 아래 부록으로만 둔다.

---

## 1. 입력/출력

> 별도 스키마 문서 참조: TODO: POST /chat 스키마 v0.3 문서 링크 연결

### 요청 (POST /chat)

| 필드 | 타입 | 필수 | 비고 |
|---|---|---|---|
| `session_id` | string (1~100자) | required | 세션 식별자. 로깅 키. |
| `turn_id` | string (1~100자) | required | 응답에 그대로 반환. |
| `message` | string (1~500자) | required | 이번 턴 사용자 발화. |
| `history` | list[ChatMessage] | required | 최근 메시지 6개 (왕복 3회 = user 3 + assistant 3). Spring이 슬라이딩 윈도우로 전달. |
| `slots` | object \| null | required | null이면 `empty_slots()`로 정규화. |
| `slots.meal_times` | list["아침"\|"점심"\|"저녁"\|"간식"\|"야식"] \| null | — | — |
| `slots.purpose` | "light"\|"protein"\|"hearty"\|"tasty" \| null | — | — |
| `slots.free_text` | string \| null | — | 누적 free_text 스냅샷. |
| `last_recommendations` | list[Recommendation] \| null | — | 직전 턴 추천 결과. refine/ask 판단용. 없으면 `[]` 또는 `null`로 전달 가능. |

### 응답

| 필드 | 타입 | 필수 | 비고 |
|---|---|---|---|
| `turn_id` | string | required | 요청과 동일. |
| `intent` | "recommend"\|"slot_fill"\|"refine"\|"ask" | required | **out_of_scope는 응답 intent에 두지 않는다.** |
| `answer` | string (min_length=1) | required | 항상 채움. 빈 문자열 금지. |
| `slots_updated` | object | required | 갱신 후 **전체 슬롯 스냅샷**. null은 "해당 슬롯 미설정"만 의미. |
| `recommendations` | list[Recommendation] | required | 추천이 없으면 `[]`. |
| `flags.needs_more_slots` | boolean | required | — |
| `flags.out_of_scope` | boolean | required | — |
| `flags.is_fallback` | boolean | required | — |

---

## 2. 처리 단계

### 0. 입력 정규화 + 타이머 시작
- `t_start` 기록.
- `request.slots`가 null이면 `empty_slots()`로 정규화.
- `request.last_recommendations`가 null이면 `[]`로 정규화.
- `history`는 방어적으로 최대 6개 메시지로 슬라이스.
- 단계별 latency 측정용 타이머 준비 (`t_intent`, `t_slot`, `t_retrieval`, `t_rerank`, `t_qa`).

### 1. 의도 분류
- `intent.py` 호출.
- 5라벨 분류: `recommend` / `slot_fill` / `refine` / `ask` / `out_of_scope`.
- 입력: `message`, `history`, `slots`, `has_last_recs` (= `len(last_recommendations) > 0`).
- 실패 시 fallback 응답 반환:
  - `intent="ask"`
  - `answer="다시 말씀해 주시겠어요?"`
  - `recommendations=[]`
  - `flags.is_fallback=true`

### 2. out_of_scope 즉시 반환
- intent.py가 `out_of_scope`를 반환하면 즉시 응답한다.
- 응답 `intent`에는 `out_of_scope`를 두지 **않는다**. `intent="ask"`로 매핑.
- `flags.out_of_scope=true`
- `answer`는 "식사/레시피 관련 질문만 도와드릴 수 있어요." 계열의 정중한 안내문.
- `recommendations=[]`

### 3. refine + last_recs=[] → recommend로 재분류
- intent가 `refine`인데 `last_recommendations`가 비어 있으면 `recommend`로 재분류한다.
- 이 케이스는 **fallback이 아닌 정상 처리**로 본다. `flags.is_fallback=false` 유지.

### 4. 슬롯 추출
- **모든 의도에서** 슬롯 추출을 시도한다.
- `slot.py` 호출. 내부 재시도 1회 포함.
- 추출 대상: `meal_times`, `purpose`, `free_text_delta`.
- 병합 규칙:
  - `meal_times`: 새 값이 있으면 덮어쓰기.
  - `purpose`: 새 값이 있으면 덮어쓰기.
  - `free_text`: 기존 `slots.free_text`에 `free_text_delta`를 정제 후 누적.
- 슬롯 추출 실패 시 `intent="slot_fill"`로 강제 전환.
- **누적 실패 카운트는 FastAPI가 관리하지 않는다.** Spring DB/서비스 레이어에서 관리.

### 5. 의도별 분기 처리
- 최종 intent 기준으로 handler 호출.
- **응답의 `intent`는 재분류 후 intent**를 사용한다.
- 응답 직전 §7의 불변식을 검증.
- `total_ms`를 포함해 전체 로깅.

---

## 3. 의도별 분기 상세

### slot_fill
- 슬롯 추출 후 슬롯 충족 여부 검사.
- 충족 기준:
  - `meal_times`가 비어 있지 않음
  - `purpose`가 비어 있지 않음
  - `free_text`는 선택값
- 충족 시 → `recommend`로 재분류하여 추천 흐름으로 진행.
- 미충족 시:
  - 부족한 슬롯에 대한 질문을 `answer`로 반환.
  - `recommendations=[]`
  - `flags.needs_more_slots=true`

### recommend
- 슬롯 충족 여부 검사. 미충족이면 `slot_fill`로 폴백.
- 충족이면 `hybrid_retrieve` 수행:
  - ChromaDB dense retrieval
  - BM25
  - RRF merge
- 검색 결과 **0건**이면:
  - `intent="slot_fill"`
  - `answer="조건에 맞는 메뉴를 못 찾았어요. 조건을 조금 풀어볼까요?"` 계열
  - `recommendations=[]`
  - `flags.needs_more_slots=true`
  - `flags.is_fallback=true`
- 정상 검색이면:
  - LLM rerank 수행
  - **top2** recommendation 반환
  - `intent="recommend"`
  - `flags.needs_more_slots=false`

### refine
- `refine + last_recs=[]` 케이스는 §2 step 3에서 `recommend`로 처리 완료되어야 한다.
- 본 분기는 `last_recommendations`가 **있다고 가정**한다.
- query 재구성:
  - 기존 `slots`
  - 이번 턴 `free_text_delta`
  - `request.message`
  - `last_recommendations`
- 직전 추천 `recipe_id`는 **exclude 대상**에 넣는다.
- `hybrid_retrieve_with_exclusion` 수행.
- 검색 결과 **0건**이면:
  - `intent="ask"`
  - `answer="조건을 더 풀어주시면 다시 찾아볼게요."` 계열
  - `recommendations=[]`
  - `flags.is_fallback=true`
  - `flags.needs_more_slots=false`
  - `flags.out_of_scope=false`
- 정상 검색이면:
  - LLM rerank 수행
  - **top2** recommendation 반환
  - `intent="refine"`

### ask
- `last_recommendations`가 비어 있으면 QA를 **수행하지 않는다**:
  - `intent="ask"`
  - `answer="먼저 추천받은 메뉴가 있어야 해당 메뉴에 대해 답변드릴 수 있어요."` 계열
  - `recommendations=[]`
- `last_recommendations`가 있으면 `qa.py` 호출.
- QA는 직전 추천 결과 또는 선택된 `recipe_id` 기반으로 답변.
- `recommendations=[]`
- `slots_updated`는 변경 없는 전체 스냅샷 반환.

---

## 4. fallback 정책

| 케이스 | 처리 | intent | flags |
|---|---|---|---|
| LLM 의도 분류 실패 | "다시 말씀해 주시겠어요?" | ask | `is_fallback=true` |
| LLM 슬롯 추출 실패 | slot_fill로 강제 전환 후 부족 슬롯 재질문 | slot_fill | `needs_more_slots=true` |
| retrieval 0건 (recommend) | 조건 완화 요청 + 추가 질문 | slot_fill | `needs_more_slots=true`, `is_fallback=true` |
| refine인데 last_recommendations=[] | recommend로 재분류 후 진행 | recommend | `is_fallback=false` |
| ask인데 last_recommendations=[] | 추천 이후 질문 가능 안내, QA 스킵 | ask | `is_fallback=false` |
| refine 검색 결과 0건 | 조건 완화 안내 | ask | `is_fallback=true`, `needs_more_slots=false`, `out_of_scope=false` |

주의:
- **누적 slot_fill 실패 카운트는 FastAPI 요청/응답 스키마에 포함하지 않는다.**
- 누적 실패 관찰 및 정책 처리는 **Spring이 담당**한다.
- FastAPI는 **단일 턴 기준으로만** 판단한다.

---

## 5. 로깅 정책

- 진입 시 `t_start` 기록.
- 단계별 latency 측정.
- 응답 직전 `total_ms` 포함 전체 로깅.

### 로그 키

| 키 | 설명 |
|---|---|
| `session_id` | 세션 식별자 |
| `turn_id` | 턴 식별자 |
| `message` | 사용자 발화 |
| `initial_intent` | intent.py 1차 분류 결과 |
| `final_intent` | 재분류/폴백 반영 후 최종 intent |
| `slots_before` | 진입 시점 슬롯 스냅샷 |
| `slots_after` | 병합 후 슬롯 스냅샷 |
| `last_recommendation_count` | 요청의 last_recommendations 길이 |
| `recommendation_count` | 응답 recommendations 길이 |
| `flags` | needs_more_slots / out_of_scope / is_fallback |
| `intent_ms` | 의도 분류 소요 시간 |
| `slot_ms` | 슬롯 추출/병합 소요 시간 |
| `retrieval_ms` | hybrid retrieval 소요 시간 |
| `rerank_ms` | LLM rerank 소요 시간 |
| `qa_ms` | QA 생성 소요 시간 |
| `total_ms` | 전체 처리 시간 |

---

## 6. flags 표준 패턴

| 상황 | needs_more_slots | out_of_scope | is_fallback |
|---|---:|---:|---:|
| 슬롯 추가 질문 | true | false | false |
| 정상 추천 | false | false | false |
| 정상 재추천 | false | false | false |
| 정상 QA | false | false | false |
| ask인데 last_recommendations=[] | false | false | false |
| 서비스 범위 밖 | false | true | false |
| 의도 분류 실패 | false | false | true |
| retrieval 0건 (recommend) | true | false | true |
| refine 검색 0건 | false | false | true |

---

## 7. 불변식

응답 직전 assertion 대상:

- `recommendations=[]` ↔ `intent ∈ {slot_fill, ask}`
- `recommendations.length=2` ↔ `intent ∈ {recommend, refine}`
- `needs_more_slots=true` → `recommendations=[]`
  - 역방향은 성립하지 않음: `recommendations=[]`이어도 `needs_more_slots=false`인 케이스 존재 (예: `ask`, `refine` 0건)
- `flags` 3개 필드는 **모두 required**:
  - `needs_more_slots`
  - `out_of_scope`
  - `is_fallback`
- `slots_updated`는 **항상 전체 스냅샷**이다.
- `slots_updated`의 `null`은 "해당 슬롯에 아직 값 없음"만 의미한다.
- 응답 `answer`는 **항상 빈 문자열이 아니어야** 한다.
- `turn_id`는 요청과 동일해야 한다.

---

## 부록. 간략 의사코드

> 실제 구현 코드가 아닌 흐름 이해용 pseudo-code. 함수는 이름만 사용.

```text
def handle_chat(request):
    # 0. 입력 정규화 + 타이머
    t_start = now()
    slots = request.slots or empty_slots()
    last_recs = request.last_recommendations or []
    history = request.history[-6:]

    # 1. 의도 분류
    try:
        intent = classify_intent(request.message, history, slots, has_last_recs=bool(last_recs))
    except Exception:
        return fallback_ask("다시 말씀해 주시겠어요?", slots)

    # 2. out_of_scope 즉시 반환
    if intent == "out_of_scope":
        return respond(intent="ask", answer=OOS_MSG, slots=slots,
                       recs=[], flags={"out_of_scope": True})

    # 3. refine + last_recs=[] → recommend
    if intent == "refine" and not last_recs:
        intent = "recommend"

    # 4. 슬롯 추출 + 병합
    try:
        delta = extract_slots(request.message, history)
        slots = merge_slots(slots, delta)
    except Exception:
        intent = "slot_fill"

    # 5. 분기 처리
    if intent == "slot_fill":
        if slots_filled(slots):
            intent = "recommend"
        else:
            return ask_for_missing_slots(slots)

    if intent == "recommend":
        if not slots_filled(slots):
            return ask_for_missing_slots(slots)
        hits = hybrid_retrieve(slots)
        if not hits:
            return retrieval_zero_fallback(slots)
        recs = rerank_top2(hits, slots)
        return respond(intent="recommend", recs=recs, slots=slots)

    if intent == "refine":
        hits = hybrid_retrieve_with_exclusion(slots, request.message,
                                              exclude=[r.recipe_id for r in last_recs])
        if not hits:
            return refine_zero_fallback(slots)
        recs = rerank_top2(hits, slots)
        return respond(intent="refine", recs=recs, slots=slots)

    if intent == "ask":
        if not last_recs:
            return respond(intent="ask", answer=ASK_NEEDS_REC_MSG, slots=slots, recs=[])
        answer = qa_answer(request.message, last_recs, history)
        return respond(intent="ask", answer=answer, slots=slots, recs=[])

    # 응답 직전: 불변식 검증 + total_ms 포함 전체 로깅
    # (각 respond() 내부에서 수행)
```
