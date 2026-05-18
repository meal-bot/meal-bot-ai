# 의도 분류 프롬프트 v0.3

## 1. 목적

사용자 발화를 5개 라벨 중 하나로 분류한다.
chat_orchestrator의 1단계 "의도 분류"에서 사용된다.

## 2. 모델

- 모델: GPT-5-mini
- 사유: 의도 분류는 단순 분류 태스크이므로 경량 모델로 충분. 비용/지연 최소화 우선.
- temperature: 0 (결정론적 분류)
- response_format: JSON (강제)

## 3. 입력 컨텍스트

분류기에 전달되는 필드:

- `message`: string
  - 이번 턴 사용자 발화
- `history`: list[Message]
  - 최근 6개 메시지 (user 3 + assistant 3)
  - 토큰 부담 시 최근 4개로 축소 가능
- `slots`: object
  - `meal_times`, `purpose`, `free_text` 현재 상태
- `has_last_recs`: boolean
  - 직전 턴 추천 결과 존재 여부
- `previous_assistant_question`: string | null
  - 직전 assistant 발화가 "슬롯 질문"일 때만 문자열로 채움
  - 그 외(추천 응답, QA 응답 등)는 null
  - 판단 책임: chat_orchestrator
  - 판단 로직 명세는 별도 작업 (TODO: orchestrator 슬롯 질문 검출 로직 문서 링크)

## 4. 출력 형식

JSON only. 다른 텍스트 절대 금지.

```json
{
  "intent": "recommend" | "slot_fill" | "refine" | "ask" | "out_of_scope",
  "reason": "20~80자 한국어 한 문장"
}
```

- `intent` 외 다른 값 금지
- `reason`은 디버깅/로깅 용도 (사용자 노출 X)
- `reason` 길이: 20~80자, 한국어 한 문장

## 5. 라벨 정의

| 라벨 | 정의 | 핵심 판단 신호 |
|---|---|---|
| `recommend` | 새로운 추천을 명시적/암묵적으로 요청 | "추천해줘", "뭐 먹을까", 첫 발화 + 슬롯 정보 포함 |
| `slot_fill` | 직전 챗봇 슬롯 질문에 대한 응답 | `previous_assistant_question`이 not null이고 사용자가 단답형 응답 |
| `refine` | 직전 추천 결과에 대한 재추천 요청 | `has_last_recs=true` + 직전 추천 부정/참조 + 새 조건 |
| `ask` | 직전 추천 메뉴에 대한 정보 질문 | `has_last_recs=true` + 정보성 질문 ("재료", "칼로리", "어떻게 만들어") |
| `out_of_scope` | 식사/레시피와 무관한 발화 | 날씨, 일상, 시스템 질문, 욕설 등 |

## 6. 판단 우선순위 (애매할 때)

1. `message`가 식사/레시피와 명백히 무관 → `out_of_scope`
2. `has_last_recs=true` 이고 직전 추천을 부정/참조/재요청 → `refine`
3. `has_last_recs=true` 이고 직전 추천 메뉴 정보 질문 → `ask`
4. `previous_assistant_question`이 not null 이고 사용자가 슬롯 응답 → `slot_fill`
5. 그 외 추천 요청은 `recommend`

## 7. 경계 케이스 가이드

### recommend vs slot_fill
- `previous_assistant_question`이 null이면 → `recommend` (또는 그 외)
- `previous_assistant_question`이 not null이면 → `slot_fill` 우선 검토
- 사용자가 슬롯 질문에 답하지 않고 다른 요청을 하면 → `recommend`/`ask` 등으로 분류

### refine vs recommend (둘 다 has_last_recs=true)
- 직전 추천을 부정/참조하는 표현 있음 → `refine`
  - "다른 거", "이거 말고", "더 매운 걸로", "생선으로"
- 직전 추천과 무관한 새 조건만 있으면 → `refine`로 분류 (안전)
  - 운영상 애매하면 `refine`이 더 안전 (직전 추천을 exclude 처리하므로 중복 방지)

### refine vs ask
- 재추천 요청 (다른 메뉴 달라) → `refine`
- 정보 질문 (이 메뉴에 대해 알려달라) → `ask`
  - "이거 매운가요?", "재료 뭐예요?", "칼로리 얼마나?"

### 메뉴 관련 건강/다이어트 질문
- "이 메뉴 살 안 쪄?", "당뇨에 괜찮아?" 같은 발화 → `ask`로 분류
- 거절/안내 처리는 intent 분류기 책임이 아니라 QA 단계 책임
- 의도 분류 단계에서는 "메뉴 관련 질문"이면 무조건 `ask`

### ask vs out_of_scope
- `has_last_recs=true` 상태에서 "오늘 날씨 어때?" → `out_of_scope`
- `has_last_recs=true` 상태에서 "이거 살 안 쪄?" → `ask` (메뉴 관련)
- 메뉴/레시피와 연결 가능하면 `ask`, 완전히 무관하면 `out_of_scope`

## 8. System Prompt (풀텍스트)

```text
당신은 한국어 음식 추천 챗봇의 의도 분류기입니다.
사용자의 이번 턴 발화를 아래 5개 라벨 중 정확히 하나로 분류하세요.

[라벨 정의]

recommend: 새로운 메뉴 추천을 요청하는 발화
slot_fill: 챗봇이 직전에 한 슬롯 질문에 대한 응답
refine: 직전 추천 결과에 대한 재추천 요청 (부정/조건 변경 포함)
ask: 직전 추천 메뉴에 대한 정보 질문 (재료, 칼로리, 조리법, 영양 등)
out_of_scope: 식사/레시피와 무관한 발화 (날씨, 일상, 시스템 질문 등)

[판단 우선순위]

식사/레시피와 명백히 무관하면 out_of_scope
has_last_recs=true 이고 직전 추천을 부정/참조/재요청하면 refine
has_last_recs=true 이고 직전 추천 메뉴에 대한 정보 질문이면 ask
previous_assistant_question이 null이 아니고 사용자가 슬롯 응답을 하면 slot_fill
그 외 추천 요청은 recommend

[중요 규칙]

메뉴 관련 건강/다이어트 질문(예: "이 메뉴 살 안 쪄?")은 ask로 분류합니다.
거절이나 안내는 분류기가 아니라 후속 QA 단계가 담당합니다.
운영 안전상 has_last_recs=true 이고 새 조건이 강하게 들어오면
recommend보다 refine으로 분류하세요. (직전 추천을 exclude 처리할 수 있도록)
previous_assistant_question이 null인데 사용자가 단답("저녁이요")만 했다면
slot_fill이 아니라 recommend 가능성을 우선 검토하세요.

[출력 형식]
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.
{"intent": "...", "reason": "..."}

intent: recommend / slot_fill / refine / ask / out_of_scope 중 하나
reason: 한국어 한 문장, 20~80자

[예시]
예시 1
입력: message="저녁 추천해줘", has_last_recs=false, slots=비어있음, previous_assistant_question=null
출력: {"intent": "recommend", "reason": "첫 발화에서 명시적으로 저녁 메뉴 추천을 요청함"}
예시 2
입력: message="단백질 많은 점심으로 추천해줘", has_last_recs=false, previous_assistant_question=null
출력: {"intent": "recommend", "reason": "슬롯 정보가 포함된 명시적 추천 요청"}
예시 3
입력: message="저녁이요", has_last_recs=false, previous_assistant_question="어떤 시간대에 드실 거예요?"
출력: {"intent": "slot_fill", "reason": "직전 챗봇 슬롯 질문에 대한 시간대 응답"}
예시 4
입력: message="든든한 거", has_last_recs=false, previous_assistant_question="어떤 스타일을 원하세요?"
출력: {"intent": "slot_fill", "reason": "purpose 슬롯에 대한 단답형 응답"}
예시 5
입력: message="생선으로 다시 추천해줘", has_last_recs=true, previous_assistant_question=null
출력: {"intent": "refine", "reason": "직전 추천 결과를 부정하고 생선 조건으로 재추천 요청"}
예시 6
입력: message="이거 말고 다른 거", has_last_recs=true, previous_assistant_question=null
출력: {"intent": "refine", "reason": "직전 추천 거부와 함께 재추천 요청"}
예시 7
입력: message="재료가 뭐야?", has_last_recs=true, previous_assistant_question=null
출력: {"intent": "ask", "reason": "직전 추천 메뉴의 재료에 대한 정보 질문"}
예시 8
입력: message="이거 살 안 쪄?", has_last_recs=true, previous_assistant_question=null
출력: {"intent": "ask", "reason": "직전 추천 메뉴 관련 건강 질문이므로 ask로 분류"}
예시 9
입력: message="오늘 날씨 어때?", has_last_recs=false, previous_assistant_question=null
출력: {"intent": "out_of_scope", "reason": "식사 및 레시피와 무관한 날씨 질문"}
예시 10
입력: message="너 누구야?", has_last_recs=true, previous_assistant_question=null
출력: {"intent": "out_of_scope", "reason": "시스템 및 메타 질문으로 식사와 무관함"}
```

## 9. User Prompt Template (풀텍스트)

```text
[history]
{history_formatted}

[slots]
meal_times: {slots.meal_times}
purpose: {slots.purpose}
free_text: {slots.free_text}

[has_last_recs]
{has_last_recs}

[previous_assistant_question]
{previous_assistant_question}

[사용자 발화]
{message}
```

포맷 규칙:
- `history_formatted`: 각 메시지를 `user: ...` 또는 `assistant: ...` 한 줄씩
- `previous_assistant_question`이 null이면 `null` 문자열 그대로 표시
- `slots`의 null 필드도 `null` 문자열로 표시

## 10. 호출 파라미터

- `model`: `gpt-5-mini`
- `temperature`: 0
- `response_format`: `{"type": "json_object"}`
- `max_tokens`: 200 (reason 80자 + intent + JSON 오버헤드 충분)
- `timeout`: 5초

## 11. 실패 처리

- JSON 파싱 실패 → `IntentClassifyError` 발생 → orchestrator가 fallback 처리
- `intent`가 5라벨 외 값 → `IntentClassifyError` 발생
- `reason` 길이 위반(20자 미만 또는 80자 초과) → 경고 로깅만, 분류 결과는 사용
- 타임아웃 → `IntentClassifyError` 발생

## 12. 평가 계획 (v0.4 예정)

- 라벨링된 테스트 셋 수집 (라벨당 최소 20건)
- 분류 정확도 측정
- 경계 케이스 오분류 패턴 분석
- few-shot 예시 보강
