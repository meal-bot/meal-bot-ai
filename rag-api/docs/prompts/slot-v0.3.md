# 슬롯 추출 프롬프트 v0.3

## 1. 목적

사용자 발화에서 슬롯 정보를 추출한다.
chat_orchestrator의 4단계 "슬롯 추출"에서 사용된다.
intent 분류기와는 **별도의 LLM 호출**이다 (분리 유지).

## 2. 모델

- 모델: GPT-5-mini
- temperature: 0
- response_format: JSON 강제
- 사유: intent 분류기와 일관성, 단가 동일, 슬롯 추출도 경량 모델로 충분

## 3. 입력 컨텍스트

- `message`: string (이번 턴 사용자 발화)
- `history`: list[Message] (최근 6개 메시지, 맥락 참고용)

intent 분류기와 달리 `slots`, `has_last_recs`, `previous_assistant_question`은 필요 없음.
slot 추출은 "이번 발화에 무슨 정보가 있나"만 판단하면 되므로 입력 단순화.

## 4. 출력 형식

JSON only. 다른 텍스트 절대 금지.

```json
{
  "meal_times": ["아침"|"점심"|"저녁"|"간식"|"야식"] | null,
  "purpose": "light"|"protein"|"hearty"|"tasty" | null,
  "free_text_delta": string | null
}
```

- `null` = "이번 턴에 추출된 정보 없음"
- `meal_times`는 배열, 복수 선택 가능 (예: `["아침", "점심"]`)
- `purpose`는 단일 값
- `free_text_delta`는 슬롯에 매핑되지 않은 추가 조건만 보존

## 5. 추출 원칙

### 원칙 1: delta만 반환
- 이번 발화에서 새로 추출된 정보만 반환.
- 기존 slots와의 병합은 orchestrator의 `merge_slots()`가 결정론적으로 처리.
- LLM은 누적 상태를 알 필요 없음.

### 원칙 2: 슬롯 매핑은 enum에 확실히 맞을 때만
- 고정 슬롯(`meal_times`, `purpose`)의 enum에 명확히 매핑되지 않으면 해당 슬롯은 `null`.
- 애매한 표현은 원 표현 그대로 `free_text_delta`에 보존.
- 예: "아점" → `meal_times=null`, `free_text_delta="아점"`

### 원칙 3: free_text_delta는 슬롯에 매핑되지 않은 추가 조건만
- `meal_times`, `purpose`에 이미 반영된 정보는 `free_text_delta`에서 제외.
- 재료, 조리 방식, 음식 유형, 질감, 맛, 매운 정도, 기타 선호는 `free_text_delta`에 보존.
- "매운", "안 매운" 등 매운맛 관련은 `spicy_max` 슬롯이 삭제되었으므로 **반드시** `free_text_delta`에 보존.

### 원칙 4: 빈 결과는 정상
- 추출할 슬롯이 없으면 모든 필드 `null`.
- 예: "안녕" → `{meal_times: null, purpose: null, free_text_delta: null}`
- 빈 결과는 **실패가 아니라 정상**.

## 6. 매핑 가이드

### meal_times 매핑

| 표현 | 매핑 |
|---|---|
| "아침", "조식" | `["아침"]` |
| "점심", "런치" | `["점심"]` |
| "저녁", "디너", "저녁밥" | `["저녁"]` |
| "간식", "스낵" | `["간식"]` |
| "야식", "밤참", "새벽" | `["야식"]` |
| "아침이랑 점심", "아침 또는 점심" | `["아침", "점심"]` |
| "아점" | `null` + `free_text_delta="아점"` (enum에 없음) |
| "오후" | `null` + `free_text_delta="오후"` (애매) |
| "브런치" | `null` + `free_text_delta="브런치"` (애매) |

### purpose 매핑

| 표현 | 매핑 |
|---|---|
| "가볍게", "간단한", "다이어트", "샐러드 같은" | `light` |
| "단백질", "고기", "헬스", "운동 후", "근육" | `protein` |
| "든든한", "푸짐한", "배 채우는", "포만감" | `hearty` |
| "맛있는", "기분 좋은", "특별한", "맛집" | `tasty` |

매핑 어려우면 `purpose=null` + `free_text_delta`에 보존.

### free_text_delta에 보존할 정보
- **재료**: "닭고기", "생선", "두부", "콩"
- **조리 방식**: "구이", "튀김", "찜", "조림"
- **음식 유형**: "한식", "양식", "면 요리", "국물"
- **매운맛**: "매운", "안 매운", "덜 매운"
- **질감/온도**: "따뜻한", "차가운", "부드러운"
- **상황/선호**: "혼자 먹을 거", "빨리 되는 거", "재료 적은 거"
- **기타 enum 외 시간 표현**: "아점", "오후", "브런치"

## 7. System Prompt (풀텍스트)

```text
당신은 한국어 음식 추천 챗봇의 슬롯 추출기입니다.
사용자의 이번 턴 발화에서 슬롯 정보를 추출해 JSON으로 반환하세요.

[추출 대상 슬롯]

meal_times: ["아침", "점심", "저녁", "간식", "야식"] 중 해당하는 값들의 배열 (복수 가능)
purpose: "light", "protein", "hearty", "tasty" 중 하나의 단일 값
free_text_delta: 위 두 슬롯에 매핑되지 않은 추가 조건 (재료, 조리법, 매운맛, 음식 유형 등)

[핵심 원칙]

이번 발화에서 새로 추출된 정보만 반환합니다. 누적은 시스템이 처리합니다.
enum에 확실히 매핑되는 표현만 슬롯에 넣습니다. 애매하면 null로 두고 free_text_delta에 원 표현을 보존합니다.
meal_times나 purpose에 이미 반영된 정보는 free_text_delta에서 제외합니다.
추출할 정보가 없으면 모든 필드를 null로 반환합니다 (정상 상황).

[meal_times 매핑]

"아침"/"조식" → ["아침"]
"점심"/"런치" → ["점심"]
"저녁"/"디너" → ["저녁"]
"간식"/"스낵" → ["간식"]
"야식"/"밤참"/"새벽" → ["야식"]
"아침이랑 점심" 같은 명시적 복수 → ["아침", "점심"]
"아점"/"오후"/"브런치" 같은 enum 외 표현 → meal_times=null, free_text_delta에 보존

[purpose 매핑]

"가볍게"/"간단한"/"다이어트" → light
"단백질"/"고기"/"헬스"/"운동 후" → protein
"든든한"/"푸짐한"/"포만감" → hearty
"맛있는"/"특별한"/"기분 좋은" → tasty
매핑 어려우면 purpose=null, free_text_delta에 원 표현 보존

[free_text_delta 보존 대상]

재료 (닭고기, 생선, 두부 등)
조리 방식 (구이, 튀김, 찜 등)
음식 유형 (한식, 양식, 면, 국물 등)
매운맛 표현 (매운, 안 매운, 덜 매운) — spicy 슬롯이 없으므로 반드시 free_text_delta에 보존
질감/온도/상황/기타 선호

[출력 형식]
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 절대 금지.
{
"meal_times": [...] | null,
"purpose": "..." | null,
"free_text_delta": "..." | null
}

[예시]
예시 1 — 모든 슬롯 + free_text
입력: "저녁에 든든한 닭고기 요리 추천"
출력: {"meal_times": ["저녁"], "purpose": "hearty", "free_text_delta": "닭고기 요리"}

예시 2 — meal_times만
입력: "저녁이요"
출력: {"meal_times": ["저녁"], "purpose": null, "free_text_delta": null}

예시 3 — purpose만
입력: "든든한 거"
출력: {"meal_times": null, "purpose": "hearty", "free_text_delta": null}

예시 4 — free_text_delta만 (refine 발화)
입력: "생선으로 다시 추천해줘"
출력: {"meal_times": null, "purpose": null, "free_text_delta": "생선"}

예시 5 — 매운맛 (spicy 슬롯 없음, free_text 보존)
입력: "안 매운 저녁 추천"
출력: {"meal_times": ["저녁"], "purpose": null, "free_text_delta": "안 매운"}

예시 6 — 모호한 시간 표현
입력: "아점 추천해줘"
출력: {"meal_times": null, "purpose": null, "free_text_delta": "아점"}

예시 7 — 복수 시간대
입력: "아침이랑 점심 같이 먹을 거"
출력: {"meal_times": ["아침", "점심"], "purpose": null, "free_text_delta": null}

예시 8 — 빈 결과 (정상)
입력: "안녕"
출력: {"meal_times": null, "purpose": null, "free_text_delta": null}
```

## 8. User Prompt Template

```text
[history]
{history_formatted}

[사용자 발화]
{message}
```

포맷 규칙:
- `history_formatted`: 각 메시지 `user: ...` / `assistant: ...` 한 줄씩
- `history`는 맥락 참고용 (직접 추출 대상 아님)

## 9. 호출 파라미터

- `model`: `gpt-5-mini`
- `temperature`: 0
- `response_format`: `{"type": "json_object"}`
- `max_tokens`: 200
- `timeout`: 5초

## 10. 실패 처리

**실패로 간주 (`SlotExtractError` 발생):**
- JSON 파싱 실패
- 필수 키 누락 (`meal_times`, `purpose`, `free_text_delta` 중 하나라도)
- enum 외 값 (`meal_times` 배열 요소 또는 `purpose` 값)
- 타입 오류 (`meal_times`가 배열 아님, `purpose`가 string/null 아님 등)
- LLM 응답이 비어 있음
- 타임아웃

**정상으로 처리:**
- 모든 필드 `null`
- 일부 필드만 값 있음
- `meal_times`가 빈 배열 `[]`은 `null`로 정규화 (orchestrator 책임)

실패 시 orchestrator는 `intent="slot_fill"`로 강제 전환하고 부족 슬롯 재질문.

## 11. merge_slots() 동작 (참고)

`slot.py`의 LLM 호출 결과를 받은 후 orchestrator가 수행하는 merge 로직:

- `meal_times`: delta가 `null`이면 유지, 배열이면 **덮어쓰기 (replace)**
- `purpose`: delta가 `null`이면 유지, 값이면 **덮어쓰기**
- `free_text`: FastAPI는 `free_text_delta`만 응답에 포함. Spring이 누적 관리.

정책: **"추가(append)"가 아니라 "덮어쓰기(replace)"**.
사용자가 "점심"이라 했다가 "저녁"이라 하면 저녁으로 바뀌어야 하기 때문.

## 12. 빈 결과 시 orchestrator 처리 (참고)

slot 추출 결과가 모두 `null`이고 `intent="slot_fill"`인 경우:
- 사용자가 슬롯 질문에 엉뚱한 답을 한 케이스
- orchestrator는 **강제 선택형 재질문** 발화 생성
- 예: "아침/점심/저녁/간식/야식 중에 선택해 주세요"
- 이 처리는 chat_orchestrator 문서에 별도 명세 (TODO 링크)

## 13. 평가 계획 (v0.4 예정)

- 라벨링된 테스트 셋 (발화 → 정답 슬롯) 수집
- 슬롯별 정확도(F1) 측정
- 매핑 가이드 보강
- few-shot 예시 추가
