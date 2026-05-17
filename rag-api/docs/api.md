# Meal-bot RAG API 명세서

> **Version:** 0.1 (draft)
> **Last updated:** 2026-05-17
> **Source of truth:** 이 문서 (`docs/api.md`)
> **자동 생성 스펙:** 서버 실행 중 `GET /openapi.json`, Swagger UI `/docs`, ReDoc `/redoc`

---

## 1. 개요

Meal-bot의 AI/RAG 모듈. 사용자 조건을 받아 메뉴를 추천하고, 추천된 메뉴에 대한 후속 질문에 답함.

- **Base URL (로컬):** `http://localhost:8000`
- **Base URL (배포):** *TBD (Phase 1 — EC2 배포 후 결정)*
- **인증:** 없음 (Phase 1 시연 범위). 추후 Spring 게이트웨이에서 처리 예정.
- **요청/응답 포맷:** JSON, UTF-8
- **세션 상태:** **stateless**. FastAPI는 세션/사용자/대화 저장하지 않음. 모든 컨텍스트는 매 요청에 포함됨 (Spring이 관리).

---

## 2. 공통 규약

### 2.1 요청 헤더

| 헤더 | 값 |
|---|---|
| `Content-Type` | `application/json` |

### 2.2 세션 식별자

| 필드 | 설명 | 제약 |
|---|---|---|
| `session_id` | 대화 세션 단위 식별자 (Spring이 발급) | 1–100자 string |
| `turn_id` | 한 세션 내 요청 단위 식별자 | 1–100자 string |

`/recommend`, `/ask` 모든 요청에 두 필드 필수. 응답에는 `turn_id`만 echo됨.

### 2.3 응답 필드 정책

본 API는 응답 필드 두 종류를 명확히 구분함.

- **분기 플래그 (required boolean):** 클라이언트가 응답 처리 분기에 사용. 항상 응답에 포함됨.
  - `is_fallback`, `insufficient_matches`, `refused`, `qa_failed`, `out_of_scope`
- **데이터 필드 (nullable / optional):** 원본 데이터 결측 또는 상황에 따라 빠질 수 있음.
  - `kcal`, `cooking_time`, `image_url`, `matched_intents`, `used_fields`, `free_text`, `chat_history`

### 2.4 에러 응답

FastAPI 표준 검증 에러 형식 사용.

**스키마:**
```json
{
  "detail": [
    {
      "type": "<error_type>",
      "loc": ["body", "<field_name>"],
      "msg": "<설명>",
      "input": "<입력값>",
      "ctx": { /* (선택) 제약 컨텍스트 */ }
    }
  ]
}
```

**주요 HTTP 상태 코드:**

| 코드 | 의미 | 발생 케이스 |
|---|---|---|
| `200` | 정상 | 비즈니스 실패(refused/qa_failed 등)도 200으로 반환됨. 응답 플래그로 판별 |
| `422` | 검증 실패 | 필수 필드 누락, enum 위반, 범위 초과 등 (Pydantic 검증) |
| `500` | 서버 에러 | LLM 호출 실패 등 예측 못한 예외 (※ 가능하면 fallback 응답으로 200 반환 시도) |

**에러 타입 예시:**

| `type` | 예시 상황 |
|---|---|
| `missing` | 필수 필드 누락 |
| `literal_error` | enum 외 값 |
| `string_too_short` / `string_too_long` | 문자열 길이 제약 위반 |
| `less_than_equal` / `greater_than_equal` | 숫자 범위 초과 |

---

## 3. 엔드포인트

### 3.1 `GET /healthz`

헬스체크. 서버 기동 확인용.

**요청:** 없음
**응답 (200):**
```json
{ "status": "ok" }
```

`status` 필드는 `"ok"` 고정 (Literal).

---

### 3.2 `POST /recommend`

사용자 조건 기반 메뉴 Top-N 추천. LLM이 vector DB 후보를 rerank하여 최종 선정.

#### 요청 스키마

| 필드 | 타입 | 필수 | 제약 | 설명 |
|---|---|---|---|---|
| `meal_times` | `array[string]` | ✅ | enum: `아침` / `점심` / `저녁` / `간식` / `야식` | 식사 시간대 (복수 가능) |
| `purpose` | `string` | ✅ | enum: `light` / `protein` / `hearty` / `tasty` | 식사 목적 |
| `spicy_max` | `integer` | ✅ | 1–4 | 허용 최대 매운맛 (1: 안 매움 ~ 4: 매우 매움) |
| `free_text` | `string` | ❌ | maxLen 200 | 자유 입력 (없으면 `""` 또는 생략) |
| `session_id` | `string` | ✅ | 1–100자 | 세션 식별자 |
| `turn_id` | `string` | ✅ | 1–100자 | 턴 식별자 |

#### 응답 스키마 (200)

**최상위:**

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `turn_id` | `string` | ✅ | 요청과 동일한 값 echo |
| `recommendations` | `array[Recommendation]` | ✅ | 추천 목록 (rank 오름차순) |
| `insufficient_matches` | `boolean` | ✅ | 조건 일치 후보가 부족했는지 |
| `is_fallback` | `boolean` | ✅ | LLM rerank 실패 등으로 fallback 사용했는지 |

**`Recommendation` 객체:**

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `rank` | `integer` | ✅ | 1부터 시작하는 순위 |
| `recipe_id` | `string` | ✅ | 레시피 고유 ID (후속 `/ask` 호출에 사용) |
| `name` | `string` | ✅ | 메뉴명 |
| `summary` | `string` | ✅ | 한 줄 요약 |
| `reason` | `string` | ✅ | LLM이 생성한 추천 이유 |
| `kcal` | `number` ⎮ `null` | nullable | 열량 (kcal) |
| `cooking_time` | `integer` ⎮ `null` | nullable | 조리 시간 (분) |
| `image_url` | `string` ⎮ `null` | nullable | 이미지 URL |
| `matched_intents` | `array[string]` | optional | 매칭된 의도 태그 (보조 정보) |

#### curl 예시

**요청:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "meal_times": ["점심"],
    "purpose": "hearty",
    "spicy_max": 2,
    "free_text": "",
    "session_id": "sess-001",
    "turn_id": "t1"
  }'
```

**응답 (요약):**
```json
{
  "turn_id": "t1",
  "recommendations": [
    {
      "rank": 1,
      "recipe_id": "220",
      "name": "토마토카레 채소볶음밥",
      "image_url": "http://www.foodsafetykorea.go.kr/uploadimg/cook/10_00220_2.png",
      "summary": "토마토와 각종 채소, 닭가슴살을 볶아 카레 소스를 얹은 든든한 볶음밥",
      "kcal": 612.1,
      "cooking_time": 35,
      "reason": "여러 채소와 닭가슴살이 더해져 한 끼로 든든하고 조리 시간도 적당함",
      "matched_intents": ["한끼", "담백한맛"]
    }
  ],
  "insufficient_matches": false,
  "is_fallback": false
}
```

---

### 3.3 `POST /ask`

추천된 메뉴에 대한 후속 질문 응답. RAG 기반 QA. **stateless** — 이전 대화는 `chat_history`로 매번 전달.

#### 요청 스키마

| 필드 | 타입 | 필수 | 제약 | 설명 |
|---|---|---|---|---|
| `question` | `string` | ✅ | 1–500자 | 사용자 질문 |
| `recipe_id` | `string` | ✅ | 1자 이상 | `/recommend` 응답의 `recipe_id` |
| `chat_history` | `array[ChatMessage]` | ❌ | 최대 20개 | 이전 대화 (없으면 `[]` 또는 생략) |
| `session_id` | `string` | ✅ | 1–100자 | 세션 식별자 |
| `turn_id` | `string` | ✅ | 1–100자 | 턴 식별자 |

**`ChatMessage` 객체:**

| 필드 | 타입 | 필수 | 제약 |
|---|---|---|---|
| `role` | `string` | ✅ | enum: `user` / `assistant` |
| `content` | `string` | ✅ | 1–2000자 |

#### 응답 스키마 (200)

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `turn_id` | `string` | ✅ | 요청과 동일한 값 echo |
| `answer` | `string` | ✅ | 답변 본문 |
| `refused` | `boolean` | ✅ | 정책상 답변 거부 (의학/효능/건강 판단 등) |
| `qa_failed` | `boolean` | ✅ | QA 처리 실패 (LLM 오류 등) |
| `is_fallback` | `boolean` | ✅ | fallback 응답인지 |
| `out_of_scope` | `boolean` | ✅ | 레시피 컨텍스트로 답하기 어려운 질문 |
| `used_fields` | `array[string]` | optional | 답변 생성에 사용된 컨텍스트 필드 (디버깅용) |

#### curl 예시

**요청:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "이 요리에 어떤 재료가 들어가나요?",
    "recipe_id": "220",
    "chat_history": [],
    "session_id": "sess-001",
    "turn_id": "t2"
  }'
```

**응답:**
```json
{
  "turn_id": "t2",
  "answer": "재료는 다음과 같습니다.\n- 감자 20g\n- 양파 10g\n...",
  "used_fields": ["ingredients_structured", "ingredients", "main_ingredients"],
  "refused": false,
  "out_of_scope": false,
  "qa_failed": false,
  "is_fallback": false
}
```

#### chat_history 사용 예시

후속 턴에서 이전 대화 포함:

```json
{
  "question": "그럼 칼로리는 얼마야?",
  "recipe_id": "220",
  "chat_history": [
    {"role": "user", "content": "재료가 뭐야?"},
    {"role": "assistant", "content": "재료는 감자, 양파, 당근..."}
  ],
  "session_id": "sess-001",
  "turn_id": "t3"
}
```

---

## 4. 플래그 매트릭스

`/ask` 응답 플래그 조합에 따른 클라이언트 처리 가이드.

| 시나리오 | `refused` | `qa_failed` | `is_fallback` | `out_of_scope` | 클라이언트 처리 |
|---|---|---|---|---|---|
| 정상 답변 | false | false | false | false | `answer` 그대로 표시 |
| 범위 밖 질문 (예: 다른 요리에 대해 질문) | false | false | false | **true** | `answer` 표시 + "이 메뉴 관련 질문만 가능" 안내 |
| 정책 거부 (의학/효능) | **true** | false | false | false | `answer`(거부 메시지) 표시 |
| QA 실패 fallback | false | **true** | **true** | false | fallback 답변 표시 + 재시도 유도 |

`/recommend` 응답 플래그:

| 시나리오 | `insufficient_matches` | `is_fallback` | 클라이언트 처리 |
|---|---|---|---|
| 정상 추천 | false | false | `recommendations` 그대로 표시 |
| 조건 부합 후보 부족 | **true** | false | 결과 표시 + "조건을 완화해 보세요" 안내 |
| rerank 실패 fallback | (varies) | **true** | 결과 표시 + 품질 저하 경고 |

### 4.1 플래그 조합 제약

`/ask` 응답 플래그는 다음 제약을 따름. 클라이언트는 이를 전제로 분기 처리해도 안전함.

- `out_of_scope`과 `refused`는 **동시에 `true`가 될 수 없음** (상호 배타)
- `out_of_scope=true`일 때 `used_fields`는 **항상 `[]`** (빈 리스트)
- `qa_failed=true`일 때 `is_fallback`도 **항상 `true`** (페어로만 발생)

---

## 5. 시나리오별 응답 예시

> ℹ️ **주의:** 아래 예시의 `answer` 본문은 LLM이 생성하므로 호출마다 미묘하게 달라질 수 있음. 클라이언트 분기는 반드시 응답 **플래그**로 판단할 것. `answer`는 사용자에게 그대로 표시하는 용도로만 사용.

### 5.1 추천 — 조건 부합 후보 부족

`spicy_max=1` + `purpose=tasty` 등 빡빡한 조건에서 발생 가능.

```json
{
  "turn_id": "t1",
  "recommendations": [ /* 일부 후보 */ ],
  "insufficient_matches": true,
  "is_fallback": false
}
```

### 5.2 /ask — 범위 밖 질문

질문: "오늘 서울 날씨가 어때?" (현재 recipe_id 컨텍스트와 무관)

```json
{
  "turn_id": "t5",
  "answer": "이 질문은 현재 추천된 레시피와 관련된 정보로 답변하기 어렵습니다. 조리법, 재료, 맛·식감, 조리 시간, 영양 정보 등 레시피와 관련된 내용을 질문해 주세요.",
  "used_fields": [],
  "refused": false,
  "qa_failed": false,
  "is_fallback": false,
  "out_of_scope": true
}
```

`out_of_scope=true`인 경우 `used_fields`는 항상 `[]`.

### 5.3 /ask — 정책 거부 (안전 단정 회피)

질문: "이거 먹으면 다이어트에 효과 있어?" 같은 개인 건강·효능 판단 질문.

LLM이 단정적 답변을 회피하고 대안(예: 영양 수치 안내)을 제시하는 톤으로 응답함.

```json
{
  "turn_id": "t6",
  "answer": "다이어트에 적합한지는 개인의 목표와 건강 상태에 따라 달라질 수 있어 제가 단정해서 답변드리기 어렵습니다. 필요하시면 이 레시피의 열량·영양 수치만 따로 알려드릴 수 있습니다.",
  "refused": true,
  "qa_failed": false,
  "is_fallback": false,
  "out_of_scope": false
}
```

### 5.4 검증 에러 — 필수 필드 누락 (422)

요청: `{"meal_times":["점심"],"purpose":"hearty","spicy_max":2}` (session_id, turn_id 누락)

```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "session_id"], "msg": "Field required"},
    {"type": "missing", "loc": ["body", "turn_id"], "msg": "Field required"}
  ]
}
```

### 5.5 검증 에러 — enum 위반 (422)

요청에 `"purpose": "invalid"` 포함 시:

```json
{
  "detail": [{
    "type": "literal_error",
    "loc": ["body", "purpose"],
    "msg": "Input should be \'light\', \'protein\', \'hearty\' or \'tasty\'",
    "input": "invalid",
    "ctx": {"expected": "\'light\', \'protein\', \'hearty\' or \'tasty\'"}
  }]
}
```

---

## 6. 변경 이력

| 날짜 | 버전 | 변경 |
|---|---|---|
| 2026-05-17 | 0.1 (draft) | 초안 작성. 파트너 검토 대기 |

---

## 7. 향후 작업 (파트너 검토 후)

- [ ] 파트너 검토 코멘트 수렴
- [ ] 합의된 스펙으로 v1.0 확정
- [ ] Notion 미러 페이지 생성
- [ ] EC2 배포 후 Base URL 업데이트
- [ ] (선택) OpenAPI YAML 별도 관리 여부 결정
