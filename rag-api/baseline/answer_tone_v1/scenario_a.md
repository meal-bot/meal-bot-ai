# Scenario A — 야식 추천 → ask → refine 2회 (5턴, v1)

- 서버: `http://127.0.0.1:8000/chat` (로컬)
- 서버 버전: `feat/answer-tone-v1` (HEAD `ddf725e` + uncommitted answer_generator/prompt 변경)
- 세션: `v1-a`
- 캡처 시각: 2026-05-28
- 변경점: recommend/refine의 answer 본문이 LLM(`gpt-5-mini`, ANSWER_TIMEOUT=8.0s) 생성으로 전환됨

---

### T1
- 입력: 점심 메뉴 추천해줘 → (오타 정정) 야식으로 든든하게 먹을거 추천해줘
- intent: slot_fill
- elapsed: 4.1s
- answer: 마지막으로, 더 알려주실 정보가 있으세요? (없으면 '없음'이나 '패스'라고 답해주세요)
- recommendations: (없음)
- slots: meal_times=["야식"], purpose="hearty"
- free_text: null
- exclude_ids: (없음)

### T2
- 입력: 없음
- intent: recommend
- elapsed: 19.3s
- answer: 야식으로 든든하게 드시기 좋은 메뉴로 컵밥과 향향 볶음밥을 골라봤어요. 간단히 배불리 먹고 싶을 때 부담 없이 고르실 수 있습니다.
- recommendations:
  - 컵밥 — 조리 시간이 약 25분으로 야식에 맞추기 쉽고 밥·달걀·베이컨을 사용해 든든한 한 끼가 됩니다.
  - 향향 볶음밥 — 닭고기와 감자를 넣어 고소하고 포만감이 높아 야식으로 든든하게 즐기기 좋습니다.
- slots: meal_times=["야식"], purpose="hearty"
- free_text: null
- exclude_ids: (없음)

### T3
- 입력: 1번 칼로리 알려줘
- intent: ask
- elapsed: 5.5s
- answer: 문서 기준으로 1번 메뉴의 열량은 약 218.2kcal입니다.
- recommendations: (없음)
- slots: meal_times=["야식"], purpose="hearty"
- free_text: null
- exclude_ids: (없음)

### T4
- 입력: 다른거 추천해줘
- intent: refine
- elapsed: 27.5s
- answer: 직전 컵밥·향향 볶음밥과 다른 방향으로, 든든한 야식으로 오색볶음면과 숙주라면을 가져왔어요.
- recommendations:
  - 오색볶음면 — 라면과 돼지고기·다양한 채소를 볶아 매콤하고 쫄깃한 식감으로 든든한 야식 한끼를 제공합니다; 볶음밥과 달리 면의 쫄깃함이 특징입니다.
  - 숙주라면 — 숙주와 홍합을 넣은 얼큰한 국물로 든든하게 먹기 좋은 야식이며, 국물요리를 원할 때 매콤한 한끼 대안이 됩니다.
- slots: meal_times=["야식"], purpose="hearty"
- free_text: null (응답 slots_updated 기준 / free_text_delta=null도 null — slot_extractor가 "다른거 추천해줘"를 이번 호출에선 메타로 처리)
- exclude_ids: 127, 931 (직전 추천)

### T5
- 입력: 더 매콤한 걸로
- intent: refine
- elapsed: 17.5s
- answer: 이번엔 더 매콤한 쪽으로 야식용 든든한 메뉴, 닭갈비볶음면과 낙지볶음면을 가져왔어요.
- recommendations:
  - 닭갈비볶음면 — 닭가슴살과 라면을 매콤하게 볶아 든든한 단백질을 제공하고 조리 시간이 적당해 야식에 어울립니다.
  - 낙지볶음면 — 낙지와 쭈꾸미의 쫄깃한 식감에 강한 매운맛이 더해져 빠른 시간에 든든한 매콤한 야식이 됩니다.
- slots: meal_times=["야식"], purpose="hearty"
- free_text_delta: 더 매콤한
- exclude_ids: 306, 549 (직전 추천)
