# Scenario A — 야식 추천 → ask → refine 2회 (5턴)

- 서버: `http://210.104.76.139:8000/chat`
- 서버 버전: develop `ddf725e`
- 세션: `baseline-a`
- 캡처 시각: 2026-05-27

---

### T1
- 입력: 야식으로 든든하게 먹을거 추천해줘
- intent: slot_fill
- elapsed: 4.7s
- answer: 마지막으로, 더 알려주실 정보가 있으세요? (없으면 '없음'이나 '패스'라고 답해주세요)
- recommendations: (없음)
- slots: meal_times=["야식"], purpose="hearty"
- free_text: null
- exclude_ids: (없음)

### T2
- 입력: 없음
- intent: recommend
- elapsed: 15.4s
- answer: 조건에 맞춰 2개 골라봤어요.
- recommendations:
  - 향향 볶음밥 — 밥과 닭고기를 볶아 25분 내에 만들 수 있어 야식으로 빠르게 준비할 수 있고 든든한 한 끼가 됩니다.
  - 니고랭 — 닭가슴살과 해산물을 넣어 20분 내 조리가 가능해 야식으로 가볍지 않게 든든히 먹기 좋습니다.
- slots: meal_times=["야식"], purpose="hearty"
- free_text: null
- exclude_ids: (없음)

### T3
- 입력: 1번 칼로리 알려줘
- intent: ask
- elapsed: 5.3s
- answer: 문서 기준으로 1번의 열량은 약 104.0kcal입니다.
- recommendations: (없음)
- slots: meal_times=["야식"], purpose="hearty"
- free_text: null
- exclude_ids: (없음)

### T4
- 입력: 다른거 추천해줘
- intent: refine
- elapsed: 16.8s
- answer: 다른거 추천해줘 반영해서 다시 골라봤어요.
- recommendations:
  - 삼겹살라면 — 삼겹살과 라면을 끓여 진한 국물로 든든함을 주는 야식이라 국물 메뉴를 원할 때 적합합니다.
  - 밥크로켓 — 밥과 메추리알을 빵가루로 입혀 튀겨 바삭하고 간단히 든든한 스낵형 야식으로 어울립니다.
- slots: meal_times=["야식"], purpose="hearty"
- free_text: null (응답의 slots_updated 기준 / free_text_delta="다른거 추천해줘")
- exclude_ids: 931, 1010 (직전 추천)

### T5
- 입력: 더 매콤한 걸로
- intent: refine
- elapsed: 14.0s
- answer: 더 매콤한 반영해서 다시 골라봤어요.
- recommendations:
  - 닭갈비볶음면 — 닭가슴살과 라면을 매콤하게 볶아 야식으로 든든하고 매운맛을 만족시키는 다른 선택입니다.
  - 골뱅이무침과 삼겹살수육 — 골뱅이의 매콤한 무침과 삶은 삼겹살의 조합으로 다른 식감과 단백질 풍부한 든든한 야식입니다.
- slots: meal_times=["야식"], purpose="hearty", free_text="다른거 추천해줘"
- free_text_delta: 더 매콤한
- exclude_ids: 309, 479 (직전 추천)
