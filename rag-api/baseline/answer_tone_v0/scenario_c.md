# Scenario C — 첫 발화에 자유 요청이 풍부한 케이스 (1턴)

- 서버: `http://210.104.76.139:8000/chat`
- 서버 버전: develop `ddf725e`
- 세션: `baseline-c`
- 캡처 시각: 2026-05-27

---

### T1
- 입력: 저녁에 가볍게 먹을건데 매운 건 빼고
- intent: recommend
- elapsed: 10.4s
- answer: 조건에 맞춰 2개 골라봤어요.
- recommendations:
  - 코다리맑은찜 — 조리 시간이 짧고 담백한 찜으로 저녁 가벼운 한끼에 적합하며 맵지 않아 요청에 부합합니다.
  - 시금치 리조또 — 부드럽고 가벼운 리조또로 저녁 가볍게 먹기 좋고 맵지 않아 매운맛 제외 조건을 만족합니다.
- slots: meal_times=["저녁"], purpose="light"
- free_text: null (응답 slots_updated 기준; free_text_delta로는 추출됨)
- free_text_delta: 매운 건 빼고
- exclude_ids: (없음)

---

## 안전망 스킵 확인
- `free_text_delta="매운 건 빼고"` (8자) → `_needs_freetext_safetynet`은 3자 미만 검사이므로 우회됨.
- 의도대로 첫 턴에서 곧장 recommend로 진입.
