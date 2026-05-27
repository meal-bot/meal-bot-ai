# Scenario C — 첫 발화에 자유 요청이 풍부한 케이스 (1턴, v1)

- 서버: `http://127.0.0.1:8000/chat` (로컬)
- 서버 버전: `feat/answer-tone-v1` (HEAD `ddf725e` + uncommitted answer_generator/prompt 변경)
- 세션: `v1-c`
- 캡처 시각: 2026-05-28

---

### T1
- 입력: 저녁에 가볍게 먹을건데 매운 건 빼고
- intent: recommend
- elapsed: 11.8s
- answer: 저녁에 맵지 않고 가볍게 드시기 좋은 메뉴로 시금치 리조또와 코다리맑은찜을 골라봤어요.
- recommendations:
  - 시금치 리조또 — 시금치와 쌀을 사용한 부드럽고 담백한 리조또로 조리 시간이 짧아 저녁 가벼운 식사에 적합하고 매운맛이 없어요.
  - 코다리맑은찜 — 코다리와 콩나물, 무로 만든 촉촉하고 담백한 찜으로 저녁 한끼로 가볍고 매운 양념이 사용되지 않았습니다.
- slots: meal_times=["저녁"], purpose="light"
- free_text: null (응답 slots_updated 기준; free_text_delta로는 추출됨)
- free_text_delta: 매운 건 빼고
- exclude_ids: (없음)

---

## 안전망 스킵 확인
- `free_text_delta="매운 건 빼고"` (8자) → `_needs_freetext_safetynet`은 3자 미만 검사이므로 우회됨.
- 의도대로 첫 턴에서 곧장 recommend로 진입.
- answer 본문에 "맵지 않고"로 자유 요청이 자연스럽게 미러링됨 (v0의 정적 "조건에 맞춰 2개 골라봤어요." 대비 개선).
