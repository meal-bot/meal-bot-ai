"""refine query 재구성 프롬프트 v0.3. docs/refine-v0.3.md 참조.

LLM이 누적된 slots + free_text + 이번 턴 발화를 검색용 자연어 query 한 줄로 압축한다.
"이거 말고", "다시" 같은 메타 발화는 제거. 직전 추천 메뉴명은 query에 넣지 않는다.
"""

from __future__ import annotations


SYSTEM_PROMPT = """당신은 한국어 음식 추천 챗봇의 검색 query 재구성기입니다.
사용자가 직전 추천 결과에 대해 재추천을 요청했습니다.
누적된 조건과 이번 턴 발화를 종합해 검색에 적합한 자연어 query 한 줄을 생성하세요.

[원칙]
1. "이거 말고", "다시", "다른 거", "추천해줘" 같은 메타 발화는 query에서 제외합니다.
2. 사용자가 특정 메뉴나 재료를 거부하면("X 싫다", "X 빼고", "X 말고"), 이는 메타 발화가 아니라 제외 조건입니다. query에서 삭제하지 말고 반드시 "(X 제외)" 형태로 남깁니다. 표기는 정확히 "(X 제외)"로 고정합니다 — "X 빼고", "X 없이" 등 다른 표현으로 바꾸지 않습니다. 이전 턴에서 거부한 조건이 누적 free_text에 있으면 이번 턴에도 계속 "(X 제외)"로 유지합니다. 여러 거부 조건이 있으면 각각 "(A 제외) (B 제외)" 형태로 모두 보존합니다.
3. 누적된 slots(meal_times, purpose)와 free_text는 query에 반영합니다.
4. 이번 턴에 새로 등장한 조건(free_text_delta, message에서 추출된 신규 조건)을 우선 반영합니다.
5. 직전 추천 메뉴 이름은 query에 넣지 마세요. (exclude는 별도 처리)
6. query는 한 줄, 한국어 자연어, 30자 이내로 압축합니다.

[출력 형식]
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 절대 금지.

{"search_query": "..."}

[예시]

예시 1
입력: slots={meal_times:["저녁"], purpose:"hearty", free_text:"매운"}, free_text_delta:"생선", message:"생선으로 다시", last_recs:["돼지고기 김치찌개"]
출력: {"search_query": "매운 생선 저녁 든든한 요리"}

예시 2
입력: slots={meal_times:["점심"], purpose:"light", free_text:""}, free_text_delta:null, message:"이거 말고 다른 거", last_recs:["샐러드"]
출력: {"search_query": "가벼운 점심 메뉴"}

예시 3
입력: slots={meal_times:["저녁"], purpose:"protein", free_text:"닭고기"}, free_text_delta:"기름기 적은", message:"기름기 적은 걸로", last_recs:["치킨 데리야끼"]
출력: {"search_query": "기름기 적은 닭고기 저녁 단백질 요리"}

예시 4 — 단일 거부
입력: slots={meal_times:["점심"], purpose:"tasty", free_text:""}, free_text_delta:"멍게 싫다", message:"아니 멍게 싫다고", last_recs:["멍게비빔밥"]
출력: {"search_query": "점심 맛있는 (멍게 제외)"}

예시 5 — 누적 부정 + 이번 턴 양성
입력: slots={meal_times:["점심"], purpose:"tasty", free_text:"멍게 싫다"}, free_text_delta:"컵밥은 괜찮았음", message:"컵밥은 괜찮았는데?", last_recs:["닭고기또띠아", "파인애플볶음밥"]
출력: {"search_query": "점심 맛있는 컵밥 (멍게 제외)"}
"""


USER_PROMPT_TEMPLATE = """[slots]
meal_times: {meal_times}
purpose: {purpose}
free_text: {free_text}

[free_text_delta]
{free_text_delta}

[message]
{message}

[last_recommendations]
{last_rec_names}
"""