# meal-bot 프로젝트 지침 (Claude Code용)

## 프로젝트 개요

식약처 조리식품 레시피 1,146건 대상 RAG 검색 서비스. 학부 졸업작품.

- **담당**: RAG 서버 (Python) + Spring Boot 백엔드 + React 프론트
- **작업 방식**: 클로드 웹(설계/흐름 주도) → 명령문 전달 → 클로드 코드(실행)
- **원칙**: 한 번에 큰 덩어리 금지. 단계 쪼개서 검증 리듬 유지.

---

## 기술 스택

- **RAG**: Python 3.13 + FastAPI (v4 완성 후 도입)
- **정형 DB**: MySQL on Docker (포트 3308, `meal_bot` DB)
- **벡터 DB**: ChromaDB (로컬 파일, `rag-api/data/chroma/`)
- **임베딩**: `jhgan/ko-sbert-nli` (768차원)
- **LLM**: OpenAI `gpt-4o-mini` (v2 이후 검색 시점 + Golden Set 판정)
- **백엔드/프론트**: Spring Boot + React (v4 완성 후 연동)
- **데이터 소스**: 식약처 COOKRCP01 API

---

## 버전 로드맵

| 버전 | 내용                                        | 상태 |
| ---- | ------------------------------------------- | ---- |
| v1   | Naive RAG (Chroma 단일 검색)                | 완료 |
| v2   | LLM 쿼리 재작성 (도메인 어휘 미스매치 해결) | 예정 |
| v3   | Re-ranking (메타필터 + 영양 수치)           | 예정 |
| v4   | Agentic (LLM 2회 앙상블, Self-Consistency)  | 예정 |

각 단계마다 **Precision@5, Recall@10으로 Ablation Study**. 공통 Golden Set 고정.

**API 엔드포인트는 v4 코어 완성 후 구축** (스펙 안정화 우선).

---

## 폴더 구조

```
meal-bot/
├── rag-api/
│   ├── scripts/
│   │   ├── _embedding_text.py
│   │   ├── collect_recipes.py
│   │   ├── load_to_mysql.py
│   │   ├── preview_embedding.py
│   │   ├── build_vector_db.py
│   │   ├── test_embedding_model.py
│   │   ├── inspect_v1_top10.py      # v1 실측 → artifacts/v1_top10.md
│   │   ├── dump_v1_samples.py
│   │   ├── label_with_llm.py        # LLM 1차 판정 (서브 챕터 3 완료)
│   │   └── evaluate_v1.py           # (예정) P@5, R@10 평가
│   ├── core/                        # 공통 모듈
│   │   ├── config.py                # 전역 상수/환경변수
│   │   ├── embedding.py             # 모델 싱글톤
│   │   ├── db.py                    # MySQL context manager + Chroma 싱글톤
│   │   └── retrieval.py             # search(query, top_k) → list[Hit]
│   ├── app/                         # (예정) FastAPI
│   ├── artifacts/                   # 산출물 (v1_top10.md, golden_set CSV 등)
│   ├── docs/                        # 판단 기준 문서 (golden_set_criteria.md)
│   ├── data/
│   │   ├── raw/recipes_raw.json
│   │   ├── processed/recipes.json
│   │   └── chroma/                  # gitignored
│   ├── venv/
│   ├── .env / .env.example
│   └── requirements.txt
├── backend/                         # (예정) Spring Boot
├── frontend/                        # (예정) React
├── docker/mysql/init/01_schema.sql
├── docker-compose.yml
└── CLAUDE.md
```

---

## MySQL 스키마 (recipe 테이블)

| 컬럼                                  | 타입         | 비고                                       |
| ------------------------------------- | ------------ | ------------------------------------------ |
| rcp_seq                               | INT PK       | 식약처 고유 ID                             |
| name                                  | VARCHAR(200) |                                            |
| category                              | VARCHAR(50)  | 인덱스 (반찬/일품/후식/국&찌개/기타)       |
| cooking_way                           | VARCHAR(50)  | 인덱스 (끓이기/기타/굽기/볶기/찌기/튀기기) |
| ingredients                           | TEXT         |                                            |
| hash_tag                              | VARCHAR(200) |                                            |
| calories, carbs, protein, fat, sodium | INT          | calories/sodium 인덱스. **1인분 기준**     |
| img_main, img_thumb                   | VARCHAR(500) |                                            |
| manuals                               | JSON         | 조리순서 배열                              |
| created_at, updated_at                | TIMESTAMP    |                                            |

**중요**:

- 식약처 원본 이상치 존재 (예: 매생이순두부탕 calories=10 kcal, 해물순두부된장찌개 protein=5g). 보정하지 않음, 보고서에 한계 명시 예정.
- **Chroma metadata**에는 `rcp_seq, name, category, cooking_way, calories, sodium`만. `protein, carbs, fat, ingredients, hash_tag`는 MySQL에만 존재. **v3 진입 시 metadata 재적재 필요**.

---

## 임베딩 텍스트 템플릿 (확정)

```python
"{name}. {category} 요리. {cooking_way} 방식. 주재료 {cleaned_ingredients}. 특징 {hash_tag}."
```

- NULL/"기타" 필드는 생략
- ingredients 전처리: 수량/단위/괄호/불릿/유니코드 분수 제거
- category `&` → 공백
- 평균 60자, 최대 139자 (토큰 128 이내)

---

## 데이터 파이프라인

```
식약처 API → collect_recipes.py → raw/recipes_raw.json
→ 정제 → processed/recipes.json (1,146건)
→ load_to_mysql.py (TRUNCATE → INSERT) → MySQL recipe
→ build_vector_db.py → Chroma recipes_v1 (1,146 × 768, cosine)
```

---

## 현재 진행 상태

**Step A 완료**: 데이터 파이프라인, core/ 모듈, v1 베이스라인 실측

**Step B 진행 중** (v1 평가 체계 구축):

- 서브 챕터 1: v1 top-10 실측 (완료) → `artifacts/v1_top10.md`
- 서브 챕터 2: 판정 기준 합의 (완료) → `docs/golden_set_criteria.md`
- 서브 챕터 3: LLM 1차 판정 50건 (완료) → `artifacts/golden_set_v1_llm_draft.csv`
- **서브 챕터 4: 사람 최종 확정 ← 진행 예정**
- 서브 챕터 5: 쿼리 20개 설계
- 서브 챕터 6: 200건 전체 라벨링

---

## core/ 사용 규약

- `retrieval.search(query, top_k)` — v1~v4 공통 부품. **시그니처 고정**.
  - 반환: `list[Hit]` (Hit: recipe_id, name, score, distance, metadata, document)
- DB 연결은 반드시 `core.db` 경유 (스크립트에서 직접 `pymysql`/`chromadb` 호출 금지)
- 모델 로드는 반드시 `core.embedding.get_model()` 경유 (싱글톤 보장)
- 모든 상수는 `core.config` (매직넘버 금지)
- `scripts/`에서 `core/` import 시 `sys.path.insert(0, str(Path(__file__).parent.parent))` 추가

---

## 보안 규칙 (엄격 준수)

**API 키 노출 사고 2회 발생 후 확립된 규칙**:

- `.env` 파일 내용 출력 명령 **절대 금지**:
  - `cat .env`, `grep <키이름> .env`, `echo $OPENAI_API_KEY` 등
- `.env` 관련 조사는 **존재 여부만** 확인:
  - `test -f .env && echo exists`
  - `grep -c "OPENAI_API_KEY" .env` (카운트만, 값 미출력)
- 파일 내용 확인이 불가피한 경우: 명령문에 **"값은 절대 출력하지 말 것"** 반복 명시
- 환경변수 디버깅 시에도 값 자체는 마스킹 또는 존재 여부만 출력

---

## 알려진 이슈

| 이슈                                           | 현황                                                                                          | 해결 시점             |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------- |
| `get_chroma_collection()` lru_cache 스테일     | 동일 프로세스에서 collection 재생성 시 캐시에 삭제된 객체 잔존. 스크립트 기반 실행이라 미발현 | FastAPI 도입 시       |
| `category` NULL 119건                          | 임베딩 텍스트에서 생략 처리 완료                                                              | v2 LLM 자동 분류 검토 |
| `calories=0` 1건 (rcp_seq=1057)                | 무시                                                                                          | —                     |
| `ingredients` 빈값 3건 (rcp_seq=691, 692, 831) | 무시                                                                                          | —                     |
| Chroma metadata 영양소 필드 누락               | `protein, carbs, fat` 등 MySQL에만 존재                                                       | v3 진입 시 재적재     |

---

## 구조적 개선 리스트 (지금 건드리지 말 것, 챕터 전환 시 반영)

1. `scripts/inspect_v1_top10.py`와 `scripts/dump_v1_samples.py`의 `TEST_QUERIES` 중복 → `core/config.py`에 `EVAL_QUERIES`로 통합
2. `label_with_llm.py` 내 `COMMON_RULES`, `QUERY_CRITERIA` 하드코딩 → v2 진입 시 `prompts.py` 또는 md 읽기로 분리 검토
3. `artifacts/` gitignore 상태 확인 필요 (CSV는 git 보관 대상)

---

## 작업 스타일

- 한 번에 한 단계씩 (큰 덩어리 금지)
- 실데이터/실출력 보고 판단 (추측 지양)
- 돌아가는 v1 먼저, 최적화는 v2 이후
- 데이터 원본 보존 (raw/ 백업 유지)
- MySQL + Chroma 역할 분리 (정형 필터 vs 의미 검색)

---

## 금지 사항

- `retrieval.search()` 시그니처 변경
- `core/` 우회해서 직접 DB/모델 접근
- 매직넘버 하드코딩
- `.env` 내용 노출 명령
- 선제적 최적화 (v2/v3에서 해결될 문제를 v1에서 땜질)
- 한 번에 여러 파일 대규모 수정

---

## 자주 쓰는 검증 명령

```bash
# Chroma 상태 확인
cd rag-api && python -c "from core.db import get_chroma_collection; c = get_chroma_collection(); print(c.count())"

# MySQL 연결 확인
cd rag-api && python -c "from core.db import mysql_connection; \
with mysql_connection() as conn: \
    with conn.cursor() as cur: \
        cur.execute('SELECT COUNT(*) FROM recipe'); \
        print(cur.fetchone())"

# v1 실측 재실행
cd rag-api && python scripts/inspect_v1_top10.py
```
