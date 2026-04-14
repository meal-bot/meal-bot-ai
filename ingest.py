import shutil
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

COLUMNS = [
    "식품명", "식품대분류명", "식품중분류명",
    "영양성분함량기준량", "식품중량",
    "에너지(kcal)", "탄수화물(g)", "단백질(g)", "지방(g)",
    "나트륨(mg)", "식이섬유(g)", "당류(g)",
    "콜레스테롤(mg)", "포화지방산(g)", "칼륨(mg)", "칼슘(mg)", "철(mg)"
]
CORE_COLS = ["에너지(kcal)", "탄수화물(g)", "단백질(g)", "지방(g)"]
NUTRIENT_LABELS = {
    "에너지(kcal)": ("에너지", "kcal"),
    "탄수화물(g)": ("탄수화물", "g"),
    "단백질(g)": ("단백질", "g"),
    "지방(g)": ("지방", "g"),
    "나트륨(mg)": ("나트륨", "mg"),
    "식이섬유(g)": ("식이섬유", "g"),
    "당류(g)": ("당류", "g"),
    "콜레스테롤(mg)": ("콜레스테롤", "mg"),
    "포화지방산(g)": ("포화지방산", "g"),
    "칼륨(mg)": ("칼륨", "mg"),
    "칼슘(mg)": ("칼슘", "mg"),
    "철(mg)": ("철", "mg"),
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, usecols=COLUMNS)
    df = df.dropna(subset=["식품명"])
    df = df[~(df[CORE_COLS].fillna(0) == 0).all(axis=1)]
    return df


def row_to_document(row: pd.Series) -> Document:
    기준량 = str(row.get("영양성분함량기준량", ""))
    식품중량 = row.get("식품중량")

    lines = [
        f"음식명: {row['식품명']}",
        f"분류: {row.get('식품대분류명', '')} > {row.get('식품중분류명', '')}",
        f"기준량: {기준량}",
    ]
    if not pd.isna(row.get("식품중량")):
        lines.append(f"식품중량(1인분 참고): {식품중량}")

    for col, (label, unit) in NUTRIENT_LABELS.items():
        value = row[col]
        if pd.isna(value):
            lines.append(f"{label}: 정보 없음")
        else:
            lines.append(f"{label}: {round(value, 1)}{unit}")

    content = "\n".join(lines)

    metadata = {
        "식품명": row["식품명"],
        "대분류": str(row.get("식품대분류명", "")),
        "중분류": str(row.get("식품중분류명", "")),
        "기준량": 기준량,
        "식품중량": "" if pd.isna(식품중량) else str(식품중량),
    }
    for col, (label, _) in NUTRIENT_LABELS.items():
        value = row[col]
        metadata[label] = float(value) if pd.notna(value) else -1.0

    return Document(page_content=content, metadata=metadata)


VECTORSTORE_DIR = "vectorstore"
BATCH_SIZE = 500


def ingest():
    print("데이터 로딩 중...")
    df = load_data("data/food_nutrition.xlsx")
    print(f"총 {len(df)}개 식품 로드 완료")

    docs = [row_to_document(row) for _, row in df.iterrows()]

    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
        print("기존 vectorstore 삭제 완료")

    print("임베딩 및 ChromaDB 저장 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=docs[:BATCH_SIZE],
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    for i in range(BATCH_SIZE, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"  {min(i + BATCH_SIZE, len(docs))}/{len(docs)} 완료")

    print("완료!")


if __name__ == "__main__":
    ingest()
