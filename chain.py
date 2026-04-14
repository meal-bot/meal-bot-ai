from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PROMPT_TEMPLATE = """
당신은 영양사 AI입니다. 아래 음식 데이터를 참고하여 사용자의 질문에 맞는 식단을 추천해주세요.

[참고 음식 데이터]
{context}

[사용자 질문]
{question}

답변 규칙:
- 추천 음식 3~5가지를 제시하세요
- 각 음식의 주요 영양성분(칼로리, 단백질, 나트륨 등)을 간략히 설명하세요
- 왜 이 음식이 적합한지 이유를 설명하세요
- 친절하고 이해하기 쉬운 말투로 답변하세요
"""

DRINK_CATEGORIES = ["음료 및 차류"]

def detect_filter(question: str) -> dict | None:
    if any(kw in question for kw in ["음료", "차", "음료수", "마실"]):
        return {"대분류": {"$in": DRINK_CATEGORIES}}
    if any(kw in question for kw in ["음식", "식사", "밥", "반찬", "요리"]):
        return {"대분류": {"$nin": DRINK_CATEGORIES}}
    return None

def build_chain(filter: dict | None = None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
    search_kwargs = {"k": 10}
    if filter:
        search_kwargs["filter"] = filter

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def ask(question: str) -> str:
    filter = detect_filter(question)
    chain = build_chain(filter)
    return chain.invoke(question)

if __name__ == "__main__":
    answer = ask("단백질이 높은 음식 추천해줘")
    print(answer)
