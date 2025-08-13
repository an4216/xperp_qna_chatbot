# LangChain 및 관련 라이브러리 import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ★ OpenAI → Ollama
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples
import os
import pickle
import time

# ===== 설정 =====
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_LLM = "gemma3:latest"          # ollama pull gpt-oss:20b
#MODEL_EMBED = "nomic-embed-text"   # ollama pull nomic-embed-text
TOP_K = 4
VECTOR_DIR = "vectorstore"
MAX_CHUNKS = 500
MODEL_EMBED = "intfloat/multilingual-e5-large-instruct"

# 세션별 대화 히스토리 저장소 (메모리 dict)
store = {}

# 1. 세션별 대화 이력 객체 반환
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # ── Device 자동 선택 (CUDA 있으면 사용)
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    # ── HuggingFace E5 임베딩 (권장 설정 포함)
    embedding = HuggingFaceEmbeddings(
        model_name=MODEL_EMBED,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # E5 권장
    )

    index_path = os.path.join(VECTOR_DIR, "index.faiss")

    if os.path.exists(index_path):
        # 기존 인덱스 로드
        vectorstore = FAISS.load_local(
            VECTOR_DIR,
            embedding,
            allow_dangerous_deserialization=True,
        )
    else:
        # 없으면 새로 생성
        documents = []
        docs_dirs = ["docs/manual", "docs/qna"]

        for docs_dir in docs_dirs:
            if not os.path.isdir(docs_dir):
                continue
            for filename in os.listdir(docs_dir):
                file_path = os.path.join(docs_dir, filename)
                if filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    manual_name = os.path.splitext(filename)[0]
                    for doc in docs:
                        doc.metadata["source"] = manual_name
                    documents.extend(docs)
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    for i, page in enumerate(pages):
                        manual_name = os.path.splitext(filename)[0]
                        page.metadata["source"] = manual_name
                        page.metadata["page"] = i + 1
                        citation = f"\n\n(출처: {manual_name} {i + 1}페이지)"
                        page.page_content += citation
                        documents.append(page)

        splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=48)
        split_docs = [
            d for d in splitter.split_documents(documents)
            if len(d.page_content.strip()) > 10
        ]

        if len(split_docs) > MAX_CHUNKS:
            split_docs = split_docs[:MAX_CHUNKS]

        vectorstore = FAISS.from_documents(split_docs, embedding)
        vectorstore.save_local(VECTOR_DIR)

    # ✅ 마지막에 한 번만 반환 (MMR 검색)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 25, "lambda_mult": 0.5},
    )



    # 없으면 새로 생성
    documents = []
    docs_dirs = ["docs/manual", "docs/qna"]

    for docs_dir in docs_dirs:
        if not os.path.isdir(docs_dir):
            continue
        for filename in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                manual_name = os.path.splitext(filename)[0]
                for doc in docs:
                    doc.metadata["source"] = manual_name
                documents.extend(docs)
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for i, page in enumerate(pages):
                    manual_name = os.path.splitext(filename)[0]
                    page.metadata["source"] = manual_name
                    page.metadata["page"] = i + 1
                    citation = f"\n\n(출처: {manual_name} {i + 1}페이지)"
                    page.page_content += citation
                    documents.append(page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=48)
    split_docs = [d for d in splitter.split_documents(documents) if len(d.page_content.strip()) > 10]

    if len(split_docs) > MAX_CHUNKS:
        split_docs = split_docs[:MAX_CHUNKS]

    vectorstore = FAISS.from_documents(split_docs, embedding)
    vectorstore.save_local(VECTOR_DIR)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# 4. LLM(챗봇) 인스턴스 생성 (Ollama 사용)
def get_llm(model: str = MODEL_LLM):
    # ChatOpenAI → ChatOllama
    # 필요시 매개변수: temperature, num_ctx 등 조절 가능
    return ChatOllama(
        model=model,
        base_url=OLLAMA_HOST,
        temperature=0.1,
        # num_ctx=8192,  # 모델/빌드에 따라 조정
    )

# 3. 대화 맥락을 반영한 retriever 반환 (standalone question 변환 + 벡터검색)
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "대화 이력과 최신 사용자 질문이 주어졌을 때, "
        "대화 이력 없이도 이해 가능한 독립 질문으로 재작성하세요. "
        "답변하지 말고 질문만 반환하세요."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        get_llm(), retriever, contextualize_q_prompt
    )
    return history_aware_retriever

# 5. 사용자의 질문을 사전 기반으로 전처리 (질문 변환 체인)
def get_dictionary_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        변경 필요가 없으면 원문 질문만 리턴하세요.

        질문: {question}
    """.strip())

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain

# 6. RAG(문서 기반 질의응답) 전체 체인
def get_rag_chain():
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "너는 Xperp 문서 기반 RAG 상담 챗봇이다.\n"
        "반드시 아래 context에 포함된 내용만 근거로 한국어로 답하라.\n"
        "context에 근거가 없거나 불충분하면 추측하지 말고, 문서에 근거가 없음을 분명히 밝혀라.\n"
        "외부 지식·경험·일반 상식은 사용하지 마라.\n"
        "\n"
        "[응답 우선순위]\n"
        "1) QnA 문서의 Q/A·키워드가 질문과 직접 관련되면 그 내용을 먼저 요약·정리하라.\n"
        "2) 그 다음 PDF 매뉴얼/기타 문서의 관련 내용을 보완하라.\n"
        "\n"
        "[사실·표현 규칙]\n"
        "- 메뉴 경로, 수치, 옵션명, 제한사항 등은 context에 있는 표현을 그대로 사용하라(의미 변경·창작 금지).\n"
        "- 여러 문서가 상충하면 각각의 근거를 분리해 제시하고, 모순이 있음을 알리되 중립적으로 기술하라.\n"
        "- 인용 또는 주장 뒤에는 가능한 한 근거 문서의 출처(source/page)를 문장 끝에 표기하라.\n"
        "- 출처 표기 형식: (출처: {source} {page}페이지)\n"
        "- source/page 정보가 없으면 출처 표기를 생략하라(임의 추정 금지).\n"
        "\n"
        "[섹션 출력 규칙]\n"
        "아래 섹션을 사용하되, 해당 내용을 뒷받침할 근거가 context에 없으면 그 섹션은 과감히 생략하라.\n"
        "1) 질문에 대한 정식 답변: 문서 근거로 핵심 개념/목적/동작 원리를 간결하게 설명\n"
        "2) 간단 요약: 한두 줄 요약\n"
        "3) 사용법 안내: 메뉴 경로와 단계(문서에 있을 때만), 필요 시 3~10단계로 명확히\n"
        "4) 유의사항: 제약·주의·예외(문서 근거가 있을 때만)\n"
        "5) 매뉴얼 참조: 본문에서 실제로 인용한 문장/항목이 있을 때만 출처를 모아 표기\n"
        "6) 예상 질문: 문서에 근거가 있는 연관 항목이 있을 때만 1~3문장 제시\n"
        "\n"
        "[금지 사항]\n"
        "- 문서에 없는 기능/정책/수치/메뉴를 만들어내지 마라.\n"
        "- 예시 프롬프트의 내용(형식 예시)은 사실 근거로 사용하지 마라. 형식만 참고하라.\n"
        "- 사용자 요청이 모호해도 임의 보완 금지. 문서 근거가 부족하면 필요한 추가 정보를 요청하라.\n"
        "\n"
        "마지막으로, 과장 없이 사실 위주로 간결하게 작성하라.\n"
        "{context}"
    )


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain

# 7. 최종 답변 생성 함수 (질문 → 답변) + 소요시간 표시
def get_ai_response(user_message):
    #dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    #tax_chain = {"input": dictionary_chain} | rag_chain
    tax_chain = rag_chain
    # 내부 스트림 생성
    inner_stream = tax_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": "abc123"}},
    )

    # 스트리밍 래퍼: 본문 스트림 그대로 흘리고, 마지막에 시간 한 줄 추가
    def timed_stream():
        start = time.perf_counter()
        for chunk in inner_stream:
            yield chunk
        elapsed = time.perf_counter() - start
        yield f"\n\n⏱ {elapsed:.2f}s"

    return timed_stream()
