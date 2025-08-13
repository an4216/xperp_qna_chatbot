# llm.py
# - QnA/매뉴얼 벡터스토어 분리
# - E5 instruct 프리픽스(query:/passage:)
# - QnA 우선 라우팅 → 실패 시 매뉴얼 폴백
# - 출처 표기는 메타 있을 때만
# - 결정적 응답(temperature=0.0)

import os
import time

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda

# ===== 환경/상수 =====
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_LLM = os.getenv("MODEL_LLM", "gemma3:latest")  # 예: "gemma3:latest"

VECTOR_ROOT = "vectorstore"
QNA_DIR = "docs/qna"
MANUAL_DIR = "docs/manual"

TOP_K_QNA = 3
TOP_K_MAN = 4
MAX_CHUNKS_MAN = 500

# 세션별 대화 이력 저장 (in-memory)
_store = {}


# =========================
# 공통 유틸/임베딩
# =========================
def _get_device():
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    return device


def _get_embedding(device: str):
    # E5 instruct 권장: normalize_embeddings=True
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _prefix_passage(text: str) -> str:
    # E5 instruct: 문서는 passage: 프리픽스
    return f"passage: {text.strip()}"


def _to_e5_query(q: str) -> str:
    # E5 instruct: 질의는 query: 프리픽스
    return f"query: {q.strip()}"


def _ensure_faiss(path_dir: str, embedding, split_docs):
    os.makedirs(path_dir, exist_ok=True)
    idx_path = os.path.join(path_dir, "index.faiss")
    if os.path.exists(idx_path):
        return FAISS.load_local(path_dir, embedding, allow_dangerous_deserialization=True)
    vs = FAISS.from_documents(split_docs, embedding)
    vs.save_local(path_dir)
    return vs


# =========================
# 문서 로딩
# =========================
def _load_qna_docs():
    docs = []
    if not os.path.isdir(QNA_DIR):
        return docs
    for fn in os.listdir(QNA_DIR):
        if not fn.endswith(".txt"):
            continue
        loader = TextLoader(os.path.join(QNA_DIR, fn), encoding="utf-8")
        for d in loader.load():
            d.metadata["doc_type"] = "qna"
            d.metadata["source"] = os.path.splitext(fn)[0]
            d.page_content = _prefix_passage(d.page_content)
            docs.append(d)
    return docs


def _load_manual_docs():
    docs = []
    if not os.path.isdir(MANUAL_DIR):
        return docs
    for fn in os.listdir(MANUAL_DIR):
        path = os.path.join(MANUAL_DIR, fn)
        base = os.path.splitext(fn)[0]
        if fn.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            for d in loader.load():
                d.metadata["doc_type"] = "manual"
                d.metadata["source"] = base
                d.page_content = _prefix_passage(d.page_content)
                docs.append(d)
        elif fn.endswith(".pdf"):
            loader = PyPDFLoader(path)
            pages = loader.load()
            for i, p in enumerate(pages, start=1):
                p.metadata["doc_type"] = "manual"
                p.metadata["source"] = base
                p.metadata["page"] = i
                citation = f"\n\n(출처: {base} {i}페이지)"
                p.page_content = _prefix_passage(p.page_content) + citation
                docs.append(p)
    return docs


# =========================
# Retriever 구성 (분리 인덱스)
# =========================
def get_retrievers():
    device = _get_device()
    embedding = _get_embedding(device)

    # QnA
    qna_docs = _load_qna_docs()
    qna_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    qna_chunks = qna_splitter.split_documents(qna_docs)
    qna_vs = _ensure_faiss(os.path.join(VECTOR_ROOT, "qna"), embedding, qna_chunks)
    qna_retriever = qna_vs.as_retriever(
        search_kwargs={
            "k": TOP_K_QNA,
            "score_threshold": 0.25  # 충분히 보수적으로 필터링
        },
        search_type="similarity_score_threshold",
    )

    # Manual
    man_docs = _load_manual_docs()
    man_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    man_chunks = man_splitter.split_documents(man_docs)
    if len(man_chunks) > MAX_CHUNKS_MAN:
        man_chunks = man_chunks[:MAX_CHUNKS_MAN]
    man_vs = _ensure_faiss(os.path.join(VECTOR_ROOT, "manual"), embedding, man_chunks)
    man_retriever = man_vs.as_retriever(
        search_kwargs={"k": TOP_K_MAN, "fetch_k": 20},
        search_type="mmr",  # 다양성 확보
    )

    return qna_retriever, man_retriever


# =========================
# LLM (Ollama)
# =========================
def get_llm(model: str = MODEL_LLM):
    return ChatOllama(
        model=model,
        base_url=OLLAMA_HOST,
        temperature=0.0,  # 결정적 응답
        # num_ctx=8192,  # 모델/빌드 환경에 맞춰 필요 시 조정
    )


# =========================
# 세션 히스토리
# =========================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


# =========================
# 프롬프트 (QnA 전용 / 매뉴얼 폴백)
# =========================
def _qna_prompt():
    system = (
        "당신은 Xperp QnA 전용 응답기입니다.\n"
        "- 제공된 QnA context만 근거로 간결하고 정확하게 답하십시오.\n"
        "- 아래 형식을 따르되, 문서에 없는 항목은 생성하지 마십시오.\n\n"
        "✅ 질문에 대한 정식 답변: 문서의 A를 한 문단으로 명확히.\n"
        "✅ 간단 요약: 한 줄 요약.\n"
        "✅ 예상 질문: 1~2개(문서에 근거 가능할 때만).\n\n"
        "출처 표기는 context의 source/page 메타가 있을 때만 문장 끝에 표시합니다.\n"
        "{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def _manual_prompt():
    system = (
        "당신은 Xperp 매뉴얼 기반 상담 챗봇입니다.\n"
        "- 제공된 매뉴얼 context에서만 정보를 발췌하십시오.\n"
        "- 사용법/유의사항/추가설명은 문서에 실제로 있을 때만 포함하십시오.\n\n"
        "✅ 질문에 대한 정식 답변: 핵심을 한 문단으로.\n"
        "✅ 간단 요약: 한 줄.\n"
        "✅ 사용법 안내: 단계·경로가 있을 때만 단계형으로.\n"
        "✅ 유의사항: 실제 제약/주의가 있을 때만.\n"
        "✅ 추가 설명: 관련 항목이 있을 때만.\n"
        "✅ 예상 질문: 문서에 근거 가능하면 1~2개, 없으면 생략.\n\n"
        "모든 출처 표기는 context의 source/page 메타가 있을 때만 문장 끝에 표시합니다.\n"
        "{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


# =========================
# 라우팅 RAG (QnA 우선 → 매뉴얼 폴백)
# =========================
def _route_and_answer(question: str, chat_history, qna_ret, man_ret):
    llm = get_llm()
    e5q = _to_e5_query(question)

    # 1) QnA 우선
    qna_docs = qna_ret.get_relevant_documents(e5q)
    if qna_docs:
        prompt = _qna_prompt()
        chain = create_stuff_documents_chain(llm, prompt)
        return chain.invoke({"input": question, "chat_history": chat_history, "context": qna_docs})

    # 2) 매뉴얼 폴백
    man_docs = man_ret.get_relevant_documents(e5q)
    if man_docs:
        prompt = _manual_prompt()
        chain = create_stuff_documents_chain(llm, prompt)
        return chain.invoke({"input": question, "chat_history": chat_history, "context": man_docs})

    # 3) 미발견
    return "문의하신 내용은 현재 자료 기준으로는 확인이 어렵습니다."


# get_rag_chain() 전체 교체
def get_rag_chain():
    qna_ret, man_ret = get_retrievers()

    # RunnableWithMessageHistory가 요구하는 형태(dict)에 맞춰 반환
    def _invoke(inputs, config):
        question = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        text = _route_and_answer(question, chat_history, qna_ret, man_ret)
        return {"answer": text}

    runnable = RunnableLambda(_invoke)

    conversational = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational



# =========================
# 외부 노출: 최종 응답 스트리밍
# =========================
def get_ai_response(user_message: str, session_id: str = "abc123"):
    chain = get_rag_chain()
    inner_stream = chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )

    def timed_stream():
        start = time.perf_counter()
        for chunk in inner_stream:
            yield chunk
        yield f"\n\n⏱ {time.perf_counter() - start:.2f}s"

    return timed_stream()
