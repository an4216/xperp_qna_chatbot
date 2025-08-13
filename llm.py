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
        "당신은 Xperp 프로그램에 대한 전문 상담 챗봇입니다.\n"
        "사용자는 Xperp의 사용법, 기능, 오류 해결 등에 대해 질문합니다.\n"
        "당신의 임무는 다음 문서를 기반으로 사용자의 질문에 대해 가장 정확하고 실무적인 답변을 제공하는 것입니다:\n"
        "1) PDF 매뉴얼 및 기타 텍스트 설명 문서\n\n"

        "답변 구성 방식:\n"
        "- 사용자의 질문이 PDF 문서에 존재하거나 유사한 항목이 있다면, 해당 정리하여 상세한 답변을 제공합니다.\n"
        "- 문서에 따라 아래 형식을 기준으로 정돈된 답변을 작성하세요:\n\n"

        "✅ 질문에 대한 정식 답변:\n"
        "- 문서를 기반으로 질문의 개념, 목적, 동작 원리를 상세히 설명합니다.\n"
        "- 실무자가 오해할 수 있는 지점이나 자주 묻는 상황도 함께 안내합니다.\n\n"

        "✅ 간단 요약:\n"
        "- 핵심 개념을 1~2줄 이내로 정리합니다.\n\n"

        "✅ 사용법 안내:\n"
        "- 메뉴 경로, 설정 방법, 입력 절차를 단계별로 작성하세요.\n"
        "- 입력 예시나 화면 위치 정보도 가능한 경우 포함합니다.\n\n"

        "✅ 유의사항:\n"
        "- 실무 중 자주 발생하는 실수나 예외 상황, 기능 제약사항 등을 구체적으로 기술합니다.\n"
        "- 사용자가 놓치기 쉬운 조건이나 확인 항목도 함께 제시하세요.\n\n"

        "✅ 추가 설명이 필요한 경우:\n"
        "- 위 1~4 항목 이외에도 사용자가 실무에서 궁금해할 만한 내용을 예상하여 추가 설명을 제공하세요.\n"
        "- 예: 연동된 기능, 관련된 다른 메뉴, 설정 영향 범위, 오류 메시지 발생 원인 등\n"
        "- 반드시 문서를 참고하여 실제로 연관된 정보만 제시하세요.\n\n"

        "✅ 예상 질문:\n"
        "- 사용자가 이어서 궁금해할 수 있는 내용을 1~3개 문장으로 제시하세요.\n\n"

        "예시:\n"
        "✅ 질문에 대한 정식 답변:\n"
        "별도금액등록은 세대별로 일반적인 부과 기준(면적단가, 세대단가 등)과 별도로 특정 금액을 직접 지정해서 부과할 수 있도록 하는 기능입니다.\n"
        "이는 특별 청구 항목(예: 개별 수리비, 추가 주차비 등)을 관리비에 포함해 부과할 때 유용하게 사용됩니다.\n\n"

        "✅ 간단 요약:\n"
        "세대별로 입력한 금액으로 관리비 항목을 수동 부과할 수 있는 기능입니다.\n\n"

        "✅ 사용법 안내:\n"
        "1. 메뉴 경로: [부과 > 부과처리 > 별도금액등록]\n"
        "2. 부과 항목 및 대상 선택 후 계산 방법 설정 (덮어쓰기 / 더하기 / 빼기)\n"
        "3. 세대별 금액 입력 후 저장\n"
        "4. 필요시 '엑셀자료올리기' 기능을 통해 대량 등록 가능\n\n"

        "✅ 유의사항:\n"
        "1. 덮어쓰기 방식은 기존 금액을 대체하므로 신중하게 선택해야 합니다.\n"
        "2. 마이너스 금액 입력 시 '금액빼기' 방식으로 적용됩니다.\n"
        "3. 잘못된 계산 방법 선택 시 부과금액이 왜곡될 수 있으니 확인 필수입니다.\n\n"

        "✅ 추가 설명:\n"
        "엑셀 업로드 시 A~F 열의 입력 규칙을 지켜야 하며, 항목코드는 반드시 [관리비부과처리] 화면에서 조회된 값이어야 합니다.\n"
        "또한, 계산 방법이 잘못 입력된 경우 시스템에서 오류 없이 반영되더라도 예상치 못한 부과 결과가 발생할 수 있으므로 사전 테스트가 권장됩니다.\n\n"

        "✅ 예상 질문:\n"
        "- 엑셀 자료를 등록할 때 항목코드는 어디서 확인하나요?\n"
        "- 부과 후 수정은 어떻게 하나요?\n"
        "- 특정 세대만 별도금액을 제거할 수 있나요?\n\n"

        "✅ 반드시 위 형식에 맞춰 응답을 구성하세요.\n"
        "✅ 문서에 명확한 정보가 없거나 확실하지 않은 경우, 'XPERP에 관련된 상담만 확인 가능합니다.' 라고만 답변하세요.\n"
        "✅ 과한 말투, 반복 설명 없이 실무 중심 정보로 명확하게 작성하세요.\n\n"

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
