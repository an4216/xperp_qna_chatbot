# llm.py

# LangChain 및 관련 라이브러리 import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough, RunnableMap


# ✅ Ollama용 LLM
from langchain_community.chat_models import ChatOllama
# ✅ HuggingFace bge-m3 임베딩
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from config import answer_examples
import os
import time
import re

# =========================================
# 환경설정
# =========================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_LLM   = os.getenv("MODEL_LLM", "gemma3:latest")  # ollama pull gemma3:latest
TOP_K       = int(os.getenv("TOP_K", "4"))
VECTOR_DIR  = os.getenv("VECTOR_DIR", "vectorstore")
REFUSAL_MSG = "죄송합니다. 업로드된 문서에서 근거를 찾지 못해 답변할 수 없습니다. 질문을 더 구체화하거나 관련 문서를 업로드해 주세요."
# 세션별 대화 히스토리 저장소
store = {}

# -------------------------------
# 유틸: few-shot 예시의 '출처' 문구 제거
# -------------------------------
def sanitize_examples(examples: list[dict]) -> list[dict]:
    """
    Few-shot 예시에 포함된 '출처/페이지' 표기(형식 예시)를 제거해서,
    실제 인용은 반드시 context의 metadata(source/page)로만 하도록 보조한다.
    """
    sanitized = []
    for ex in examples:
        inp = ex.get("input", "")
        ans = ex.get("answer", "")

        # 1) '✅ 매뉴얼 참조:' 라인 제거
        ans = re.sub(r'^\s*✅\s*매뉴얼\s*참조:.*$', '', ans, flags=re.MULTILINE)

        # 2) 본문 내 임의 출처 괄호 제거: (출처: ...페이지)
        ans = re.sub(r'\(출처:\s*[^)]+\)', '', ans)

        # 3) '...페이지 참조' 류 문구 제거 (선택적)
        ans = re.sub(r'[(（]?\s*[^)\n]*매뉴얼[^)\n]*\d+\s*페이지\s*참조[)）]?', '', ans)

        # 4) 여분 공백 정리
        ans = re.sub(r'\n{3,}', '\n\n', ans).strip()

        sanitized.append({"input": inp, "answer": ans})
    return sanitized

# 1. 세션별 대화 이력 객체 반환
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ✅ bge-m3 임베딩 인스턴스 생성
def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True}  # 코사인 유사도 안정화
    )

# 2. 문서 로드 + 벡터스토어 생성 + retriever 반환
def get_retriever():
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # 저장된 벡터스토어 로드
    if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        vectorstore = FAISS.load_local(
            VECTOR_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={'k': TOP_K})

    # 없으면 새로 생성
    embedding = get_embeddings()
    documents = []
    docs_dirs = ["docs/manual", "docs/qna"]

    for docs_dir in docs_dirs:
        if not os.path.isdir(docs_dir):
            continue
        for filename in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, filename)
            manual_name = os.path.splitext(filename)[0]

            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = manual_name
                documents.extend(docs)

            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for i, page in enumerate(pages):
                    page.metadata["source"] = manual_name
                    page.metadata["page"] = i + 1
                    # ✅ 출처 정보 삽입 (실제 인용은 여기서만)
                    citation = f"\n\n(출처: {manual_name} {i + 1}페이지)"
                    page.page_content += citation
                    documents.append(page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 10]

    MAX_CHUNKS = 500
    if len(split_docs) > MAX_CHUNKS:
        split_docs = split_docs[:MAX_CHUNKS]

    vectorstore = FAISS.from_documents(split_docs, embedding)
    vectorstore.save_local(VECTOR_DIR)

    return vectorstore.as_retriever(search_kwargs={'k': TOP_K})

# 4. LLM(챗봇) 인스턴스 생성 → Ollama
def get_llm():
    # 필요 시 num_predict, temperature, keep_alive 등 파라미터 추가 가능
    return ChatOllama(
        base_url=OLLAMA_HOST,
        model=MODEL_LLM,
        # 아래는 선택: 속도/안정성 튜닝 예시
        temperature=0.0,
        # top_p=0.9,
        # num_predict=256,
        # num_ctx=4096,
        # keep_alive="30m",
    )

# 3. 대화 맥락을 반영한 retriever 반환 (standalone question 변환 + 벡터검색)
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

# 6. RAG 체인
def get_rag_chain():
    llm = get_llm()

    cleaned_examples = sanitize_examples(answer_examples)

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{answer}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=cleaned_examples,
    )

    # 👉 프롬프트 '원문 그대로' 유지 (형님이 쓰던 system_prompt 그대로)
    system_prompt = (
        "당신은 Xperp 프로그램에 대한 전문 상담 챗봇입니다.\n"
        "사용자는 Xperp의 사용법, 기능, 오류 해결 등에 대해 질문합니다.\n"
        "당신의 임무는 다음 문서를 기반으로 사용자의 질문에 대해 가장 정확하고 실무적인 답변을 제공하는 것입니다:\n"
        "1) 질문(Q)과 답변(A), 키워드(T)가 포함된 QnA 문서\n"
        "2) PDF 매뉴얼 및 기타 텍스트 설명 문서\n\n"
        "답변 구성 방식 (qna.txt 우선):\n"
        "- 사용자의 질문이 qna.txt 문서에 존재하거나 키워드를 참고하여 유사한 항목이 있다면, 해당 A 내용을 우선적으로 정리하여 답변의 맨 처음에 제공합니다.\n"
        "- 이후 PDF 매뉴얼 등 기타 문서를 참고하여 보완 설명을 이어서 작성합니다.\n"
        "- 문서에 따라 아래 형식을 기준으로 정돈된 답변을 가독성을 고려하여 작성하세요:\n\n"
        "✅ 질문에 대한 정식 답변:\n"
        "- 문서를 기반으로 질문의 개념, 목적, 동작 원리를 상세히 설명합니다.\n"
        "- 실무자가 오해할 수 있는 지점이나 자주 묻는 상황도 함께 안내합니다.\n\n"
        "✅ 간단 요약:\n"
        "- 핵심 개념을 1~2줄 이내로 정리합니다.\n\n"
        "✅ 사용법 안내:\n"
        "- 메뉴 경로, 설정 방법, 입력 절차를 문서에 있는 내용으로 단계별로 작성하세요.\n"
        "- 입력 예시나 화면 위치 정보도 가능한 경우 포함합니다.\n\n"
        "✅ 유의사항:\n"
        "- 실무 중 자주 발생하는 실수나 예외 상황, 기능 제약사항 등을 구체적으로 기술합니다.\n"
        "- 사용자가 놓치기 쉬운 조건이나 확인 항목도 함께 제시하세요.\n\n"
        "✅ 추가 설명이 필요한 경우:\n"
        "- 위 1~4 항목 이외에도 사용자가 실무에서 궁금해할 만한 내용을 예상하여 추가 설명을 제공하세요.\n"
        "- 예: 연동된 기능, 관련된 다른 메뉴, 설정 영향 범위, 오류 메시지 발생 원인 등\n"
        "- 반드시 문서를 참고하여 실제로 연관된 정보만 제시하세요.\n\n"
        "✅ 예상 질문:\n"
        "- 사용자가 이어서 궁금해할 수 있는 내용을 1~3개 문장으로 제시하세요.\n"
        "- qna와 매뉴얼에서 답변할 수 있는 내용을 발췌하여 제시하세요.\n"
        "- qna와 매뉴얼에 나온 예상질문이 추가로 없다면, 예상질문을 생략해주세요.\n\n"
        "✅ 매뉴얼 참조 출력 지침:\n"
        "- 아래 조건을 반드시 지켜야 합니다:\n"
        "  1. 반드시 'context'의 문서 metadata(source/page)에서만 출처를 가져오세요.\n"
        "  2. few-shot 예시(answer_examples) 안의 출처/페이지 표기는 모두 무시하세요. (형식 예시일 뿐 실제 인용 아님)\n"
        "  3. 문서명이나 페이지를 임의로 추측하거나 생성하지 마세요.\n"
        "  4. 각 설명이 어떤 문서에서 유래했는지 사용자에게 명확히 전달해야 합니다.\n"
        "- 사용법 안내 등의 답변 본문에서도 관련 설명 끝에 (출처: 문서명 n페이지) 형식으로 표시해 주세요.\n\n"
        "출력 형식 규칙(매우 중요):\n"
        "- 반드시 Markdown을 사용하세요.\n"
        "- 각 섹션 제목(예: '✅ 질문에 대한 정식 답변:') 뒤에는 빈 줄 1개를 두세요.\n"
        "- '사용법 안내'는 번호 목록(1., 2., 3., ...)으로, 항목마다 새 줄에서 시작하세요.\n"
        "- '유의사항', '예상 질문'은 불릿 목록(- )으로, 항목마다 새 줄에서 시작하세요.\n"
        "- 한 줄에 여러 항목을 이어 쓰지 마세요. 각 항목은 반드시 줄바꿈으로 구분합니다.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # ✅ 런타임 가드: 문서가 0개면 즉시 거부문구 반환, 아니면 QA 체인 실행
    guard_input = RunnableMap({
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", []),
        "context": history_aware_retriever,   # ← retriever가 리스트[Document] 반환
    })

    refuse = RunnableLambda(lambda _: {"answer": REFUSAL_MSG})

    guarded_chain = (
        guard_input
        | RunnableBranch(
            # 조건: context 비었으면 거부
            (lambda x: len(x.get("context", [])) == 0, refuse),
            # 기본: QA 실행
            question_answer_chain
        )
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        guarded_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain

# 7. 최종 답변 생성 함수
def get_ai_response(user_message):
    rag_chain = get_rag_chain()

    # ✅ 'input' 키로 전달
    stream = rag_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": "abc123"}},
    )

    def timed_stream():
        start = time.perf_counter()
        for chunk in stream:
            yield chunk
        elapsed = time.perf_counter() - start
        yield f"\n\n⏱ {elapsed:.2f}s"

    return timed_stream()
