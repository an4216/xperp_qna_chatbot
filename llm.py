# LangChain 및 관련 라이브러리 import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ★ OpenAI → Ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples
import os
import pickle

# ===== 설정 =====
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_LLM = "gpt-oss:20b"          # ollama pull gpt-oss:20b
MODEL_EMBED = "nomic-embed-text"   # ollama pull nomic-embed-text
TOP_K = 4
VECTOR_DIR = "vectorstore"
MAX_CHUNKS = 500

# 세션별 대화 히스토리 저장소 (메모리 dict)
store = {}

# 1. 세션별 대화 이력 객체 반환
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 2. 문서 로드 + 벡터스토어 생성 + retriever 반환
def get_retriever():
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # Ollama Embeddings 인스턴스 (OpenAI Embeddings 대체)
    embedding = OllamaEmbeddings(
        model=MODEL_EMBED,
        base_url=OLLAMA_HOST,
    )

    index_path = os.path.join(VECTOR_DIR, "index.faiss")
    if os.path.exists(index_path):
        # allow_dangerous_deserialization=True 필요 (FAISS 메타 로드)
        vectorstore = FAISS.load_local(
            VECTOR_DIR,
            embedding,
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

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
                    # 출처 정보 삽입
                    citation = f"\n\n(출처: {manual_name} {i + 1}페이지)"
                    page.page_content += citation
                    documents.append(page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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
        temperature=0.2,
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
        "당신은 Xperp 프로그램에 대한 전문 상담 챗봇입니다.\n"
        "사용자는 Xperp의 사용법, 기능, 오류 해결 등에 대해 질문합니다.\n"
        "다음 문서를 기반으로 가장 정확하고 실무적인 답변을 제공하세요:\n"
        "1) Q/A/키워드가 포함된 QnA 문서\n"
        "2) PDF 매뉴얼 및 기타 텍스트 문서\n\n"
        "답변 구성 방식 (qna.txt 우선):\n"
        "- qna에 해당 항목이 있으면 그 A내용을 맨 앞에 요약/정리\n"
        "- 이후 매뉴얼 등으로 보완 설명\n"
        "- 문서에 출처/페이지가 있으면 (출처: 문서명 N페이지) 표기. 추측 금지\n\n"
        "✅ 정식 답변 / ✅ 간단 요약 / ✅ 사용법 안내 / ✅ 유의사항 / ✅ 추가 설명 / ✅ 예상 질문(문서 근거 있을 때만)\n"
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

# 7. 최종 답변 생성 함수 (질문 → 답변)
def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"input": dictionary_chain} | rag_chain

    ai_response = tax_chain.stream(
        {"question": user_message},
        config={"configurable": {"session_id": "abc123"}},
    )
    return ai_response
