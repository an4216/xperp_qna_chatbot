# LangChain 및 관련 라이브러리 import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples
import os

# 세션별 대화 히스토리 저장소 (메모리 dict)
store = {}

# 1. 세션별 대화 이력 객체 반환
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 2. 문서 로드 + 벡터스토어 생성 + retriever 반환
def get_retriever():
    from tqdm import tqdm

    # OpenAI 임베딩 모델 정의
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    documents = []
    docs_dirs = ["docs/manual", "docs/qna"]  # 매뉴얼/QA 폴더 둘 다 로드

    # 문서 로드
    for docs_dir in docs_dirs:
        for filename in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, filename)

            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())

            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

    # 문서 분할(청크)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    # ✅ 빈 문서 제거
    split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 10]

    # ✅ 최대 청크 개수 제한 (예: 1000개)
    MAX_CHUNKS = 500
    if len(split_docs) > MAX_CHUNKS:
        split_docs = split_docs[:MAX_CHUNKS]

    # ✅ FAISS 저장
    vectorstore = FAISS.from_documents(split_docs, embedding)

    # ✅ 검색기 생성
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    return retriever


# 3. 대화 맥락을 반영한 retriever 반환 (standalone question 변환 + 벡터검색)
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    # 대화 맥락+최신 질문 → 완전한 질문으로 변환 프롬프트
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

    # LangChain history aware retriever 생성
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

# 4. LLM(챗봇) 인스턴스 생성 (모델 선택 가능)
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

# 5. 사용자의 질문을 사전 기반으로 전처리 (질문 변환 체인)
def get_dictionary_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요

        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()

    return dictionary_chain

# 6. RAG(문서 기반 질의응답) 전체 체인
def get_rag_chain():
    llm = get_llm()
    # Q/A 예시(few-shot)로 LLM의 답변 퀄리티 향상
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

    # 챗봇 역할 및 답변 스타일 정의(system 프롬프트)
    system_prompt = (
        "당신은 Xperp 프로그램에 대한 전문 상담 챗봇입니다.\n"
        "사용자는 Xperp의 사용법, 기능, 오류 해결 등에 대해 질문합니다.\n"
        "당신의 임무는 문서에 포함된 질문(Q), 답변(A), 태그(T) 또는 PDF 매뉴얼을 참고하여,\n"
        "가장 정확하고 실무적으로 도움이 되는 정보를 다음 형식에 맞춰 작성하는 것입니다:\n\n"

        "질문에 대한 정식 답변:\n"
        "- 질문 의도를 명확히 파악하여 배경 설명과 함께 답변합니다.\n"
        "- 기능의 목적, 사용 예시, 적용 상황 등을 구체적으로 설명합니다.\n"
        "- 사용자 실수나 혼동이 잦은 부분은 '주의사항' 형태로 안내합니다.\n\n"

        "✅ 간단 요약:\n"
        "- 질문의 핵심 개념을 1~2줄 요약하여 제공합니다.\n\n"

        "🔹 사용법 안내:\n"
        "📌 1. 메뉴 경로: 기능이 속한 실제 화면 경로를 명시합니다.\n"
        "📌 2. 입력 절차: 항목별 입력 순서 및 설명을 번호순으로 나열합니다.\n"
        "📌 3. 예시 제공: 가능한 경우 입력 예시를 보여줍니다.\n"
        "📌 4. 계산방법 구분: 옵션(덮어쓰기/더하기/빼기 등)을 명확히 비교합니다.\n"
        "📌 5. 주의사항: 실무에서 발생할 수 있는 오류나 유의점은 강조해 표시합니다.\n\n"

        "🔸 참고사항:\n"
        "- 엑셀 업로드, 항목코드 조회, 세대구분 등 실무에 필요한 부가 정보를 함께 제공합니다.\n"
        "- 문서에 정확한 정보가 없거나 애매할 경우, 'XPERP에 관련된 상담만 확인 가능합니다.' 라고 답변합니다.\n"
        "- 설명은 친절하지만 간결하게, 불필요한 말투나 과한 반복은 지양합니다.\n\n"

        "{context}"
    )


    # QA 프롬프트(시스템 + few-shot + 히스토리 + 질문)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 대화형 retriever(standalone question → 벡터검색)
    history_aware_retriever = get_history_retriever()
    # LLM 답변 생성 체인 (문서 context 주입)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    # Retrieval → LLM QA 체인
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 대화 이력 Session별로 관리 (RunnableWithMessageHistory)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain

# 7. 최종 답변 생성 함수 (질문 → 답변)
def get_ai_response(user_message):
    # step 1: 사전 기반 질문 전처리
    dictionary_chain = get_dictionary_chain()
    # step 2: RAG 문서기반 QA 체인
    rag_chain = get_rag_chain()
    # step 3: 사전 체인 결과를 rag_chain의 input으로 연결
    tax_chain = {"input": dictionary_chain} | rag_chain

    # step 4: 답변 생성 (streaming)
    ai_response = tax_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response
