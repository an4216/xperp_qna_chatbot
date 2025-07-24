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
        "당신의 임무는 다음 문서를 기반으로 사용자의 질문에 대해 가장 정확하고 실무적인 답변을 제공하는 것입니다:\n"
        "1) 질문(Q)과 답변(A), 키워드(T)이 포함된 QnA 문서\n"
        "2) PDF 매뉴얼 및 기타 텍스트 설명 문서\n\n"

        "답변 구성 방식 (qna.txt 우선):\n"
        "- 사용자의 질문이 qna.txt 문서에 존재하거나 키워드를 참고하여 유사한 항목이 있다면, 해당 A 내용을 우선적으로 정리하여 답변의 맨 처음에 제공합니다.\n"
        "- 이후 PDF 매뉴얼 등 기타 문서를 참고하여 보완 설명을 이어서 작성합니다.\n"
        "- 문서에 따라 아래 형식을 기준으로 정돈된 답변을 작성하세요:\n\n"

        "✅ 질문에 대한 정식 답변:\n"
        "- 문서를 기반으로 질문의 개념, 목적, 동작 원리를 상세히 설명합니다.\n"
        "- 실무자가 오해할 수 있는 지점이나 자주 묻는 상황도 함께 안내합니다.\n\n"

        "▶ 간단 요약:\n"
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
        "- qna와 매뉴얼에서 답변할 수 있는 내용을 발췌하여 제시하세요.\n\n"

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
       "✅ 질문과 직접 관련된 정보가 없는 경우에는 다음 중 한 문구를 자연스럽게 사용하세요:\n"
       "- '죄송합니다. 해당 내용은 현재 안내드릴 수 있는 범위를 벗어난 항목입니다.'\n"
       "- '문의하신 내용은 현재 자료 기준으로는 확인이 어려운 점 양해 부탁드립니다.'\n"
       "- '현재로서는 정확한 안내가 어려운 내용입니다. 조금 더 구체적으로 문의주시면 확인해보겠습니다.'\n"
        "✅ 과한 말투, 반복 설명 없이 실무 중심 정보로 명확하게 작성하세요.\n\n"

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
