# LangChain 및 관련 라이브러리 import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ★ OpenAI → Ollama
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS
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
MODEL_LLM = "llama2:latest"          # ollama pull gpt-oss:20b
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

# 2. 문서 로드 + 벡터스토어 생성 + retriever 반환
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
            "당신의 임무는 다음 문서를 기반으로 사용자의 질문에 대해 가장 정확하고 실무적인 답변을 제공하는 것입니다:\n"
            "1) 질문(Q)과 답변(A), 키워드(T)가 포함된 QnA 문서\n"
            "2) PDF 매뉴얼 및 기타 텍스트 설명 문서\n\n"

            "답변 구성 방식 (qna.txt 우선):\n"
            "- 사용자의 질문이 qna.txt 문서에 존재하거나 키워드를 참고하여 유사한 항목이 있다면, 해당 A 내용을 우선적으로 정리하여 답변의 맨 처음에 제공합니다.\n"
            "- 이후 PDF 매뉴얼 등 기타 문서를 참고하여 보완 설명을 이어서 작성합니다.\n"
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
            "- qna와 매뉴얼에서 답변할 수 있는 내용을 발췌하여 제시하세요.\n\n"
            "- qna와 매뉴얼에 나온 예상질문이 추가로 없다면, 예상질문을 생략해주세요.\n\n"

            "✅ 매뉴얼 참조 출력 지침:\n"
            "- 출처가 있는 경우, 반드시 다음과 같은 형식으로 명시하세요:  \n"
            "  예시: (출처: 해당 메뉴의 매뉴얼 15페이지)\n"
            "- 아래 조건을 반드시 지켜야 합니다:\n"
            "  1. context에 명시된 문서 출처(source)와 페이지(page) 정보만 사용하세요.\n"
            "  2. 출처나 페이지 정보가 누락된 경우 이 항목은 생략해야 합니다.\n"
            "  3. 문서명이나 페이지를 임의로 추측하거나 생성하지 마세요.\n"
            "  4. 각 설명이 어떤 문서에서 유래했는지 사용자에게 명확히 전달해야 합니다.\n"
            "\n"
            "- 사용법안내 등의 답변 본문에서도 관련 설명 끝에 (출처: 해당 메뉴의 매뉴얼 15페이지) 형식으로 표시해 주세요.\n"
            "- 본문내용에 참조한 문서가 있을경우 매뉴얼 참조 항목은 반드시 표시해주세요.\n"


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

            "✅ 매뉴얼 참조: (출처: 해당 메뉴의 매뉴얼 15페이지)\n\n"

            "✅ 예상 질문:\n"
            "- 엑셀 자료를 등록할 때 항목코드는 어디서 확인하나요?\n"
            "- 부과 후 수정은 어떻게 하나요?\n"
            "- 특정 세대만 별도금액을 제거할 수 있나요?\n\n"

           "✅ 위 형식에 맞춰 응답을 구성하되, 매뉴얼참조 항목은 조건에 따라 생략해 주세요.\n"
           "✅ 질문과 직접 관련된 정보가 문서에 없는 경우에는 다음 중 한 문구를 자연스럽게 사용하세요:\n"
           "- '죄송합니다. 해당 내용은 현재 안내드릴 수 있는 범위를 벗어난 항목입니다.'\n"
           "- '문의하신 내용은 현재 자료 기준으로는 확인이 어려운 점 양해 부탁드립니다.'\n"
           "- '현재로서는 정확한 안내가 어려운 내용입니다. 조금 더 구체적으로 문의주시면 확인해보겠습니다.'\n"
            "✅ 과한 말투, 반복 설명 없이 실무 중심 정보로 명확하게 작성하세요.\n\n"

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
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"input": dictionary_chain} | rag_chain

    # 내부 스트림 생성
    inner_stream = tax_chain.stream(
        {"question": user_message},
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
