# llm.py

# LangChain 및 관련 라이브러리 import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ✅ vLLM(OpenAI 호환)용 LLM
from langchain_openai import ChatOpenAI
# ✅ HuggingFace bge-m3 임베딩
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from pathlib import Path

from config import answer_examples
import os
import time
import re
import json


# =========================================
# 환경설정 (RunPod vLLM OpenAI 호환)
# =========================================
# 반드시 /v1 포함
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "https://zc2liu1ru5cjgm-8000.proxy.runpod.net/v1")
# /v1/models 의 data[].id 값과 정확히 일치해야 함
MODEL_LLM     = os.getenv("MODEL_LLM", "unsloth/gemma-3-27b-it")
# 키 검증을 안 해도 ChatOpenAI에는 문자열이 필요 → 더미키 사용
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")

TOP_K       = int(os.getenv("TOP_K", "4"))
VECTOR_DIR  = os.getenv("VECTOR_DIR", "vectorstore")

# 세션별 대화 히스토리 저장소
store = {}

# =========================================
# 전역 캐싱 변수
# =========================================
_cached_embeddings = None
_cached_retriever = None
_cached_llm = None
_cached_rag_chain = None

META_PATH = Path("data/artifacts/index_meta.json")
_cached_retriever = None
_cached_fingerprint = None

def _load_fingerprint():
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            return meta.get("fingerprint")
        except Exception:
            return None
    return None

# -------------------------------
# 유틸: few-shot 예시의 '출처' 문구 제거
# -------------------------------
def sanitize_examples(examples: list[dict]) -> list[dict]:
    start = time.perf_counter()
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
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[TIMER] sanitize_examples 완료 ({elapsed:.2f} ms)")
    return sanitized


# 1. 세션별 대화 이력 객체 반환
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ✅ bge-m3 임베딩 인스턴스 생성 (전역 캐싱)
def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        start = time.perf_counter()
        _cached_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True}  # 코사인 유사도 안정화
        )
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[TIMER] get_embeddings 최초 로드 완료 ({elapsed:.2f} ms)")
    return _cached_embeddings


# 2. 문서 로드 + 벡터스토어 로드 (전역 캐싱)
def get_retriever():
    global _cached_retriever, _cached_fingerprint

    start = time.perf_counter()
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # ✅ index_meta.json fingerprint 로드
    current_fp = None
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            current_fp = meta.get("docs_fingerprint")   # ✅ ingest.py와 키 통일
        except Exception:
            current_fp = None

    # retriever가 없거나, fingerprint가 바뀐 경우만 새로 로드
    if _cached_retriever is None or _cached_fingerprint != current_fp:
        print(f"[INFO] retriever reload triggered (old={_cached_fingerprint}, new={current_fp})")

        index_path = os.path.join(VECTOR_DIR, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ 벡터스토어 없음: {index_path}. 먼저 01_ingest.py 실행 필요")

        # ✅ 기존 벡터스토어만 로드
        vectorstore = FAISS.load_local(
            VECTOR_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
        _cached_retriever = vectorstore.as_retriever(search_kwargs={'k': TOP_K})
        _cached_fingerprint = current_fp

        elapsed = (time.perf_counter() - start) * 1000
        print(f"[TIMER] get_retriever: 로컬 벡터스토어 로드 완료 ({elapsed:.2f} ms)")
    else:
        print(f"[INFO] retriever reuse (fingerprint={_cached_fingerprint})")

    return _cached_retriever


# 3. 대화 맥락을 반영한 retriever 반환 (standalone question 변환 + 벡터검색)
def get_history_retriever():
    start = time.perf_counter()
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
    elapsed = (time.perf_counter() - start) * 1000
    return history_aware_retriever

# 4. LLM(챗봇) 인스턴스 생성 → vLLM(OpenAI 호환) (전역 캐싱)
def get_llm():
    global _cached_llm
    if _cached_llm is None:
        start = time.perf_counter()
        _cached_llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=MODEL_LLM,
        )
        elapsed = (time.perf_counter() - start) * 1000
    return _cached_llm

# 6. RAG 체인 (전역 캐싱)
def get_rag_chain():
    global _cached_rag_chain
    if _cached_rag_chain is not None:
        return _cached_rag_chain

    start = time.perf_counter()
    llm = get_llm()

    # ✅ 예시를 클린업해서 사용
    cleaned_examples = sanitize_examples(answer_examples)

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=cleaned_examples,
    )

    system_prompt = (
        """
        당신은 Xperp 프로그램에 대한 전문 상담 챗봇입니다.
        사용자는 Xperp의 사용법, 기능, 오류 해결 등에 대해 질문합니다.
        당신의 임무는 다음 문서를 기반으로 사용자의 질문에 대해 가장 정확하고 실무적인 답변을 제공하는 것입니다:
        1) 질문(Q)과 답변(A), 키워드(T)가 포함된 QnA 문서
        2) PDF 매뉴얼 및 기타 텍스트 설명 문서

        답변 구성 방식 (qna.txt 우선):
        - 사용자의 질문이 qna.txt 문서에 존재하거나 키워드를 참고하여 유사한 항목이 있다면, 해당 A 내용을 우선적으로 정리하여 답변의 맨 처음에 제공합니다.
        - 이후 PDF 매뉴얼 등 기타 문서를 참고하여 보완 설명을 이어서 작성해 주세요.
        - 문서에 따라 아래 형식을 기준으로 정돈된 답변을 가독성을 고려하여 작성하세요:

        ### 질문에 대한 정식 답변
        - 문서를 기반으로 질문의 개념, 목적, 동작 원리를 상세히 설명합니다.
        - 실무자가 오해할 수 있는 지점이나 자주 묻는 상황도 함께 안내합니다.
        - 한 문장이 끝나면 줄바꿈을 통해 가독성을 높여주세요.
        ---

        ### 간단 요약
        - 핵심 개념을 1~2줄 이내로 정리합니다.
        ---

        ### 사용법 안내
        1. 메뉴 경로, 설정 방법, 입력 절차를 문서에 있는 내용으로 단계별로 작성하세요.
        2. 화면 위치 정보도 가능한 경우 포함합니다.
        ---

        ### 유의사항
        - 실무 중 자주 발생하는 실수나 예외 상황, 기능 제약사항 등을 구체적으로 기술합니다.
        - 사용자가 놓치기 쉬운 조건이나 확인 항목도 함께 제시하세요.
        ---


        ### 매뉴얼 참조 출력 지침:
        - 반드시 'context'의 문서 metadata(source/page)에서만 출처를 가져오세요.
        - few-shot 예시 안의 출처/페이지 표기는 무시하세요.
        - 문서명이나 페이지를 임의로 추측하거나 생성하지 마세요.
        - qna.txt를 참조한 경우는 출처 생략하세요.

        출력 형식 규칙(매우 중요):
        - 반드시 Markdown을 사용하세요.
        - 각 섹션 제목은 무조건 `### 제목` 형식을 사용하세요. (다른 굵기나 H4~H6 사용 금지)
        - 본문에는 `**굵게**` 마크다운을 사용하지 마세요. (제목만 굵게 보이도록 제한)
        - 한 문장이 끝나면 줄바꿈하여 가독성을 높이세요.

        ✅ 질문과 직접 관련된 XPERP 정보가 없거나, 학습된 문서에서 근거를 찾을 수 없는 경우:
        - '죄송합니다. 해당 내용은 현재 안내드릴 수 있는 범위를 벗어난 항목입니다.'
        - '문의하신 내용은 현재 자료 기준으로는 확인이 어려운 점 양해 부탁드립니다.'
        - '현재로서는 정확한 안내가 어려운 내용입니다. 조금 더 구체적으로 문의주시면 확인해보겠습니다.'

         {context}
        """
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

    _cached_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    elapsed = (time.perf_counter() - start) * 1000
    return _cached_rag_chain


# 7. 최종 답변 생성 함수
def get_ai_response(user_message):
    start = time.perf_counter()
    rag_chain = get_rag_chain()

    # ✅ 'input' 키로 전달
    stream = rag_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": "abc123"}},
    )

    def timed_stream():
        inner_start = time.perf_counter()
        for chunk in stream:
            yield chunk
        elapsed_inner = time.perf_counter() - inner_start
        yield f"\n\n⏱ 소요시간: {elapsed_inner:.2f}s"

    elapsed = (time.perf_counter() - start) * 1000
    return timed_stream()
