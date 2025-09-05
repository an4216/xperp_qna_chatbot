# llm.py

# LangChain 및 관련 라이브러리 import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ✅ vLLM(OpenAI 호환)용 LLM
from langchain_openai import ChatOpenAI
# ✅ HuggingFace bge-m3 임베딩
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from pathlib import Path
from dotenv import load_dotenv
from collections import OrderedDict
from typing import List, Dict
from config import answer_examples
import os
import time
import re
import json

# =========================================
# 환경설정 (.env 로드)
# =========================================
load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")  # .env 관리
MODEL_LLM     = os.getenv("MODEL_LLM")      # .env 관리
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")  # .env 관리

TOP_K       = int(os.getenv("TOP_K", "4"))              # 기본값 유지 가능
VECTOR_DIR  = os.getenv("VECTOR_DIR", "vectorstore")    # 기본값 유지 가능

# 세션별 대화 히스토리 저장소 (리소스 관리 개선)
class SessionStore:
    """리소스 관리가 가능한 세션 저장소"""
    def __init__(self, max_sessions=100, session_timeout=3600):
        self.store = OrderedDict()
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.session_timestamps = {}

    def get_session(self, session_id: str):
        self._cleanup_old_sessions()

        if session_id not in self.store:
            if len(self.store) >= self.max_sessions:
                # LRU 방식으로 가장 오래된 세션 제거
                oldest_session = next(iter(self.store))
                del self.store[oldest_session]
                self.session_timestamps.pop(oldest_session, None)

            self.store[session_id] = ChatMessageHistory()

        self.session_timestamps[session_id] = time.time()
        # Move to end (LRU)
        self.store.move_to_end(session_id)
        return self.store[session_id]

    def _cleanup_old_sessions(self):
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, timestamp in self.session_timestamps.items()
            if current_time - timestamp > self.session_timeout
        ]

        for session_id in expired_sessions:
            self.store.pop(session_id, None)
            self.session_timestamps.pop(session_id, None)
            print(f"[INFO] 만료된 세션 제거: {session_id}")

# 전역 세션 저장소 인스턴스
session_store = SessionStore()

# =========================================
# 전역 캐싱 변수
# =========================================
_cached_embeddings = None
_cached_retriever = None
_cached_llm = None
_cached_rag_chain = None

META_PATH = Path("data/artifacts/index_meta.json")
_cached_fingerprint = None

# 정규표현식 컴파일 (성능 최적화)
MANUAL_REF_PATTERN = re.compile(r'^\s*✅\s*매뉴얼\s*참조:.*$', re.MULTILINE)
SOURCE_PATTERN = re.compile(r'\(출처:\s*[^)]+\)')
PAGE_REF_PATTERN = re.compile(r'[(（]?\s*[^)\n]*매뉴얼[^)\n]*\d+\s*페이지\s*참조[)）]?')
WHITESPACE_PATTERN = re.compile(r'\n{3,}')

def _load_metadata():
    """메타데이터 로드 통합 함수"""
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"[WARNING] 메타데이터 로드 실패: {e}")
        return {}
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류: {e}")
        return {}

def _load_fingerprint():
    """문서 핑거프린트 로드"""
    return _load_metadata().get("docs_fingerprint")

# -------------------------------
# 유틸: few-shot 예시의 '출처' 문구 제거 (성능 최적화)
# -------------------------------
def sanitize_examples(examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    start = time.perf_counter()
    sanitized = []
    for ex in examples:
        inp = ex.get("input", "")
        ans = ex.get("answer", "")

        ans = MANUAL_REF_PATTERN.sub('', ans)
        ans = SOURCE_PATTERN.sub('', ans)
        ans = PAGE_REF_PATTERN.sub('', ans)
        ans = WHITESPACE_PATTERN.sub('\n\n', ans).strip()

        sanitized.append({"input": inp, "answer": ans})
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[TIMER] sanitize_examples 완료 ({elapsed:.2f} ms)")
    return sanitized

# 1. 세션별 대화 이력 객체 반환 (리소스 관리 개선)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return session_store.get_session(session_id)

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

    os.makedirs(VECTOR_DIR, exist_ok=True)
    current_fp = _load_fingerprint()

    # retriever가 없거나, fingerprint가 바뀐 경우만 새로 로드
    if _cached_retriever is None or _cached_fingerprint != current_fp:
        print(f"[INFO] retriever reload triggered (old={_cached_fingerprint}, new={current_fp})")

        index_path = os.path.join(VECTOR_DIR, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ 벡터스토어 없음: {index_path}. 먼저 01_ingest.py 실행 필요")

        try:
            print(f"[DEBUG] 임베딩 로드 시작...")
            embeddings = get_embeddings()
            print(f"[DEBUG] 임베딩 로드 완료, FAISS 로드 시작...")

            vectorstore = FAISS.load_local(
                VECTOR_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[DEBUG] FAISS 로드 완료")
        except Exception as e:
            print(f"[ERROR] 벡터스토어 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

        _cached_retriever = vectorstore.as_retriever(search_kwargs={'k': TOP_K})
        _cached_fingerprint = current_fp
    else:
        print(f"[INFO] retriever reuse (fingerprint={_cached_fingerprint})")

    return _cached_retriever

# 3. LLM(챗봇) 인스턴스 생성 → vLLM(OpenAI 호환) (전역 캐싱)
def get_llm():
    global _cached_llm
    if _cached_llm is None:
        _cached_llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=MODEL_LLM,
        )
    return _cached_llm

# 4. 대화 맥락을 반영한 retriever 반환 (standalone question 변환 + 벡터검색)
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

    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# 5. RAG 체인 (전역 캐싱)
def get_rag_chain():
    global _cached_rag_chain
    if _cached_rag_chain is not None:
        return _cached_rag_chain

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
        당신의 임무는 아래 문서를 기반으로 가장 정확하고 실무적인 답변을 제공하는 것입니다:
        1) 질문(Q)과 답변(A), 키워드(T)가 포함된 QnA 문서
        2) PDF 매뉴얼 및 기타 텍스트 설명 문서

        답변 작성 규칙:
        - 답변 템플릿은 다음 중 하나를 선택하세요:
          1) [알려드릴게요 + 짧게 요약하면] → 단순 개념/이유 설명
          2) [알려드릴게요 + 이렇게 해보세요] → 메뉴 경로나 절차 안내
          3) [알려드릴게요 + 짧게 요약하면 + 꼭 알아두세요] → 오류/주의사항 관련
        - 질문 성격에 따라 가장 적절한 템플릿을 사용하세요.
        - 불필요한 섹션은 포함하지 마세요.

        ### 각 섹션 작성 상세 지침:
        - `### 알려드릴게요`
          - 문서를 기반으로 질문의 개념, 목적, 동작 원리를 상세히 설명합니다.
          - 실무자가 오해할 수 있는 지점이나 자주 묻는 상황도 함께 안내합니다.
          - 한 문장이 끝나면 줄바꿈을 통해 가독성을 높여주세요.

        - `### 짧게 요약하면`
          - 핵심 개념을 1~2줄 이내로 정리합니다.

        - `### 이렇게 해보세요`
          1. 메뉴 경로, 설정 방법, 입력 절차를 문서에 있는 내용으로 단계별로 작성하세요.
          2. 화면 위치 정보도 가능한 경우 포함합니다.
          3. 메뉴 경로는 절대 유추하지말고 문서에 있는 내용으로만 답변해주세요.

        - `### 꼭 알아두세요`
          - 실무 중 자주 발생하는 실수나 예외 상황, 기능 제약사항 등을 구체적으로 기술합니다.
          - 사용자가 놓치기 쉬운 조건이나 확인 항목도 함께 제시하세요.

        ### 매뉴얼 참조 출력 지침:
        - 반드시 'context'의 문서 metadata(source/page)에서만 출처를 가져오세요.
        - few-shot 예시 안의 출처/페이지 표기는 무시하세요.
        - 문서명이나 페이지를 임의로 추측하거나 생성하지 마세요.

        출력 형식 규칙(매우 중요):
        - 반드시 Markdown을 사용하세요.
        - 각 섹션 제목은 무조건 `### 제목` 형식을 사용하세요.
        - 본문에는 `**굵게**` 마크다운을 사용하지 마세요. (제만 굵게)
        - 한 문장이 끝나면 줄바꿈을 통해 가독성을 높이세요.

        ✅ 질문과 직접 관련된 XPERP 정보가 없거나, 문서에서 근거를 찾을 수 없는 경우:
        - '죄송합니다. 해당 내용은 현재 안내드릴 수 있는 범위를 벗어난 항목입니다.\n
        본 챗봇 서비스는 XpERP 사용과 관련한 답변만 제공하도록 설계되어 있습니다.\n
        XpERP와 관련한 질의가 있으시면 다시 질문해주시길 바랍니다.'

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

    return _cached_rag_chain

# 6. 최종 답변 생성 함수 (FE가 준 session_id 사용)
def get_ai_response(user_message: str, session_id: str):
    """
    AI 답변 생성 함수 (스트리밍 지원 + 소요시간 표시)
    - FE에서 받은 session_id로 히스토리를 이어서 관리
    """
    rag_chain = get_rag_chain()

    if not session_id:
        raise ValueError("session_id is required (FE에서 생성하여 전달하세요).")

    start_time = time.perf_counter()

    stream = rag_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )

    for chunk in stream:
        yield chunk

    elapsed = time.perf_counter() - start_time
    yield f"\n\n⏱ 소요시간: {elapsed:.2f}초"

# 리소스 관리 함수들
def cleanup_resources():
    """전역 캐시 및 세션 리소스 정리"""
    global _cached_embeddings, _cached_retriever, _cached_llm, _cached_rag_chain, _cached_fingerprint

    _cached_embeddings = None
    _cached_retriever = None
    _cached_llm = None
    _cached_rag_chain = None
    _cached_fingerprint = None

    # 세션 저장소 정리
    session_store.store.clear()
    session_store.session_timestamps.clear()

    print("[INFO] 모든 캐시 및 세션 리소스 정리 완료")

def get_cache_info():
    """캐시 상태 정보 반환"""
    cache_status = {
        "embeddings_cached": _cached_embeddings is not None,
        "retriever_cached": _cached_retriever is not None,
        "llm_cached": _cached_llm is not None,
        "rag_chain_cached": _cached_rag_chain is not None,
        "fingerprint": _cached_fingerprint,
        "active_sessions": len(session_store.store)
    }
    return cache_status
