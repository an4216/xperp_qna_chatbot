# llm.py

# =========================================
# LangChain 및 관련 라이브러리 import
# =========================================
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict
import os, time, re, json
from functools import wraps

from dotenv import load_dotenv
from rapidfuzz import fuzz

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

from config import answer_examples

# =========================================
# 함수 실행시간 측정 데코레이터
# =========================================
def log_execution_time(func):
    """함수 실행 시간을 서버 로그로 출력하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[TIMING] {func.__name__}: {elapsed*1000:.0f}ms")
        return result
    return wrapper

# =========================================
# 환경설정 (.env 로드)
# =========================================
load_dotenv()

VLLM_BASE_URL   = os.getenv("VLLM_BASE_URL")
MODEL_LLM       = os.getenv("MODEL_LLM")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "not-needed")

TOP_K           = int(os.getenv("TOP_K", "4"))
VECTOR_DIR      = os.getenv("VECTOR_DIR", "vectorstore")
MENU_FILE_PATH  = os.getenv("MENU_FILE_PATH", "docs/menus/menus.txt")
META_PATH       = Path("data/artifacts/index_meta.json")
USE_HYDE        = os.getenv("USE_HYDE", "true").lower() == "true"
USE_RERANK      = os.getenv("USE_RERANK", "true").lower() == "true"
RERANK_TOP_K    = int(os.getenv("RERANK_TOP_K", "10"))  # rerank 전 검색할 문서 수
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))  # 대화 이력 최대 메시지 수

# =========================================
# 세션 관리
# =========================================
class SessionStore:
    """세션 기반 대화 기록 관리 (LRU + Timeout)"""
    def __init__(self, max_sessions=100, session_timeout=3600):
        self.store = OrderedDict()
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.session_timestamps = {}

    def get_session(self, session_id: str):
        self._cleanup_old_sessions()

        if session_id not in self.store:
            if len(self.store) >= self.max_sessions:
                oldest_session = next(iter(self.store))
                del self.store[oldest_session]
                self.session_timestamps.pop(oldest_session, None)

            self.store[session_id] = ChatMessageHistory()

        self.session_timestamps[session_id] = time.time()
        self.store.move_to_end(session_id)  # LRU 갱신

        # 대화 이력 길이 제한 적용
        self._trim_history(session_id)

        return self.store[session_id]

    def _trim_history(self, session_id: str):
        """대화 이력을 최근 N개 메시지로 제한"""
        if session_id in self.store:
            chat_history = self.store[session_id]
            messages = chat_history.messages

            if len(messages) > MAX_HISTORY_MESSAGES:
                # 최근 MAX_HISTORY_MESSAGES개만 유지
                chat_history.clear()
                for msg in messages[-MAX_HISTORY_MESSAGES:]:
                    chat_history.add_message(msg)

    def _cleanup_old_sessions(self):
        current_time = time.time()
        expired = [
            sid for sid, ts in self.session_timestamps.items()
            if current_time - ts > self.session_timeout
        ]
        for sid in expired:
            self.store.pop(sid, None)
            self.session_timestamps.pop(sid, None)
            print(f"[INFO] 만료된 세션 제거: {sid}")

session_store = SessionStore()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return session_store.get_session(session_id)

# =========================================
# 전역 캐싱 변수
# =========================================
_cached_embeddings = None
_cached_retriever = None
_cached_llm = None
_cached_rag_chain = None
_cached_vectorstore = None
_cached_fingerprint = None
_cached_menu_dict = None
_cached_hyde_chain_simple = None
_cached_hyde_chain_detailed = None
_cached_rag_chain_simple = None
_cached_rag_chain_detailed = None
_cached_reranker = None

# =========================================
# Reranker (재순위화)
# =========================================
class CrossEncoderReranker:
    """Cross-encoder 기반 문서 재순위화"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 4):
        self.model_name = model_name
        self.top_k = top_k
        self.model = None
        self.device = self._get_device()

    def _get_device(self):
        """사용 가능한 디바이스 확인"""
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[INFO] GPU 감지됨: {torch.cuda.get_device_name(0)}")
                return "cuda"
            else:
                print("[INFO] GPU 없음, CPU 사용")
                return "cpu"
        except ImportError:
            print("[INFO] PyTorch 없음, CPU 사용")
            return "cpu"

    def _load_model(self):
        """모델 로드 (최초 1회만)"""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[INFO] Reranker 모델 로드 중 (device: {self.device})...")
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=512,
                    device=self.device
                )
                print(f"[INFO] Reranker 모델 로드 완료")
            except ImportError:
                print("[ERROR] sentence-transformers 미설치. pip install sentence-transformers 필요")
                raise
            except Exception as e:
                print(f"[ERROR] Reranker 모델 로드 실패: {e}")
                raise

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """문서를 재순위화하여 상위 k개 반환"""
        if not documents:
            return documents

        self._load_model()

        # 질문-문서 쌍 생성
        pairs = [[query, doc.page_content] for doc in documents]

        # 점수 계산 (배치 크기 지정으로 최적화)
        scores = self.model.predict(
            pairs,
            batch_size=32,  # 배치 크기 증가로 처리 속도 향상
            show_progress_bar=False
        )

        # 점수와 문서 결합 후 정렬
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 선택
        reranked_docs = [doc for doc, score in doc_score_pairs[:self.top_k]]

        return reranked_docs

@log_execution_time
def get_reranker():
    """Reranker 인스턴스 반환 (캐시)"""
    global _cached_reranker
    if _cached_reranker is None:
        _cached_reranker = CrossEncoderReranker(top_k=TOP_K)
    return _cached_reranker

# =========================================
# 임베딩 & 벡터스토어
# =========================================
@log_execution_time
def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        # GPU 사용 가능 여부 확인
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] 임베딩 모델 device: {device}")
        except ImportError:
            device = "cpu"
            print(f"[INFO] 임베딩 모델 device: {device}")

        _cached_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _cached_embeddings

def _load_metadata():
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARNING] 메타데이터 로드 실패: {e}")
        return {}

def _load_fingerprint():
    return _load_metadata().get("docs_fingerprint")

class RerankRetriever(BaseRetriever):
    """Reranking을 적용하는 Retriever 래퍼"""

    base_retriever: object
    reranker: object
    search_k: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """검색 후 재순위화"""
        # 1차 검색 (더 많은 문서)
        search_start = time.perf_counter()
        docs = self.base_retriever.invoke(query)
        search_time = time.perf_counter() - search_start

        # Reranking 적용
        if self.reranker and len(docs) > 0:
            rerank_start = time.perf_counter()
            docs = self.reranker.rerank_documents(query, docs)
            rerank_time = time.perf_counter() - rerank_start
            print(f"[TIMING] 벡터검색: {search_time*1000:.0f}ms, 리랭킹: {rerank_time*1000:.0f}ms")
        else:
            print(f"[TIMING] 벡터검색: {search_time*1000:.0f}ms")

        return docs

@log_execution_time
def get_retriever():
    global _cached_retriever, _cached_vectorstore, _cached_fingerprint
    os.makedirs(VECTOR_DIR, exist_ok=True)
    current_fp = _load_fingerprint()

    if _cached_retriever is None or _cached_fingerprint != current_fp:
        index_path = os.path.join(VECTOR_DIR, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ 벡터스토어 없음: {index_path}. 01_ingest.py 실행 필요")

        embeddings = get_embeddings()
        _cached_vectorstore = FAISS.load_local(
            VECTOR_DIR, embeddings, allow_dangerous_deserialization=True
        )

        # Rerank 사용 여부에 따라 retriever 설정
        if USE_RERANK:
            base_retriever = _cached_vectorstore.as_retriever(search_kwargs={'k': RERANK_TOP_K})
            reranker = get_reranker()
            _cached_retriever = RerankRetriever(
                base_retriever=base_retriever,
                reranker=reranker,
                search_k=RERANK_TOP_K
            )
        else:
            _cached_retriever = _cached_vectorstore.as_retriever(search_kwargs={'k': TOP_K})

        _cached_fingerprint = current_fp
    return _cached_retriever

# =========================================
# LLM 초기화
# =========================================
@log_execution_time
def get_llm():
    global _cached_llm
    if _cached_llm is None:
        _cached_llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=MODEL_LLM,
            timeout=120.0,  # 타임아웃 120초로 증가 (긴 응답 대응)
            max_retries=2,  # 실패 시 2번까지 재시도
        )
    return _cached_llm

# =========================================
# Dictionary 기반 질문 보정
# =========================================
@log_execution_time
def load_menu_dict(path=MENU_FILE_PATH):
    global _cached_menu_dict
    if _cached_menu_dict is not None:
        return _cached_menu_dict

    if not Path(path).exists():
        raise FileNotFoundError(f"❌ 메뉴 파일 없음: {path}")

    with open(path, "r", encoding="utf-8") as f:
        _cached_menu_dict = json.load(f)

    print(f"[INFO] MENU_DICT 최초 로드 완료 (size={len(_cached_menu_dict)})")
    return _cached_menu_dict

def tokenize_korean(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text)

def rewrite_with_dictionary(question: str, dictionary: dict, threshold: int = 80) -> str:
    tokens = tokenize_korean(question)
    best_match, best_score, best_category = None, 0, None

    for category, keywords in dictionary.items():
        for kw in keywords:
            if question.strip() == kw:
                return f"[{category}] {question}"
            if len(kw) > 1 and kw in tokens:
                return f"[{category}] {question}"
            score = fuzz.partial_ratio(kw, question)
            if score > best_score:
                best_match, best_score, best_category = kw, score, category

    if best_match and best_score >= threshold:
        return f"[{best_category}] {question} (※ {best_match} 로 인식)"
    return question

def get_dictionary_chain():
    llm = get_llm()
    template = """
    너는 '질문 재작성기' 역할을 한다. 사전(dictionary)의 키워드와 대메뉴 정보를 참고해 사용자의 질문을 더 명확하고 구체적인 "질문 문장"으로 다시 작성한다.

    규칙:
    - 반드시 질문 형태로 출력한다. (답변 금지)
    - 질문을 구체적으로 만들어라. ("왜 그런지", "제공된 문서들을 기반으로 설명해주세요" 등을 붙여라)
    - 대메뉴 태그가 있으면 질문에 포함시켜라.
    - 불필요하게 길게 풀지 말고, 한 문장 안에서 간결하지만 구체적으로 표현해라.

    예시:
    입력: [입주자] 차량등록은 어디서해?
    출력: 입주자 메뉴에서 차량등록은 어디서 하는지 제공된 문서들을 기반으로 구체적으로 설명해주세요.
    입력: [수납] 연체료는 어디서 확인해?
    출력: 수납 메뉴에서 연체료는 어디서 확인하는지 제공된 문서들을 기반으로 알려주세요.
    입력: [입주자] 중간정산할때 전기검침 사용량입력 후 계산을 하면 금액이 안맞아요
    출력: 입주자 메뉴에서 중간정산 시 전기검침 사용량 입력 후 계산 금액이 왜 맞지 않는지 제공된 문서들에 기반해서 구체적으로 설명해주세요.

    [사전] {dictionary}
    [사용자질문] {question}
    """.strip()

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

@log_execution_time
def process_question(question: str):
    dictionary = load_menu_dict()   # ✅ 최초 1회만 로드, 이후 캐시 사용
    rewritten = rewrite_with_dictionary(question, dictionary)

    if rewritten != question:
        try:
            dict_chain = get_dictionary_chain()
            llm_rewrite = dict_chain.invoke({
                "dictionary": json.dumps(dictionary, ensure_ascii=False, indent=2),
                "question": rewritten
            })
            if llm_rewrite:
                return llm_rewrite
        except Exception as e:
            print(f"[WARN] 2차 보정 실패: {e}")
        return rewritten

    return question

# =========================================
# RAG 체인
# =========================================
@log_execution_time
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def _get_simple_prompt():
    """단순 질문용 프롬프트 (간단한 답변만)"""
    system_prompt = (
        """
        당신은 Xperp 프로그램에 대한 전문 상담 챗봇입니다.
        사용자의 질문에 대해 문서 기반으로 **간단하고 명확하게** 답변하세요.

        답변 규칙:
        - 질문에 대한 핵심 정보만 간결하게 제공하세요
        - 불필요한 섹션 구조(### 알려드릴게요 등)는 사용하지 마세요
        - 전화번호, 주소, 이메일 등은 그대로 제공하세요
        - 1-3문장 이내로 간단히 답변하세요
        - 출처가 있다면 간단히 표기하세요

        예시:
        질문: 고객센터 전화번호 알려줘
        답변: 고객센터 전화번호는 1588-1234입니다.

        질문: 영업시간은?
        답변: 평일 오전 9시부터 오후 6시까지입니다.

        ✅ 문서에 정보가 없는 경우:
        - '죄송합니다. 해당 정보를 찾을 수 없습니다. XpERP 관련 다른 질문이 있으시면 말씀해주세요.'

         {context}
        """
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def _get_detailed_prompt():
    """상세 질문용 프롬프트 (템플릿 기반 답변)"""
    examples = [
        {"input": ex.get("input", ""), "answer": re.sub(r"\(출처:[^)]+\)", "", ex.get("answer", ""))}
        for ex in answer_examples
    ]
    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{answer}")])
    few_shot = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)

    system_prompt = (
            """
            당신은 Xperp 프로그램에 대한 전문 상담 챗봇입니다.
            사용자는 Xperp의 사용법, 기능, 오류 해결 등에 대해 질문합니다.
            당신의 임무는 아래 문서를 기반으로 가장 정확하고 실무적인 답변을 제공하는 것입니다:
            1) 질문(Q)과 답변(A), 키워드(T)가 포함된 QnA 문서
            2) PDF 매뉴얼 및 기타 텍스트 설명 문서

            답변 구성 방식 (qna.txt 우선):
           - 사용자의 질문이 qna.txt 문서에 존재하거나 키워드를 참고하여 유사한 항목이 있다면, 해당 A 내용을 우선적으로 정리하여 답변의 맨 처음에 제공합니다.
           - 이후 PDF 매뉴얼 등 기타 문서를 참고하여 보완 설명을 이어서 작성합니다.
           - 문서에 따라 아래 형식을 기준으로 정돈된 답변을 가독성을 고려하여 작성하세요:

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

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def _get_qa_prompt(question_type: str = "detailed"):
    """질문 유형에 따라 적절한 프롬프트 반환"""
    if question_type == "simple":
        return _get_simple_prompt()
    else:
        return _get_detailed_prompt()

@log_execution_time
def get_hyde_chain(question_type: str = "detailed"):
    """HyDE 전용 캐시된 chain (검색 결과를 직접 전달)"""
    global _cached_hyde_chain_simple, _cached_hyde_chain_detailed

    if question_type == "simple":
        if _cached_hyde_chain_simple:
            return _cached_hyde_chain_simple
        llm = get_llm()
        qa_prompt = _get_qa_prompt("simple")
        _cached_hyde_chain_simple = create_stuff_documents_chain(llm, qa_prompt)
        return _cached_hyde_chain_simple
    else:
        if _cached_hyde_chain_detailed:
            return _cached_hyde_chain_detailed
        llm = get_llm()
        qa_prompt = _get_qa_prompt("detailed")
        _cached_hyde_chain_detailed = create_stuff_documents_chain(llm, qa_prompt)
        return _cached_hyde_chain_detailed

@log_execution_time
def get_rag_chain(question_type: str = "detailed"):
    global _cached_rag_chain_simple, _cached_rag_chain_detailed

    if question_type == "simple":
        if _cached_rag_chain_simple:
            return _cached_rag_chain_simple
        llm = get_llm()
        qa_prompt = _get_qa_prompt("simple")
        retriever = get_history_retriever()
        chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, chain)
        _cached_rag_chain_simple = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        ).pick("answer")
        return _cached_rag_chain_simple
    else:
        if _cached_rag_chain_detailed:
            return _cached_rag_chain_detailed
        llm = get_llm()
        qa_prompt = _get_qa_prompt("detailed")
        retriever = get_history_retriever()
        chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, chain)
        _cached_rag_chain_detailed = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        ).pick("answer")
        return _cached_rag_chain_detailed

# =========================================
# 질문 유형 분류
# =========================================
@log_execution_time
def classify_question_type(question: str) -> str:
    """
    질문 유형을 분류
    - 'simple': 단순 사실 확인 질문 (전화번호, 주소, 간단한 정보 조회만)
    - 'detailed': 상세 설명이 필요한 질문 (절차, 방법, 오류 해결)
    """
    # ✅ 상세 질문 패턴 먼저 체크 (우선순위)
    detailed_patterns = [
        r"어떻게|방법|절차|어찌",
        r"왜|이유|원인",
        r"안.*돼|안.*되|오류|에러|문제|안.*나와",
        r"어디서.*하|어떻게.*하|어디.*입력|어디.*등록",
        r"중간정산|계산|입력|등록|수정|삭제|조회|확인.*하|설정",
        r"~하려면|~하는.*방법|~할.*때",
    ]

    for pattern in detailed_patterns:
        if re.search(pattern, question):
            return "detailed"

    # ✅ 단순 질문 패턴 (조회성 키워드와 함께 있을 때만)
    simple_patterns = [
        r"(전화|연락).*번호.*(알려|뭐|몇|무엇)",  # 전화번호 알려줘
        r"(이메일|메일).*(알려|뭐|무엇)",  # 이메일 알려줘
        r"^(언제|어디|누구|몇.*시)$",  # 단독 의문사
        r"^.{1,8}$",  # 8글자 이하 매우 짧은 질문
        r"(고객|상담).*센터.*(번호|연락)",  # 고객센터 번호
    ]

    for pattern in simple_patterns:
        if re.search(pattern, question):
            return "simple"

    # 기본값: 중간 길이 이상은 상세 질문으로 처리
    if len(question) >= 12:
        return "detailed"

    return "simple"

# =========================================
# HyDE (Hypothetical Document Embeddings)
# =========================================
@log_execution_time
def should_use_hyde(question: str) -> bool:
    """
    HyDE를 사용할지 판단
    - 간단한 키워드 질문은 HyDE 불필요
    - 복잡하거나 모호한 질문은 HyDE 사용
    """
    # 간단한 키워드 질문 패턴 (HyDE 불필요)
    simple_patterns = [
        r"전화번호",
        r"고객센터",
        r"문의.*어디",
        r"\d{3,4}-\d{4}",  # 전화번호 포함
        r"^.{1,5}$",  # 5글자 이하 짧은 질문
    ]

    for pattern in simple_patterns:
        if re.search(pattern, question):
            return False

    # 복잡한 질문 패턴 (HyDE 사용)
    complex_patterns = [
        r"어떻게|방법|어디서|왜|이유",  # How, Why 질문
        r"안.*돼|오류|에러|문제",  # 문제/오류 관련
        r".*는.*는",  # 복합 질문 ("이거는 저거는")
    ]

    for pattern in complex_patterns:
        if re.search(pattern, question):
            return True

    # 기본값: 중간 길이 질문은 HyDE 사용
    if len(question) >= 10:
        return True

    return False

@log_execution_time
def hyde_transform(question: str, max_retries: int = 2) -> str:
    """
    질문을 가상의 답변으로 변환 (재시도 로직 포함)
    """
    llm = get_llm()

    prompt = f"""
    당신은 Xperp 프로그램 전문가입니다.
    다음 질문에 대한 답변을 **상상해서** 작성하세요.

    중요 규칙:
    - 실제로 정확한 정보가 아니어도 괜찮습니다
    - Xperp 매뉴얼에 있을 법한 답변의 "형식과 스타일"만 맞추면 됩니다
    - 메뉴 경로, 절차, 주의사항 등을 포함하여 자연스럽게 작성하세요
    - 100-200단어 정도로 작성하세요

    질문: {question}

    가상 답변:
    """.strip()

    for attempt in range(max_retries):
        try:
            result = llm.invoke(prompt)
            hypothetical_answer = result.content
            return hypothetical_answer

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 지수 백오프: 1초, 2초
                time.sleep(wait_time)
            else:
                return question  # 모든 재시도 실패 시 원본 질문 반환

    return question

# =========================================
# 최종 응답 함수
# =========================================
def get_ai_response(user_message: str, session_id: str, use_hyde: bool = None):
    if not session_id:
        raise ValueError("session_id is required.")

    # HyDE 기본값: 환경변수 사용
    if use_hyde is None:
        use_hyde = USE_HYDE

    # 1. 질문 보정
    try:
        effective_message = process_question(user_message)
    except Exception as e:
        effective_message = user_message

    # 1.5. 질문 유형 분류 (단순 vs 상세)
    question_type = classify_question_type(effective_message)

    # 2. HyDE 적용 여부 판단 및 실행
    search_query = effective_message  # 기본값: 보정된 질문으로 검색

    if use_hyde and should_use_hyde(effective_message):
        try:
            # HyDE: 가상 답변 생성
            hypothetical_answer = hyde_transform(effective_message)
            search_query = hypothetical_answer  # 가상 답변으로 검색

            # 디버깅용 출력 (필요시 주석 해제)
            # yield f"🔍 HyDE 변환됨\n가상 답변: {hypothetical_answer[:150]}...\n\n"

        except Exception as e:
            print(f"[WARN] HyDE 변환 실패, 원본 질문 사용: {e}")
            search_query = effective_message

    # 3. RAG 체인 실행
    # HyDE를 사용한 경우, 검색은 가상 답변으로 하되 최종 응답은 원본 질문 기반

    if use_hyde and search_query != effective_message:
        # HyDE 모드: 가상 답변으로 1회 검색 → 캐시된 chain으로 답변 생성
        retriever = get_retriever()
        docs = retriever.invoke(search_query)

        # 캐시된 HyDE chain 사용 (질문 유형에 따라 선택)
        hyde_chain = get_hyde_chain(question_type)
        chat_history = get_session_history(session_id)

        # 스트리밍 답변 생성
        stream = hyde_chain.stream({
            "input": effective_message,  # 원본 질문
            "context": docs,  # 검색된 문서
            "chat_history": chat_history.messages
        })
    else:
        # 일반 모드 (질문 유형에 따라 선택)
        rag_chain = get_rag_chain(question_type)

        stream = rag_chain.stream(
            {"input": effective_message},
            config={"configurable": {"session_id": session_id}},
        )

    # 스트림 처리 및 대화 이력 저장 (에러 핸들링 강화)
    full_response = ""
    try:
        for chunk in stream:
            full_response += chunk
            yield chunk

        # HyDE 모드일 때 대화 이력 수동 저장
        if use_hyde and search_query != effective_message:
            chat_history = get_session_history(session_id)
            chat_history.add_user_message(effective_message)
            chat_history.add_ai_message(full_response)

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)[:200]
        print(f"[ERROR] 스트리밍 중 오류 발생: {error_type} - {error_msg}")

        # 부분 응답이라도 있으면 저장
        if full_response:
            if use_hyde and search_query != effective_message:
                chat_history = get_session_history(session_id)
                chat_history.add_user_message(effective_message)
                chat_history.add_ai_message(full_response)
            yield f"\n\n⚠️ 응답 중 연결이 끊어졌습니다. 부분 응답만 표시됩니다."
        else:
            yield f"\n\n❌ 오류가 발생했습니다: {error_type}\n서버 연결을 확인하고 다시 시도해주세요."

# =========================================
# 유틸
# =========================================
def cleanup_resources():
    global _cached_embeddings, _cached_retriever, _cached_llm, _cached_rag_chain, _cached_fingerprint
    global _cached_hyde_chain_simple, _cached_hyde_chain_detailed
    global _cached_rag_chain_simple, _cached_rag_chain_detailed, _cached_reranker

    _cached_embeddings = _cached_retriever = _cached_llm = _cached_rag_chain = _cached_fingerprint = None
    _cached_hyde_chain_simple = _cached_hyde_chain_detailed = None
    _cached_rag_chain_simple = _cached_rag_chain_detailed = None
    _cached_reranker = None

    session_store.store.clear()
    session_store.session_timestamps.clear()
    print("[INFO] 모든 캐시 및 세션 리소스 정리 완료")

def get_cache_info():
    return {
        "embeddings_cached": _cached_embeddings is not None,
        "retriever_cached": _cached_retriever is not None,
        "llm_cached": _cached_llm is not None,
        "rag_chain_cached": _cached_rag_chain is not None,
        "hyde_chain_simple_cached": _cached_hyde_chain_simple is not None,
        "hyde_chain_detailed_cached": _cached_hyde_chain_detailed is not None,
        "rag_chain_simple_cached": _cached_rag_chain_simple is not None,
        "rag_chain_detailed_cached": _cached_rag_chain_detailed is not None,
        "reranker_cached": _cached_reranker is not None,
        "use_rerank": USE_RERANK,
        "fingerprint": _cached_fingerprint,
        "active_sessions": len(session_store.store)
    }
