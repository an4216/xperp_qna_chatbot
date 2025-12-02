# llm.py

# =========================================
# LangChain 및 관련 라이브러리 import
# =========================================
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Optional, Any
import os, time, re, json
from functools import wraps
import asyncio

from dotenv import load_dotenv
from rapidfuzz import fuzz

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
import boto3
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

from src.config.config import answer_examples
from src.llm.prompts import (
    FOLLOWUP_MERGE_PROMPT,
    QUESTION_REFINE_PROMPT,
    HISTORY_RETRIEVER_PROMPT,
    SIMPLE_QA_PROMPT,
    DETAILED_QA_PROMPT,
    HYDE_TRANSFORM_PROMPT
)
from src.core.guardrails import validate_yearend_tax_topic

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

AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL   = os.getenv("BEDROCK_MODEL", "huggingface-vlm-gemma-3-27b-instruct")

TOP_K           = int(os.getenv("TOP_K", "4"))
YEAREND_KEYWORDS_FILE = os.getenv("YEAREND_KEYWORDS_FILE", "docs/yearend_keywords.txt")
USE_HYDE        = os.getenv("USE_HYDE", "true").lower() == "true"
USE_RERANK      = os.getenv("USE_RERANK", "true").lower() == "true"
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))  # 대화 이력 최대 메시지 수

# Bedrock Knowledge Base 설정
BEDROCK_KB_ID   = os.getenv("BEDROCK_KB_ID", "")
BEDROCK_KB_RESULTS = int(os.getenv("BEDROCK_KB_RESULTS", "4"))

# 성능 최적화 설정
USE_LLM_QUESTION_REWRITE = os.getenv("USE_LLM_QUESTION_REWRITE", "false").lower() == "true"
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "4096"))

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
_cached_retriever = None
_cached_llm = None
_cached_rag_chain = None
_cached_yearend_keywords = None
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
# Bedrock Knowledge Base Retriever
# =========================================

def get_bedrock_agent_client():
    """타임아웃 설정이 적용된 Bedrock Agent 클라이언트 생성"""
    from botocore.config import Config

    config = Config(
        connect_timeout=10,  # 연결 타임아웃: 10초
        read_timeout=60,     # 읽기 타임아웃: 60초
        retries={
            'max_attempts': 3,
            'mode': 'adaptive'
        }
    )

    return boto3.client(
        service_name='bedrock-agent-runtime',
        region_name=AWS_REGION,
        config=config
    )

class RetryRetriever(BaseRetriever):
    """재시도 로직을 적용하는 기본 Retriever 래퍼"""

    base_retriever: Any
    max_retries: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """재시도 로직이 적용된 검색"""
        search_start = time.perf_counter()
        docs = []

        for attempt in range(self.max_retries):
            try:
                docs = self.base_retriever.invoke(query)
                break  # 성공 시 루프 종료
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"[ERROR] Retriever 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {error_type} - {error_msg}")

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 지수 백오프: 1초, 2초, 4초
                    print(f"[INFO] {wait_time}초 후 재시도합니다...")
                    time.sleep(wait_time)
                else:
                    # 모든 재시도 실패 시 빈 문서 리스트 반환
                    print(f"[WARN] 모든 재시도 실패. 빈 문서 리스트 반환합니다.")
                    docs = []

        search_time = time.perf_counter() - search_start
        print(f"[TIMING] 벡터검색: {search_time*1000:.0f}ms")

        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """비동기 검색"""
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)

class RerankRetriever(BaseRetriever):
    """Reranking을 적용하는 Retriever 래퍼 (재시도 로직 포함)"""

    base_retriever: Any
    reranker: Any
    search_k: int
    max_retries: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """검색 후 재순위화 (재시도 로직 포함)"""
        # 1차 검색 (더 많은 문서) - 재시도 로직 포함
        search_start = time.perf_counter()
        docs = []

        for attempt in range(self.max_retries):
            try:
                docs = self.base_retriever.invoke(query)
                break  # 성공 시 루프 종료
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"[ERROR] Base retriever 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {error_type} - {error_msg}")

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 지수 백오프: 1초, 2초, 4초
                    print(f"[INFO] {wait_time}초 후 재시도합니다...")
                    time.sleep(wait_time)
                else:
                    # 모든 재시도 실패 시 빈 문서 리스트 반환
                    print(f"[WARN] 모든 재시도 실패. 빈 문서 리스트 반환합니다.")
                    docs = []

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

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """비동기 검색 후 재순위화"""
        # 동기 버전을 비동기로 래핑
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)

@log_execution_time
def get_retriever():
    """
    Amazon Bedrock Knowledge Base Retriever 반환 (캐시됨)
    """
    global _cached_retriever

    if not BEDROCK_KB_ID:
        raise ValueError(
            "❌ BEDROCK_KB_ID 환경 변수가 설정되지 않았습니다.\n"
            "   .env 파일에 BEDROCK_KB_ID=your-kb-id 추가 필요"
        )

    if _cached_retriever is None:
        print(f"[INFO] Bedrock Knowledge Base 사용: {BEDROCK_KB_ID}")

        # 타임아웃 설정이 적용된 클라이언트 생성
        bedrock_agent_client = get_bedrock_agent_client()

        # AmazonKnowledgeBasesRetriever 생성
        base_retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=BEDROCK_KB_ID,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": BEDROCK_KB_RESULTS
                }
            },
            region_name=AWS_REGION,
            client=bedrock_agent_client  # 타임아웃 설정된 클라이언트 주입
        )

        # Reranking 적용 (선택적)
        if USE_RERANK:
            print(f"[INFO] Bedrock KB + Reranker 조합 사용 (results: {BEDROCK_KB_RESULTS} → top_k: {TOP_K})")
            reranker = get_reranker()
            _cached_retriever = RerankRetriever(
                base_retriever=base_retriever,
                reranker=reranker,
                search_k=BEDROCK_KB_RESULTS,
                max_retries=3
            )
        else:
            print(f"[INFO] Bedrock KB 단독 사용 (results: {BEDROCK_KB_RESULTS}) + 재시도 로직")
            _cached_retriever = RetryRetriever(
                base_retriever=base_retriever,
                max_retries=3
            )

    return _cached_retriever

# =========================================
# 커스텀 SageMaker Bedrock LLM (boto3 converse API 사용)
# =========================================
class BedrockConverseLLM(BaseChatModel):
    """boto3 Bedrock Runtime converse API를 사용하는 LangChain 호환 LLM"""

    model_id: str
    region_name: str = "us-east-1"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    client: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name
        )

    @property
    def _llm_type(self) -> str:
        return "bedrock-converse"

    def _convert_messages_to_converse_format(self, messages: list[BaseMessage]) -> tuple[list[dict], list[dict]]:
        """
        LangChain 메시지를 Bedrock Converse 형식으로 변환

        Returns:
            tuple: (converse_messages, system_prompts)
        """
        converse_messages = []
        system_prompts = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System 메시지는 별도로 수집
                system_prompts.append({"text": msg.content})
            elif isinstance(msg, HumanMessage):
                converse_messages.append({
                    "role": "user",
                    "content": [{"text": msg.content}]
                })
            elif isinstance(msg, AIMessage):
                converse_messages.append({
                    "role": "assistant",
                    "content": [{"text": msg.content}]
                })
            else:
                # 기타 메시지는 user로 처리
                converse_messages.append({
                    "role": "user",
                    "content": [{"text": msg.content}]
                })

        return converse_messages, system_prompts

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs) -> ChatResult:
        """메시지 생성 (동기)"""
        converse_messages, system_prompts = self._convert_messages_to_converse_format(messages)

        try:
            # API 호출 파라미터 구성
            api_params = {
                "modelId": self.model_id,
                "messages": converse_messages,
                "inferenceConfig": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": self.top_p
                }
            }

            # System 메시지가 있으면 추가
            if system_prompts:
                api_params["system"] = system_prompts

            response = self.client.converse(**api_params)

            content = response['output']['message']['content'][0]['text']
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        except Exception as e:
            print(f"[ERROR] Bedrock Converse API 호출 실패: {e}")
            raise

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs) -> ChatResult:
        """메시지 생성 (비동기)"""
        # 동기 버전을 비동기로 래핑
        return await asyncio.to_thread(self._generate, messages, stop, **kwargs)

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs):
        """
        스트리밍 생성

        참고: SageMaker 엔드포인트는 Bedrock Converse API를 통한 스트리밍을 지원하지 않을 수 있습니다.
        이 경우 invoke 결과를 문자 단위로 청크를 나누어 반환합니다.
        """
        from langchain_core.outputs import ChatGenerationChunk

        converse_messages, system_prompts = self._convert_messages_to_converse_format(messages)

        try:
            # API 호출 파라미터 구성
            api_params = {
                "modelId": self.model_id,
                "messages": converse_messages,
                "inferenceConfig": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": self.top_p
                }
            }

            # System 메시지가 있으면 추가
            if system_prompts:
                api_params["system"] = system_prompts

            # invoke를 사용해서 전체 응답 받기
            response = self.client.converse(**api_params)

            content = response['output']['message']['content'][0]['text']

            # 응답을 작은 청크로 나누어 yield (의사 스트리밍)
            chunk_size = 5  # 5글자씩
            for i in range(0, len(content), chunk_size):
                chunk_text = content[i:i+chunk_size]
                message = AIMessageChunk(content=chunk_text)
                yield ChatGenerationChunk(message=message)

        except Exception as e:
            print(f"[ERROR] Bedrock Converse API 호출 실패: {e}")
            raise

# =========================================
# LLM 초기화
# =========================================
@log_execution_time
def get_llm():
    global _cached_llm
    if _cached_llm is None:
        # SageMaker 엔드포인트 ARN인지 확인
        if BEDROCK_MODEL and "sagemaker" in BEDROCK_MODEL.lower():
            print(f"[INFO] SageMaker 엔드포인트 사용: {BEDROCK_MODEL}")
            _cached_llm = BedrockConverseLLM(
                model_id=BEDROCK_MODEL,
                region_name=AWS_REGION,
                temperature=0.7,
                top_p=0.9,
                max_tokens=MAX_TOKENS,
            )
        else:
            # Bedrock Foundation Model 사용
            print(f"[INFO] Bedrock Foundation Model 사용: {BEDROCK_MODEL}")
            _cached_llm = ChatBedrock(
                model_id=BEDROCK_MODEL,
                region_name=AWS_REGION,
                model_kwargs={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": MAX_TOKENS,
                },
            )
    return _cached_llm

# =========================================
# 연말정산 키워드 로딩
# =========================================
@log_execution_time
def load_yearend_keywords():
    """파일에서 연말정산 키워드 로드"""
    global _cached_yearend_keywords
    if _cached_yearend_keywords is not None:
        return _cached_yearend_keywords

    keywords = []
    with open(YEAREND_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            keyword = line.strip()
            if keyword:
                keywords.append(keyword)

    _cached_yearend_keywords = keywords
    print(f"[INFO] 연말정산 키워드 로드 완료 (count={len(_cached_yearend_keywords)})")
    return _cached_yearend_keywords

# =========================================
# 오타 보정 (키워드 기반 유사도 매칭)
# =========================================
@log_execution_time
def correct_typos(question: str, similarity_threshold: int = 80) -> str:
    """
    질문에서 연말정산 키워드의 오타를 자동으로 보정

    Args:
        question: 사용자 질문
        similarity_threshold: 유사도 임계값 (0-100, 기본값 80)

    Returns:
        오타가 보정된 질문

    Example:
        "현금여수증 공제받을 수 있나요?" → "현금영수증 공제받을 수 있나요?"
    """
    keywords = load_yearend_keywords()
    corrected_question = question
    corrections_made = []

    # 질문을 단어로 분리 (공백 및 특수문자 기준)
    import re
    words = re.findall(r'[가-힣a-zA-Z0-9]+', question)

    for word in words:
        # 2글자 이상인 단어만 검사 (너무 짧은 단어는 오탐 가능성 높음)
        if len(word) < 2:
            continue

        # 정확히 일치하는 키워드는 건너뛰기
        if word in keywords:
            continue

        # 한글 조사 제거 (이, 가, 은, 는, 을, 를, 와, 과, 에, 에서, 으로, 로)
        base_word = word
        particles = ['에서', '으로', '이', '가', '은', '는', '을', '를', '와', '과', '에', '로', '도', '만', '의']
        for particle in particles:
            if len(base_word) > 2 and base_word.endswith(particle):
                base_word = base_word[:-len(particle)]
                break

        # 조사 제거 후에도 2글자 이상이어야 검사
        if len(base_word) < 2:
            continue

        # 각 키워드와 유사도 계산
        best_match = None
        best_score = 0

        for keyword in keywords:
            # 단어 길이 차이가 너무 크면 건너뛰기 (성능 최적화)
            if abs(len(base_word) - len(keyword)) > 3:
                continue

            # 유사도 계산 (ratio: 0-100) - 조사 제거된 base_word 사용
            score = fuzz.ratio(base_word, keyword)

            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = keyword

        # 오타로 판단되면 교체 (원본 word를 best_match로 교체)
        if best_match:
            corrected_question = corrected_question.replace(word, best_match)
            corrections_made.append(f"{word} → {best_match} (유사도: {best_score}%)")

    # 보정 내역 로그 출력
    if corrections_made:
        print(f"[INFO] 오타 보정: {', '.join(corrections_made)}")

    return corrected_question

# =========================================
# 후속 질문 병합 (이전 질문 + 후속 질문 → 자연스러운 완전한 질문)
# =========================================
@log_execution_time
def merge_followup_question(previous_question: str, current_question: str) -> str:
    """
    이전 질문과 후속 질문을 자연스럽게 병합

    Args:
        previous_question: 이전 사용자 질문
        current_question: 현재 후속 질문

    Returns:
        자연스럽게 재구성된 질문
    """
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(FOLLOWUP_MERGE_PROMPT)
        chain = prompt | llm | StrOutputParser()

        merged = chain.invoke({
            "previous_question": previous_question,
            "current_question": current_question
        })

        if merged and len(merged.strip()) > 0:
            return merged.strip()

    except Exception as e:
        print(f"[WARN] 질문 병합 실패: {e}")
        # 실패 시 현재 질문 반환
        return current_question

    return current_question

# =========================================
# 질문 보정
# =========================================
@log_execution_time
def refine_question(question: str) -> str:
    """
    연말정산 키워드 기반 질문 보정
    - 키워드를 참고하여 질문을 더 명확하고 구체적으로 개선
    """
    if not USE_LLM_QUESTION_REWRITE:
        return question

    try:
        keywords = load_yearend_keywords()
        keywords_str = ", ".join(keywords[:20])  # 처음 20개만 사용

        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(QUESTION_REFINE_PROMPT)
        chain = prompt | llm | StrOutputParser()

        refined = chain.invoke({
            "keywords": keywords_str,
            "question": question
        })

        if refined and len(refined.strip()) > 0:
            print(f"[INFO] 질문 보정: {question[:30]}... → {refined[:30]}...")
            return refined.strip()

    except Exception as e:
        print(f"[WARN] 질문 보정 실패: {e}")

    return question

# =========================================
# RAG 체인
# =========================================
@log_execution_time
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", HISTORY_RETRIEVER_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def _get_simple_prompt():
    """단순 질문용 프롬프트 (간단한 답변만)"""
    return ChatPromptTemplate.from_messages([
        ("system", SIMPLE_QA_PROMPT),
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

    return ChatPromptTemplate.from_messages([
        ("system", DETAILED_QA_PROMPT),
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
    if len(question) >= 10:
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
    - 복잡하거나 모호한 질문만 HyDE 사용
    """
    # 간단한 키워드 질문 패턴 (HyDE 불필요) - 확대
    simple_patterns = [
        r"전화번호|연락처|이메일|주소",
        r"고객센터|문의처",
        r"문의.*어디",
        r"\d{3,4}-\d{4}",  # 전화번호 포함
        r"^.{1,8}$",  # 8글자 이하 짧은 질문
        r"뭐|무엇|누구|언제|어디",  # 단순 의문사
        r"알려|확인|조회",  # 단순 조회
    ]

    for pattern in simple_patterns:
        if re.search(pattern, question):
            return False

    # 복잡한 질문 패턴 (HyDE 사용) - 축소
    complex_patterns = [
        r"어떻게.*하|.*하는.*방법|.*하려면",  # 절차/방법 질문
        r"왜.*안.*돼|왜.*안.*되|왜.*오류|왜.*에러",  # 오류 원인 질문
        r".*와.*차이|.*비교|.*다른점",  # 비교 질문
    ]

    for pattern in complex_patterns:
        if re.search(pattern, question):
            return True

    # 기본값: HyDE 사용 안함 (속도 우선)
    return False

@log_execution_time
def hyde_transform(question: str, max_retries: int = 2) -> str:
    """
    질문을 가상의 답변으로 변환 (재시도 로직 포함)
    """
    llm = get_llm()

    # 간결한 프롬프트로 빠른 생성
    prompt = HYDE_TRANSFORM_PROMPT.format(question=question)

    for attempt in range(max_retries):
        try:
            result = llm.invoke(
                prompt,
                max_tokens=100,  # 토큰 제한으로 빠른 생성
                temperature=0.3   # 낮은 temperature로 빠른 생성
            )
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

    # 0. 오타 보정 (가드레일 검증 전에 먼저 수행)
    typo_corrected_input = correct_typos(user_message)
    if typo_corrected_input != user_message:
        print(f"[INFO] ✓ 입력 오타 보정")
        print(f"  원본: {user_message}")
        print(f"  보정: {typo_corrected_input}")

    # 1. 연말정산 주제 가드레일 검증 (오타 보정된 질문으로 검증) + 이전 질문 추출
    is_valid, error_msg, previous_question = validate_yearend_tax_topic(
        typo_corrected_input,  # 오타 보정된 질문으로 검증
        session_id,
        keyword_loader=load_yearend_keywords,
        session_history_getter=get_session_history
    )
    if not is_valid:
        # 가드레일 거부 메시지에 특수 마커 추가 (DB 저장 방지용)
        yield "__GUARDRAIL_REJECT__"
        yield error_msg
        return

    # 2. 후속 질문이면 이전 질문과 병합하여 자연스러운 질문 생성
    enhanced_message = typo_corrected_input
    if previous_question:
        # LLM을 사용해서 이전 질문과 후속 질문을 자연스럽게 병합
        enhanced_message = merge_followup_question(previous_question, typo_corrected_input)
        print(f"[INFO] 후속 질문 병합:")
        print(f"  이전: {previous_question}")
        print(f"  후속: {typo_corrected_input}")
        print(f"  병합: {enhanced_message}")

    # HyDE 기본값: 환경변수 사용
    if use_hyde is None:
        use_hyde = USE_HYDE

    # 3. 질문 보정 (키워드가 병합된 질문으로 보정)
    refined_message = refine_question(enhanced_message)

    # 질문 보정 결과 로그
    if refined_message != enhanced_message:
        print(f"[INFO] ✓ 질문 보정 완료")
        print(f"  입력: {enhanced_message}")
        print(f"  보정: {refined_message}")

    # 4. 질문 유형 분류 (단순 vs 상세)
    question_type = classify_question_type(refined_message)

    # 5. HyDE 적용 여부 판단 및 실행
    search_query = refined_message  # 기본값: 보정된 질문으로 검색

    if use_hyde and should_use_hyde(refined_message):
        try:
            # HyDE: 가상 답변 생성
            hypothetical_answer = hyde_transform(refined_message)
            search_query = hypothetical_answer  # 가상 답변으로 검색

            # 디버깅용 출력 (필요시 주석 해제)
            # yield f"🔍 HyDE 변환됨\n가상 답변: {hypothetical_answer[:150]}...\n\n"

        except Exception as e:
            print(f"[WARN] HyDE 변환 실패, 보정된 질문 사용: {e}")
            search_query = refined_message

    # 6. RAG 체인 실행
    # HyDE를 사용한 경우, 검색은 가상 답변으로 하되 최종 응답은 보정된 질문 기반

    if use_hyde and search_query != refined_message:
        # HyDE 모드: 가상 답변으로 1회 검색 → 캐시된 chain으로 답변 생성
        retriever = get_retriever()

        # Retriever 호출 시 재시도 로직 포함
        docs = []
        max_retries = 3
        for attempt in range(max_retries):
            try:
                docs = retriever.invoke(search_query)
                break  # 성공 시 루프 종료
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"[ERROR] Retriever 호출 실패 (시도 {attempt + 1}/{max_retries}): {error_type} - {error_msg}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 지수 백오프: 1초, 2초
                    time.sleep(wait_time)
                else:
                    # 모든 재시도 실패 시 빈 문서 리스트로 대체
                    print(f"[WARN] 모든 재시도 실패. 빈 문서 리스트로 대체합니다.")
                    docs = []
                    yield f"⚠️ 벡터 데이터베이스 연결에 문제가 발생했습니다. 일반적인 답변을 제공합니다.\n\n"

        # 캐시된 HyDE chain 사용 (질문 유형에 따라 선택)
        hyde_chain = get_hyde_chain(question_type)
        chat_history = get_session_history(session_id)

        # 스트리밍 답변 생성
        try:
            stream = hyde_chain.stream({
                "input": refined_message,  # 보정된 질문
                "context": docs,  # 검색된 문서
                "chat_history": chat_history.messages
            })
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            print(f"[ERROR] HyDE chain 스트리밍 시작 실패: {error_type} - {error_msg}")
            yield f"❌ 답변 생성 중 오류가 발생했습니다: {error_type}\n벡터 데이터베이스 연결을 확인해주세요."
            return
    else:
        # 일반 모드 (질문 유형에 따라 선택)
        rag_chain = get_rag_chain(question_type)

        try:
            stream = rag_chain.stream(
                {"input": refined_message},
                config={"configurable": {"session_id": session_id}},
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            print(f"[ERROR] RAG chain 스트리밍 시작 실패: {error_type} - {error_msg}")

            # ValidationException 또는 벡터 DB 관련 오류인 경우
            if "ValidationException" in error_type or "vector database" in error_msg.lower() or "aurora" in error_msg.lower():
                yield f"❌ 벡터 데이터베이스 연결 오류가 발생했습니다.\n"
                yield f"오류 유형: {error_type}\n"
                yield f"Aurora DB 인스턴스 연결을 확인하고 잠시 후 다시 시도해주세요."
            else:
                yield f"❌ 답변 생성 중 오류가 발생했습니다: {error_type}\n서버 연결을 확인하고 다시 시도해주세요."
            return

    # 스트림 처리 및 대화 이력 저장 (에러 핸들링 강화)
    full_response = ""
    try:
        for chunk in stream:
            full_response += chunk
            yield chunk

        # HyDE 모드일 때 대화 이력 수동 저장 (원본 질문 저장)
        if use_hyde and search_query != refined_message:
            chat_history = get_session_history(session_id)
            chat_history.add_user_message(user_message)  # 원본 질문 저장
            chat_history.add_ai_message(full_response)

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)[:200]
        print(f"[ERROR] 스트리밍 중 오류 발생: {error_type} - {error_msg}")

        # 부분 응답이라도 있으면 저장
        if full_response:
            if use_hyde and search_query != refined_message:
                chat_history = get_session_history(session_id)
                chat_history.add_user_message(user_message)  # 원본 질문 저장
                chat_history.add_ai_message(full_response)
            yield f"\n\n⚠️ 응답 중 연결이 끊어졌습니다. 부분 응답만 표시됩니다."
        else:
            yield f"\n\n❌ 오류가 발생했습니다: {error_type}\n서버 연결을 확인하고 다시 시도해주세요."

# =========================================
# 유틸
# =========================================
def cleanup_resources():
    global _cached_retriever, _cached_llm, _cached_rag_chain, _cached_yearend_keywords
    global _cached_hyde_chain_simple, _cached_hyde_chain_detailed
    global _cached_rag_chain_simple, _cached_rag_chain_detailed, _cached_reranker

    _cached_retriever = _cached_llm = _cached_rag_chain = _cached_yearend_keywords = None
    _cached_hyde_chain_simple = _cached_hyde_chain_detailed = None
    _cached_rag_chain_simple = _cached_rag_chain_detailed = None
    _cached_reranker = None

    session_store.store.clear()
    session_store.session_timestamps.clear()
    print("[INFO] 모든 캐시 및 세션 리소스 정리 완료")

def get_cache_info():
    return {
        "retriever_cached": _cached_retriever is not None,
        "llm_cached": _cached_llm is not None,
        "rag_chain_cached": _cached_rag_chain is not None,
        "yearend_keywords_cached": _cached_yearend_keywords is not None,
        "hyde_chain_simple_cached": _cached_hyde_chain_simple is not None,
        "hyde_chain_detailed_cached": _cached_hyde_chain_detailed is not None,
        "rag_chain_simple_cached": _cached_rag_chain_simple is not None,
        "rag_chain_detailed_cached": _cached_rag_chain_detailed is not None,
        "reranker_cached": _cached_reranker is not None,
        "use_rerank": USE_RERANK,
        "active_sessions": len(session_store.store)
    }
