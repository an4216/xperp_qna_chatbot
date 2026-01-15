# llm.py

# =========================================
# LangChain 및 관련 라이브러리 import
# =========================================
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict
import os, time, re, json

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

from config import answer_examples

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
        return self.store[session_id]

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

# =========================================
# 임베딩 & 벡터스토어
# =========================================
def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        start = time.perf_counter()
        _cached_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True}
        )
        print(f"[TIMER] get_embeddings 로드 완료 ({(time.perf_counter() - start) * 1000:.2f} ms)")
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

def get_retriever():
    global _cached_retriever, _cached_vectorstore, _cached_fingerprint
    os.makedirs(VECTOR_DIR, exist_ok=True)
    current_fp = _load_fingerprint()

    if _cached_retriever is None or _cached_fingerprint != current_fp:
        print(f"[INFO] retriever reload (old={_cached_fingerprint}, new={current_fp})")
        index_path = os.path.join(VECTOR_DIR, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ 벡터스토어 없음: {index_path}. 01_ingest.py 실행 필요")

        embeddings = get_embeddings()
        _cached_vectorstore = FAISS.load_local(
            VECTOR_DIR, embeddings, allow_dangerous_deserialization=True
        )
        _cached_retriever = _cached_vectorstore.as_retriever(search_kwargs={'k': TOP_K})
        _cached_fingerprint = current_fp
    return _cached_retriever

# =========================================
# LLM 초기화
# =========================================
def get_llm():
    global _cached_llm
    if _cached_llm is None:
        _cached_llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=MODEL_LLM,
        )
    return _cached_llm

# =========================================
# Dictionary 기반 질문 보정
# =========================================
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
    - 질문을 구체적으로 만들어라. ("왜 그런지", "메뉴얼 기반으로 설명해주세요" 등을 붙여라)
    - 대메뉴 태그가 있으면 질문에 포함시켜라.
    - 불필요하게 길게 풀지 말고, 한 문장 안에서 간결하지만 구체적으로 표현해라.

    예시:
    입력: [입주자] 차량등록은 어디서해?
    출력: 입주자 메뉴에서 차량등록은 어디서 하는지 메뉴얼을 기반으로 구체적으로 설명해주세요.
    입력: [수납] 연체료는 어디서 확인해?
    출력: 수납 메뉴에서 연체료는 어디서 확인하는지 메뉴얼을 기반으로 알려주세요.
    입력: [입주자] 중간정산할때 전기검침 사용량입력 후 계산을 하면 금액이 안맞아요
    출력: 입주자 메뉴에서 중간정산 시 전기검침 사용량 입력 후 계산 금액이 왜 맞지 않는지 메뉴얼에 기반해서 구체적으로 설명해주세요.

    [사전] {dictionary}
    [사용자질문] {question}
    """.strip()

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

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

def get_rag_chain():
    global _cached_rag_chain
    if _cached_rag_chain:
        return _cached_rag_chain

    llm = get_llm()
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

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    retriever = get_history_retriever()
    chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, chain)

    _cached_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return _cached_rag_chain

# =========================================
# 최종 응답 함수
# =========================================
def get_ai_response(user_message: str, session_id: str):
    if not session_id:
        raise ValueError("session_id is required.")

    # ✅ 세션 강제 초기화: 같은 session_id라도 항상 새로운 히스토리로 시작
    session_store.store.pop(session_id, None)
    session_store.session_timestamps.pop(session_id, None)

    rag_chain = get_rag_chain()

    try:
        effective_message = process_question(user_message)
    except Exception as e:
        print(f"[WARN] 질문 보정 실패: {e}")
        effective_message = user_message

#     yield f"🔧 보정된 질문: {effective_message}\n\n"

    start = time.perf_counter()
    stream = rag_chain.stream(
        {"input": effective_message},
        config={"configurable": {"session_id": session_id}},
    )
    for chunk in stream:
        yield chunk

    yield f"\n\n⏱ 소요시간: {time.perf_counter() - start:.2f}초"

# =========================================
# 유틸
# =========================================
def cleanup_resources():
    global _cached_embeddings, _cached_retriever, _cached_llm, _cached_rag_chain, _cached_fingerprint
    _cached_embeddings = _cached_retriever = _cached_llm = _cached_rag_chain = _cached_fingerprint = None
    session_store.store.clear()
    session_store.session_timestamps.clear()
    print("[INFO] 모든 캐시 및 세션 리소스 정리 완료")

def get_cache_info():
    return {
        "embeddings_cached": _cached_embeddings is not None,
        "retriever_cached": _cached_retriever is not None,
        "llm_cached": _cached_llm is not None,
        "rag_chain_cached": _cached_rag_chain is not None,
        "fingerprint": _cached_fingerprint,
        "active_sessions": len(session_store.store)
    }
