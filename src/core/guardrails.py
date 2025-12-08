"""
가드레일 모듈

이 모듈은 사용자 입력 검증 및 주제 관련성 검증을 담당합니다.
- 악성 패턴 감지 (Prompt Injection, XSS, SQL Injection)
- 입력 메시지 검증 (길이, 형식, 의미 있는 입력)
- 연말정산 주제 관련성 검증 (키워드 기반 + 대화 이력 고려)
"""

import re
from typing import Callable, Optional

from src.core.messages import (
    get_validation_message,
    get_topic_message,
)


# =========================================
# 1. 악성 패턴 감지
# =========================================
# 용도: Prompt Injection, XSS, SQL Injection 등 보안 위협 패턴 탐지
# 사용처: _validate_input() 함수 내부
def contains_malicious_pattern(text: str) -> bool:
    """
    악성 패턴 감지 (Prompt Injection, XSS 등)

    Args:
        text: 검증할 텍스트

    Returns:
        True: 악성 패턴 감지됨
        False: 정상 입력
    """
    malicious_patterns = [
        # Prompt Injection 패턴
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?instructions?",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now",
        r"new\s+role",
        r"act\s+as",
        r"pretend\s+to\s+be",
        r"system\s*:",
        r"assistant\s*:",

        # XSS 패턴
        r"<script[^>]*>",
        r"javascript\s*:",
        r"on\w+\s*=",  # onclick, onerror 등
        r"<iframe",  # iframe 태그 (속성 무관)
        r"<object",
        r"<embed",

        # SQL Injection 기본 패턴 (참고용)
        r"(union\s+select|drop\s+table|insert\s+into|delete\s+from)",
    ]

    text_lower = text.lower()

    for pattern in malicious_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    return False


# =========================================
# 2. 입력 메시지 검증
# =========================================
# 용도: API 요청 시 사용자 입력의 기본적인 형식과 안전성 검증
# 사용처: /chat, /chat-test 엔드포인트
def validate_input(
    message: str,
    min_length: int = 5,
    max_length: int = 2000
) -> tuple[bool, str]:
    """
    입력 메시지 검증

    Args:
        message: 검증할 메시지
        min_length: 최소 길이 (기본값: 5)
        max_length: 최대 길이 (기본값: 2000)

    Returns:
        (검증 성공 여부, 에러 메시지)
    """
    # 0. 공백만 있는 입력 차단 (최우선)
    if message.strip() == "":
        return False, get_validation_message("EMPTY_OR_SHORT_INPUT")

    # 1. 길이 검증
    if len(message) < min_length:
        return False, get_validation_message("EMPTY_OR_SHORT_INPUT")

    if len(message) > max_length:
        return False, get_validation_message("TOO_LONG_INPUT")

    # 2. 의미 없는 입력 패턴 차단
    stripped = message.strip()

    # 숫자만 있는 입력 (예: "1", "123")
    if stripped.isdigit():
        return False, get_validation_message("EMPTY_OR_SHORT_INPUT")

    # 특수문자만 있는 입력 (예: "!!!", "???")
    if re.match(r'^[^\w\s가-힣]+$', stripped, re.UNICODE):
        return False, get_validation_message("EMPTY_OR_SHORT_INPUT")

    # 같은 문자 반복 (예: "aaaa", "1111")
    if len(set(stripped)) == 1 and len(stripped) > 2:
        return False, get_validation_message("EMPTY_OR_SHORT_INPUT")

    # 3. 악성 패턴 검증
    if contains_malicious_pattern(message):
        return False, get_validation_message("MALICIOUS_PATTERN")

    # 4. 제어 문자 검증
    if any(ord(char) < 32 and char not in ['\n', '\r', '\t'] for char in message):
        return False, get_validation_message("INVALID_FORMAT")

    return True, get_validation_message("SUCCESS")


# =========================================
# 3. 연말정산 주제 관련성 검증
# =========================================
# 용도: 질문이 연말정산 주제와 관련 있는지 검증 (키워드 + 대화 이력 고려)
# 사용처: get_ai_response() 함수 - 응답 생성 전 주제 검증
def validate_yearend_tax_topic(
    question: str,
    session_id: Optional[str] = None,
    keyword_loader: Optional[Callable[[], list[str]]] = None,
    session_history_getter: Optional[Callable[[str], any]] = None
) -> tuple[bool, str, str]:
    """
    연말정산 주제 관련성 검증 (대화 이력 고려)

    Args:
        question: 현재 질문
        session_id: 세션 ID (대화 이력 조회용)
        keyword_loader: 연말정산 키워드 로드 함수
        session_history_getter: 세션 히스토리 조회 함수

    Returns:
        (검증 성공 여부, 에러 메시지, 이전 사용자 질문)
    """
    # 키워드 로드 (함수가 제공되지 않으면 기본 동작 수행)
    if keyword_loader:
        yearend_keywords = keyword_loader()
    else:
        # 기본값: 빈 리스트 (실제 사용 시 keyword_loader는 필수)
        yearend_keywords = []

    question_lower = question.lower()

    # 1. 현재 질문에 키워드가 있으면 바로 허용
    for keyword in yearend_keywords:
        if keyword in question_lower:
            return True, get_topic_message("SUCCESS"), ""

    # 2. 일반적인 인사말이나 도움 요청은 허용
    greeting_patterns = [
        r"^안녕",
        r"^반가",
        r"^도와",
        r"^도움",
        r"^문의",
        r"^궁금",
        r"어떻게\s*(사용|이용|하면|하는)",
        r"^알려"
    ]

    for pattern in greeting_patterns:
        if re.search(pattern, question_lower):
            return True, get_topic_message("SUCCESS"), ""

    # 3. 후속 질문 패턴 감지 (대명사/지시어/맥락 의존 명사)
    followup_patterns = [
        r"^(그거|그건|그게|그럼|그러면)",  # 지시대명사
        r"(항목|내용|방법|절차|메뉴)",  # 맥락 의존 명사
        r"^(어디|어떻게|뭐|무엇|언제|왜)",  # 의문사로 시작
        r"(어디서|어떻게)\s*(해|하|봐|보)",  # 의문사 + 동사
    ]

    has_followup_pattern = any(re.search(pattern, question_lower) for pattern in followup_patterns)

    # 후속 질문 패턴이 있으면 대화 이력에서 이전 질문 추출
    if has_followup_pattern and session_id and session_history_getter:
        try:
            # 대화 이력에서 최근 메시지 확인
            chat_history = session_history_getter(session_id)
            recent_messages = chat_history.messages[-6:]  # 최근 3턴 (user + ai)

            # 최근 대화에서 연말정산 키워드가 있는지 확인
            previous_user_question = ""
            for msg in recent_messages:
                msg_content = msg.content.lower()
                # 사용자 메시지만 확인
                if hasattr(msg, 'type') and msg.type == 'human':
                    # 키워드 체크
                    has_keyword = any(keyword in msg_content for keyword in yearend_keywords)
                    if has_keyword:
                        previous_user_question = msg.content
                        break

            # 이전 질문을 찾았으면 후속 질문 허용
            if previous_user_question:
                print(f"[INFO] 후속 질문 감지 - 이전 질문: {previous_user_question[:30]}...")
                return True, get_topic_message("SUCCESS"), previous_user_question

        except Exception as e:
            print(f"[WARN] 대화 이력 확인 실패: {e}")

    # 4. 모든 조건 불만족 시 거부
    return False, get_topic_message("OFF_TOPIC"), ""
