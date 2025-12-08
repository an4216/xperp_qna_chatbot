"""
가드레일 메시지 상수

이 모듈은 가드레일에서 사용되는 모든 메시지를 중앙에서 관리합니다.
메시지 변경 시 이 파일만 수정하면 전체 시스템에 반영됩니다.
"""

# =========================================
# 입력 검증 메시지
# =========================================

VALIDATION_MESSAGES = {
    # 공백/짧은 입력/의미 없는 입력
    "EMPTY_OR_SHORT_INPUT": "XPERP 연말정산에 대해 구체적으로 질문해주세요.",

    # 긴 입력
    "TOO_LONG_INPUT": "질문이 너무 깁니다. 간결하게 요약해서 다시 입력해주세요.",

    # 악성 패턴 감지
    "MALICIOUS_PATTERN": "부적절한 입력이 감지되었습니다. XPERP 연말정산 관련 질문만 입력해주세요.",

    # 제어 문자 포함
    "INVALID_FORMAT": "올바른 형식으로 질문을 입력해주세요.",

    # 검증 성공
    "SUCCESS": "OK",
}


# =========================================
# 주제 관련성 검증 메시지
# =========================================

TOPIC_MESSAGES = {
    # 연말정산 무관 질문
    "OFF_TOPIC": "안녕하세요! 😊 저는 연말정산 전문 상담 챗봇입니다.\n연말정산에 관해 궁금하신 점을 말씀해주시면 친절하게 안내해드릴게요!",

    # 주제 검증 성공
    "SUCCESS": "OK",
}


# =========================================
# Rate Limit 메시지
# =========================================

RATE_LIMIT_MESSAGES = {
    # 요청 제한 초과
    "RATE_LIMIT_EXCEEDED": "너무 많은 요청을 보내셨습니다. 잠시 후 다시 시도해주세요.",
}


# =========================================
# 메시지 접근 함수 (타입 안전성)
# =========================================

def get_validation_message(key: str) -> str:
    """
    입력 검증 메시지 조회

    Args:
        key: 메시지 키 (예: "EMPTY_OR_SHORT_INPUT")

    Returns:
        메시지 문자열
    """
    return VALIDATION_MESSAGES.get(key, VALIDATION_MESSAGES["INVALID_FORMAT"])


def get_topic_message(key: str) -> str:
    """
    주제 관련성 메시지 조회

    Args:
        key: 메시지 키 (예: "OFF_TOPIC")

    Returns:
        메시지 문자열
    """
    return TOPIC_MESSAGES.get(key, TOPIC_MESSAGES["OFF_TOPIC"])


def get_rate_limit_message(key: str) -> str:
    """
    Rate Limit 메시지 조회

    Args:
        key: 메시지 키 (예: "RATE_LIMIT_EXCEEDED")

    Returns:
        메시지 문자열
    """
    return RATE_LIMIT_MESSAGES.get(key, RATE_LIMIT_MESSAGES["RATE_LIMIT_EXCEEDED"])
