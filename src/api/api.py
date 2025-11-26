from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
from src.llm.llm import get_ai_response, get_cache_info, get_llm, get_retriever, get_reranker, load_yearend_keywords, USE_RERANK
import asyncio
import json, os, time, re
from httpx import AsyncClient, HTTPError
from rate_limiter import RateLimiter


# FastAPI 앱 생성
app = FastAPI()
# 환경변수 로드
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# =========================================
# 가드레일 설정
# =========================================
# 입력 검증
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
MIN_MESSAGE_LENGTH = int(os.getenv("MIN_MESSAGE_LENGTH", "5"))

# Rate Limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Rate Limiter 인스턴스 생성
rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS
)

# =========================================
# 서버 시작 시 모델 사전 로드
# =========================================
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모든 모델 사전 로드 (첫 요청 시간 단축)"""
    print("\n" + "="*70)
    print("[INFO] 모델 사전 로드 시작...")
    print("="*70)
    start = time.time()

    try:
        # 1. Retriever 로드 (Bedrock KB)
        print("  [1/4] Retriever 로드 중...", end=" ", flush=True)
        t = time.time()
        get_retriever()
        print(f"완료 ({time.time()-t:.1f}초)")

        # 2. Reranker 로드 (USE_RERANK=true일 때만)
        if USE_RERANK:
            print("  [2/4] Reranker 로드 중...", end=" ", flush=True)
            t = time.time()
            get_reranker()
            print(f"완료 ({time.time()-t:.1f}초)")
        else:
            print("  [2/4] Reranker 로드 생략 (USE_RERANK=false)")

        # 3. LLM 로드
        print("  [3/4] LLM 로드 중...", end=" ", flush=True)
        t = time.time()
        get_llm()
        print(f"완료 ({time.time()-t:.1f}초)")

        # 4. 연말정산 키워드 로드
        print("  [4/4] 연말정산 키워드 로드 중...", end=" ", flush=True)
        t = time.time()
        load_yearend_keywords()
        print(f"완료 ({time.time()-t:.1f}초)")

        elapsed = time.time() - start
        print("="*70)
        print(f"[INFO] 모델 사전 로드 완료! (총 {elapsed:.1f}초)")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n[ERROR] 모델 사전 로드 실패: {e}")
        print("첫 요청 시 모델이 로드됩니다.")

# =========================================
# 입력 검증 함수
# =========================================
def _contains_malicious_pattern(text: str) -> bool:
    """
    악성 패턴 감지 (Prompt Injection, XSS 등)

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

def _validate_input(message: str) -> tuple[bool, str]:
    """
    입력 메시지 검증

    Returns:
        (검증 성공 여부, 에러 메시지)
    """
    # 0. 공백만 있는 입력 차단 (최우선)
    if message.strip() == "":
        return False, "XPERP 연말정산에 대해 구체적으로 질문해주세요."

    # 2. 길이 검증
    if len(message) < MIN_MESSAGE_LENGTH:
        return False, "XPERP 연말정산에 대해 구체적으로 질문해주세요."

    if len(message) > MAX_MESSAGE_LENGTH:
        return False, "질문이 너무 깁니다. 간결하게 요약해서 다시 입력해주세요."

    # 3. 의미 없는 입력 패턴 차단
    stripped = message.strip()

    # 숫자만 있는 입력 (예: "1", "123")
    if stripped.isdigit():
        return False, "XPERP 연말정산에 대해 구체적으로 질문해주세요."

    # 특수문자만 있는 입력 (예: "!!!", "???")
    if re.match(r'^[^\w\s가-힣]+$', stripped, re.UNICODE):
        return False, "XPERP 연말정산에 대해 구체적으로 질문해주세요."

    # 같은 문자 반복 (예: "aaaa", "1111")
    if len(set(stripped)) == 1 and len(stripped) > 2:
        return False, "XPERP 연말정산에 대해 구체적으로 질문해주세요."

    # 4. 악성 패턴 검증
    if _contains_malicious_pattern(message):
        return False, "부적절한 입력이 감지되었습니다. XPERP 연말정산 관련 질문만 입력해주세요."

    # 5. 제어 문자 검증
    if any(ord(char) < 32 and char not in ['\n', '\r', '\t'] for char in message):
        return False, "올바른 형식으로 질문을 입력해주세요."

    return True, "OK"


# HTML 템플릿 설정
templates = Jinja2Templates(directory="templates")
# 👉 환경 변수 로드 (.env 사용 시)
load_dotenv()
# 정적 파일(CSS, JS) 설정 (필요시)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 경로 ("/")
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# ✅ 채팅 API 엔드포인트 (스트리밍 버전)
@app.post("/chat")
async def chat(message: str = Form(...), session_id: str = Form(None)):
    # 1. session_id 검증
    if not session_id:
        return PlainTextResponse("session_id is required", status_code=400)

    # 2. Rate Limiting 체크
    allowed, rate_limit_msg = rate_limiter.is_allowed(session_id)
    if not allowed:
        return PlainTextResponse(rate_limit_msg, status_code=429)

    # 3. 입력 검증
    is_valid, error_msg = _validate_input(message)
    if not is_valid:
        return PlainTextResponse(error_msg, status_code=400)

    # 4. AI 응답 생성
    ai_response_generator = get_ai_response(message, session_id=session_id)

    async def event_stream():
        for chunk in ai_response_generator:
            yield chunk
            await asyncio.sleep(0.01)

    return StreamingResponse(event_stream(), media_type="text/plain")


# 🔎 호출 테스트용 API (전체 모아서 반환)
@app.post("/chat-test")
async def chat_test(message: str = Form(...), session_id: str = Form("test-session")):
    # 1. Rate Limiting 체크
    allowed, rate_limit_msg = rate_limiter.is_allowed(session_id)
    if not allowed:
        return PlainTextResponse(rate_limit_msg, status_code=429)

    # 2. 입력 검증
    is_valid, error_msg = _validate_input(message)
    if not is_valid:
        return PlainTextResponse(error_msg, status_code=400)

    # 3. AI 응답 생성
    ai_response_generator = get_ai_response(message, session_id=session_id)
    chunks = []
    for chunk in ai_response_generator:
        chunks.append(chunk)
    return PlainTextResponse("".join(chunks))
# ✅ 사용자 피드백 저장 API
# ✅ 사용자 피드백 저장 API
@app.post("/feedback")
async def feedback(data: dict = Body(...)):

    # 1) 피드백 로그 파일 저장
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/feedback.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # 2) 👎 down 피드백이면 Slack 알림 전송
    if data.get("feedback") == "down" and SLACK_WEBHOOK_URL:
        user_name = data.get("name", "익명")   # ✅ 이름 필드 (없으면 '익명')
        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "⚠️ Xperp 챗봇 Down Feedback 발생"}
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🙋 질문자:* {user_name}\n*🙋 질문:*\n>{data.get('message')}"
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*💬 답변 (요약):*\n>{data.get('response')[:300]}..."
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":warning: *사유:*\n```{data.get('reason', '사유 미작성')}```"
                    },
                },
                {   # ✅ 코멘트 추가
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"✍️ *코멘트:*\n>{data.get('comment', '코멘트 없음')}"
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": ":loudspeaker: <!channel> 모든 분 확인 바랍니다."}
                    ],
                },
            ],
        }

        async def send_slack_webhook(payload: dict) -> None:
            """Slack Webhook을 비동기로 전송하면서 기본 재시도를 수행합니다."""
            async with AsyncClient(timeout=5.0) as client:
                for attempt in range(3):
                    try:
                        response = await client.post(SLACK_WEBHOOK_URL, json=payload)
                        response.raise_for_status()
                        return
                    except HTTPError as exc:
                        if attempt == 2:
                            print(f"Slack 전송 실패: {exc}")
                        else:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                    except Exception as exc:
                        if attempt == 2:
                            print(f"Slack 전송 예기치 못한 실패: {exc}")
                        else:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                    # 마지막 시도까지 실패하면 루프 종료
                    break

        await send_slack_webhook(message)
    return {"status": "ok"}
