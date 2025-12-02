from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
from src.llm.llm import get_ai_response, get_cache_info, get_llm, get_retriever, get_reranker, load_yearend_keywords, USE_RERANK
import asyncio
import json, os, time, re
from src.api.middleware.rate_limiter import RateLimiter
from src.core.guardrails import validate_input
from src.core.database import execute_query
from datetime import datetime


# FastAPI 앱 생성
app = FastAPI()
# 환경변수 로드
load_dotenv()

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

# HTML 템플릿 설정
templates = Jinja2Templates(directory="src/templates")
# 👉 환경 변수 로드 (.env 사용 시)
load_dotenv()
# 정적 파일(CSS, JS) 설정 (필요시)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 경로 ("/")
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, userId: str = None):
    """
    챗봇 화면 렌더링 (userId 파라미터 필수)

    Args:
        request: FastAPI Request 객체
        userId: 사용자 ID (query parameter)

    Returns:
        chat.html 템플릿 또는 에러 메시지
    """
    # userId 파라미터 검증
    if not userId:
        return PlainTextResponse(
            "접근이 거부되었습니다. userId 파라미터가 필요합니다.\n\n"
            "올바른 접근 방법: /?userId=your_user_id",
            status_code=403
        )

    # userId 길이 검증 (최소 3자, 최대 50자)
    if len(userId) < 3 or len(userId) > 50:
        return PlainTextResponse(
            "유효하지 않은 userId입니다. (3-50자 필요)",
            status_code=400
        )

    print(f"[INFO] 챗봇 접속: userId={userId}")

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "userId": userId
    })


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
    is_valid, error_msg = validate_input(
        message,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH
    )
    if not is_valid:
        return PlainTextResponse(error_msg, status_code=400)

    # 4. AI 응답 생성
    ai_response_generator = get_ai_response(message, session_id=session_id)

    # 전체 응답 수집용 변수
    full_response = ""
    is_guardrail_reject = False

    async def event_stream():
        nonlocal full_response, is_guardrail_reject

        # 답변 스트리밍
        for chunk in ai_response_generator:
            # 가드레일 거부 마커 감지
            if chunk == "__GUARDRAIL_REJECT__":
                is_guardrail_reject = True
                continue  # 마커는 클라이언트에 전송하지 않음

            full_response += chunk
            yield chunk
            await asyncio.sleep(0.01)

        # 가드레일 거부 메시지는 DB에 저장하지 않음
        if is_guardrail_reject:
            print(f"[INFO] 가드레일 거부 메시지 - DB 저장 생략")
            return

        # 스트리밍 완료 후 DB에 저장 (정상 응답만)
        try:
            now = datetime.now()
            user_id = session_id[:20] if session_id else "anonymous"

            # INSERT 쿼리 (uuid는 DB에서 자동 생성됨)
            query = """
                INSERT INTO feedback
                (user_id, message, response, created_at)
                VALUES (%s, %s, %s, %s)
                RETURNING uuid
            """
            params = (user_id, message, full_response, now)

            # UUID 반환 받기
            from src.core.database import fetch_one
            result = fetch_one(query, params)

            if result:
                conversation_uuid = str(result[0])
                # UUID를 특수 형식으로 전달 (프론트엔드에서 파싱)
                yield f"\n__UUID:{conversation_uuid}"

        except Exception as e:
            print(f"[ERROR] 대화 저장 실패: {e}")

    return StreamingResponse(event_stream(), media_type="text/plain")


# 🔎 호출 테스트용 API (전체 모아서 반환)
@app.post("/chat-test")
async def chat_test(message: str = Form(...), session_id: str = Form("test-session")):
    # 1. Rate Limiting 체크
    allowed, rate_limit_msg = rate_limiter.is_allowed(session_id)
    if not allowed:
        return PlainTextResponse(rate_limit_msg, status_code=429)

    # 2. 입력 검증
    is_valid, error_msg = validate_input(
        message,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH
    )
    if not is_valid:
        return PlainTextResponse(error_msg, status_code=400)

    # 3. AI 응답 생성
    ai_response_generator = get_ai_response(message, session_id=session_id)
    chunks = []
    for chunk in ai_response_generator:
        chunks.append(chunk)
    return PlainTextResponse("".join(chunks))
# ✅ 사용자 피드백 저장 API
@app.post("/feedback")
async def feedback(data: dict = Body(...)):

    # 1) conversation_uuid 필수 확인
    conversation_uuid = data.get("conversation_uuid")
    if not conversation_uuid:
        return {"status": "error", "message": "conversation_uuid is required"}

    # 2) 피드백 데이터베이스에 업데이트
    try:
        now = datetime.now()
        user_name = data.get("name", "")  # ✅ 빈 값 허용 (up 피드백은 이름 없음)

        # UPDATE 쿼리 (uuid 기준으로 레코드 업데이트)
        query = """
            UPDATE feedback
            SET feedback = %s,
                reason = %s,
                comment = %s,
                name = %s,
                timestamp = %s,
                feedback_updated_at = %s
            WHERE uuid = %s
        """
        params = (
            data.get("feedback"),
            data.get("reason"),
            data.get("comment"),
            user_name,
            now,
            now,
            conversation_uuid
        )

        success = execute_query(query, params)
        if not success:
            print(f"[ERROR] 피드백 DB 업데이트 실패: {data}")
            return {"status": "error", "message": "Database update failed"}
    except Exception as e:
        print(f"[ERROR] 피드백 업데이트 중 오류 발생: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "ok"}
