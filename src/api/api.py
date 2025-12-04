from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
from src.llm.llm import get_ai_response, get_cache_info, get_llm, get_retriever, get_reranker, load_yearend_keywords, USE_RERANK
import asyncio
import json, os, time, re
from src.api.middleware.rate_limiter import RateLimiter
from src.core.guardrails import validate_input
from src.core.database import execute_query, test_connection
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

# Rate Limiting (테스트용 - 1분에 100회로 완화)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
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
            "접근이 거부되었습니다.",
            status_code=403
        )

    # userId 길이 검증 (최소 3자, 최대 50자)
    if len(userId) < 3 or len(userId) > 50:
        return PlainTextResponse(
            "접근이 거부되었습니다.",
            status_code=403
        )

    print(f"[INFO] 챗봇 접속: userId={userId}")

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "userId": userId
    })


# ✅ 채팅 API 엔드포인트 (스트리밍 버전)
@app.post("/chat")
async def chat(message: str = Form(...), session_id: str = Form(None), user_id: str = Form(None)):
    # ⏱️ 성능 측정 시작
    request_start = time.time()
    print(f"\n{'='*70}")
    print(f"[PERF] 요청 시작: {message[:50]}...")
    print(f"{'='*70}")

    # 1. session_id 검증
    if not session_id:
        return PlainTextResponse("session_id is required", status_code=400)

    # 2. Rate Limiting 체크
    step_start = time.time()
    allowed, rate_limit_msg = rate_limiter.is_allowed(session_id)
    print(f"[PERF] Rate Limiting 체크: {(time.time() - step_start)*1000:.2f}ms")
    if not allowed:
        return PlainTextResponse(rate_limit_msg, status_code=429)

    # 3. 입력 검증
    step_start = time.time()
    is_valid, error_msg = validate_input(
        message,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH
    )
    print(f"[PERF] 입력 검증: {(time.time() - step_start)*1000:.2f}ms")
    if not is_valid:
        return PlainTextResponse(error_msg, status_code=400)

    # 4. AI 응답 생성
    step_start = time.time()
    ai_response_generator = get_ai_response(message, session_id=session_id)
    print(f"[PERF] AI 응답 생성기 초기화: {(time.time() - step_start)*1000:.2f}ms")

    # 전체 응답 수집용 변수
    full_response = ""
    is_guardrail_reject = False
    first_token_received = False

    async def event_stream():
        nonlocal full_response, is_guardrail_reject, first_token_received

        # 답변 스트리밍
        for chunk in ai_response_generator:
            # 가드레일 거부 마커 감지
            if chunk == "__GUARDRAIL_REJECT__":
                is_guardrail_reject = True
                continue  # 마커는 클라이언트에 전송하지 않음

            # ⏱️ 첫 토큰 도달 시간 측정
            if not first_token_received and chunk.strip():
                first_token_time = (time.time() - request_start) * 1000
                print(f"[PERF] ✨ 첫 토큰 도달: {first_token_time:.2f}ms")
                print(f"{'='*70}\n")
                first_token_received = True

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
            # user_id: URL 파라미터에서 전달된 값 사용 (없으면 session_id에서 추출)
            db_user_id = user_id if user_id else (session_id[:20] if session_id else "anonymous")

            # conversation_id = session_id 사용
            conversation_id = session_id if session_id else f"conv_{db_user_id}_{int(now.timestamp() * 1000)}"

            # 첫 메시지 여부 확인 (해당 conversation_id로 기존 메시지가 있는지)
            from src.core.database import fetch_one
            check_query = """
                SELECT COUNT(*)
                FROM feedback
                WHERE conversation_id = %s
            """
            count_result = fetch_one(check_query, (conversation_id,))
            is_first_message = (count_result[0] == 0) if count_result else True

            # INSERT 쿼리 (conversation_id, is_first_message 추가)
            query = """
                INSERT INTO feedback
                (user_id, conversation_id, is_first_message, message, response, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING uuid
            """
            params = (db_user_id, conversation_id, is_first_message, message, full_response, now)

            # UUID 반환 받기
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


# ✅ 대화 목록 조회 API
@app.get("/conversations")
async def get_conversations(user_id: str):
    """
    사용자의 대화 목록 조회 (ChatGPT 스타일)

    Args:
        user_id: 사용자 ID (URL 파라미터)

    Returns:
        JSON 응답:
        - conversations: 대화 목록 배열
          - conversation_id: 대화 고유 ID
          - first_message: 첫 메시지 (제목 대신 사용)
          - message_count: 대화 내 메시지 수
          - created_at: 대화 시작 시간
          - updated_at: 마지막 메시지 시간
    """
    try:
        from src.core.database import fetch_all_dict

        query = """
            SELECT
                f1.conversation_id,
                (
                    SELECT message
                    FROM feedback f2
                    WHERE f2.conversation_id = f1.conversation_id
                    AND f2.is_first_message = true
                    LIMIT 1
                ) as first_message,
                COUNT(*) as message_count,
                MIN(f1.created_at) as created_at,
                MAX(f1.created_at) as updated_at
            FROM feedback f1
            WHERE f1.user_id = %s
            AND f1.conversation_id IS NOT NULL
            GROUP BY f1.conversation_id
            ORDER BY MAX(f1.created_at) DESC
            LIMIT 50
        """

        conversations = fetch_all_dict(query, (user_id,))

        # 시간 포맷 변환 (ISO 형식)
        for conv in conversations:
            if conv.get('created_at'):
                conv['created_at'] = conv['created_at'].isoformat()
            if conv.get('updated_at'):
                conv['updated_at'] = conv['updated_at'].isoformat()

        return {
            "status": "ok",
            "conversations": conversations
        }

    except Exception as e:
        print(f"[ERROR] 대화 목록 조회 실패: {e}")
        return {
            "status": "error",
            "message": str(e),
            "conversations": []
        }


# ✅ 특정 대화의 메시지 조회 API
@app.get("/conversation/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, user_id: str = None):
    """
    특정 대화의 모든 메시지 조회

    Args:
        conversation_id: 대화 고유 ID (path parameter)
        user_id: 사용자 ID (query parameter, 선택)

    Returns:
        JSON 응답:
        - messages: 메시지 배열 (시간순)
          - message: 사용자 질문
          - response: AI 응답
          - created_at: 생성 시간
    """
    try:
        from src.core.database import fetch_all_dict

        # user_id 검증이 필요한 경우 WHERE 조건 추가
        if user_id:
            query = """
                SELECT uuid, message, response, created_at, feedback, reason, comment
                FROM feedback
                WHERE conversation_id = %s AND user_id = %s
                ORDER BY created_at ASC
            """
            params = (conversation_id, user_id)
        else:
            query = """
                SELECT uuid, message, response, created_at, feedback, reason, comment
                FROM feedback
                WHERE conversation_id = %s
                ORDER BY created_at ASC
            """
            params = (conversation_id,)

        messages = fetch_all_dict(query, params)

        # 시간 포맷 변환 (ISO 형식)
        for msg in messages:
            if msg.get('created_at'):
                msg['created_at'] = msg['created_at'].isoformat()

        return {
            "status": "ok",
            "conversation_id": conversation_id,
            "messages": messages
        }

    except Exception as e:
        print(f"[ERROR] 대화 메시지 조회 실패: {e}")
        return {
            "status": "error",
            "message": str(e),
            "messages": []
        }


# ✅ 대화 삭제 API
@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str = None):
    """
    특정 대화 삭제 (해당 대화의 모든 메시지 삭제)

    Args:
        conversation_id: 삭제할 대화 ID (path parameter)
        user_id: 사용자 ID (query parameter, 선택 - 보안을 위해 권장)

    Returns:
        JSON 응답:
        - status: "ok" 또는 "error"
        - deleted_count: 삭제된 메시지 수
    """
    try:
        from src.core.database import execute_query, fetch_one

        # user_id가 제공된 경우 소유권 확인
        if user_id:
            check_query = """
                SELECT COUNT(*) as count
                FROM feedback
                WHERE conversation_id = %s AND user_id = %s
            """
            result = fetch_one(check_query, (conversation_id, user_id))

            if not result or result[0] == 0:
                return {
                    "status": "error",
                    "message": "대화를 찾을 수 없거나 권한이 없습니다.",
                    "deleted_count": 0
                }

        # 삭제 실행
        delete_query = """
            DELETE FROM feedback
            WHERE conversation_id = %s
        """

        if user_id:
            delete_query += " AND user_id = %s"
            params = (conversation_id, user_id)
        else:
            params = (conversation_id,)

        success = execute_query(delete_query, params)

        if success:
            return {
                "status": "ok",
                "message": "대화가 삭제되었습니다.",
                "conversation_id": conversation_id
            }
        else:
            return {
                "status": "error",
                "message": "대화 삭제에 실패했습니다."
            }

    except Exception as e:
        print(f"[ERROR] 대화 삭제 실패: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ✅ 헬스 체크 API
@app.get("/health")
async def health_check():
    """
    서버 헬스 체크 엔드포인트

    Returns:
        JSON 응답:
        - status: 서버 상태 (ok/degraded)
        - database_connected: DB 연결 상태 (true/false)
        - response_time_ms: 응답 시간 (밀리초)
    """
    start_time = time.time()

    # DB 연결 체크
    db_connected, _ = test_connection()

    # 전체 상태 결정
    if db_connected:
        overall_status = "ok"
    else:
        overall_status = "degraded"  # DB 없어도 챗봇은 작동하므로 degraded

    response_time_ms = round((time.time() - start_time) * 1000, 2)

    return JSONResponse(
        status_code=200,
        content={
            "status": overall_status,
            "database_connected": db_connected,
            "response_time_ms": response_time_ms
        }
    )
