from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
from llm import get_ai_response, get_cache_info, get_embeddings, get_llm, get_retriever, get_reranker
import asyncio
import json, os, time
from httpx import AsyncClient, HTTPError


# FastAPI 앱 생성
app = FastAPI()
# 환경변수 로드
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

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
        # 1. 임베딩 모델 로드
        print("  [1/4] 임베딩 모델 로드 중...", end=" ", flush=True)
        t = time.time()
        get_embeddings()
        print(f"완료 ({time.time()-t:.1f}초)")

        # 2. Retriever 로드
        print("  [2/4] Retriever 로드 중...", end=" ", flush=True)
        t = time.time()
        get_retriever()
        print(f"완료 ({time.time()-t:.1f}초)")

        # 3. Reranker 로드
        print("  [3/4] Reranker 로드 중...", end=" ", flush=True)
        t = time.time()
        get_reranker()
        print(f"완료 ({time.time()-t:.1f}초)")

        # 4. LLM 로드
        print("  [4/4] LLM 로드 중...", end=" ", flush=True)
        t = time.time()
        get_llm()
        print(f"완료 ({time.time()-t:.1f}초)")

        elapsed = time.time() - start
        print("="*70)
        print(f"[INFO] 모델 사전 로드 완료! (총 {elapsed:.1f}초)")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n[ERROR] 모델 사전 로드 실패: {e}")
        print("첫 요청 시 모델이 로드됩니다.")

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
    # FE가 반드시 session_id를 넘기게 하고, 없으면 에러 리턴(개발 중엔 기본값 줄 수도 있음)
    if not session_id:
        return PlainTextResponse("session_id is required", status_code=400)

    ai_response_generator = get_ai_response(message, session_id=session_id)

    async def event_stream():
        for chunk in ai_response_generator:
            yield chunk
            await asyncio.sleep(0.01)

    return StreamingResponse(event_stream(), media_type="text/plain")


# 🔎 호출 테스트용 API (전체 모아서 반환)
@app.post("/chat-test")
async def chat_test(message: str = Form(...), session_id: str = Form("test-session")):
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



# 서버 실행 (개발용)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
