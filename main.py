from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
from llm import get_ai_response, get_cache_info
import asyncio
import json, os, requests   # ✅ requests 추가


# FastAPI 앱 생성
app = FastAPI()
# 환경변수 로드
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

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
async def chat(message: str = Form(...)):
    ai_response_generator = get_ai_response(message)

    async def event_stream():
        for chunk in ai_response_generator:
            yield chunk
            await asyncio.sleep(0.01)

    return StreamingResponse(event_stream(), media_type="text/plain")


# 🔎 호출 테스트용 API (전체 모아서 반환)
@app.post("/chat-test")
async def chat_test(message: str = Form(...)):
    ai_response_generator = get_ai_response(message)
    chunks = []
    for chunk in ai_response_generator:
        chunks.append(chunk)
    return PlainTextResponse("".join(chunks))
# ✅ 사용자 피드백 저장 API
# ✅ 사용자 피드백 저장 API
@app.post("/feedback")
async def feedback(data: dict = Body(...)):
    import requests

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
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*💬 답변 (요약):*\n>{data.get('response')[:300]}..."
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":warning: *사유:*\n```{data.get('reason', '사유 미작성')}```"
                    }
                },
                {   # ✅ 코멘트 추가
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"✍️ *코멘트:*\n>{data.get('comment', '코멘트 없음')}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": ":loudspeaker: <!channel> 모든 분 확인 바랍니다."}
                    ]
                }
            ]
        }
        try:
            requests.post(SLACK_WEBHOOK_URL, json=message)
        except Exception as e:
            print(f"Slack 전송 오류: {e}")

    return {"status": "ok"}



# 서버 실행 (개발용)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
