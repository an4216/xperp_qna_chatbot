from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
from llm import get_ai_response
import asyncio

# FastAPI 앱 생성
app = FastAPI()

# HTML 템플릿 설정
templates = Jinja2Templates(directory="templates")
# 👉 환경 변수 로드 (.env 사용 시)
load_dotenv()
# 정적 파일(CSS, JS) 설정 (필요시)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 경로 ("/")
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # templates/chat.html 렌더링
    return templates.TemplateResponse("chat.html", {"request": request})

# 채팅 API 엔드포인트 (SSE)
@app.post("/chat")
async def chat(message: str = Form(...)):
    ai_response_generator = get_ai_response(message)

    # 제너레이터에서 모두 모아서 문자열로 변환
    chunks = []
    for chunk in ai_response_generator:
        chunks.append(chunk)

    full_response = "".join(chunks)

    return PlainTextResponse(full_response)

# 🔎 호출 테스트용 API (SSE 없이 즉시 텍스트 반환)
# - Postman에서 본문이 바로 보이도록 PlainText로 응답
@app.post("/chat-test")
async def chat_test(message: str = Form(...)):
    ai_response_generator = get_ai_response(message)
    chunks = []
    for chunk in ai_response_generator:
        chunks.append(chunk)
    return PlainTextResponse("".join(chunks))

# 서버 실행 (개발용)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
