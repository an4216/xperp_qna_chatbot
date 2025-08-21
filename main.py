from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
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
# app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 경로 ("/")
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # templates/chat.html 렌더링
    return templates.TemplateResponse("chat.html", {"request": request})

# 채팅 API 엔드포인트 ("/chat")
@app.post("/chat")
async def chat(request: Request, message: str = Form(...) ):
    # llm.py의 get_ai_response 함수 호출
    ai_response_generator = get_ai_response(message)

    # 스트리밍 응답 생성
    async def stream_generator():
        response_content = ""
        for chunk in ai_response_generator:
            response_content += chunk
            # 스트리밍 효과를 위해 작은 지연 추가 (실제로는 필요 없을 수 있음)
            await asyncio.sleep(0.01)
        yield response_content

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# 서버 실행 (개발용)
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8051)
