import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response
import base64

# 👉 페이지 기본 설정
st.set_page_config(page_title="Xperp 문의 챗봇", page_icon="", layout="wide")

# 👉 로고 이미지를 base64로 변환하는 함수
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 👉 로고 경로 설정 (assets 폴더 내)
logo_base64 = get_image_base64("assets/m_logo.png")

# 👉 상단 헤더 영역 구성
st.markdown(f"""
    <style>
    .header-left {{flex: 1; width: 100%;}}
    .aegisep-link {{color: #fff !important; border-radius: 8px !important; text-decoration: none !important; margin-left: auto; display: inline-flex; align-items: center;}}
    .aegisep-link:hover {{color: #262626 !important;  }}
    .header-container {{
        width: 100%;
        background-color: #002c5f;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-sizing: border-box;
    }}
    .header-left {{
        display: flex;
        align-items: center;
    }}
    .header-logo {{
        height: 40px;
        margin-right: 18px;
    }}
    .header-link {{
        font-size: 14px;
        font-weight: 500;
        color: white;
        padding: 6px 14px;
        border: 1px solid white;
        border-radius: 4px;
        text-decoration: none;  /* ✅ 밑줄 제거 */
        transition: background-color 0.3s, color 0.3s;
    }}
    .header-link:hover {{
        background-color: white;
        color: #002c5f;
    }}
    </style>
    <div class="header-container">
        <div class="header-left">
            <img src="data:image/png;base64,{logo_base64}" class="header-logo" alt="Xperp Logo" />
            <a href="https://www.aegisep.com/aegisep/business/biz_apterp_info.jsp" target="_blank" rel="noopener noreferrer" class="header-link aegisep-link">
                        공식 홈페이지 바로가기
                    </a>
        </div>
    </div>
""", unsafe_allow_html=True)



# 👉 타이틀 영역
st.title("🤖 Xperp 문의 챗봇")
st.caption("Xperp 사용법이나 오류가 궁금하신가요? 저에게 물어보세요!")

# 👉 환경 변수 로드 (.env 사용 시)
load_dotenv()

# 👉 세션 상태 초기화
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 👉 이전 대화 메시지 출력
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 👉 사용자 입력 처리
if user_question := st.chat_input(placeholder="Xperp 사용법이나 오류에 대해 물어보세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
