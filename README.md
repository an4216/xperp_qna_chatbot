
## 📘 Xperp 문의 챗봇 (Streamlit + LangChain 기반)

### ✅ Overview

이 프로젝트는 **LangChain + Streamlit**을 기반으로 동작하는 **RAG(Retrieval Augmented Generation)** 챗봇입니다.
사용자 질문에 대해 Xperp 프로그램의 **사용법 및 오류 해결 방법**을 제공하며,
대화 이력과 사전 예시 템플릿을 활용해 보다 정확한 답변을 생성합니다.

---

### ✅ Features

* 🧠 **LangChain 통합**: 벡터 DB 및 LLM 기반 검색/생성 기능
* 💬 **Streamlit 웹 인터페이스**: 직관적인 웹 기반 챗봇 UI
* 🔍 **RAG 방식 적용**: 문서 기반 답변 생성
* 🗂️ **Xperp 매뉴얼 기반 지식탐색**: `docs/sample.txt`에 저장된 매뉴얼 내용을 바탕으로 답변
* 🧾 **챗 이력 기반 문맥 이해**
* 📄 **Few-shot 예시 템플릿**으로 정제된 답변 제공

---

### ✅ Installation

#### 1. 프로젝트 클론

```bash
git clone https://github.com/your-org/xperp-chatbot.git
cd xperp-chatbot
```

#### 2. 가상환경 생성 및 활성화 (Windows 기준)

```bash
python -m venv venv
venv\\Scripts\\activate
```

#### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

---

### ✅ Usage

#### ▶ 실행 명령어 (Windows 기준)

```bash
streamlit run chat.py
```

브라우저가 자동으로 열리지 않으면 안내된 `http://localhost:8501` 주소로 직접 접속하세요.

---

### ✅ Project Structure

| 파일명                 | 설명                          |
| ------------------- | --------------------------- |
| `chat.py`           | Streamlit UI 구성 및 사용자 인터페이스 |
| `llm.py`            | LLM, RAG chain, 벡터 검색 설정    |
| `config.py`         | Few-shot 예시(입력/출력 템플릿) 정의   |
| `docs/sample.txt`   | Xperp 관련 문서 (검색용 문서)        |
| `assets/m_logo.png` | 상단 헤더 로고 이미지                |

---

### ✅ How It Works

1. **문서 로딩**
   → `docs/sample.txt`를 FAISS 벡터로 변환

2. **사용자 질문 입력**
   → 질문을 LangChain이 벡터 검색

3. **관련 문서 검색 → LLM으로 응답 생성**
   → Few-shot 예시 포함된 prompt 템플릿 사용

4. **Streamlit에서 채팅 형태로 응답 제공**

---

### ✅ 예시 질문

* Xperp 로그인 오류 해결 방법은?
* 거래처 등록은 어디서 하나요?
* 세금계산서 출력은 어떤 메뉴에 있나요?

---

### ✅ Contributing

* PR 및 이슈 환영합니다!
* 개선사항이나 버그는 언제든 공유해주세요 🙌

---

### ✅ Acknowledgments

* [LangChain](https://github.com/langchain-ai/langchain)
* [Streamlit](https://github.com/streamlit/streamlit)
