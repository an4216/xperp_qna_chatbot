# pipelines/02_generate_eval.py
import os
import json
import re
import random
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm import get_llm

DOCS_DIRS = ["docs/manual", "docs/qna"]
EVAL_FILE = "data/eval/eval_set.json"
MAX_TOTAL_QA = 10      # 최종 평가셋 크기
MAX_RAND_QA = 5        # 랜덤으로 생성할 개수

FEEDBACK_FILE = "logs/feedback.jsonl"

# ================================
# 문서 로딩
# ================================
def load_documents():
    documents = []
    for docs_dir in DOCS_DIRS:
        if not os.path.isdir(docs_dir):
            continue
        for filename in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
    return documents

# ================================
# Feedback 로딩
# ================================
def load_feedback():
    feedback_data = []
    if not os.path.exists(FEEDBACK_FILE):
        return feedback_data
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                fb = json.loads(line.strip())
                if fb.get("feedback") == "down" and fb.get("message") and fb.get("response"):
                    feedback_data.append({
                        "question": fb["message"],
                        "expected": fb["response"]
                    })
            except:
                continue
    return feedback_data

# ================================
# 랜덤 QA 생성
# ================================
def generate_random_eval(documents, num_questions=5):
    llm = get_llm()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # 여러 문서에서 랜덤 샘플 선택
    random.shuffle(chunks)
    selected_chunks = chunks[:min(len(chunks), num_questions*2)]

    eval_data = []
    for chunk in selected_chunks:
        text = chunk.page_content.strip()
        prompt = f"""
        다음 문서 내용을 기반으로 질문과 정식 답변 쌍을 1개 생성하세요.

        ⚠️ 조건:
        - 질문은 문서 내용을 이해해야 답할 수 있는 형태 (페이지/위치 질문 금지)
        - expected는 문서에 근거한 정식 답변 (설명/절차/기능)
        - 출력은 JSON 배열 ONLY
        - 예시:
        [
          {{"question": "검침 환경등록은 언제 사용하나요?",
            "expected": "검침환경등록은 전기와 수도 검침 방식을 설정할 때 사용합니다."}}
        ]

        문서 내용:
        {text}
        """
        try:
            result = llm.invoke(prompt)
            raw_content = result.content.strip()
            cleaned = re.sub(r"^```[a-zA-Z]*", "", raw_content, flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

            qa_pairs = json.loads(cleaned)
            if isinstance(qa_pairs, dict):
                qa_pairs = [qa_pairs]

            qa_pairs = [
                qa for qa in qa_pairs
                if not str(qa["expected"]).isdigit() and "페이지" not in qa["expected"]
            ]

            eval_data.extend(qa_pairs)
            if len(eval_data) >= num_questions:
                break

        except Exception as e:
            print(f"⚠️ 랜덤 QA 생성 실패: {e}")
            continue

    return eval_data[:num_questions]

# ================================
# Main
# ================================
if __name__ == "__main__":
    print("🚀 문서 로딩 시작...")
    docs = load_documents()
    print(f"📄 총 {len(docs)} 페이지 로드됨")

    print("👍 feedback 불러오기...")
    feedback_qas = load_feedback()

    print("🎲 랜덤 QA 생성...")
    random_qas = generate_random_eval(docs, num_questions=MAX_RAND_QA)

    # feedback 최대 5개만 사용
    feedback_qas = feedback_qas[:(MAX_TOTAL_QA - MAX_RAND_QA)]

    # 합치기
    eval_data = feedback_qas + random_qas
    eval_data = eval_data[:MAX_TOTAL_QA]

    if not eval_data:
        print("❌ 생성된 평가 데이터가 없습니다.")
        exit(1)

    os.makedirs(Path(EVAL_FILE).parent, exist_ok=True)
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 평가셋 저장 완료: {EVAL_FILE} (총 {len(eval_data)}개 질문)")
