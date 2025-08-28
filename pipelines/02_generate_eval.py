# pipelines/02_generate_eval.py
import os
import json
import random
from pathlib import Path

QNA_FILE = "docs/qna/qna.txt"
EVAL_FILE = "data/eval/eval_set.json"
MAX_TOTAL_QA = 10

def load_qna_pairs():
    qa_pairs = []
    with open(QNA_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(len(lines)):
        if lines[i].startswith("Q") and i+1 < len(lines) and lines[i+1].startswith("A"):
            q_text = lines[i].split(":", 1)[1].strip(" .\"")
            a_text = lines[i+1].split(":", 1)[1].strip(" .\"")
            qa_pairs.append({
                "question": q_text,
                "expected": a_text
            })
    return qa_pairs

if __name__ == "__main__":
    print("🚀 qna.txt에서 Q/A 매핑 로딩...")
    qa_pairs = load_qna_pairs()
    print(f"📄 총 {len(qa_pairs)}개 Q/A 쌍 로드됨")

    if not qa_pairs:
        print("❌ Q/A 데이터 없음")
        exit(1)

    # 랜덤 샘플링
    eval_data = random.sample(qa_pairs, min(MAX_TOTAL_QA, len(qa_pairs)))

    # 저장
    os.makedirs(Path(EVAL_FILE).parent, exist_ok=True)
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 평가셋 저장 완료: {EVAL_FILE} (총 {len(eval_data)}개)")
