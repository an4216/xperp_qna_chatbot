# pipelines/02_generate_eval.py
import os
import json
import random
from pathlib import Path

QNA_DIR = Path("docs/qna")
EVAL_FILE = Path("data/eval/eval_set.json")
MAX_TOTAL_QA = 10

def parse_line(line: str) -> str:
    """Q/A/T 라인을 안전하게 파싱해서 내용만 반환"""
    parts = line.split(":", 1)
    if len(parts) == 2:
        return parts[1].strip(" .\"\t")
    return line.strip(" .\"\t")

def load_qna_pairs():
    qa_pairs = []

    if not QNA_DIR.exists():
        print(f"❌ QNA 디렉토리 없음: {QNA_DIR}")
        return qa_pairs

    for qna_file in QNA_DIR.glob("*.txt"):
        print(f"📂 파일 로딩 중: {qna_file}")
        try:
            with open(qna_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            i = 0
            while i < len(lines):
                if lines[i].startswith("Q") and i + 2 < len(lines):
                    q_text = parse_line(lines[i])
                    a_text = parse_line(lines[i + 1]) if lines[i + 1].startswith("A") else ""
                    t_text = parse_line(lines[i + 2]) if lines[i + 2].startswith("T") else ""

                    qa_pairs.append({
                        "question": q_text,
                        "expected": a_text,
                        "tags": t_text,
                        "source": qna_file.name
                    })
                    i += 3
                else:
                    i += 1

        except Exception as e:
            print(f"⚠️ 파일 로딩 실패: {qna_file} ({e})")

    return qa_pairs

if __name__ == "__main__":
    print("🚀 docs/qna/ 디렉토리 내 Q/A 매핑 로딩...")
    qa_pairs = load_qna_pairs()
    print(f"📄 총 {len(qa_pairs)}개 Q/A 쌍 로드됨")

    if not qa_pairs:
        print("❌ Q/A 데이터 없음")
        exit(1)

    eval_data = random.sample(qa_pairs, min(MAX_TOTAL_QA, len(qa_pairs)))

    os.makedirs(EVAL_FILE.parent, exist_ok=True)
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 평가셋 저장 완료: {EVAL_FILE} (총 {len(eval_data)}개)")
