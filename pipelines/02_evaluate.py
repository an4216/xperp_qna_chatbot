import os
import json
import time
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import subprocess
import sys

from llm import get_ai_response, get_embeddings, get_llm

# MLflow 추가
import mlflow

LOG_LOW_SCORE_DIR = "logs/low_scores"
LOG_FEEDBACK_DOWN = "logs/feedback_down.json"
LOW_SCORE_FILE = "logs/low_score.json"
EVAL_FILE = "data/eval/eval_set.json"
LOW_SCORE_THRESHOLD = 70   # LLM Judge 점수 기준

# ================================
# Cosine Similarity 기반 평가
# ================================
embeddings = get_embeddings()

def embed_text(text: str):
    return embeddings.embed_query(text)

def similarity_score(expected: str, actual: str) -> float:
    vec1 = np.array(embed_text(expected)).reshape(1, -1)
    vec2 = np.array(embed_text(actual)).reshape(1, -1)
    score = cosine_similarity(vec1, vec2)[0][0]
    return float(score * 100)

# ================================
# LLM Judge 기반 평가
# ================================
def llm_judge_score(expected: str, actual: str) -> float:
    llm = get_llm()
    prompt = f"""
    당신은 평가 전문가로서, 두 텍스트(기대 답변과 실제 답변)의 의미적 일치도를 평가해야 합니다.
    이 평가는 **NLI(Natural Language Inference)** 및 **Semantic Textual Similarity(STS)** 원칙을 기반으로 합니다.
    또한, 기대 답변(expected)은 '골든 레퍼런스(golden reference)'로 간주됩니다.

    ### 평가 기준:
    - 100점: 의미적으로 완전히 동일 (표현이 달라도 핵심 의도/정보가 완벽히 일치)
    - 70~99점: 핵심 의미는 동일하나, 일부 표현·세부 설명·부가 맥락에서 차이가 있음
    - 40~69점: 부분적으로만 일치 (일부는 맞지만 중요한 사실이 누락되거나 잘못됨)
    - 1~39점: 대부분 불일치 (거의 다른 내용이나 잘못된 설명)
    - 0점: 완전히 불일치 (전혀 다른 답변)

    기대 답변은 항상 기준(golden reference)이며, 실제 답변이 얼마나 잘 부합하는지 의미적으로 평가하세요.
    평가 시 단순한 단어 겹침이 아니라 **논리적 포함관계(entailment)**와 **의미 유사성**을 중점적으로 고려하세요.

    ✅ 기대 답변 (Golden Reference):
    {expected}

    🤖 실제 답변 (Model Output):
    {actual}

    출력 형식: 숫자만 (예: 85)
    """
    result = llm.invoke(prompt)
    try:
        score = int("".join([c for c in result.content if c.isdigit()]))
        return min(max(score, 0), 100)
    except:
        return 0

# ================================
# Feedback 로드 (down만 추출)
# ================================
def load_down_feedback(feedback_file="logs/feedback.jsonl"):
    down_list = []
    if not os.path.exists(feedback_file):
        return down_list

    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                fb = json.loads(line.strip())
                if fb.get("feedback") == "down":
                    down_list.append({
                        "message": fb.get("message", ""),
                        "response": fb.get("response", ""),
                        "reason": fb.get("reason", ""),
                        "comment": fb.get("comment", ""),
                        "timestamp": fb.get("timestamp", "")
                    })
            except:
                continue
    return down_list

# ================================
# Low score 관리
# ================================
def normalize_q(s: str) -> str:
    """질문 텍스트 정규화 (공백, 대소문자)"""
    return " ".join(s.strip().split()).lower()

def update_low_score(low_score_logs):
    """ 새로 발견된 low_score QA를 low_score.json에 병합 (중복 제거) """
    if not low_score_logs:
        return

    existing = []
    if os.path.exists(LOW_SCORE_FILE):
        with open(LOW_SCORE_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # 기존 + 신규 통합 (중복 제거)
    seen = {normalize_q(x["question"]): x for x in existing}
    for item in low_score_logs:
        key = normalize_q(item["question"])
        seen[key] = item  # 최신 기록으로 갱신

    merged = list(seen.values())

    with open(LOW_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"⚠️ low_score.json 갱신됨 (총 {len(merged)}개)")

def clean_low_score(results):
    """ 재평가 결과에서 점수 >= 70인 항목은 low_score.json에서 제거하고 해소된 항목 추적 """
    if not os.path.exists(LOW_SCORE_FILE):
        return []

    with open(LOW_SCORE_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)

    remaining = []
    resolved_items = []

    for item in existing:
        q_key = normalize_q(item["question"])
        match = next((r for r in results if normalize_q(r["question"]) == q_key), None)
        if match and match["llm_judge_score"] >= LOW_SCORE_THRESHOLD:
            resolved_item = {
                "question": item["question"],
                "previous_score": item.get("llm_judge_score", 0),
                "new_score": match["llm_judge_score"],
                "resolved_at": datetime.now().isoformat(),
                "improvement": match["llm_judge_score"] - item.get("llm_judge_score", 0)
            }
            resolved_items.append(resolved_item)
            print(f"✅ 개선됨 → low_score.json에서 제거: {item['question']} ({item.get('llm_judge_score', 0):.1f} → {match['llm_judge_score']:.1f})")
            continue
        remaining.append(item)

    with open(LOW_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)

    print(f"🧹 low_score.json 정리 완료 (남은 항목 {len(remaining)}개, 해소된 항목 {len(resolved_items)}개)")
    return resolved_items

# ================================
# 평가 실행
# ================================
def run_evaluation(eval_data):
    results = []
    low_score_logs = []
    low_file = None

    for idx, item in enumerate(eval_data, 1):
        question = item["question"]
        expected = item["expected"]

        print(f"\n🔎 [{idx}] 질문: {question}")
        print(f"✅ 기대 답변: {expected}")

        start = time.perf_counter()
        response = "".join(get_ai_response(question))
        elapsed = time.perf_counter() - start

        print(f"🤖 모델 답변: {response}")

        sim_score = similarity_score(expected, response)
        print(f"📊 유사도 점수: {sim_score:.2f}")

        judge_score = llm_judge_score(expected, response)
        print(f"🧑‍⚖️ LLM Judge 점수: {judge_score:.2f}")

        result_item = {
            "question": question,
            "expected": expected,
            "response": response,
            "similarity_score": sim_score,
            "llm_judge_score": judge_score,
            "time_taken": elapsed
        }
        results.append(result_item)

        if judge_score < LOW_SCORE_THRESHOLD:
            low_score_logs.append(result_item)

    # 점수 낮은 QA 로그 저장 (이번 실행만 기록)
    if low_score_logs:
        os.makedirs(LOG_LOW_SCORE_DIR, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        low_file = os.path.join(LOG_LOW_SCORE_DIR, f"low_score_{today}.json")
        with open(low_file, "w", encoding="utf-8") as f:
            json.dump(low_score_logs, f, ensure_ascii=False, indent=2)
        print(f"⚠️ 낮은 점수 QA {len(low_score_logs)}개 저장됨 → {low_file}")

    return results, low_score_logs, low_file

# ================================
# Main
# ================================
if __name__ == "__main__":
    eval_data = []

    # 1) eval_set.json 로드
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            eval_data.extend(json.load(f))

    # 2) 기존 low_score.json도 합쳐서 평가
    if os.path.exists(LOW_SCORE_FILE):
        with open(LOW_SCORE_FILE, "r", encoding="utf-8") as f:
            eval_data.extend(json.load(f))

    if not eval_data:
        print("❌ 평가할 데이터가 없습니다.")
        exit(1)

    print(f"🚀 총 {len(eval_data)}개 질문에 대해 평가 시작...")

    # 평가 실행
    results, low_score_logs, low_file = run_evaluation(eval_data)

    # low_score.json 업데이트 & 정리
    update_low_score(low_score_logs)
    clean_low_score(results)

    # 결과 저장
    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 평가 완료! 결과 저장됨: {out_file}")

    # MLflow 로깅
    avg_similarity = sum(r["similarity_score"] for r in results) / len(results)
    avg_judge = sum(r["llm_judge_score"] for r in results) / len(results)

    mlflow.set_experiment("xperp_chatbot_eval")
    with mlflow.start_run():
        mlflow.log_param("embedding_model", "BAAI/bge-m3")
        mlflow.log_param("llm_model", "unsloth/gemma-3-27b-it")
        mlflow.log_param("eval_size", len(eval_data))

        mlflow.log_metric("avg_similarity", avg_similarity)
        mlflow.log_metric("avg_judge", avg_judge)

        mlflow.log_artifact(out_file)

        # ✅ 이번 실행에서 생성된 low_score 파일만 업로드
        if low_file:
            mlflow.log_artifact(low_file)
            print(f"⚠️ 이번 검사 low_score 로그 업로드 완료 → {low_file}")

        down_feedback = load_down_feedback()
        if down_feedback:
            with open(LOG_FEEDBACK_DOWN, "w", encoding="utf-8") as f:
                json.dump(down_feedback, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(LOG_FEEDBACK_DOWN)
            print(f"⚠️ down feedback {len(down_feedback)}개 저장됨 → {LOG_FEEDBACK_DOWN}")

    print("📊 MLflow에 평가 결과 기록 완료!")

    # ✅ 인덱스 재생성 (low_score 반영)
    subprocess.run([sys.executable, "-m", "pipelines.01_ingest"], check=True)
    print("🔄 인덱스 재생성 완료 (low_score 반영)")
