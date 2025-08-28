import os
import json
import time
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from llm import get_ai_response, get_embeddings, get_llm

# MLflow 추가
import mlflow

LOG_LOW_SCORE = "logs/low_score.json"
LOG_FEEDBACK_DOWN = "logs/feedback_down.json"
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
    다음은 사용자가 기대한 답변과 실제 모델의 답변입니다.
    두 답변의 의미가 얼마나 유사한지 0~100점으로 채점하세요.
    - 100점: 의미적으로 완전히 동일
    - 70점 이상: 핵심 의미는 동일하나 표현이나 부가 설명이 조금 다름
    - 40점 이상: 일부는 맞지만 중요한 차이가 있음
    - 0점: 전혀 다른 답변

    ✅ 기대 답변:
    {expected}

    🤖 실제 답변:
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
# 평가 실행
# ================================
def run_evaluation(eval_data):
    results = []
    low_score_logs = []

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

        # ✅ LLM Judge 기준 70점 미만만 저장
        if judge_score < LOW_SCORE_THRESHOLD:
            low_score_logs.append(result_item)

    # 점수 낮은 QA 로그 저장
    if low_score_logs:
        os.makedirs(Path(LOG_LOW_SCORE).parent, exist_ok=True)
        with open(LOG_LOW_SCORE, "w", encoding="utf-8") as f:
            json.dump(low_score_logs, f, ensure_ascii=False, indent=2)
        print(f"⚠️ 낮은 점수 QA {len(low_score_logs)}개 저장됨 → {LOG_LOW_SCORE}")

    return results

# ================================
# Main
# ================================
if __name__ == "__main__":
    eval_file = "data/eval/eval_set.json"
    if not os.path.exists(eval_file):
        print(f"❌ 평가셋 파일 없음: {eval_file}")
        exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"🚀 총 {len(eval_data)}개 질문에 대해 평가 시작...")

    # 평가 실행
    results = run_evaluation(eval_data)

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
        if os.path.exists(LOG_LOW_SCORE):
            mlflow.log_artifact(LOG_LOW_SCORE)

        # ✅ down feedback도 MLflow에 기록 (comment 포함)
        down_feedback = load_down_feedback()
        if down_feedback:
            with open(LOG_FEEDBACK_DOWN, "w", encoding="utf-8") as f:
                json.dump(down_feedback, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(LOG_FEEDBACK_DOWN)
            print(f"⚠️ down feedback {len(down_feedback)}개 저장됨 → {LOG_FEEDBACK_DOWN}")

    print("📊 MLflow에 평가 결과 기록 완료!")
