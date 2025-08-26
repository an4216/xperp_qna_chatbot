import os
import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from llm import get_ai_response, get_embeddings, get_llm

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
    """
    LLM에게 직접 채점을 요청 (0~100점)
    """
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
        return min(max(score, 0), 100)  # 0~100 범위 보정
    except:
        return 0


# ================================
# 평가 실행
# ================================
def run_evaluation(eval_data):
    results = []
    for idx, item in enumerate(eval_data, 1):
        question = item["question"]
        expected = item["expected"]

        print(f"\n🔎 [{idx}] 질문: {question}")
        print(f"✅ 기대 답변: {expected}")

        start = time.perf_counter()
        response = "".join(get_ai_response(question))
        elapsed = time.perf_counter() - start

        print(f"🤖 모델 답변: {response}")

        # Cosine Similarity 점수
        sim_score = similarity_score(expected, response)
        print(f"📊 유사도 점수: {sim_score:.2f}")

        # LLM Judge 점수
        judge_score = llm_judge_score(expected, response)
        print(f"🧑‍⚖️ LLM Judge 점수: {judge_score:.2f}")

        results.append({
            "question": question,
            "expected": expected,
            "response": response,
            "similarity_score": sim_score,
            "llm_judge_score": judge_score,
            "time_taken": elapsed
        })
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

    results = run_evaluation(eval_data)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ 평가 완료! 결과 저장됨: outputs/eval_results.json")
