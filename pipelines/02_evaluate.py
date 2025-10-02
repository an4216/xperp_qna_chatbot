# evaluate.py
import os
import json
import time
import re
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
# 유틸
# ================================
_TIME_LINE_RE = re.compile(r"^\s*⏱\s*소요시간:.*$", re.MULTILINE)

def sanitize_response(s: str) -> str:
    """평가에 불필요한 라인 제거 및 정리."""
    if not s:
        return ""
    # get_ai_response 마지막 타임라인 제거
    s = _TIME_LINE_RE.sub("", s)
    # 제어문자 제거
    s = "".join(ch for ch in s if ch == "\n" or ord(ch) >= 32)
    # 양끝 공백 정리
    return s.strip()

def safe_collect_response_from_stream(question: str, session_id: str = "eval", retries: int = 1) -> str:
    """
    스트리밍 수집 중 예외가 나도 폴백으로 이어서 평가가 가능하도록 함.
    1) get_ai_response 스트림 수집 시도
    2) 실패하면 get_llm().invoke(question) 폴백
    3) 두 경로 모두 실패 시 빈 문자열 반환
    """
    # 1) 스트림 수집
    attempt = 0
    while attempt <= retries:
        try:
            buf_parts = []
            for chunk in get_ai_response(question, session_id=session_id):
                # chunk가 dict/객체일 가능성 방지
                if chunk is None:
                    continue
                buf_parts.append(str(chunk))
            return sanitize_response("".join(buf_parts))
        except Exception as e:
            print(f"[WARN] stream 실패 (attempt={attempt}/{retries}): {type(e).__name__} - {e}")
            attempt += 1
            if attempt <= retries:
                time.sleep(0.3)

    # 2) 폴백: LLM 직접 단발 호출 (RAG 미사용)
    try:
        llm = get_llm()
        result = llm.invoke(question)
        text = getattr(result, "content", None)
        if not text:
            # openai 호환 드라이버가 dict로 줄 가능성 방지
            text = str(result)
        return sanitize_response(text)
    except Exception as e:
        print(f"[ERROR] fallback LLM 호출도 실패: {type(e).__name__} - {e}")
        return ""

# ================================
# LLM Judge 기반 평가
# ================================
def llm_judge_score(expected: str, actual: str) -> float:
    llm = get_llm()
    prompt = f"""
    당신은 평가 전문가로서, 두 텍스트(기대 답변과 실제 답변)의 의미적 일치도를 평가해야 합니다.
    이 평가는 NLI 및 STS 원칙을 기반으로 하며, 기대 답변(expected)은 골든 레퍼런스입니다.

    ### 기준:
    - 100: 의미 완전 동일
    - 70~99: 핵심 동일, 일부 표현/세부 차이
    - 40~69: 일부만 일치(중요 사실 누락/오류)
    - 1~39: 대부분 불일치
    - 0: 완전 불일치

    ✅ 기대 답변:
    {expected}

    🤖 실제 답변:
    {actual}

    출력 형식: 숫자만 (예: 85)
    """.strip()

    try:
        result = llm.invoke(prompt)
        raw = getattr(result, "content", "") if result is not None else ""
        digits = "".join(c for c in str(raw) if c.isdigit())
        score = int(digits) if digits else 0
        return max(0, min(100, score))
    except Exception as e:
        print(f"[WARN] llm_judge_score 실패: {type(e).__name__} - {e}")
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
            except Exception:
                continue
    return down_list

# ================================
# Low score 관리
# ================================
def normalize_q(s: str) -> str:
    return " ".join(s.strip().split()).lower()

def update_low_score(low_score_logs):
    if not low_score_logs:
        return

    existing = []
    if os.path.exists(LOW_SCORE_FILE):
        with open(LOW_SCORE_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    seen = {normalize_q(x["question"]): x for x in existing}
    for item in low_score_logs:
        seen[normalize_q(item["question"])] = item

    merged = list(seen.values())
    with open(LOW_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"⚠️ low_score.json 갱신됨 (총 {len(merged)}개)")

def clean_low_score(results):
    if not os.path.exists(LOW_SCORE_FILE):
        return []

    with open(LOW_SCORE_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)

    remaining, resolved_items = [], []
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
            print(f"✅ 개선됨 → 제거: {item['question']} ({item.get('llm_judge_score', 0):.1f} → {match['llm_judge_score']:.1f})")
        else:
            remaining.append(item)

    with open(LOW_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)

    print(f"🧹 정리 완료 (남음 {len(remaining)}개, 해소 {len(resolved_items)}개)")
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
        # ✅ 스트림 안전 수집 + 폴백
        response = safe_collect_response_from_stream(question, session_id="eval", retries=1)
        elapsed = time.perf_counter() - start

        # 비어있어도 전체 평가는 계속 진행 (0점 처리)
        print(f"🤖 모델 답변: {response if response else '[EMPTY RESPONSE]'}")

        sim_score = similarity_score(expected, response) if response else 0.0
        print(f"📊 유사도 점수: {sim_score:.2f}")

        judge_score = llm_judge_score(expected, response) if response else 0.0
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

    # 이번 실행의 low score만 따로 저장
    if low_score_logs:
        os.makedirs(LOG_LOW_SCORE_DIR, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        low_file = os.path.join(LOG_LOW_SCORE_DIR, f"low_score_{today}.json")
        with open(low_file, "w", encoding="utf-8") as f:
            json.dump(low_score_logs, f, ensure_ascii=False, indent=2)
        print(f"⚠️ 낮은 점수 QA {len(low_score_logs)}개 저장 → {low_file}")

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
        sys.exit(1)

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

        # 이번 실행에서 생성된 low_score 파일만 업로드
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

    # 인덱스 재생성 (low_score 반영)
    subprocess.run([sys.executable, "-m", "pipelines.01_ingest"], check=True)
    print("🔄 인덱스 재생성 완료 (low_score 반영)")
