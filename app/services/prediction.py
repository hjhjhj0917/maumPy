import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1. 전역 변수 설정
SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 파일이 임포트될 때(서버 부팅 시) 모델을 단 한 번만 메모리에 적재합니다.
print("--- [prediction.py] AI 모델 전역 로딩 시작 ---")
try:
    sentiment_pipeline = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512
    )

    default_disease = "depression"
    model_path = f"./models/trained_model_{default_disease}_binary"

    models = {
        default_disease: AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    }
    models[default_disease].eval()

    tokenizers = {
        default_disease: AutoTokenizer.from_pretrained(model_path)
    }
    print("--- [prediction.py] AI 모델 전역 로딩 완벽하게 성공! ---")
except Exception as e:
    print(f"--- [prediction.py] 모델 로딩 중 에러 발생 (경로/이름 확인 필요): {e} ---")
    sentiment_pipeline = None
    models = {}
    tokenizers = {}


def get_analysis_summary(final_level):
    summary_map = {
        0: "정상 범위 내의 정서 상태",
        1: "우울 증상 의심 및 전문가 상담 권장"
    }
    return summary_map.get(final_level, "분석 결과 없음")


def split_into_chunks(text, window=3, step=2):
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]

    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + window])
        chunks.append(chunk)
        if i + window >= len(sentences):
            break
    return chunks


def analyze_diary(content: str, disease_type: str = "depression"):
    # 3. 함수 안에서는 매번 로드하지 않고, 이미 올라간 전역 변수를 '사용'만 합니다.
    global sentiment_pipeline
    global models
    global tokenizers

    if not sentiment_pipeline or disease_type not in models:
        print("[WARN] 모델이 아직 로드되지 않았습니다.")
        return {
            "summary": get_analysis_summary(0),
            "dep_res": {"final_level": 0, "raw_score": 0.0, "is_symptom": False}
        }

    chunks = split_into_chunks(content, window=3, step=2)
    is_all_clearly_positive = True

    # 1차 분류: 긍정적인 내용인지 빠른 확인
    for chunk in chunks:
        sentiment_result = sentiment_pipeline(chunk)[0]
        # (기존 로직 유지) 라벨이나 스코어 기준
        if sentiment_result['label'] != '1' or sentiment_result['score'] <= 0.70:
            is_all_clearly_positive = False
            break

    if is_all_clearly_positive:
        return {
            "summary": get_analysis_summary(0),
            "dep_res": {
                "final_level": 0,
                "raw_score": 0.0,
                "is_symptom": False
            }
        }

    # 2차 분류: 본격적인 우울증 모델 추론 (초고속 진행)
    chunk_probs = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizers[disease_type](
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE)

            outputs = models[disease_type](**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()
            chunk_probs.append(float(probs[1]))

    final_prob = sum(chunk_probs) / len(chunk_probs)
    dep_lvl = 1 if final_prob >= 0.50 else 0

    return {
        "summary": get_analysis_summary(dep_lvl),
        "dep_res": {
            "final_level": int(dep_lvl),
            "raw_score": round(final_prob, 4),
            "is_symptom": bool(dep_lvl == 1)
        }
    }