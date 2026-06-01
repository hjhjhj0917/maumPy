import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
tokenizers = {}

# [추가] 1차 방어선용 감성 분석 파이프라인 (전역 변수)
sentiment_pipeline = None


def get_analysis_summary(final_level):
    summary_map = {
        0: "정상 범위 내의 정서 상태",
        1: "우울 증상 의심 및 전문가 상담 권장"
    }
    return summary_map.get(final_level, "분석 결과 없음")


def split_into_chunks(text, sentences_per_chunk=3):
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks


def analyze_diary(content: str, disease_type: str = "depression"):
    global sentiment_pipeline

    # ========================================================
    # [STEP 1] 1차 방어선: Koelectra 감성 분석 (긍정 필터링)
    # ========================================================
    if sentiment_pipeline is None:
        # 한국어 긍정/부정에 잘 학습된 공개 Koelectra 모델을 사용 (필요시 변경 가능)
        sentiment_model_name = SENTIMENT_MODEL_NAME
        sentiment_pipeline = pipeline(
            "text-classification",
            model=sentiment_model_name,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,  # 앞부분 문맥만 보고도 긍정 파악이 가능하므로 자름
            max_length=512
        )

    # 텍스트 감성 예측
    sentiment_result = sentiment_pipeline(content)[0]

    # 해당 모델은 '1'이 긍정, '0'이 부정을 의미합니다.
    # 만약 확고한 긍정(예: 긍정 확률 70% 이상)이라면 뒤쪽의 우울증 모델을 아예 돌리지 않습니다.
    is_clearly_positive = (sentiment_result['label'] == '1') and (sentiment_result['score'] > 0.70)

    if is_clearly_positive:
        print(f"[INFO] 긍정 텍스트 필터링됨 (확신도: {sentiment_result['score']:.2f})")
        return {
            "summary": get_analysis_summary(0),
            "dep_res": {
                "final_level": 0,
                "raw_score": 0.0,  # 긍정이므로 우울 확률 0%
                "is_symptom": False
            }
        }

    # ========================================================
    # [STEP 2] 2차 정밀 검사: RoBERTa 우울증 모델 (청킹 분석)
    # ========================================================
    if disease_type not in models:
        model_path = f"./models/trained_model_{disease_type}_binary"
        models[disease_type] = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        models[disease_type].eval()
        tokenizers[disease_type] = AutoTokenizer.from_pretrained(model_path)

    # 긍정이 아닌 텍스트(부정, 중립 등)만 잘게 쪼개서 심층 분석
    chunks = split_into_chunks(content, sentences_per_chunk=3)
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

    # 조각들의 평균 확률 계산
    final_prob = sum(chunk_probs) / len(chunk_probs)

    # 이진 분류 임계값 적용
    if final_prob >= 0.50:
        dep_lvl = 1
    else:
        dep_lvl = 0

    return {
        "summary": get_analysis_summary(dep_lvl),
        "dep_res": {
            "final_level": int(dep_lvl),
            "raw_score": round(final_prob, 4),
            "is_symptom": bool(dep_lvl == 1)
        }
    }