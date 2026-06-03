import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
tokenizers = {}

sentiment_pipeline = None


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
    global sentiment_pipeline

    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )

    chunks = split_into_chunks(content, window=3, step=2)

    is_all_clearly_positive = True

    for chunk in chunks:
        sentiment_result = sentiment_pipeline(chunk)[0]
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

    if disease_type not in models:
        model_path = f"./models/trained_model_{disease_type}_binary"
        models[disease_type] = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        models[disease_type].eval()
        tokenizers[disease_type] = AutoTokenizer.from_pretrained(model_path)

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