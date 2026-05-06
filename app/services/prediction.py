import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
tokenizers = {}

def get_analysis_summary(final_level):
    summary_map = {
        0: "정상 범위 내의 정서 상태",
        1: "경미한 우울 증상 의심",
        2: "중간 수준의 우울 증상 의심",
        3: "심화된 우울 증상 의심"
    }
    return summary_map.get(final_level, "분석 결과 없음")

def analyze_diary(content: str, disease_type: str = "depression"):

    if disease_type not in models:
        model_path = f"./models/trained_model_kluebert_{disease_type}"
        models[disease_type] = BertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        models[disease_type].eval()

        tokenizers[disease_type] = BertTokenizer.from_pretrained(model_path)

    inputs = tokenizers[disease_type](content, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)

    with torch.no_grad():
        outputs = models[disease_type](**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()
    raw_score = float(sum(i * prob for i, prob in enumerate(probs)))

    if raw_score < 0.6: dep_lvl = 0
    elif raw_score < 1.65: dep_lvl = 1
    elif raw_score < 2.45: dep_lvl = 2
    else: dep_lvl = 3

    return {
        "summary": get_analysis_summary(dep_lvl),
        "dep_res": {
            "final_level": int(dep_lvl),
            "raw_score": round(raw_score, 4),
            "is_symptom": bool(dep_lvl != 0)
        }
    }