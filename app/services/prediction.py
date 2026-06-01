import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
tokenizers = {}


# [수정 1] 4단계였던 상태 메시지를 깔끔하게 2단계로 축소
def get_analysis_summary(final_level):
    summary_map = {
        0: "정상 범위 내의 정서 상태",
        1: "우울 증상 의심"
    }
    return summary_map.get(final_level, "분석 결과 없음")


def analyze_diary(content: str, disease_type: str = "depression"):
    if disease_type not in models:
        model_path = f"./models/trained_model_{disease_type}_binary"
        models[disease_type] = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        models[disease_type].eval()

        tokenizers[disease_type] = AutoTokenizer.from_pretrained(model_path)

    # 사용자 텍스트 토큰화
    inputs = tokenizers[disease_type](content, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
        DEVICE)

    # 결과 예측
    with torch.no_grad():
        outputs = models[disease_type](**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()

    prob_depressed = float(probs[1])

    if prob_depressed >= 0.50:
        dep_lvl = 1  # 우울 증상 있음
    else:
        dep_lvl = 0  # 정상

    return {
        "summary": get_analysis_summary(dep_lvl),
        "dep_res": {
            "final_level": int(dep_lvl),
            "raw_score": round(prob_depressed, 4),  # 우울증일 확률을 그대로 반환
            "is_symptom": bool(dep_lvl == 1)
        }
    }