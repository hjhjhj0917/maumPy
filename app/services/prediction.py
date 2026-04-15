import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 모델 로드 (앱 시작 시 한 번만 로드하는 것이 효율적입니다)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {}
tokenizers = {}

def load_model(disease_type):
    if disease_type not in models:
        model_path = f"./models/trained_model_kluebert_{disease_type}"
        models[disease_type] = BertForSequenceClassification.from_pretrained(model_path).to(device)
        models[disease_type].eval()
        tokenizers[disease_type] = BertTokenizer.from_pretrained(model_path)

def get_analysis_summary(final_level):
    summary_map = {
        0: "정상 범위 내의 정서 상태",
        1: "경미한 우울 증상 의심",
        2: "중간 수준의 우울 증상 의심",
        3: "심화된 우울 증상 의심"
    }
    return summary_map.get(final_level, "분석 결과 없음")

def predict(sentence, model, tokenizer, disease_type):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    dep_lvl = int(torch.argmax(probs).item())
    raw_score = float(sum(i * prob for i, prob in enumerate(probs)))

    result = {
        "ANALYSIS_SUM": get_analysis_summary(dep_lvl),
        "DEP_RES": {
            "DISEASE_TYPE": disease_type,
            "DEP_LVL": int(dep_lvl),
            "DEP_SCORE": round(raw_score, 2),
            "IS_SYMPTOM": bool(dep_lvl != 0)
        }
    }
    return result

def analyze_diary(content: str, disease_type: str = "depression"):
    load_model(disease_type)
    return predict(content, models[disease_type], tokenizers[disease_type], disease_type)