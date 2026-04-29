import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification

# 장치 설정하는 코드로 GPU사용 여부를 확인, 사용 불가능시 CPU 사용
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
tokenizers = {}

# 결과 요약 함수
# 모델이 출력한 숫자 형태의 레벨(0, 1, 2, 3)을 사용자가 이해하기 쉬운 한국어 문장으로 변환해 주는 맵핑 함수입니다
def get_analysis_summary(final_level):
    summary_map = {0: "정상 범위 내의 정서 상태", 1: "경미한 우울 증상 의심", 2: "중간 수준의 우울 증상 의심", 3: "심화된 우울 증상 의심"}
    return summary_map.get(final_level, "분석 결과 없음")

# 핵심 분석 함수
def analyze_diary(content: str, disease_type: str = "depression"):

    # 학습된 모델을 로드
    if disease_type not in models:
        model_path = f"./models/trained_model_kluebert_{disease_type}"
        models[disease_type] = BertForSequenceClassification.from_pretrained(model_path).to(DEVICE)

        # 학습 시에만 필요한 기능을 끄고 추론에 최적화된 상태를 만듬
        models[disease_type].eval()

        tokenizers[disease_type] = BertTokenizer.from_pretrained(model_path)

    # 입력된 문장을 토큰으로 나누고, 길이를 맞추며(padding),
    # 너무 길면 자릅니다(truncation). 결과를 파이토치 텐서(pt) 형식으로 변환하여 지정된 장치(GPU/CPU)로 보냅니다
    inputs = tokenizers[disease_type](content, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)

    # 추론 과정에서는 가중치를 업데이트할 필요가 없으므로 메모리 절약을 위해 기울기 계산을 끕니다.
    with torch.no_grad():
        outputs = models[disease_type](**inputs)

    # softmax: 모델의 결과물(Logits)을 4개 클래스별 확률 값(합계 1.0)으로 변환합니다.
    # raw_score: 각 클래스의 인덱스(0, 1, 2, 3)에 해당 확률을 곱해 더한 가중 평균 점수입니다.
    # 예를 들어 '심화(3)'의 확률이 높을수록 이 점수는 3에 가까워집니다.
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()
    raw_score = float(sum(i * prob for i, prob in enumerate(probs)))

    # 임계값 적용
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