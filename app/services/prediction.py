import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 전역 변수 설정
SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # AutoModelForSequenceClassification는 Hugging Face 라이브러리 기능
    # 모델의 종류를 자동으로 파악해서 텍스트 분류용 구조로 불러옴
    # from_pretrained(model_path)는 지정한 경로에 모델의 가중치를 불러옴
    # .to(DEVICE)는 앞서 선언한 GPU나 CPU 메모리에 모델을 올림
    models = {
        default_disease: AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    }
    models[default_disease].eval() # .eval기존에 학습에 치중된 모델을 평가 모드로 전환하는 함수

    # 텍스트를 모델이 인식가능한 숫자 형태로 변환 하는 토큰화 과정 세팅
    # 같은 토크나이저를 사용해서 분석할 수 있게 해당 경로에 있는 토크나이저 설정을 찾아서 세팅함
    tokenizers = {
        default_disease: AutoTokenizer.from_pretrained(model_path)
    }
    print("--- [prediction.py] AI 모델 전역 로딩 완벽하게 성공! ---")
except Exception as e:
    print(f"--- [prediction.py] 모델 로딩 중 에러 발생 (경로/이름 확인 필요): {e} ---")
    sentiment_pipeline = None
    models = {}
    tokenizers = {}


# 분석된 결과를 요약하는 함수
def get_analysis_summary(final_level):
    summary_map = {
        0: "정상 범위 내의 정서 상태",
        1: "우울 증상 의심 및 전문가 상담 권장"
    }
    return summary_map.get(final_level, "분석 결과 없음")


# 문장 별로 나누는 함수
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

    # 전역으로 선언된 모델과 파이프라인 토크나이저를 불러옴
    global sentiment_pipeline
    global models
    global tokenizers

    # 파이프라인과 선택한 모델이 잘 로드가 되었는지 검증
    if not sentiment_pipeline or disease_type not in models:
        print("[WARN] 모델이 아직 로드되지 않았습니다.")
        return {
            "summary": get_analysis_summary(0),
            "dep_res": {"final_level": 0, "raw_score": 0.0, "is_symptom": False}
        }

    # content를 문장단위로 3문장씩 그리고 다은 2문장은 건너뛰고 다시 3문장씩 반복
    # 굳이 겹치게 설정한 이유는 글 전체에 흐름이 끊기지 않기 위함
    chunks = split_into_chunks(content, window=3, step=2)
    # 이 부분은 통해서 전체 적으로 분석을 처리할때 false 값이 존재해야지만 모델을 실행하는 역할로 시간을 단축함
    is_all_clearly_positive = True

    # 1차 분류: 긍정적인 내용인지 빠른 확인
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

    # 2차 분류: 본격적인 우울증 모델 추론
    chunk_probs = []

    # with torch.no_grad() 이거는 추론 모델을 활성화해서 메모리 성능을 올림 (no gradient 기울기 계산)
    with torch.no_grad():
        for chunk in chunks:
            # inputs는 모델 분석에 사용될 텍스트를 토큰화 하는 과정임
            inputs = tokenizers[disease_type](
                chunk,
                return_tensors="pt",  # 반환 객체를 텐서 객체로 변환하라
                padding=True,  # 모델이 처리할 수 있는 최대 길이에 맞춰서 짧은 텍스트는 공백을 채워 길이를 맞춤
                truncation=True,  # 최대 길이가 넘어가면 자르는 설정
                max_length=512  # 토큰 수 최대 길이 설정
            ).to(DEVICE)  # 연산 기기 설정

            # ** 은 딕셔너리 구조를 풀어헤쳐 인자로 전달함
            outputs = models[disease_type](**inputs) # 모델에 inputs 넣어서 분석하여 Logits 형태의 값을 얻음

            # softmax는 분석된 결과의 합이 정확히 1이 되도록 하는 함수
            # cpu는 결과를 cpu 메모리로 가져오기 위함이고 그 이유는 numpy를 사용하기 위함이다
            # numpy는 최종적인 결과를 다른 라이브러리에서 활용하기 위한 표준언어이기 때문에 사용함
            # 모델에 출력구조는 2차원 배열이므로 flatten을 사용해서 1차원 배열로 변환해 데이터 구조를 단순화 함
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()

            # 분석된 우울증 수치만 chunk_probs에 담기 위해서 probs[1]을 사용했고,
            # float은 probs[1]은 파이토치 객체 내부 숫자라 일반 파이썬 리스트에 담기 위해서 다루기 편한 소수현태로 변환함
            chunk_probs.append(float(probs[1]))

    final_prob = sum(chunk_probs) / len(chunk_probs) # 마지막으로 조각난 분석 결과들의 평균을 구함
    dep_lvl = 1 if final_prob >= 0.50 else 0 # 최종적으로 분석된 결과를 이진 분류를 통해서 증상 유무를 판단함

    return {
        "summary": get_analysis_summary(dep_lvl),
        "dep_res": {
            "final_level": int(dep_lvl),
            "raw_score": round(final_prob, 4),
            "is_symptom": bool(dep_lvl == 1)
        }
    }