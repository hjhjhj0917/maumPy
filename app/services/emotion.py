import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "searle-j/kote_for_easygoing_people"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOTION_GROUPS = {
    "기쁨": {"color": "#FFD700", "labels": ['즐거움/신남', '행복', '기쁨', '뿌듯함', '흐뭇함(귀여움/예쁨)', '감동/감탄', '고마움', '환영/호의']},
    "신뢰": {"color": "#66CDAA", "labels": ['안심/신뢰', '존경', '아껴주는', '편안/쾌적']},
    "공포": {"color": "#4B0082", "labels": ['공포/무서움', '불안/걱정', '부담/안_내킴', '의심/불신']},
    "놀람": {"color": "#00BFFF", "labels": ['놀람', '신기함/관심', '어이없음', '경악', '당황/난처']},
    "슬픔": {"color": "#1E3A8A", "labels": ['슬픔', '절망', '서러움', '불쌍함/연민', '안타까움/실망', '패배/자기혐오', '힘듦/지침']},
    "혐오": {"color": "#556B2F", "labels": ['역겨움/징그러움', '증오/혐오', '지긋지긋', '한심함']},
    "분노": {"color": "#FF3B30", "labels": ['화남/분노', '짜증', '불평/불만']},
    "기대": {"color": "#FFA500", "labels": ['기대감', '비장함', '깨달음']},
    "무감정": {"color": "#9E9E9E", "labels": ['없음', '귀찮음', '재미없음', '우쭐댐/무시함', '의심/불신']}
}

KOTE_LABELS = ['불평/불만', '환영/호의', '감동/감탄', '지긋지긋', '고마움', '슬픔', '화남/분노', '존경', '기대감', '우쭐댐/무시함', '안타까움/실망', '비장함',
               '의심/불신', '뿌듯함', '편안/쾌적', '신기함/관심', '아껴주는', '부끄러움', '공포/무서움', '절망', '한심함', '역겨움/징그러움', '짜증', '어이없음', '없음',
               '패배/자기혐오', '귀찮음', '힘듦/지침', '즐거움/신남', '깨달음', '죄책감', '증오/혐오', '흐뭇함(귀여움/예쁨)', '당황/난처', '경악', '부담/안_내킴',
               '서러움', '재미없음', '불쌍함/연민', '놀람', '행복', '불안/걱정', '기쁨', '안심/신뢰']

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


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


def analyze_emotions(text):
    # content를 문장단위로 3문장씩 그리고 다은 2문장은 건너뛰고 다시 3문장씩 반복
    # 굳이 겹치게 설정한 이유는 글 전체에 흐름이 끊기지 않기 위함
    chunks = split_into_chunks(text, window=3, step=2)
    all_probs = []

    # with torch.no_grad() 이거는 추론 모델을 활성화해서 메모리 성능을 올림 (no gradient 기울기 계산)
    with torch.no_grad():
        for chunk in chunks:
            # inputs는 모델 분석에 사용될 텍스트를 토큰화 하는 과정임
            inputs = tokenizer(
                chunk,
                return_tensors="pt",  # 반환 객체를 텐서 객체로 변환하라
                padding=True,  # 모델이 처리할 수 있는 최대 길이에 맞춰서 짧은 텍스트는 공백을 채워 길이를 맞춤
                truncation=True,  # 최대 길이가 넘어가면 자르는 설정
                max_length=512  # 토큰 수 최대 길이 설정
            ).to(DEVICE)  # 연산 기기 설정

            # ** 은 딕셔너리 구조를 풀어헤쳐 인자로 전달함
            outputs = model(**inputs) # 모델에 inputs 넣어서 분석하여 Logits 형태의 값을 얻음

            # 각 감정 카테고리별 독립 확률을 Sigmoid로 산출하여 복합적인 감정 상태를 추출
            # cpu는 결과를 cpu 메모리로 가져오기 위함이고 그 이유는 numpy를 사용하기 위함이다
            # numpy는 최종적인 결과를 다른 라이브러리에서 활용하기 위한 표준언어이기 때문에 사용함
            # 모델에 출력구조는 2차원 배열이므로 flatten을 사용해서 1차원 배열로 변환해 데이터 구조를 단순화 함
            probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()

            all_probs.append(probs)

    # chuking으로 분리돼서 분석된 결과를 axis=0(세로방향) 감정별로 평균을 내는 함수
    avg_probs = np.mean(all_probs, axis=0)

    # 분석된 감정 결과를 사용자가 확인 가능한 형태의 감정이름으로 매핑
    raw_results = {KOTE_LABELS[i]: float(avg_probs[i]) for i in range(len(KOTE_LABELS))}

    # 메인 감정을 찾기위해서 분석된 감정들 중에서 최댓값을 찾는데
    # key=raw_results.get 이 옵션을 통해 분석된 값끼리 최댓값을 찾음
    main_emotion = max(raw_results, key=raw_results.get)

    # 위 에서 찾은 메인 감정명을 통해서 그 값을 구함
    max_prob = raw_results[main_emotion]

    # 메인 감정 색상 매핑
    if max_prob < 0.15 or main_emotion == '없음':
        main_emotion = "무감정"
        main_color = "#9E9E9E"
    else:
        main_color = "#9E9E9E"
        for group_name, info in EMOTION_GROUPS.items():
            if main_emotion in info["labels"]:
                main_color = info["color"]
                break

    return {
        "main_emotion": main_emotion,
        "main_color": main_color,
        "raw_emotions": raw_results
    }