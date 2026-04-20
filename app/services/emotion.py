import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "searle-j/kote_for_easygoing_people"

# 플루칙 이론에 기반한 8대 감정 그룹 및 색상 매핑 Plutchik's Wheel of Emotions
EMOTION_GROUPS = {
    "Joy": {"color": "#FFD700", "labels": ['즐거움/신남', '행복', '기쁨', '뿌듯함', '흐뭇함(귀여움/예쁨)', '감동/감탄', '고마움', '환영/호의']},
    "Trust": {"color": "#9ACD32", "labels": ['안심/신뢰', '존경', '아껴주는', '편안/쾌적']},
    "Fear": {"color": "#008000", "labels": ['공포/무서움', '불안/걱정', '부담/안_내킴', '당황/난처']},
    "Surprise": {"color": "#00BFFF", "labels": ['놀람', '신기함/관심', '어이없음', '경악']},
    "Sadness": {"color": "#4169E1", "labels": ['슬픔', '절망', '서러움', '불쌍함/연민', '안타까움/실망', '패배/자기혐오']},
    "Disgust": {"color": "#8A2BE2", "labels": ['역겨움/징그러움', '증오/혐오', '지긋지긋', '한심함', '부끄러움', '죄책감']},
    "Anger": {"color": "#FF4500", "labels": ['화남/분노', '짜증', '불평/불만']},
    "Anticipation": {"color": "#FFA500", "labels": ['기대감', '비장함', '깨달음']}
}

# KOTE 모델의 원본 라벨 순서 (기존과 동일)
LABELS = [
    '불평/불만', '환영/호의', '감동/감탄', '지긋지긋', '고마움', '슬픔', '화남/분노',
    '존경', '기대감', '우쭐댐/무시함', '안타까움/실망', '비장함', '의심/불신', '뿌듯함',
    '편안/쾌적', '신기함/관심', '아껴주는', '부끄러움', '공포/무서움', '절망', '한심함',
    '역겨움/징그러움', '짜증', '어이없음', '없음', '패배/자기혐오', '귀찮음', '힘듦/지침',
    '즐거움/신남', '깨달음', '죄책감', '증오/혐오', '흐뭇함(귀여움/예쁨)', '당황/난처',
    '경악', '부담/안_내킴', '서러움', '재미없음', '불쌍함/연민', '놀람', '행복',
    '불안/걱정', '기쁨', '안심/신뢰'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion_model = None
emotion_tokenizer = None


def load_emotion_model():
    global emotion_model, emotion_tokenizer
    if emotion_model is None:
        emotion_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
        emotion_model.eval()


def analyze_emotions(text):
    load_emotion_model()
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    # 1. 개별 감정 점수 매핑
    raw_results = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    # 2. 플루칙 그룹별 점수 합산 (평균값 활용)
    group_scores = {}
    for group_name, info in EMOTION_GROUPS.items():
        relevant_scores = [raw_results[label] for label in info["labels"] if label in raw_results]
        group_scores[group_name] = {
            "score": float(np.mean(relevant_scores)) if relevant_scores else 0.0,
            "color": info["color"]
        }

    # 3. 가장 점수가 높은 메인 그룹 추출
    main_group = max(group_scores, key=lambda k: group_scores[k]["score"])

    # Spring 서버에 전달할 최종 데이터 구조
    return {
        "main_emotion": main_group,
        "emotion_color": group_scores[main_group]["color"],
        "group_details": group_scores,
        "top_3_labels": sorted(raw_results.items(), key=lambda x: x[1], reverse=True)[:3]
    }