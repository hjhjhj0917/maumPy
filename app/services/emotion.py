import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "searle-j/kote_for_easygoing_people"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 플루칙 그룹화 정보
EMOTION_GROUPS = {
    "기쁨": {"color": "#FFD700", "labels": ['즐거움/신남', '행복', '기쁨', '뿌듯함', '흐뭇함(귀여움/예쁨)', '감동/감탄', '고마움', '환영/호의']},
    "신뢰": {"color": "#9ACD32", "labels": ['안심/신뢰', '존경', '아껴주는', '편안/쾌적']},
    "공포": {"color": "#008000", "labels": ['공포/무서움', '불안/걱정', '부담/안_내킴', '당황/난처']},
    "놀람": {"color": "#00BFFF", "labels": ['놀람', '신기함/관심', '어이없음', '경악']},
    "슬픔": {"color": "#4169E1", "labels": ['슬픔', '절망', '서러움', '불쌍함/연민', '안타까움/실망', '패배/자기혐오']},
    "혐오": {"color": "#8A2BE2", "labels": ['역겨움/징그러움', '증오/혐오', '지긋지긋', '한심함', '부끄러움', '죄책감']},
    "분노": {"color": "#FF4500", "labels": ['화남/분노', '짜증', '불평/불만']},
    "기대": {"color": "#FFA500", "labels": ['기대감', '비장함', '깨달음']}
}

KOTE_LABELS = ['불평/불만', '환영/호의', '감동/감탄', '지긋지긋', '고마움', '슬픔', '화남/분노', '존경', '기대감', '우쭐댐/무시함', '안타까움/실망', '비장함',
               '의심/불신', '뿌듯함', '편안/쾌적', '신기함/관심', '아껴주는', '부끄러움', '공포/무서움', '절망', '한심함', '역겨움/징그러움', '짜증', '어이없음', '없음',
               '패배/자기혐오', '귀찮음', '힘듦/지침', '즐거움/신남', '깨달음', '죄책감', '증오/혐오', '흐뭇함(귀여움/예쁨)', '당황/난처', '경악', '부담/안_내킴',
               '서러움', '재미없음', '불쌍함/연민', '놀람', '행복', '불안/걱정', '기쁨', '안심/신뢰']

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def analyze_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()

    raw_results = {KOTE_LABELS[i]: float(probs[i]) for i in range(len(KOTE_LABELS))}

    group_scores = {}
    for group_name, info in EMOTION_GROUPS.items():
        vals = [raw_results[label] for label in info["labels"] if label in raw_results]
        group_scores[group_name] = float(np.mean(vals)) if vals else 0.0

    main_emotion = max(group_scores, key=group_scores.get)
    return {
        "main_emotion": main_emotion,
        "main_color": EMOTION_GROUPS[main_emotion]["color"],
        "raw_emotions": raw_results
    }