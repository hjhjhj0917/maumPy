import torch
import numpy as np
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


def analyze_emotions(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()

    raw_results = {KOTE_LABELS[i]: float(probs[i]) for i in range(len(KOTE_LABELS))}

    main_emotion = max(raw_results, key=raw_results.get)
    max_prob = raw_results[main_emotion]

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