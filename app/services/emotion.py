import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "searle-j/kote_for_easygoing_people"
LABELS = [
    '불평/불만', '환영/호의', '감동/감탄', '지긋지긋', '고마움', '슬픔', '화남/분노',
    '존경', '기대감', '우쭐댐/무시함', '안타까움/실망', '비장함', '의심/불신', '뿌듯함',
    '편안/쾌적', '신기함/관심', '아껴주는', '부끄러움', '공포/무서움', '절망', '한심함',
    '역겨움/징그러움', '짜증', '어이없음', '없음', '패배/자기혐오', '귀찮음', '힘듦/지침',
    '즐거움/신남', '깨달음', '죄책감', '증오/혐오', '흐뭇함(귀여움/예쁨)', '당황/난처',
    '경악', '부담/안_내킴', '서러움', '재미없음', '불쌍함/연민', '놀람', '행복',
    '불안/걱정', '기쁨', '안심/신뢰'
]

# 전역 변수로 선언하여 한 번만 로드되도록 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion_model = None
emotion_tokenizer = None


def load_emotion_model():
    global emotion_model, emotion_tokenizer
    if emotion_model is None:
        print("Loading Emotion Model (KOTE)...")
        emotion_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
        emotion_model.eval()


def analyze_emotions(text):
    # 모델이 안 켜져있으면 켭니다.
    load_emotion_model()

    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0]  # Multi-label 이므로 Sigmoid

    results = []
    for i, score in enumerate(probs):
        results.append({
            "emotion": LABELS[i],
            "score": round(float(score), 4)
        })

    # 높은 점수 순으로 정렬
    results.sort(key=lambda x: x["score"], reverse=True)
    return results