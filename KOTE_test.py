import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import json

# 1. 모델 및 설정 초기화
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

# 모델 로드 (학교 보안망 이슈가 해결된 상태이므로 정상 작동)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


# 2. 감정 분석 함수 (44개 전수 조사)
def predict_all_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Multi-label 분류이므로 Sigmoid 적용
        probs = torch.sigmoid(outputs.logits)[0]

    results = []
    for i, score in enumerate(probs):
        score = float(score)
        results.append({
            "emotion": LABELS[i],
            "score": round(score, 4)
        })

    # 높은 점수 순으로 정렬
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# 3. MongoDB Document 형식 가공 함수
def format_for_mongodb(user_no, title, text, emotion_results):
    """
    분석 결과를 MongoDB 컬렉션 구조에 맞게 가공합니다.
    """
    # 가장 높은 점수의 감정을 메인 감정으로 설정
    main_emotion = emotion_results[0]['emotion'] if emotion_results else "없음"

    document = {
        "user_no": user_no,  # 회원 번호 (FK 역할)
        "title": title,  # 일기 제목
        "content": text,  # 일기 본문
        "main_emotion": main_emotion,  # 대표 감정
        "emotions_detail": emotion_results,  # 44개 전체 분석 결과 (Embedded List)
        "created_at": datetime.now(),  # 생성 일시 (Local Time)
        "version": "kote_v1.0"  # 분석 엔진 버전
    }

    return document


# --- 4. 실행 및 결과 출력 ---
if __name__ == "__main__":
    # 테스트 문장
    input_text = " 오늘은 평소보다 알람 소리가 조금 더 가깝게 들리는 아침이었다. 몸이 천근만근이었지만 겨우 일어나 기지개를 켜고 하루를 시작했다.점심에는 뭘 먹을까 고민하다가 근처 식당에서 늘 먹던 메뉴를 골랐다. 아는 맛이 무섭다더니, 역시 실패 없는 선택이었다. 식후에 마신 시원한 아이스 아메리카노 한 잔이 오늘 하루 중 가장 큰 활력소였던 것 같다.오후에는 밀린 일들을 하나씩 처리했다. 중간에 조금 지치기도 했지만, 창밖으로 들어오는 햇살이 나쁘지 않아 견딜만했다. 퇴근길에는 좋아하는 노래 플레이리스트를 들으며 걸었는데, 가사 하나하나가 평소보다 더 귀에 잘 들어오는 기분이었다.집에 돌아와 씻고 누우니 이제야 진정한 휴식이라는 느낌이 든다. 거창한 이벤트는 없었지만, 큰 사고 없이 무사히 하루를 마쳤다는 사실만으로도 충분히 감사한 일이다. 내일도 오늘만 같았으면 좋겠다."

    # [STEP 1] 감정 분석 수행
    analysis_results = predict_all_emotions(input_text)

    # [STEP 2] MongoDB 저장용 데이터 생성 (예시 데이터)
    mongo_doc = format_for_mongodb(
        user_no=1,
        title="무기력한 오후",
        text=input_text,
        emotion_results=analysis_results
    )

    # 결과 확인 - 터미널 출력용
    print("\n" + "=" * 50)
    print(f"입력 문장: {input_text}")
    print("=" * 50)
    print(f"[상위 5개 감정 결과]")
    for i, r in enumerate(analysis_results[:5]):
        print(f"{i + 1}. {r['emotion']:15s} : {r['score']:.4f}")

    print("\n" + "=" * 50)
    print("[MongoDB 저장용 JSON 스키마 예시]")
    print("=" * 50)
    # datetime 객체 출력을 위해 default=str 사용
    print(json.dumps(mongo_doc, indent=4, ensure_ascii=False, default=str))