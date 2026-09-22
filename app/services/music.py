import os
import time

import requests
from dotenv import load_dotenv

from app.services.summary import GEMINI_API_URL, _get_access_token

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

RECOMMEND_TRACK_COUNT = 5

# Spotify Search API가 Development Mode 앱에는 limit을 최대 10까지만 허용해서(11 이상은 400 에러) 10으로 고정함.
# 참고: 이 앱 등급에서는 track/artist의 popularity 필드도 항상 null로 내려와서(추가로 막혀있음)
# 인기도 기준 정렬이 불가능함 - 대신 검색어 자체를 영어 장르/무드 단어로 만들어서 결과 품질을 높임
SEARCH_POOL_SIZE = 10

# 동요/키즈 콘텐츠, 커버/반주 라이브러리 등 감정 추천 취지에 안 맞는 결과를 걸러내는 차단 키워드
# (제목/앨범명에 포함되면 후보에서 제외). 한글 검색어를 쓰면 Spotify가 음절 단위로 부분매칭해서
# 엉뚱한 동요가 섞이는 걸 확인했기 때문에, 검색어 자체를 영어로 바꾼 뒤에도 안전장치로 유지함
_BLOCKED_KEYWORDS = [
    "동화", "동요", "율동", "키즈", "유아", "베이비", "자장가", "아이들",
    "backing track", "instrumental", "karaoke", "tribute", "cover version",
    "lullaby", "nursery", "sing along", "in the style of", "made famous by",
]

# 0.8 이상으로 뚜렷하게 감지된 감정만 "지금 이 상황의 진짜 감정"으로 보고 LLM에게 넘김.
# 애매하게 낮은 확률의 감정까지 다 넣으면 상황 판단이 흐려지므로 강한 신호만 사용함
STRONG_EMOTION_THRESHOLD = 0.8

# LLM 호출이 실패하거나 뚜렷한 감정이 하나도 없을 때 쓰는 안전망 - 세부 감정을
# DiaryService.getEmotionColor()와 동일한 8개 그룹으로 묶어 분위기 키워드로 대응함.
# 검색어는 영어 장르/무드 단어로 구성함 - 한글 설명 문구는 Spotify가 음절 단위로 부분매칭해서
# (예: "동기부여 팝" → "동"이 들어간 동요까지 매칭) 엉뚱한 결과가 섞이는 걸 확인했기 때문
_EMOTION_QUERY_GROUPS = {
    "upbeat k-pop dance": ["즐거움/신남", "행복", "기쁨", "뿌듯함", "흐뭇함(귀여움/예쁨)", "감동/감탄", "고마움", "환영/호의"],
    "calm healing pop": ["안심/신뢰", "존경", "아껴주는", "편안/쾌적"],
    "mellow lofi chill": ["공포/무서움", "불안/걱정", "부담/안_내킴", "의심/불신"],
    "emotional indie": ["놀람", "신기함/관심", "어이없음", "경악", "당황/난처"],
    "sad ballad": ["슬픔", "절망", "서러움", "불쌍함/연민", "안타까움/실망", "패배/자기혐오", "힘듦/지침"],
    "stress relief rock": ["역겨움/징그러움", "증오/혐오", "지긋지긋", "한심함"],
    "angry hip hop": ["화남/분노", "짜증", "불평/불만"],
    "motivational pop": ["기대감", "비장함", "깨달음"],
}

_DEFAULT_QUERY = "calm healing pop"

_token_cache = {"access_token": None, "expires_at": 0}


# ★ 즐겨찾기 이후 추가/수정
def _get_fallback_query(main_emotion: str) -> str:
    for query, emotions in _EMOTION_QUERY_GROUPS.items():
        if main_emotion in emotions:
            return query
    return _DEFAULT_QUERY


# ★ 즐겨찾기 이후 추가/수정
def _get_strong_emotions(raw_emotions: dict, threshold: float = STRONG_EMOTION_THRESHOLD):
    strong = [(emo, prob) for emo, prob in raw_emotions.items() if prob >= threshold]
    return sorted(strong, key=lambda x: x[1], reverse=True)


# ★ 즐겨찾기 이후 추가/수정
def _generate_music_query(content: str, strong_emotions: list) -> str:
    """
    일기 원문 + 뚜렷한(0.8 이상) 감정들을 Gemini에게 보여주고, 감정을 그대로 반영하는 게 아니라
    "지금 이 상황에 어울리는 음악 분위기"를 판단하게 함 (예: 슬프지만 위로가 필요한지, 힘을 내고
    싶은 상황인지는 감정 수치만으로는 구분 못하고 일기 내용을 읽어야 알 수 있음 - iso 원칙 적용).
    """
    emotions_str = ", ".join(f"{emo}({round(prob * 100, 1)}%)" for emo, prob in strong_emotions)

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {_get_access_token()}'
    }

    payload = {
        "systemInstruction": {
            "parts": [{
                "text": (
                    "당신은 사용자의 일기를 읽고 지금 상황에 어울리는 음악을 추천하는 음악 큐레이터입니다. "
                    "감정을 그대로 반영한 노래(예: 슬프면 무조건 슬픈 노래)를 고르지 말고, 일기의 맥락을 읽고 "
                    "지금 이 사람에게 실제로 도움이 될 음악의 분위기를 판단하세요. "
                    "예를 들어 슬프지만 다시 힘을 내려는 내용이면 잔잔한 위로보다 동기부여가 되는 분위기가, "
                    "그냥 지쳐서 쉬고 싶은 내용이면 편안하고 잔잔한 분위기가 더 어울립니다.\n\n"
                    "반드시 Spotify 검색창에 그대로 입력할 짧은 영어 키워드 구문 하나만 답하세요 "
                    "(예: 'motivational pop', 'calm acoustic comfort', 'upbeat k-pop dance'). "
                    "영어로 답하는 이유는 한국어 설명 문구를 검색어로 쓰면 Spotify가 음절 단위로 "
                    "부분매칭해서 전혀 관계없는 동요나 아동 콘텐츠가 섞이기 때문입니다. "
                    "장르명과 무드 형용사만 사용하고, '이야기'나 'story'처럼 은유적인 단어, "
                    "곡 제목, 아티스트명, 설명, 따옴표, 마침표는 절대 포함하지 마세요."
                )
            }]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": f"일기 원문:\n{content}\n\n뚜렷하게 감지된 감정:\n{emotions_str}\n\n위 내용을 바탕으로 지금 이 사람에게 어울리는 음악 분위기 키워드를 알려주세요."
                }]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 60,
            "temperature": 0.4,
            "topP": 0.8,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    res_data = response.json()

    candidates = res_data.get('candidates', [])
    parts = candidates[0].get('content', {}).get('parts', []) if candidates else []
    query = parts[0].get('text', '').strip() if parts else ''

    return query.replace('"', '').replace("'", '').strip()


# ★ 즐겨찾기 이후 추가/수정
def _get_spotify_token() -> str:
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    response = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data["expires_in"] - 60  # 만료 60초 전에 미리 갱신

    return _token_cache["access_token"]


# ★ 즐겨찾기 이후 추가/수정
def _is_blocked(item: dict) -> bool:
    text = (item.get("name", "") + " " + item.get("album", {}).get("name", "")).lower()
    return any(keyword.lower() in text for keyword in _BLOCKED_KEYWORDS)


# ★ 즐겨찾기 이후 추가/수정
def _search_tracks(query: str, limit: int = RECOMMEND_TRACK_COUNT):
    token = _get_spotify_token()
    headers = {"Authorization": f"Bearer {token}"}
    # 인기도/키즈 콘텐츠 필터링을 거치고도 limit장이 남도록 넉넉히 받아옴
    params = {"q": query, "type": "track", "limit": SEARCH_POOL_SIZE, "market": "KR"}

    response = requests.get(SPOTIFY_SEARCH_URL, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    items = response.json().get("tracks", {}).get("items", [])

    # 동요/키즈 콘텐츠, 반주/커버 라이브러리 제외 (popularity 필드가 이 앱에서는 항상 null로
    # 내려와서 인기도 기준 정렬은 불가능함 - Spotify 자체 관련성 순서를 그대로 신뢰하고 필터링만 함)
    items = [item for item in items if not _is_blocked(item)]

    tracks = []
    seen = set()  # (곡명, 아티스트) 기준 중복 제거 - 같은 곡이 리마스터/라이브 버전 등으로
                  # 트랙 ID만 다르게 여러 개 올라와 있어서 그대로 두면 같은 곡이 중복 추천됨
    for item in items:
        key = (item["name"].strip().lower(), tuple(a["name"] for a in item.get("artists", [])))
        if key in seen:
            continue
        seen.add(key)

        images = item.get("album", {}).get("images", [])
        tracks.append({
            "track_id": item["id"],
            "track_name": item["name"],
            "artist_name": ", ".join(a["name"] for a in item.get("artists", [])),
            "album_image_url": images[0]["url"] if images else None,
            "spotify_url": item.get("external_urls", {}).get("spotify"),
        })

        if len(tracks) >= limit:
            break

    return tracks


# ★ 즐겨찾기 이후 추가/수정
def get_music_recommendations(content: str, raw_emotions: dict, main_emotion: str):
    """
    일기 원문 + 0.8 이상으로 뚜렷하게 감지된 감정들을 LLM에게 보여줘서 상황에 맞는 음악
    분위기 키워드를 판단하게 하고, 그 키워드로 Spotify에서 곡을 검색함.
    뚜렷한 감정이 없거나 LLM/검색이 실패하면 main_emotion 기반 고정 키워드 매핑으로 대체함.
    실패해도 빈 리스트를 반환함 (일기 저장 자체는 막지 않기 위함).

    참고: 한국/해외/J-POP처럼 국적 기준으로 카테고리를 나눠보려 했으나, Spotify Search는
    "k-pop"/"j-pop" 같은 한정어를 붙여도 그 태그가 걸린 아무 곡이나 매칭할 뿐 실제 아티스트
    국적을 판별해주지 않아 결과가 부정확했음 (예: J-POP 카테고리에 미국 힙합 곡이 섞임).
    아티스트 국적을 확실히 판별하려면 추가 API 호출이 필요한데 신뢰할 만한 방법이 없어서
    카테고리 구분 없이 단일 목록으로 되돌림.
    """
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("[MUSIC ERROR] SPOTIFY_CLIENT_ID/SECRET이 설정되지 않음")
        return []

    strong_emotions = _get_strong_emotions(raw_emotions)
    query = None

    if strong_emotions:
        try:
            query = _generate_music_query(content, strong_emotions)
        except Exception as e:
            print(f"[MUSIC ERROR] LLM 키워드 생성 실패, 고정 매핑으로 대체: {e}")

    if not query:
        query = _get_fallback_query(main_emotion)

    # 순수 영어 무드 키워드(예: "motivational pop")만으로 검색하면 market=KR이어도 결과가
    # 대부분 해외 팝 위주로 나옴 (market은 "한국에서 재생 가능한지"만 볼 뿐 국적을 우선하지
    # 않음). "k-pop"을 붙이는 건 오히려 그 문자열이 제목에 박힌 엉뚱한 곡(예: Travis Scott의
    # 곡 "K-POP")을 끌어와서 실패했고, 대신 "가요"를 붙이니 실제 한국 대중가요가 훨씬 잘
    # 나오는 것을 테스트로 확인함 (예: "sad ballad 가요" → 브라운아이드소울, 윤종신 등)
    korean_biased_query = query if "가요" in query else f"{query} 가요"

    try:
        tracks = _search_tracks(korean_biased_query)
        if not tracks:
            tracks = _search_tracks(query)
        if not tracks and query != _get_fallback_query(main_emotion):
            # 그래도 결과가 없으면, 마지막으로 고정 매핑 키워드로 한 번 더 시도
            tracks = _search_tracks(_get_fallback_query(main_emotion))
        return tracks
    except Exception as e:
        print(f"[MUSIC ERROR] 음악 추천 실패: {e}")
        return []
