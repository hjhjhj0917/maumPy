import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

RECOMMEND_TRACK_COUNT = 5

# 감정 세부 라벨을 노래 검색 키워드로 묶는 그룹 - DiaryService.getEmotionColor()의
# 감정 그룹 분류와 동일한 카테고리를 사용해서 일관성을 맞춤 (Recommendations API는
# 2024.11부터 신규 앱에 막혀서 못 쓰고, 대신 Search API로 감정별 대표 키워드를 검색함)
_EMOTION_QUERY_GROUPS = {
    "신나는 K-pop 댄스": ["즐거움/신남", "행복", "기쁨", "뿌듯함", "흐뭇함(귀여움/예쁨)", "감동/감탄", "고마움", "환영/호의"],
    "잔잔한 힐링 팝": ["안심/신뢰", "존경", "아껴주는", "편안/쾌적"],
    "차분한 로파이": ["공포/무서움", "불안/걱정", "부담/안_내킴", "의심/불신"],
    "감성 인디": ["놀람", "신기함/관심", "어이없음", "경악", "당황/난처"],
    "슬픈 발라드": ["슬픔", "절망", "서러움", "불쌍함/연민", "안타까움/실망", "패배/자기혐오", "힘듦/지침"],
    "스트레스 해소 락": ["역겨움/징그러움", "증오/혐오", "지긋지긋", "한심함"],
    "감정 분출 힙합": ["화남/분노", "짜증", "불평/불만"],
    "동기부여 팝": ["기대감", "비장함", "깨달음"],
}

_DEFAULT_QUERY = "잔잔한 힐링 팝"

_token_cache = {"access_token": None, "expires_at": 0}


def _get_query_for_emotion(main_emotion: str) -> str:
    for query, emotions in _EMOTION_QUERY_GROUPS.items():
        if main_emotion in emotions:
            return query
    return _DEFAULT_QUERY


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


def _search_tracks(query: str, limit: int = RECOMMEND_TRACK_COUNT):
    token = _get_spotify_token()
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": limit, "market": "KR"}

    response = requests.get(SPOTIFY_SEARCH_URL, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    items = response.json().get("tracks", {}).get("items", [])

    tracks = []
    for item in items:
        images = item.get("album", {}).get("images", [])
        tracks.append({
            "track_id": item["id"],
            "track_name": item["name"],
            "artist_name": ", ".join(a["name"] for a in item.get("artists", [])),
            "album_image_url": images[0]["url"] if images else None,
            "spotify_url": item.get("external_urls", {}).get("spotify"),
        })

    return tracks


def get_music_recommendations(main_emotion: str):
    """
    일기의 대표 감정에 맞는 분위기 키워드로 Spotify에서 곡을 검색해 추천 목록을 반환함.
    실패하면 빈 리스트를 반환함 (일기 저장 자체는 막지 않기 위함).
    """
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("[MUSIC ERROR] SPOTIFY_CLIENT_ID/SECRET이 설정되지 않음")
        return []

    query = _get_query_for_emotion(main_emotion)

    try:
        return _search_tracks(query)
    except Exception as e:
        print(f"[MUSIC ERROR] 음악 추천 실패: {e}")
        return []
