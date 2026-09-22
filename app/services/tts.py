import os
import json
import re
import base64

import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account

load_dotenv()

# 문장 종결 부호(. ! ?) 또는 줄바꿈 + 그 뒤 공백까지 통째로 한 조각으로 잡아냄.
# re.split은 구분자 자체(특히 줄바꿈)를 결과에서 없애버려서 문단/목록 구분이 사라지는 문제가 있었기 때문에,
# 구분자를 버리지 않고 조각 끝에 그대로 남겨두는 findall 방식으로 씀
SENTENCE_CHUNK_PATTERN = re.compile(r'.*?(?:[.!?]+\s*|\n+\s*)|.+$', flags=re.DOTALL)

# TTS로 보내기 전 이모지를 제거하기 위한 패턴 (화면 표시용 텍스트에는 영향 없음)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # 이모지 전체 대역 (표정, 사물, 동물, 기호 등)
    "\U00002600-\U000027BF"  # 기타 기호 및 딩뱃 (☀ ✨ ❤ 등)
    "\U0001F1E0-\U0001F1FF"  # 국기
    "\U00002B00-\U00002BFF"  # 화살표/별 등 잡기호
    "\U0000FE0F"             # variation selector (이모지 렌더링 지시자)
    "]+",
    flags=re.UNICODE
)


# ★ 즐겨찾기 이후 추가/수정
def strip_emoji_for_tts(text):
    # TTS 합성용으로만 이모지를 제거하고 앞뒤 공백을 정리함
    return EMOJI_PATTERN.sub('', text).strip()


# 마크다운 기호(굵게, 목록, 헤더)가 그대로 있으면 TTS가 "별표 별표"처럼 읽어버리므로,
# 화면 표시용 텍스트는 그대로 두고 TTS로 보내기 직전에만 이 함수로 제거함
MARKDOWN_PATTERN = re.compile(r'\*\*|\*|^#{1,6}\s*|^[-•]\s*', flags=re.MULTILINE)


# ★ 즐겨찾기 이후 추가/수정
def strip_markdown_for_tts(text):
    return MARKDOWN_PATTERN.sub('', text).strip()


# Google Cloud TTS는 요청 하나당 입력 텍스트를 5000바이트(UTF-8)까지만 허용함.
# 문장 하나씩 따로 합성하면 호출 횟수가 너무 많아지고, 특정 호출이 실패하면 그 문장만
# 조용히 빠지는(synthesize_speech가 실패 시 None을 반환) 문제가 있어서, 이 바이트 한도
# 안에서 문장을 최대한 묶어 호출 횟수를 줄이고 응답 전체가 안정적으로 합성되게 함
_MAX_TTS_CHUNK_BYTES = 4500


# ★ 즐겨찾기 이후 추가/수정
def split_into_tts_chunks(text: str, max_bytes: int = _MAX_TTS_CHUNK_BYTES) -> list[str]:
    sentences = [s for s in SENTENCE_CHUNK_PATTERN.findall(text) if s.strip()]

    chunks = []
    current = ""
    for sentence in sentences:
        candidate = current + sentence
        if current and len(candidate.encode('utf-8')) > max_bytes:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current)

    return chunks

# 서비스 계정 JSON 키의 "내용 전체"를 환경변수 값으로 저장 (.env의 GCP_TTS_CREDENTIALS_JSON)
GCP_TTS_CREDENTIALS_JSON = os.getenv("GCP_CREDENTIALS_JSON")

TTS_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

# 챗봇 답변에 사용할 음색 (환경변수로 오버라이드 가능)
# 기본값은 Chirp 3: HD 라인 (2026년 기준 Google이 제공하는 가장 자연스럽고 고품질인 음성)
TTS_VOICE_NAME = os.getenv("TTS_VOICE_NAME", "ko-KR-Chirp3-HD-Leda")
TTS_LANGUAGE_CODE = "ko-KR"

# 말하기 속도 (0.25 ~ 4.0, 1.0이 기본 속도). 환경변수로 오버라이드 가능
# 주의: Chirp3-HD 음성은 speakingRate를 지원하지 않아, 이 음성을 쓸 때는 자동으로 무시됨
TTS_SPEAKING_RATE = float(os.getenv("TTS_SPEAKING_RATE", "1.5"))

# 지금 실제로 어떤 값이 로드됐는지 서버 시작 시점에 눈으로 바로 확인할 수 있도록 출력
print(f"[TTS INIT] 사용 중인 음성: {TTS_VOICE_NAME}")

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_credentials_info = json.loads(GCP_TTS_CREDENTIALS_JSON)
_credentials = service_account.Credentials.from_service_account_info(_credentials_info, scopes=_SCOPES)


# ★ 즐겨찾기 이후 추가/수정
def _get_access_token() -> str:
    if not _credentials.valid:
        _credentials.refresh(Request())
    return _credentials.token


# ★ 즐겨찾기 이후 추가/수정
def synthesize_speech(text: str) -> bytes | None:
    if not text or not text.strip():
        return None

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {_get_access_token()}'
    }

    audio_config = {"audioEncoding": "MP3"}

    # Chirp3-HD 음성은 speakingRate(속도) 파라미터를 지원하지 않아서, 이 경우엔 아예 빼고 요청함
    if "Chirp3-HD" not in TTS_VOICE_NAME:
        audio_config["speakingRate"] = TTS_SPEAKING_RATE

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": TTS_LANGUAGE_CODE,
            "name": TTS_VOICE_NAME
        },
        "audioConfig": audio_config
    }

    try:
        response = requests.post(TTS_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        res_data = response.json()
        audio_content_b64 = res_data.get("audioContent")

        if not audio_content_b64:
            raise Exception(f"API 응답에 오디오 데이터가 없습니다. 응답 내용: {res_data}")

        return base64.b64decode(audio_content_b64)

    except Exception as e:
        print(f"[TTS ERROR] 음성 합성 실패: {e}")
        return None