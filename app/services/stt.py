import os
import json
import base64

import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account

load_dotenv()

# TTS/embedding과 동일한 서비스 계정 JSON을 그대로 재사용함
GCP_CREDENTIALS_JSON = os.getenv("GCP_CREDENTIALS_JSON")

STT_API_URL = "https://speech.googleapis.com/v1/speech:recognize"
STT_LANGUAGE_CODE = "ko-KR"

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_credentials_info = json.loads(GCP_CREDENTIALS_JSON)
_credentials = service_account.Credentials.from_service_account_info(_credentials_info, scopes=_SCOPES)


# ★ 즐겨찾기 이후 추가/수정
def _get_access_token() -> str:
    if not _credentials.valid:
        _credentials.refresh(Request())
    return _credentials.token


# ★ 즐겨찾기 이후 추가/수정
def transcribe_speech(audio_bytes: bytes) -> str:
    """
    브라우저 MediaRecorder가 만든 오디오(webm/opus)를 받아 텍스트로 변환함.
    짧은 문장(한 번의 마이크 녹음) 처리가 목적이라 동기식 speech:recognize를 사용 —
    긴 오디오(1분 이상)라면 longrunningrecognize로 바꿔야 함
    """
    if not audio_bytes:
        return ""

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {_get_access_token()}'
    }

    payload = {
        "config": {
            "encoding": "WEBM_OPUS",
            "languageCode": STT_LANGUAGE_CODE,
            "enableAutomaticPunctuation": True
        },
        "audio": {
            "content": base64.b64encode(audio_bytes).decode('utf-8')
        }
    }

    try:
        response = requests.post(STT_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()

        res_data = response.json()
        results = res_data.get("results", [])

        if not results:
            return ""

        # 여러 문장으로 나뉘어 인식된 경우를 대비해 다 이어붙임
        transcript = " ".join(
            r["alternatives"][0]["transcript"]
            for r in results
            if r.get("alternatives")
        )
        return transcript.strip()

    except Exception as e:
        print(f"[STT ERROR] 음성 인식 실패: {e}")
        return ""
