import os
import json
import time

import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account

load_dotenv()

# TTS와 동일한 서비스 계정 JSON을 그대로 재사용함
# (단, GCP 콘솔에서 이 서비스 계정에 "Vertex AI User" 역할이 추가로 부여되어 있어야 하고,
#  프로젝트에 Vertex AI API가 활성화되어 있어야 함 — Text-to-Speech API 권한만으로는 호출 불가)
GCP_CREDENTIALS_JSON = os.getenv("GCP_CREDENTIALS_JSON")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# Google이 제공하는 다국어 지원 최신 정식(GA) 임베딩 모델
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# 결과 벡터 차원 (Matryoshka 방식으로 축소 가능, 기본 3072 중 일부만 사용)
# 이 값을 바꾸면 MongoDB Atlas의 벡터 검색 인덱스도 반드시 같은 차원으로 다시 만들어야 함
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

EMBEDDING_API_URL = (
    f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}"
    f"/locations/{GCP_LOCATION}/publishers/google/models/{EMBEDDING_MODEL}:predict"
)

print(f"[EMBEDDING INIT] 모델: {EMBEDDING_MODEL}, 차원: {EMBEDDING_DIMENSION}, 프로젝트: {GCP_PROJECT_ID}")

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_credentials_info = json.loads(GCP_CREDENTIALS_JSON)
_credentials = service_account.Credentials.from_service_account_info(_credentials_info, scopes=_SCOPES)


def _get_access_token() -> str:
    if not _credentials.valid:
        _credentials.refresh(Request())
    return _credentials.token


def generate_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
    """
    텍스트를 받아 Google Vertex AI(gemini-embedding-001)로 임베딩 벡터를 생성함

    task_type:
      - "RETRIEVAL_DOCUMENT": 검색 대상이 되는 원본 데이터(일기, 정책, 상담소 정보)를 저장할 때
      - "RETRIEVAL_QUERY": 사용자의 질문으로 저장된 데이터를 검색할 때
      같은 텍스트라도 용도에 맞는 task_type을 넘기면 검색 정확도가 더 좋아짐
    """

    if not text or not text.strip():
        raise Exception("임베딩할 텍스트가 비어 있습니다.")

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {_get_access_token()}'
    }

    payload = {
        "instances": [
            {"content": text, "task_type": task_type}
        ],
        "parameters": {
            "outputDimensionality": EMBEDDING_DIMENSION
        }
    }

    # 429(요청 제한)를 만나면 잠깐 기다렸다가 재시도함 (최대 5번, 대기 시간을 점점 늘림)
    MAX_RETRIES = 5
    WAIT_SECONDS = 15

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=15)

            # 429는 별도로 잡아서 재시도 대상으로 처리 (raise_for_status로 바로 예외를 던지지 않음)
            if response.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait_time = WAIT_SECONDS * attempt  # 15초, 30초, 45초... 점점 늘려가며 대기
                    print(f"[EMBEDDING] 429 요청 제한 — {wait_time}초 대기 후 재시도 ({attempt}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()  # 재시도 다 소진했으면 그때는 예외로 처리

            response.raise_for_status()

            res_data = response.json()
            predictions = res_data.get("predictions", [])

            if not predictions:
                raise Exception(f"API 응답에 예측 결과가 없습니다. 응답 내용: {res_data}")

            embedding_vector = predictions[0].get("embeddings", {}).get("values", [])

            if not embedding_vector:
                raise Exception(f"API 응답에 임베딩 데이터가 없습니다. 응답 내용: {res_data}")

            return embedding_vector

        except requests.exceptions.HTTPError:
            # 429 외의 HTTP 에러(403, 404 등)는 재시도해도 소용없으니 바로 실패 처리
            print(f"[EMBEDDING ERROR] Vertex AI 임베딩 생성 실패: {response.status_code} {response.text}")
            raise Exception(f"임베딩 생성 실패: {response.status_code}")
        except Exception as e:
            print(f"[EMBEDDING ERROR] Vertex AI 임베딩 생성 실패: {e}")
            raise Exception(f"임베딩 생성 실패: {e}")


# 일기 저장처럼 "문서를 저장하는" 용도이므로 기본 task_type인 RETRIEVAL_DOCUMENT를 그대로 씀
def generate_diary_embedding(text: str) -> list:
    return generate_embedding(text, task_type="RETRIEVAL_DOCUMENT")