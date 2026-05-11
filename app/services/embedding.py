import os
import requests
from dotenv import load_dotenv

load_dotenv()

HCX_EMBEDDING_API_URL = os.getenv("HCX_EMBEDDING_API_URL")
HCX_API_KEY = os.getenv("HCX_API_KEY")
HCX_REQUEST_ID = os.getenv("HCX_REQUEST_ID", "d487b3b7d1dc4fe6b183f0d881e75d17")


def generate_hcx_embedding(text: str) -> list:
    """
    HyperCLOVA X Embedding v2 API를 사용하여 텍스트를 벡터로 변환합니다.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HCX_API_KEY}',
        'X-NCP-CLOVASTUDIO-REQUEST-ID': HCX_REQUEST_ID
    }

    payload = {
        "text": text
    }

    try:
        response = requests.post(HCX_EMBEDDING_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        res_data = response.json()
        embedding_vector = res_data.get('result', {}).get('embedding', [])

        if not embedding_vector:
            raise Exception(f"API 응답에 임베딩 데이터가 없습니다. 응답 내용: {res_data}")

        return embedding_vector

    except Exception as e:
        print(f"HCX Embedding API 호출 에러: {e}")
        raise Exception(f"임베딩 생성 실패: {e}")