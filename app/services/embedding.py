import os
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

HCX_EMBEDDING_API_URL = os.getenv("HCX_EMBEDDING_API_URL")
HCX_API_KEY = os.getenv("HCX_API_KEY")


def generate_hcx_embedding(text: str) -> list:

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {HCX_API_KEY}',
        'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4()) # 매 요청마다 독립된 ID를 부여하여 디버깅을 하기 위함 따라서 세계적으로 유일한 난수를 생성함
    }

    payload = {
        "text": text
    }

    try:
        # 요청 URL과 인증 정보가 든 헤더와 페이로드를 추가해 요청함
        response = requests.post(HCX_EMBEDDING_API_URL, headers=headers, json=payload)

        # 응답 받은 결과에서 상태코드를 확인해서 에러가 났다면 에러를 발생함
        response.raise_for_status()

        # 응답 받은 JSON 형태의 텍스트를 .json을 통해서 구조에 따라 딕셔너리 형태로 바뀌어 변수에 저장함
        res_data = response.json()

        # 응답 구조에 맞게 result에 embedding으로 접근해서 임베딩 결과를 불러옴
        embedding_vector = res_data.get('result', {}).get('embedding', [])

        if not embedding_vector:
            raise Exception(f"API 응답에 임베딩 데이터가 없습니다. 응답 내용: {res_data}")

        return embedding_vector

    except Exception as e:
        print(f"HCX Embedding API 호출 에러: {e}")
        raise Exception(f"임베딩 생성 실패: {e}")