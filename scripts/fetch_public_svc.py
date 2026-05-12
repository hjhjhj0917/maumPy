import os
import requests
import certifi
import time
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "maum_db")

GOV24_BASE_URL = os.getenv("GOV24_BASE_URL", "https://api.odcloud.kr/api")
GOV24_API_KEY = os.getenv("GOV24_API_KEY")

HCX_EMBEDDING_API_URL = os.getenv("HCX_EMBEDDING_API_URL")
HCX_API_KEY = os.getenv("HCX_API_KEY")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[MONGO_DB_NAME]
collection = db["PUBLIC_SVC"]


def get_embedding(text):
    if not text or len(text.strip()) == 0:
        return []

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HCX_API_KEY}'
    }

    payload = {"text": text[:1000]}

    for attempt in range(3):
        try:
            response = requests.post(HCX_EMBEDDING_API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json().get('result', {}).get('embedding', [])

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"  ⏳ API 속도 제한(429). 5초 대기 후 재시도... ({attempt + 1}/3)")
                time.sleep(5)
            else:
                raise e

    return []


def fetch_service_detail(svc_id):
    try:
        url = f"{GOV24_BASE_URL}/gov24/v3/serviceDetail"
        params = {
            "serviceKey": GOV24_API_KEY,
            "cond[SVC_ID::EQ]": svc_id,
            "returnType": "json"
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json().get('data', [])
        if data:
            return data[0]
        return {}
    except Exception:
        return {}


def fetch_and_save_data():
    print("공공서비스(혜택) 전체 데이터 수집을 시작합니다...")

    success_count = 0
    page = 1
    list_url = f"{GOV24_BASE_URL}/gov24/v3/serviceList"

    while True:
        print(f"\n--- [ {page} 페이지 ] 공공서비스 목록 요청 중 ---")

        params = {
            "serviceKey": GOV24_API_KEY,
            "page": page,
            "perPage": 100,
            "returnType": "json"
        }

        try:
            response = requests.get(list_url, params=params, timeout=10)
            response.raise_for_status()
            api_data = response.json().get('data', [])

            if not api_data:
                print("\n더 이상 가져올 데이터가 없습니다. 수집을 종료합니다.")
                break

            print(f"목록에서 {len(api_data)}개의 서비스를 찾았습니다. 상세 정보 조회 및 저장을 시작합니다.")

        except Exception as e:
            print(f"API 목록 호출 실패 (수집 강제 종료): {e}")
            break

        for item in api_data:
            try:
                svc_id = item.get('SVC_ID') or item.get('서비스ID', '')
                svc_nm = item.get('서비스명', '')
                svc_sum = item.get('서비스목적요약', '')

                if not svc_id or not svc_nm:
                    continue

                existing_doc = collection.find_one({"SVC_ID": svc_id})

                if existing_doc and "EMBEDDING" in existing_doc and existing_doc["EMBEDDING"]:
                    print(f"이미 처리된 서비스입니다 (스킵): {svc_nm}")
                    continue

                print(f"새로운 서비스 분석 중... : {svc_nm}")

                detail_info = fetch_service_detail(svc_id)

                svc_dtl = detail_info.get('지원내용', '') or item.get('지원내용', '')
                target = detail_info.get('지원대상', '') or item.get('지원대상', '')
                method = detail_info.get('신청방법', '') or item.get('신청방법', '')
                docs = detail_info.get('구비서류', '')
                url_link = detail_info.get('온라인신청사이트URL', '')
                contact = detail_info.get('접수기관명', '') or detail_info.get('문의처', '')

                cond_data = {
                    "MIN_AGE": None,
                    "MAX_AGE": None,
                    "GENDER": "",
                    "INCOME": ""
                }

                search_text = f"{svc_nm} {svc_sum} {svc_dtl} {target}"
                embedding_vector = get_embedding(search_text)

                if not embedding_vector:
                    print(f"임베딩 실패하여 스킵: {svc_nm}")
                    continue

                document = {
                    "SVC_ID": svc_id,
                    "SVC_NM": svc_nm,
                    "SVC_SUM": svc_sum,
                    "SVC_DTL": svc_dtl,
                    "TARGET": target,
                    "EMBEDDING": embedding_vector,
                    "METHOD": method,
                    "DOCS": docs,
                    "URL": url_link,
                    "COND": cond_data,
                    "CONTACT": contact,
                    "REG_DT": datetime.utcnow(),
                    "CHG_DT": datetime.utcnow()
                }

                collection.update_one(
                    {"SVC_ID": svc_id},
                    {"$set": document},
                    upsert=True
                )
                success_count += 1
                print(f"저장 성공: {svc_nm}")

                time.sleep(0.3)

            except requests.exceptions.RequestException as e:
                print(f"통신 에러 발생 (스킵됨): {item.get('서비스명', 'Unknown')} - {e}")
                continue
            except Exception as e:
                print(f"데이터 처리 에러 (스킵됨): {item.get('서비스명', 'Unknown')} - {e}")
                continue

        page += 1

    print(f"\n모든 작업 완료! 총 {success_count}개의 공공서비스가 PUBLIC_SVC 컬렉션에 저장되었습니다.")


if __name__ == "__main__":
    fetch_and_save_data()