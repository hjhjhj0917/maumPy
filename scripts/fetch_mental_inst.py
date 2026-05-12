import os
import requests
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "maum_db")

PUBLIC_DATA_URL = os.getenv("PUBLIC_DATA_URL")
PUBLIC_API_KEY = os.getenv("PUBLIC_API_KEY")

HCX_EMBEDDING_API_URL = os.getenv("HCX_EMBEDDING_API_URL")
HCX_API_KEY = os.getenv("HCX_API_KEY")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[MONGO_DB_NAME]
collection = db["MENTAL_INST"]


def get_embedding(text):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HCX_API_KEY}'
    }
    payload = {"text": text}

    response = requests.post(HCX_EMBEDDING_API_URL, headers=headers, json=payload, timeout=5)
    response.raise_for_status()

    return response.json().get('result', {}).get('embedding', [])


def get_coordinates(address):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json"
        headers = {'User-Agent': 'MauM-Mental-Health-App/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        data = response.json()
        if data:
            return [float(data[0]['lon']), float(data[0]['lat'])]
        return None
    except Exception:
        return None


def fetch_and_save_data():
    print("전체 공공데이터 수집 및 저장을 시작합니다...")

    success_count = 0
    page = 1

    while True:
        print(f"\n--- [ {page} 페이지 ] 데이터 요청 중 ---")

        params = {
            "serviceKey": PUBLIC_API_KEY,
            "page": page,
            "perPage": 100,
            "returnType": "json"
        }

        try:
            response = requests.get(PUBLIC_DATA_URL, params=params, timeout=10)
            response.raise_for_status()
            api_data = response.json().get('data', [])

            if not api_data:
                print("\n더 이상 가져올 데이터가 없습니다. 수집을 종료합니다.")
                break

            print(f"해당 페이지에서 {len(api_data)}개의 데이터를 가져왔습니다. DB 저장을 시작합니다.")

        except Exception as e:
            print(f"공공데이터 API 호출 실패 (수집 강제 종료): {e}")
            break

        for item in api_data:
            try:
                category = item.get('기관구분', '')
                name = item.get('기관명', '')
                addr = item.get('주소', '')
                homepage = item.get('홈페이지', '')

                if not name or not addr:
                    continue

                existing_doc = collection.find_one({"NAME": name, "ADDR": addr})

                if existing_doc and "EMBEDDING" in existing_doc and existing_doc["EMBEDDING"]:
                    print(f"이미 임베딩 완료된 기관입니다 (스킵): {name}")
                    continue

                print(f"새로운 데이터 임베딩 중... : {name}")
                search_text = f"{category} {name} {addr}"
                embedding_vector = get_embedding(search_text)

                if not embedding_vector:
                    print(f"임베딩 실패하여 스킵: {name}")
                    continue

                coordinates = get_coordinates(addr)
                location_data = None
                if coordinates:
                    location_data = {
                        "type": "Point",
                        "coordinates": coordinates
                    }

                document = {
                    "CATEGORY": category,
                    "NAME": name,
                    "ADDR": addr,
                    "EMBEDDING": embedding_vector,
                    "HOMEPAGE": homepage,
                    "LOCATION": location_data
                }

                collection.update_one(
                    {"NAME": name, "ADDR": addr},
                    {"$set": document},
                    upsert=True
                )
                success_count += 1
                print(f"저장 성공: {name}")

            except requests.exceptions.RequestException as e:
                print(f"통신 에러 발생 (스킵됨): {name} - {e}")
                continue
            except Exception as e:
                print(f"데이터 처리 에러 (스킵됨): {name} - {e}")
                continue

        page += 1

    print(f"\n모든 작업 완료! 총 {success_count}개의 신규 기관이 MENTAL_INST 컬렉션에 저장되었습니다.")


if __name__ == "__main__":
    fetch_and_save_data()