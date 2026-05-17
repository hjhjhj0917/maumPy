import os
import time
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

KAKAO_REST_KEY = os.getenv("KAKAO_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URI)
db = client["maum_db"]
collection = db["MENTAL_INST"]


def get_coordinates(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    params = {"query": address}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("documents"):
                doc = data["documents"][0]
                return float(doc["x"]), float(doc["y"])
        else:
            print(f"카카오 API 에러 코드: {response.status_code}, 내용: {response.text}")
    except Exception as e:
        print(f"시스템 에러: {e}")
        pass
    return None, None


def migrate():
    query = {
        "ADDR": {"$exists": True},
        "$or": [
            {"LOCATION": {"$exists": False}},
            {"LOCATION": None}
        ]
    }
    docs = list(collection.find(query))
    total_docs = len(docs)

    print(f"Total: {total_docs}")

    success_count = 0
    for idx, doc in enumerate(docs):
        addr = doc.get("ADDR")
        if not addr or not addr.strip():
            continue

        lng, lat = get_coordinates(addr.strip())
        if lng and lat:
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "LOCATION": {
                        "type": "Point",
                        "coordinates": [lng, lat]
                    }
                }}
            )
            success_count += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == total_docs:
            print(f"[{idx + 1}/{total_docs}] Success: {success_count}")

        time.sleep(0.05)

    print(f"Completed: {success_count}/{total_docs}")


if __name__ == "__main__":
    migrate()