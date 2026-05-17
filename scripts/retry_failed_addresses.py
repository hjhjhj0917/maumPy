import os
import time
import re
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

KAKAO_REST_KEY = os.getenv("KAKAO_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URI)
db = client["maum_db"]
collection = db["MENTAL_INST"]


# 1. 괄호와 하위 상세 주소(층, 호)를 제거하는 정규식 함수
def clean_address(address):
    # 괄호와 괄호 안의 내용 전체 삭제 (예: (아미동2가) -> 삭제)
    address = re.sub(r'\(.*?\)', '', address)
    # 주소 뒤쪽에 붙은 층, 호, 상세 건물명 지우기
    address = re.sub(r'\s+\d+층.*|\s+\d+호.*', '', address)
    return address.strip()


# 2. 카카오 키워드(장소) 검색 API (최종 병기)
def get_coordinates_by_keyword(region, name):
    # 정제된 기관명 (앞뒤 특수문자나 복지 등의 불필요 문구 제거용)
    clean_name = re.sub(r'\(.*?\)', '', name).strip()

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    # 예: "부산광역시 서구 예사랑병원" 형태로 검색어 조합
    search_query = f"{region} {clean_name}"

    params = {"query": search_query}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("documents"):
                doc = data["documents"][0]
                return float(doc["x"]), float(doc["y"])
    except Exception:
        pass
    return None, None


# 3. 기존 주소 검색 API (정제된 주소로 재시도용)
def get_coordinates_by_address(address):
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
    except Exception:
        pass
    return None, None


def rescue_mission():
    # 💡 LOCATION이 없는 137건만 타겟팅
    query = {
        "ADDR": {"$exists": True},
        "$or": [
            {"LOCATION": {"$exists": False}},
            {"LOCATION": None}
        ]
    }
    docs = list(collection.find(query))
    total_docs = len(docs)

    print(f"구출 대상 데이터: {total_docs}건")

    saved_count = 0
    for idx, doc in enumerate(docs):
        addr = doc.get("ADDR", "")
        name = doc.get("NAME", "")

        # [시도 1] 주소 텍스트 정제 후 재검색
        cleaned_addr = clean_address(addr)
        lng, lat = get_coordinates_by_address(cleaned_addr)

        # [시도 2] 실패 시 키워드(장소) 기반 검색으로 우회
        if not lng or not lat:
            # 주소에서 "시/도 구/군" 까지만 잘라내어 검색 정확도 높임
            addr_parts = addr.split()
            region = f"{addr_parts[0]} {addr_parts[1]}" if len(addr_parts) > 1 else ""
            lng, lat = get_coordinates_by_keyword(region, name)

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
            saved_count += 1
            print(f"✅ 구출 성공 [{saved_count}]: {name} -> ({lat}, {lng})")
        else:
            print(f"❌ 최종 실패: {name} | 주소: {addr}")

        time.sleep(0.05)

    print(f"🔥 작업 완료! 137건 중 {saved_count}건 구출 성공!")


if __name__ == "__main__":
    rescue_mission()