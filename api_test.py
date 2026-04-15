import requests


def find_gangseo_institutions():
    base_url = "https://api.odcloud.kr/api/3049990/v1/uddi:14a6ea21-af95-4440-bb05-81698f7a1987"
    service_key = "3512cd4289ac57691effbc320fb7064c69701a7e3de510fa8951dc8fe97f559e"

    # 강서구 근처 동네 키워드
    nearby_keywords = ["우장산", "화곡", "내발산", "발산"]
    results = []

    print("🔎 강서구 지역 데이터를 찾는 중입니다. 잠시만 기다려주세요...")

    # 전체를 다 뒤지기 위해 perPage를 최대치(100)로 잡고 반복문 실행
    for page_num in range(1, 30):  # 2,786개이므로 약 28페이지까지 존재
        params = {
            'page': page_num,
            'perPage': 100,
            'serviceKey': service_key
        }

        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            items = data.get('data', [])

            if not items:
                break

            for item in items:
                address = str(item.get('주소', ''))
                if "강서구" in address:
                    # 캠퍼스 인근 동네인지 확인
                    is_nearby = any(kw in address for kw in nearby_keywords)
                    results.append({
                        "name": item.get('기관명'),
                        "addr": address,
                        "type": item.get('기관구분'),
                        "nearby": is_nearby
                    })

            # 진행 상황 표시
            if page_num % 5 == 0:
                print(f"   > {page_num * 100}번째 데이터까지 확인 완료...")

        except Exception as e:
            print(f"에러 발생: {e}")
            break

    # 결과 출력
    print("\n" + "=" * 50)
    print(f"📍 폴리텍 강서캠퍼스 인근 기관 리스트")
    print("=" * 50)

    nearby_count = 0
    for r in results:
        if r['nearby']:
            nearby_count += 1
            print(f"[{nearby_count}] {r['name']} ({r['type']})")
            print(f"   🏠 {r['addr']}")
            print("-" * 30)

    if nearby_count == 0:
        print("💡 강서구 전체 데이터는 찾았으나, 캠퍼스 바로 인근 동네 데이터는 없네요.")
        print("   강서구 소재 다른 기관들을 확인해보세요.")


if __name__ == "__main__":
    find_gangseo_institutions()