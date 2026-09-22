"""
기존 Clova(HCX) 임베딩으로 저장된 DIARY_LOGS, PUBLIC_SVC, MENTAL_INST 컬렉션을
Google Vertex AI(gemini-embedding-001) 임베딩으로 전부 다시 계산해서 덮어쓰는 일회성 스크립트

실행 전 꼭 확인할 것:
1. .env에 GCP_PROJECT_ID, GCP_LOCATION, EMBEDDING_DIMENSION이 설정되어 있는지
2. MongoDB Compass 등으로 3개 컬렉션 전체를 미리 JSON으로 내보내(백업해)뒀는지
   (이 스크립트는 EMBEDDING 필드를 되돌릴 수 없게 덮어씀)

실행 방법 (maumPy 루트에서):
    python -m scripts.reembed_all
"""

import time

from app.core.database import db
from app.services.embedding import generate_embedding, EMBEDDING_MODEL, EMBEDDING_DIMENSION

# API 호출 사이 최소 간격 (초) — 429(요청 제한)를 최대한 안 만나도록 넉넉하게 잡음
REQUEST_DELAY = 2.0


# ★ 즐겨찾기 이후 추가/수정
def _already_migrated(doc):
    # 이전 실행에서 이미 새 모델로 성공한 문서는 재시도 시 건너뛰기 위한 체크
    return doc.get("EMBEDDING_MODEL") == EMBEDDING_MODEL and doc.get("EMBEDDING_DIM") == EMBEDDING_DIMENSION


# ★ 즐겨찾기 이후 추가/수정
def reembed_diary_logs():
    collection = db["DIARY_LOGS"]
    documents = list(collection.find({}))
    print(f"\n[DIARY_LOGS] 대상 문서 수: {len(documents)}")

    success, fail = 0, 0

    for doc in documents:
        if _already_migrated(doc):
            print(f"[SKIP] DIARY_NO {doc.get('DIARY_NO')}: 이미 재임베딩 완료됨")
            continue

        try:
            title = doc.get("TITLE", "")
            content = doc.get("CONTENT", "")
            summary = doc.get("ANALYSIS_SUM", "")

            # analyze.py에서 최초 임베딩할 때 쓰던 것과 동일한 조합 방식을 그대로 재현함
            combined_text = f"제목: {title}\n내용: {content}\n요약: {summary}"

            if not combined_text.strip():
                print(f"[SKIP] DIARY_NO {doc.get('DIARY_NO')}: 임베딩할 텍스트가 없음")
                continue

            new_vector = generate_embedding(combined_text, task_type="RETRIEVAL_DOCUMENT")

            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "EMBEDDING": new_vector,
                    "EMBEDDING_MODEL": EMBEDDING_MODEL,
                    "EMBEDDING_DIM": EMBEDDING_DIMENSION
                }}
            )
            success += 1
            print(f"[OK] DIARY_NO {doc.get('DIARY_NO')} 재임베딩 완료")

        except Exception as e:
            fail += 1
            print(f"[FAIL] DIARY_NO {doc.get('DIARY_NO')}: {e}")

        time.sleep(REQUEST_DELAY)

    print(f"[DIARY_LOGS] 완료 — 성공 {success}건, 실패 {fail}건")


# ★ 즐겨찾기 이후 추가/수정
def reembed_public_svc():
    collection = db["PUBLIC_SVC"]
    documents = list(collection.find({}))
    print(f"\n[PUBLIC_SVC] 대상 문서 수: {len(documents)}")

    success, fail = 0, 0

    for doc in documents:
        if _already_migrated(doc):
            print(f"[SKIP] {doc.get('SVC_NM')}: 이미 재임베딩 완료됨")
            continue

        try:
            svc_nm = doc.get("SVC_NM", "")
            svc_sum = doc.get("SVC_SUM", "")
            svc_dtl = doc.get("SVC_DTL", "")
            target = doc.get("TARGET", "")

            # scripts/fetch_public_svc.py에서 최초 임베딩할 때 쓰던 것과 동일한 조합 방식
            search_text = f"{svc_nm} {svc_sum} {svc_dtl} {target}"

            if not search_text.strip():
                print(f"[SKIP] {doc.get('SVC_ID')}: 임베딩할 텍스트가 없음")
                continue

            new_vector = generate_embedding(search_text, task_type="RETRIEVAL_DOCUMENT")

            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "EMBEDDING": new_vector,
                    "EMBEDDING_MODEL": EMBEDDING_MODEL,
                    "EMBEDDING_DIM": EMBEDDING_DIMENSION
                }}
            )
            success += 1
            print(f"[OK] {svc_nm} 재임베딩 완료")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {doc.get('SVC_NM')}: {e}")

        time.sleep(REQUEST_DELAY)

    print(f"[PUBLIC_SVC] 완료 — 성공 {success}건, 실패 {fail}건")


# ★ 즐겨찾기 이후 추가/수정
def reembed_mental_inst():
    collection = db["MENTAL_INST"]
    documents = list(collection.find({}))
    print(f"\n[MENTAL_INST] 대상 문서 수: {len(documents)}")

    success, fail = 0, 0

    for doc in documents:
        if _already_migrated(doc):
            print(f"[SKIP] {doc.get('NAME')}: 이미 재임베딩 완료됨")
            continue

        try:
            category = doc.get("CATEGORY", "")
            name = doc.get("NAME", "")
            addr = doc.get("ADDR", "")

            # scripts/fetch_mental_inst.py에서 최초 임베딩할 때 쓰던 것과 동일한 조합 방식
            search_text = f"{category} {name} {addr}"

            if not search_text.strip():
                print(f"[SKIP] {doc.get('NAME')}: 임베딩할 텍스트가 없음")
                continue

            new_vector = generate_embedding(search_text, task_type="RETRIEVAL_DOCUMENT")

            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "EMBEDDING": new_vector,
                    "EMBEDDING_MODEL": EMBEDDING_MODEL,
                    "EMBEDDING_DIM": EMBEDDING_DIMENSION
                }}
            )
            success += 1
            print(f"[OK] {name} 재임베딩 완료")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {doc.get('NAME')}: {e}")

        time.sleep(REQUEST_DELAY)

    print(f"[MENTAL_INST] 완료 — 성공 {success}건, 실패 {fail}건")


if __name__ == "__main__":
    print("=" * 60)
    print(f"재임베딩 시작 — 모델: {EMBEDDING_MODEL}, 차원: {EMBEDDING_DIMENSION}")
    print("=" * 60)

    reembed_diary_logs()
    reembed_public_svc()
    reembed_mental_inst()

    print("\n모든 컬렉션 재임베딩 완료.")
    print("다음 단계: MongoDB Atlas 콘솔에서 vector_index를 삭제하고")
    print(f"새 차원({EMBEDDING_DIMENSION})에 맞춰 다시 만들어야 검색이 정상 동작합니다.")