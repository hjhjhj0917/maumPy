from fastapi import APIRouter, BackgroundTasks
import sys
import os

from scripts.fetch_public_svc import fetch_and_save_data as update_public
from scripts.fetch_mental_inst import fetch_and_save_data as update_mental
from scripts.migrate_addresses import migrate as migrate_addr

router = APIRouter(prefix="/batch", tags=["Batch Jobs"])


# ★ 즐겨찾기 이후 추가/수정
def run_all_data_updates():
    print("--- 백그라운드 데이터 업데이트 시작 ---")
    try:
        update_public()
        update_mental()
        migrate_addr()
        print("--- 백그라운드 데이터 업데이트 완료 ---")
    except Exception as e:
        print(f"배치 작업 중 에러 발생: {e}")
        # TODO: 슬랙 알림이나 로그 저장 로직 추가 고려


# ★ 즐겨찾기 이후 추가/수정
@router.post("/update-data")
async def trigger_data_update(background_tasks: BackgroundTasks):
    # 무거운 작업이라 즉시 202를 응답하고 실제 처리는 백그라운드로 넘김
    background_tasks.add_task(run_all_data_updates)

    return {"message": "Data update batch job started in background.", "status": 202}