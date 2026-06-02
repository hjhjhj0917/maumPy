# api/batch.py
from fastapi import APIRouter, BackgroundTasks
import sys
import os

# 기존 스크립트들이 루트 폴더에 있다면 경로 추가가 필요할 수 있습니다.
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_public_svc import fetch_and_save_data as update_public
from scripts.fetch_mental_inst import fetch_and_save_data as update_mental
from scripts.migrate_addresses import migrate as migrate_addr

# prefix를 지정하여 URL을 깔끔하게 관리합니다. (예: /api/batch/...)
router = APIRouter(prefix="/batch", tags=["Batch Jobs"])


def run_all_data_updates():
    print("--- 백그라운드 데이터 업데이트 시작 ---")
    try:
        update_public()
        update_mental()
        migrate_addr()
        print("--- 백그라운드 데이터 업데이트 완료 ---")
    except Exception as e:
        print(f"배치 작업 중 에러 발생: {e}")
        # 여기에 슬랙 알림이나 로그 저장 로직을 추가하면 좋습니다.


@router.post("/update-data")
async def trigger_data_update(background_tasks: BackgroundTasks):
    # 실제 무거운 작업은 백그라운드 스레드에 넘깁니다.
    background_tasks.add_task(run_all_data_updates)

    # Spring 서버에게는 즉시 202 응답을 줍니다.
    return {"message": "Data update batch job started in background.", "status": 202}