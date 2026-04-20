from fastapi import FastAPI
from app.api import analyze
import uvicorn

# 수정된 서비스들에서 모델 로드 함수 가져오기
# emotion.py와 prediction.py에 해당 함수가 정의되어 있어야 합니다.
from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions

app = FastAPI()

# 분석 라우터 등록
app.include_router(analyze.router)

@app.on_event("startup")
async def startup_event():
    print("--- 서버 부팅 중: AI 모델을 메모리에 적재합니다 ---")
    try:
        # 서비스 파일들이 임포트될 때 이미 모델을 로드하도록 설계했지만,
        # 부팅 시점에 확인을 위해 간단한 텍스트로 테스트 호출을 한 번씩 해줍니다.
        # 이렇게 하면 실제 요청 시 첫 로딩 지연(Cold Start)을 방지할 수 있습니다.
        analyze_diary("모델 로딩 테스트")
        analyze_emotions("모델 로딩 테스트")
        print("--- AI 모델 적재 완료! 서비스 준비 끝 ---")
    except Exception as e:
        print(f"--- 모델 적재 중 오류 발생: {e} ---")

@app.get("/")
async def root():
    return {"message": "Maum AI Server is running"}

if __name__ == "__main__":
    # 포트 번호는 준모님이 설정하신 8000번 그대로 유지합니다.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)