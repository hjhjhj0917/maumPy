from fastapi import FastAPI
from app.api import analyze
import uvicorn

from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions

app = FastAPI()

app.include_router(analyze.router)

@app.on_event("startup")
async def startup_event():
    print("--- 서버 부팅 중: AI 모델을 메모리에 적재합니다 ---")
    try:
        analyze_diary("모델 로딩 테스트")
        analyze_emotions("모델 로딩 테스트")
        print("--- AI 모델 적재 완료! 서비스 준비 끝 ---")
    except Exception as e:
        print(f"--- 모델 적재 중 오류 발생: {e} ---")

@app.get("/")
async def root():
    return {"message": "Maum AI Server is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)