from fastapi import FastAPI
from app.api import analyze
from app.services.prediction import load_model
from app.services.emotion import load_emotion_model

app = FastAPI()

app.include_router(analyze.router)

# 서버가 켜질 때 모델들을 메모리에 미리 세팅합니다.
@app.on_event("startup")
async def startup_event():
    print("--- 서버 부팅 중: AI 모델을 메모리에 적재합니다 ---")
    load_model("depression") # 우울증 모델 로드
    load_emotion_model()     # 44개 감정 모델 로드
    print("--- AI 모델 적재 완료! 서비스 준비 끝 ---")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)