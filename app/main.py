from fastapi import FastAPI
from app.api import analyze
import uvicorn

from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions

app = FastAPI()

# 분석 라우터 등록
app.include_router(analyze.router)

# @app.on_event("startup"): 서버가 완전히 켜지기 직전에 실행되는 함수입니다.
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
    # 서버 접속 시 가장 먼저 보이는 화면 (Health Check)
    return {"message": "Maum AI Server is running"}

if __name__ == "__main__":
    # host="0.0.0.0": 외부 장치(예: Spring 서버, 모바일 앱)에서 이 서버에 접속할 수 있도록 모든 IP의 접근을 허용합니다. 개발 편의를 위한 포트 설정
    # port=8000: 서버가 통신에 사용할 포트 번호입니다.
    # reload=True: 코드를 수정하고 저장하면 서버가 자동으로 재시작되는 개발 편의 기능입니다.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)