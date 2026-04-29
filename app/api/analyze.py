# 기능을 단위별로 나누어 관리할 수 있게 해주는 FastAPI의 도구
from fastapi import APIRouter, HTTPException

# 사용자가 보낸 데이터가 올바른 형식인지 검사(Validation)하는 Pydantic 라이브러리
from pydantic import BaseModel

# KlueBERT 우울증 분석 서비스
from app.services.prediction import analyze_diary

# KOTE 감정 분석 서비스
from app.services.emotion import analyze_emotions

router = APIRouter()

class DiaryRequest(BaseModel):
    content: str # 사용자가 쓴 일기 내용
    disease_type: str = "depression" # 분석할 질환 종류: 우울증

class DiaryResponse(BaseModel):
    analysis_summary: str  # 우울증 분석 결과 요약 텍스트
    main_emotion: str  # 8개 감정 중 대표 감정
    main_color: str  # 대표 감정의 색상 코드
    dep_res: dict  # 우울증 상세 수치 데이터

# @router.post: 사용자가 데이터를 보낼 때(POST 요청) 사용하는 주소를 지정합니다.
# response_model=DiaryResponse: 최종 결과가 위에서 정의한 응답 규격에 맞는지 자동으로 확인해줍니다.
@router.post("/api/analyze", response_model=DiaryResponse)
async def analyze_text(request: DiaryRequest):
    try:
        # 1. 우울증 분석 (prediction.py 호출)
        dep_data = analyze_diary(request.content, request.disease_type)

        # 2. 감정 분석 (emotion.py 호출)
        emo_data = analyze_emotions(request.content)

        # 3. 합쳐서 Spring 서버 형식으로 반환
        return DiaryResponse(
            analysis_summary=dep_data["summary"],
            main_emotion=emo_data["main_emotion"],
            main_color=emo_data["main_color"],
            dep_res=dep_data["dep_res"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc() # 서버 콘솔에 에러 상세 내용을 출력
        raise HTTPException(status_code=500, detail=str(e))