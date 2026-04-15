from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions

router = APIRouter()


# 스프링부트에서 받을 데이터 형식
class DiaryRequest(BaseModel):
    content: str
    disease_type: str = "depression"


# 스프링부트로 돌려줄 합쳐진 데이터 형식
class DiaryResponse(BaseModel):
    analysis_summary: str
    dep_res: dict
    main_emotion: str
    emotions_detail: List[Dict[str, Any]]


@router.post("/api/analyze", response_model=DiaryResponse)
async def analyze_text(request: DiaryRequest):
    try:
        # 1. 우울증 분석 실행
        dep_result = analyze_diary(request.content, request.disease_type)

        # 2. 감정 분석 실행
        emo_result = analyze_emotions(request.content)

        # 가장 점수가 높은 감정을 대표 감정으로 설정
        main_emotion = emo_result[0]['emotion'] if emo_result else "없음"

        # 3. 합쳐서 반환
        return DiaryResponse(
            analysis_summary=dep_result["ANALYSIS_SUM"],
            dep_res=dep_result["DEP_RES"],
            main_emotion=main_emotion,
            emotions_detail=emo_result  # 44개 감정 리스트 전체
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))