from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions # 이제 함수가 존재하므로 에러 안 남

router = APIRouter()

class DiaryRequest(BaseModel):
    content: str
    disease_type: str = "depression"

class DiaryResponse(BaseModel):
    analysis_summary: str
    main_emotion: str
    main_color: str
    dep_res: dict

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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))