from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions
from app.services.summary import generate_hcx_summary

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
        dep_data = analyze_diary(request.content, request.disease_type)

        emo_data = analyze_emotions(request.content)

        hcx_summary = generate_hcx_summary(
            content=request.content,
            dep_level=dep_data["dep_res"]["final_level"],
            raw_emotions=emo_data["raw_emotions"]
        )

        return DiaryResponse(
            analysis_summary=hcx_summary,
            main_emotion=emo_data["main_emotion"],
            main_color=emo_data["main_color"],
            dep_res=dep_data["dep_res"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))