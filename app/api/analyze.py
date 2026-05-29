from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone

from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions
from app.services.summary import generate_hcx_summary
from app.services.embedding import generate_hcx_embedding
from app.core.database import diary_logs_collection

router = APIRouter()


class DiaryRequest(BaseModel):
    diary_no: int
    user_no: int
    title: str
    content: str
    disease_type: str = "depression"
    created_at: datetime

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

        combined_text = f"제목: {request.title}\n내용: {request.content}\n요약: {hcx_summary}"
        embedding_vector = generate_hcx_embedding(combined_text)

        update_query = {
            "$set": {
                "TITLE": request.title,
                "CONTENT": request.content,
                "EMBEDDING": embedding_vector,
                "MAIN_EMOTION": emo_data["main_emotion"],
                "ANALYSIS_SUM": hcx_summary,
                "EMO_RES": emo_data["raw_emotions"],
                "DEP_RES": {
                    "DISEASE_TYPE": request.disease_type,
                    "DEP_LVL": dep_data["dep_res"]["final_level"],
                    "DEP_SCORE": float(dep_data["dep_res"]["raw_score"]),
                    "IS_SYMPTOM": dep_data["dep_res"]["is_symptom"]
                },
                "CHG_DT": datetime.now(timezone.utc),
                "VERSION": "HCX-Emb-v2"
            },
            "$setOnInsert": {
                "DIARY_NO": request.diary_no,
                "USER_NO": request.user_no,
                "REG_DT": request.created_at
            }
        }

        diary_logs_collection.update_one(
            {"DIARY_NO": request.diary_no},
            update_query,
            upsert=True
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