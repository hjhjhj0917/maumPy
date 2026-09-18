from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone

from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions
from app.services.summary import generate_diary_summary
from app.services.embedding import generate_diary_embedding
from app.services.music import get_music_recommendations
from app.core.database import diary_logs_collection

router = APIRouter()

class DiaryRequest(BaseModel):
    diary_no: int
    user_no: int
    title: str
    content: str
    disease_type: str = "depression"
    created_at: datetime

class TrackDTO(BaseModel):
    trackId: str
    trackName: str
    artistName: str
    albumImageUrl: Optional[str] = None
    spotifyUrl: Optional[str] = None

class DiaryResponse(BaseModel):
    analysis_summary: str
    main_emotion: str
    main_color: str
    dep_res: dict
    tracks: List[TrackDTO] = []


@router.post("/api/analyze", response_model=DiaryResponse)
async def analyze_text(request: DiaryRequest):
    try:
        dep_data = analyze_diary(request.content, request.disease_type)
        emo_data = analyze_emotions(request.content)

        diary_summary = generate_diary_summary(
            content=request.content,
            dep_level=dep_data["dep_res"]["final_level"],
            raw_emotions=emo_data["raw_emotions"]
        )

        combined_text = f"제목: {request.title}\n내용: {request.content}\n요약: {diary_summary}"

        embedding_vector = generate_diary_embedding(combined_text)

        music_results = get_music_recommendations(
            content=request.content,
            raw_emotions=emo_data["raw_emotions"],
            main_emotion=emo_data["main_emotion"]
        )
        tracks = [
            TrackDTO(
                trackId=t["track_id"],
                trackName=t["track_name"],
                artistName=t["artist_name"],
                albumImageUrl=t["album_image_url"],
                spotifyUrl=t["spotify_url"],
            )
            for t in music_results
        ]

        update_query = {
            "$set": {
                "TITLE": request.title,
                "CONTENT": request.content,
                "EMBEDDING": embedding_vector,
                "MAIN_EMOTION": emo_data["main_emotion"],
                "ANALYSIS_SUM": diary_summary,
                "EMO_RES": emo_data["raw_emotions"],
                "DEP_RES": {
                    "DISEASE_TYPE": request.disease_type,
                    "DEP_LVL": dep_data["dep_res"]["final_level"],
                    "DEP_SCORE": float(dep_data["dep_res"]["raw_score"]),
                    "IS_SYMPTOM": dep_data["dep_res"]["is_symptom"]
                },
                "CHG_DT": datetime.now(timezone.utc),
                "VERSION": "MAUM-Ensemble-v1.0"
            },
            "$setOnInsert": {
                "DIARY_NO": request.diary_no,
                "USER_NO": request.user_no,
                "REG_DT": request.created_at
            }
        }

        # upsert=True: DIARY_NO가 이미 있으면 update, 없으면 insert
        diary_logs_collection.update_one(
            {"DIARY_NO": request.diary_no},
            update_query,
            upsert=True
        )

        return DiaryResponse(
            analysis_summary=diary_summary,
            main_emotion=emo_data["main_emotion"],
            main_color=emo_data["main_color"],
            dep_res=dep_data["dep_res"],
            tracks=tracks
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))