from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.report import generate_weekly_report

router = APIRouter()


class DiaryEntryDTO(BaseModel):
    title: str
    summary: Optional[str] = None
    main_emotion: Optional[str] = None
    dep_score: Optional[float] = None


class WeeklyReportRequest(BaseModel):
    user_no: int
    entries: List[DiaryEntryDTO] = []


class WeeklyReportResponse(BaseModel):
    comment: str


@router.post("/api/weekly-report", response_model=WeeklyReportResponse)
async def weekly_report(request: WeeklyReportRequest):
    try:
        diary_entries = [
            {
                "title": e.title,
                "summary": e.summary or "",
                "main_emotion": e.main_emotion or ""
            }
            for e in request.entries
        ]

        comment = generate_weekly_report(diary_entries)

        return WeeklyReportResponse(comment=comment)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
