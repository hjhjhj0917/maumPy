from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from app.services.stt import transcribe_speech

router = APIRouter()


class SttResponse(BaseModel):
    text: str


@router.post("/api/stt", response_model=SttResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    text = transcribe_speech(audio_bytes)
    return SttResponse(text=text)
