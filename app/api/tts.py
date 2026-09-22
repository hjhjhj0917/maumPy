import base64

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.tts import strip_emoji_for_tts, strip_markdown_for_tts, split_into_tts_chunks, synthesize_speech

router = APIRouter()


class TtsRequest(BaseModel):
    text: str


class TtsResponse(BaseModel):
    # Java 쪽 DTO(TtsResponseDTO.audioChunks())와 그대로 매칭되도록 camelCase로 맞춤
    # (이 프로젝트의 다른 Python 응답 DTO도 Java와 직접 매칭될 때는 camelCase를 씀 — 예: analyze.py의 TrackDTO)
    audioChunks: list[str]


# ★ 즐겨찾기 이후 추가/수정
# 채팅 스트리밍(rag.py)에서는 답변 생성과 동시에 오디오를 합성해 보내지만,
# 과거 대화 내역은 오디오를 다시 저장해두지 않으므로 재진입 시 텍스트만으로 이 API를 호출해
# 오디오를 다시 합성함 (Spring이 저장된 답변 텍스트를 그대로 넘겨줌)
@router.post("/api/tts", response_model=TtsResponse)
async def text_to_speech(body: TtsRequest):
    cleaned = strip_markdown_for_tts(strip_emoji_for_tts(body.text.strip()))
    if not cleaned:
        return TtsResponse(audioChunks=[])

    audio_chunks = []
    for chunk in split_into_tts_chunks(cleaned):
        audio_bytes = synthesize_speech(chunk)
        if audio_bytes:
            audio_chunks.append(base64.b64encode(audio_bytes).decode('utf-8'))

    return TtsResponse(audioChunks=audio_chunks)
