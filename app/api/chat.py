from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.rag import generate_rag_response_stream

router = APIRouter()


class ChatRequest(BaseModel):
    userNo: str
    message: str


@router.post("/api/rag-chat")
async def process_rag_chat(request: ChatRequest):
    print(f"[{request.userNo}의 메시지]: {request.message}")

    async def event_generator():
        for chunk in generate_rag_response_stream(request.userNo, request.message):
            yield f"{chunk}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )