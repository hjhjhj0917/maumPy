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

    # StreamingResponse를 사용해 데이터가 생성되는 즉시 Spring(또는 프론트)으로 쏴줍니다.
    return StreamingResponse(
        generate_rag_response_stream(request.userNo, request.message),
        media_type="text/plain"
    )