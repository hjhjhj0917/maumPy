from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.rag import generate_rag_response_stream

router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # "user" 또는 "bot" (Spring의 ChatMessageDTO와 동일한 값)
    content: str


class ChatRequest(BaseModel):
    userNo: str
    message: str
    history: list[ChatMessage] = []  # 최근 대화 기록 (Spring이 Redis에서 최근 4턴까지 조회해서 넘겨줌)


# ★ 즐겨찾기 이후 추가/수정
@router.post("/api/rag-chat")
async def process_rag_chat(request: ChatRequest):
    print(f"[{request.userNo}의 메시지]: {request.message}")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    # ★ 즐겨찾기 이후 추가/수정
    async def event_generator():  # 전체 응답이 완성되기 전에 chunk 단위로 즉시 흘려보냄
        for chunk in generate_rag_response_stream(request.userNo, request.message, history):
            yield f"{chunk}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )