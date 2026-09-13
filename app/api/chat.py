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


@router.post("/api/rag-chat")
async def process_rag_chat(request: ChatRequest): # 매개변수 부분에 request: DiaryRequest 이 부분은 자바에서 @RquestBody ResponseDTO response와 같은 역할을 함
    print(f"[{request.userNo}의 메시지]: {request.message}")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    async def event_generator(): # chunk로 응답이 다 생성되지 않아도 단어가 생성될 때 마다 가져옴
        for chunk in generate_rag_response_stream(request.userNo, request.message, history):
            yield f"{chunk}\n"

    return StreamingResponse( # 생성된 응답을 연결 유지하면서 보냄
        event_generator(),
        media_type="text/plain"
    )