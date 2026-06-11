from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.rag import generate_rag_response_stream

router = APIRouter()


class ChatRequest(BaseModel):
    userNo: str
    message: str


@router.post("/api/rag-chat")
async def process_rag_chat(request: ChatRequest): # 매개변수 부분에 request: DiaryRequest 이 부분은 자바에서 @RquestBody ResponseDTO response와 같은 역할을 함
    print(f"[{request.userNo}의 메시지]: {request.message}")

    async def event_generator(): # chunk로 응답이 다 생성되지 않아도 단어가 생성될 때 마다 가져옴
        for chunk in generate_rag_response_stream(request.userNo, request.message):
            yield f"{chunk}\n"

    return StreamingResponse( # 생성된 응답을 연결 유지하면서 보냄
        event_generator(),
        media_type="text/plain"
    )