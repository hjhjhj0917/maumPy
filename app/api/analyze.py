from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone

from app.services.prediction import analyze_diary
from app.services.emotion import analyze_emotions
from app.services.summary import generate_hcx_summary
from app.services.embedding import generate_hcx_embedding
from app.core.database import diary_logs_collection

router = APIRouter()

# 자바에 DTO 같은 역할을 하는 클래스를 정의함
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


# api 요청이 들어오면, 여기서 실행을 함, response_model=DiaryResponse는 위에서 선언한 클래스 형태로 반환하겠다는 설정
@router.post("/api/analyze", response_model=DiaryResponse)
async def analyze_text(request: DiaryRequest): # 매개변수 부분에 request: DiaryRequest 이 부분은 자바에서 @RquestBody ResponseDTO response와 같은 역할을 함
    try:
        # 우울증과 감정을 분석하는 파이썬 함수를 실행해서 그 결과를 변수에 저장함
        dep_data = analyze_diary(request.content, request.disease_type)
        emo_data = analyze_emotions(request.content)

        hcx_summary = generate_hcx_summary(
            content=request.content,
            dep_level=dep_data["dep_res"]["final_level"],
            raw_emotions=emo_data["raw_emotions"]
        )

        # 원본 제목과, 내용 그리고 요약한 일기 내용을 한 문장으로 합침
        combined_text = f"제목: {request.title}\n내용: {request.content}\n요약: {hcx_summary}"

        # 합친 문장을 임베딩 함수로 전달하여 실행
        embedding_vector = generate_hcx_embedding(combined_text)

        # MongoDB에 저장한 내용과 쿼리문을 정의
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
                "VERSION": "MAUM-Ensemble-v1.0"
            },
            "$setOnInsert": {
                "DIARY_NO": request.diary_no,
                "USER_NO": request.user_no,
                "REG_DT": request.created_at
            }
        }

        # 실제 MongoDB에 컬렉션을 업데이트 하는 로직, upsert=True로 기존 데이터가 존재하면 update 아니면 insert
        diary_logs_collection.update_one(
            {"DIARY_NO": request.diary_no},
            update_query,
            upsert=True
        )

        # 스프링 서버로 필요한 데이터를 지정한 클래스 형식에 맞게 반환함
        return DiaryResponse(
            analysis_summary=hcx_summary,
            main_emotion=emo_data["main_emotion"],
            main_color=emo_data["main_color"],
            dep_res=dep_data["dep_res"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc() # 에러가 발생하 경로를 보여줌
        raise HTTPException(status_code=500, detail=str(e)) # 클라이언트에게도 문제를 알림