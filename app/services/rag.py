import os
import json
import time
import requests
from datetime import datetime
from app.services.embedding import generate_hcx_embedding
from app.core.database import db

HCX_API_KEY = os.getenv("HCX_API_KEY")
HCX_RAG_URL = os.getenv("HCX_RAG_URL")

def get_user_context(user_id):
    try:
        # user_id를 정수로 변환하여 조회 시도 (문자열로 들어왔을 가능성 대비)
        try:
            user_no_int = int(user_id)
        except ValueError:
            user_no_int = user_id

        print(f"--- [DB 조회 시작] USER_NO: {user_no_int} (Type: {type(user_no_int)}) ---")
        
        # 숫자 타입으로 먼저 조회
        diaries = list(
            db["DIARY"]
            .find({"USER_NO": user_no_int})
            .sort("REG_DT", -1)
            .limit(3)
        )

        if not diaries:
            # 문자열 타입으로도 조회 시도 (안전장치)
            diaries = list(
                db["DIARY"]
                .find({"USER_NO": str(user_id)})
                .sort("REG_DT", -1)
                .limit(3)
            )

        if not diaries:
            return "최근 작성된 일기가 없습니다."

        context = "사용자의 최근 기록(개인적인 기억):\n"

        for d in diaries:
            date_value = d.get("DATE") or d.get("date") or d.get("REG_DT")

            if isinstance(date_value, datetime):
                date_str = date_value.strftime("%Y년 %m월 %d일")
            else:
                date_str = str(date_value)

            emotion = d.get("MAIN_EMOTION") or d.get("EMOTION") or d.get("emotion", "")
            title = d.get("TITLE") or d.get("title", "")
            content = d.get("CONTENT") or d.get("content", "")

            context += (
                f"- 날짜: {date_str}\n"
                f"  제목: {title}\n"
                f"  감정: {emotion}\n"
                f"  내용: {content}\n"
            )

        return context

    except Exception as e:
        print(f"Diary Context Error: {e}")
        return ""

def execute_vector_search(query_text, collection_name):
    try:
        query_vector = generate_hcx_embedding(query_text)

        if not query_vector:
            return ""

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "EMBEDDING",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": 5
                }
            }
        ]

        results = list(db[collection_name].aggregate(pipeline))

        if not results:
            return ""

        context = ""

        for doc in results:
            name = doc.get("NAME") or doc.get("SVC_NM") or "정보"
            info = doc.get("SVC_DTL") or doc.get("ADDR") or ""
            context += f"[{name}]\n{info}\n\n"

        return context.strip()

    except Exception as e:
        print(f"Vector Search Error: {e}")
        return ""

def build_system_prompt(diary_context):
    return f"""
당신은 사려 깊고 따뜻한 AI 친구 '마음'입니다.

[사용자 문맥 (개인적 일기)]
{diary_context}

[행동 지침]
1. [사용자 문맥]에 있는 내용은 사용자가 실제로 겪은 '개인적인 과거 기억'입니다. 질문에 답변할 때 이 내용을 우선적으로 활용하세요.
2. 만약 [사용자 문맥]에 관련 내용이 없다면, 절대로 과거 기억을 지어내지 마세요. 대신 "아직 그 부분에 대해서는 들은 적이 없지만, 어떤 일이 있었는지 말해줄 수 있어?"라고 정중하게 물어보세요.
3. 도구(search_hospital, search_welfare)를 통해 얻은 검색 결과는 '외부 지원 기관 정보'일 뿐입니다. 이를 사용자의 개인적인 친구나 기억으로 착각하여 답변하지 마세요. (예: '좋은날'이라는 기관 정보를 보고 "예전에 '좋은날'에 가서 즐거웠잖아"라고 말하면 안 됨)
4. 사용자가 우울하거나 즐거운 일이 없다고 하면 따뜻하게 공감하고, 필요한 경우에만 검색된 외부 정보를 '추천'하듯 전달하세요.
5. 마크다운을 사용하여 가독성 좋게 답변하세요.
"""

def create_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_hospital",
                "description": "우울증, 심리 상담, 병원, 정신건강 관련 지원 기관 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_welfare",
                "description": "취업, 주거, 생활비, 지원금, 복지 정책 등 현실적인 지원 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]


def stream_response(payload, headers):
    previous_text = ""
    # 헤더가 None이면 빈 딕셔너리로 초기화
    if headers is None:
        headers = {}
    
    # Accept 헤더를 강제로 설정
    headers["Accept"] = "text/event-stream"
    payload["stream"] = True

    print(f"--- HCX 스트리밍 호출 시작 ---")
    print(f"URL: {HCX_RAG_URL}")
    print(f"Headers: {json.dumps({k: v if k != 'Authorization' else 'Bearer ***' for k, v in headers.items()}, ensure_ascii=False)}")

    with requests.post(
        HCX_RAG_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=30
    ) as response:

        print(f"HCX 응답 상태: {response.status_code}")
        
        if response.status_code != 200:
            print(f"HCX 에러 상세: {response.text}")
            yield "정보를 가져오는 중 오류가 발생했습니다."
            return

        for line in response.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8")

            if not decoded.startswith("data:"):
                if decoded.strip():
                    print(f"비정상 라인: {decoded}")
                continue

            data_str = decoded[5:].strip()

            if data_str == "[DONE]":
                print("--- HCX 스트림 완료 [DONE] ---")
                break

            try:
                parsed = json.loads(data_str)

                message_obj = (
                    parsed.get("message")
                    or parsed.get("result", {}).get("message", {})
                )

                current_text = message_obj.get("content", "")

                delta = current_text[len(previous_text):]

                if delta:
                    previous_text = current_text
                    yield delta

            except Exception as e:
                print(f"파싱 에러: {e}")
                continue

def generate_rag_response_stream(user_id, user_input):
    try:
        diary_context = get_user_context(user_id)
        print(f"--- [DIARY CONTEXT] ---\n{diary_context}\n-----------------------")
        system_prompt = build_system_prompt(diary_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        tools = create_tools()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HCX_API_KEY}"
        }

        payload = {
            "messages": messages,
            "tools": tools,
            "toolChoice": "auto",
            "topP": 0.8,
            "temperature": 0.7,
            "maxTokens": 1024
        }

        response = requests.post(
            HCX_RAG_URL,
            headers=headers,
            json=payload,
            timeout=20
        )

        response.raise_for_status()
        result_json = response.json()
        print(f"--- HCX 1차 응답 데이터: {json.dumps(result_json, ensure_ascii=False)} ---")
        
        result_message = (
            result_json.get("result", {}).get("message", {}) 
            or result_json.get("message", {})
        )
        
        tool_calls = result_message.get("tool_calls") or result_message.get("toolCalls") or []

        if tool_calls:
            yield "<think>마음이 필요한 정보를 찾고 있어요...</think>"
            messages.append(result_message)

            for tool in tool_calls:
                tool_name = tool["function"]["name"]
                tool_query = tool["function"]["arguments"]["query"]
                tool_id = tool.get("id") or tool.get("toolCallId")

                collection_name = (
                    "MENTAL_INST" if tool_name == "search_hospital" else "PUBLIC_SVC"
                )

                search_result = execute_vector_search(tool_query, collection_name)
                print(f"--- [SEARCH RESULT ({collection_name})] ---\n{search_result}\n-----------------------")
                
                tool_content = search_result if search_result else "관련 정보를 찾지 못했습니다. 따뜻하게 위로해 주세요."

                messages.append({
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tool_id
                })

            second_payload = {
                "messages": messages,
                "topP": 0.8,
                "temperature": 0.7,
                "maxTokens": 1024,
                "stream": False # 406 에러 해결을 위해 False로 시도
            }

            # 2차 호출은 스트리밍이 아닌 일반 호출로 수행
            stream_headers = headers.copy()
            stream_headers["Accept"] = "application/json"

            print(f"--- HCX 2차 호출(비스트리밍) 시작 ---")
            second_res = requests.post(
                HCX_RAG_URL,
                headers=stream_headers,
                json=second_payload,
                timeout=30
            )
            
            if second_res.status_code == 200:
                res_data = second_res.json()
                final_content = (
                    res_data.get("result", {}).get("message", {}).get("content", "")
                    or res_data.get("message", {}).get("content", "")
                    or "도움이 될 만한 정보를 찾았습니다."
                )
                for char in final_content:
                    yield char
                    time.sleep(0.005)
            else:
                print(f"HCX 2차 호출 실패: {second_res.status_code}, {second_res.text}")
                yield "정보를 정리하는 중에 문제가 발생했습니다."
        else:
            final_text = result_message.get("content", "") or "오늘 하루도 정말 고생 많았어요."
            for char in final_text:
                yield char
                time.sleep(0.01)

    except Exception as e:
        print(f"RAG API ERROR: {e}")
        fallback_message = "지금 응답 연결이 잠시 불안정해요. 조금 뒤에 다시 이야기해볼까요?"
        for char in fallback_message:
            yield char
            time.sleep(0.01)