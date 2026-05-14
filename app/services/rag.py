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
        diaries = list(db["DIARY"].find({"USER_ID": user_id}).sort("DATE", -1).limit(3))
        if not diaries: return "최근 일기 없음"
        context = "내 친구의 최근 일상과 감정이야:\n"
        for d in diaries:
            date_str = d.get("DATE", datetime.now()).strftime("%m월 %d일")
            context += f"- {date_str} / 감정: {d.get('EMOTION', '비밀')} / 내용: {d.get('CONTENT', '')}\n"
        return context
    except Exception:
        return ""


def execute_vector_search(query_text, collection_name):
    try:
        query_vector = generate_hcx_embedding(query_text)
        if not query_vector: return ""
        pipeline = [{
            "$vectorSearch": {
                "index": "vector_index",
                "path": "EMBEDDING",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5
            }
        }]
        results = db[collection_name].aggregate(pipeline)
        context = ""
        for doc in results:
            name = doc.get("NAME") or doc.get("SVC_NM") or "정책 정보"
            info = doc.get("SVC_DTL") or doc.get("ADDR") or ""
            context += f"[{name}] {info}\n"
        return context
    except Exception:
        return ""


def generate_rag_response_stream(user_id, user_input):
    diary_context = get_user_context(user_id)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HCX_API_KEY}"
    }
    system_prompt = f"""당신은 사려 깊은 상담가 '담소'입니다.
1. 사용자가 슬프거나 힘든 상황(예: 면접 탈락, 돈 문제)을 말하면 정책 유무와 상관없이 반드시 따뜻하게 먼저 위로하세요.
2. '취업', '면접', '지원', '돈' 관련 키워드에는 반드시 검색 도구를 사용하여 정보를 찾으세요.
3. 검색 결과가 없더라도 "정보가 없다"고만 하지 마세요. "지금 딱 맞는 정책은 못 찾았지만 제가 계속 찾아볼게요. 오늘 정말 고생 많았어요"라고 다정하게 말하세요.
4. 답변은 가독성 있게 줄바꿈과 마크다운을 사용하세요.
{diary_context}"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    tools = [
        {"type": "function", "function": {"name": "search_hospital", "description": "병원 검색",
                                          "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                                         "required": ["query"]}}},
        {"type": "function", "function": {"name": "search_welfare", "description": "취업/복지 정책 검색",
                                          "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                                         "required": ["query"]}}}
    ]
    payload = {"messages": messages, "tools": tools, "toolChoice": "auto", "topP": 0.8, "maxTokens": 1024,
               "temperature": 0.6}

    try:
        response = requests.post(HCX_RAG_URL, headers=headers, json=payload, timeout=15).json()
        res_msg = response.get("result", {}).get("message", {})

        if "toolCalls" in res_msg:
            messages.append(res_msg)
            for tool in res_msg["toolCalls"]:
                t_name, t_query, t_id = tool["function"]["name"], tool["function"]["arguments"]["query"], tool["id"]
                target_col = "MENTAL_INST" if t_name == "search_hospital" else "PUBLIC_SVC"
                search_res = execute_vector_search(t_query, target_col)
                tool_content = search_res if search_res else "적절한 정책을 찾지 못했습니다. 따뜻한 위로와 응원을 중심으로 답변하세요."
                messages.append({"role": "tool", "content": tool_content, "toolCallId": t_id})

            headers["Accept"] = "text/event-stream"
            payload["messages"] = messages
            payload.pop("tools", None)
            payload.pop("toolChoice", None)

            previous_text = ""
            with requests.post(HCX_RAG_URL, headers=headers, json=payload, stream=True) as stream_res:
                for line in stream_res.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith('data:'):
                            data_str = decoded[5:].strip()
                            if data_str == '[DONE]': break
                            try:
                                current_text = json.loads(data_str).get("message", {}).get("content", "")
                                delta = current_text[len(previous_text):]
                                if delta:
                                    previous_text = current_text
                                    yield delta
                            except:
                                pass
        else:
            final_text = res_msg.get("content", "") or "마음이 많이 힘들죠. 제가 곁에 있을게요."
            for char in final_text:
                yield char
                time.sleep(0.01)
    except Exception:
        yield "잠시 마음을 정리할 시간이 필요해요. 다시 말씀해 주시겠어요?"