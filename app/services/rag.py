import os
import json
import time
import requests
from datetime import datetime
from app.services.embedding import generate_hcx_embedding
from app.core.database import db

HCX_API_KEY = os.getenv("HCX_API_KEY")
HCX_RAG_URL = os.getenv("HCX_RAG_URL")

WELFARE_KEYWORDS = [
    "지원금",
    "복지",
    "정책",
    "청년지원",
    "혜택",
    "센터",
    "고용",
    "실업급여",
    "국민취업지원",
    "주거지원",
    "생활비",
    "대출",
    "정부지원"
]

HOSPITAL_KEYWORDS = [
    "병원",
    "정신과",
    "상담센터",
    "우울",
    "불안",
    "자살",
    "상담",
    "치료",
    "심리"
]


def get_user_context(user_id):
    try:
        diaries = list(
            db["DIARY"]
            .find({"USER_ID": user_id})
            .sort("DATE", -1)
            .limit(3)
        )

        if not diaries:
            return ""

        context = "사용자의 최근 기록:\n"

        for d in diaries:
            date_value = d.get("DATE")

            if isinstance(date_value, datetime):
                date_str = date_value.strftime("%m월 %d일")
            else:
                date_str = "최근"

            emotion = d.get("EMOTION", "")
            content = d.get("CONTENT", "")

            context += (
                f"- 날짜: {date_str}\n"
                f"감정: {emotion}\n"
                f"내용: {content}\n"
            )

        return context

    except Exception as e:
        print(f"Diary Context Error: {e}")
        return ""


def classify_intent(user_input):
    text = user_input.lower()

    if any(keyword in text for keyword in HOSPITAL_KEYWORDS):
        return "hospital_search"

    if any(keyword in text for keyword in WELFARE_KEYWORDS):
        return "welfare_search"

    emotional_keywords = [
        "힘들",
        "우울",
        "슬퍼",
        "외로",
        "죽고",
        "불안",
        "지쳤",
        "괴로",
        "무기력",
        "스트레스"
    ]

    if any(keyword in text for keyword in emotional_keywords):
        return "emotional_support"

    diary_keywords = [
        "일기",
        "오늘 있었던 일",
        "기억나",
        "내 이야기",
        "예전에"
    ]

    if any(keyword in text for keyword in diary_keywords):
        return "diary_chat"

    return "casual_chat"


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
            name = (
                doc.get("NAME")
                or doc.get("SVC_NM")
                or "정보"
            )

            info = (
                doc.get("SVC_DTL")
                or doc.get("ADDR")
                or ""
            )

            context += f"[{name}]\n{info}\n\n"

        return context.strip()

    except Exception as e:
        print(f"Vector Search Error: {e}")
        return ""


def build_system_prompt(intent, diary_context):
    base_prompt = """
당신은 사용자와 자연스럽게 대화하는 AI 친구 '마음'입니다.

규칙:
- 일반 대화에서는 과도한 위로를 하지 마세요.
- 사용자가 실제로 힘들어할 때만 공감해주세요.
- 정책/기관 질문일 때만 정보를 제공하세요.
- 검색 결과가 없으면 억지로 위로하지 말고 자연스럽게 말하세요.
- 답변은 짧고 자연스럽게 작성하세요.
- 같은 표현을 반복하지 마세요.
- "제가 곁에 있을게요" 같은 문장을 반복 사용하지 마세요.
- 상황에 맞는 현실적인 답변을 우선하세요.
"""

    intent_prompt = {
        "casual_chat": """
현재 상황은 일반 대화입니다.
친구처럼 자연스럽게 대화하세요.
""",

        "emotional_support": """
현재 상황은 감정 공감 대화입니다.
과하지 않은 공감과 현실적인 대화를 함께 제공하세요.
""",

        "welfare_search": """
현재 상황은 정책/복지 정보 요청입니다.
검색 결과를 기반으로 핵심 정보 위주로 설명하세요.
""",

        "hospital_search": """
현재 상황은 병원/상담기관 요청입니다.
기관 정보와 도움받을 방법을 안내하세요.
""",

        "diary_chat": """
현재 상황은 사용자의 일기 기반 대화입니다.
최근 기록을 참고해서 자연스럽게 대화하세요.
"""
    }

    return (
        base_prompt
        + "\n"
        + intent_prompt.get(intent, "")
        + "\n"
        + diary_context
    )


def create_tools(intent):
    tools = []

    if intent == "hospital_search":
        tools.append({
            "type": "function",
            "function": {
                "name": "search_hospital",
                "description": "병원 및 상담기관 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    if intent == "welfare_search":
        tools.append({
            "type": "function",
            "function": {
                "name": "search_welfare",
                "description": "복지 및 정책 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    return tools


def stream_response(payload, headers):
    previous_text = ""

    headers["Accept"] = "text/event-stream"

    with requests.post(
        HCX_RAG_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=30
    ) as response:

        for line in response.iter_lines():

            if not line:
                continue

            decoded = line.decode("utf-8")

            if not decoded.startswith("data:"):
                continue

            data_str = decoded[5:].strip()

            if data_str == "[DONE]":
                break

            try:
                parsed = json.loads(data_str)

                current_text = (
                    parsed.get("message", {})
                    .get("content", "")
                )

                delta = current_text[len(previous_text):]

                if delta:
                    previous_text = current_text
                    yield delta

            except Exception:
                continue


def generate_rag_response_stream(user_id, user_input):
    try:
        intent = classify_intent(user_input)

        print(f"USER INPUT: {user_input}")
        print(f"CLASSIFIED INTENT: {intent}")

        diary_context = get_user_context(user_id)

        system_prompt = build_system_prompt(
            intent,
            diary_context
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        tools = create_tools(intent)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HCX_API_KEY}"
        }

        payload = {
            "messages": messages,
            "topP": 0.8,
            "temperature": 0.7,
            "maxTokens": 1024
        }

        if tools:
            payload["tools"] = tools
            payload["toolChoice"] = "auto"

        response = requests.post(
            HCX_RAG_URL,
            headers=headers,
            json=payload,
            timeout=20
        )

        response.raise_for_status()

        result_json = response.json()

        result_message = (
            result_json.get("result", {})
            .get("message", {})
        )

        tool_calls = result_message.get("toolCalls", [])

        if tool_calls:
            messages.append(result_message)

            for tool in tool_calls:

                tool_name = tool["function"]["name"]
                tool_query = tool["function"]["arguments"]["query"]
                tool_id = tool["id"]

                print(f"TOOL: {tool_name}")
                print(f"QUERY: {tool_query}")

                collection_name = (
                    "MENTAL_INST"
                    if tool_name == "search_hospital"
                    else "PUBLIC_SVC"
                )

                search_result = execute_vector_search(
                    tool_query,
                    collection_name
                )

                print(f"SEARCH RESULT: {search_result}")

                tool_content = (
                    search_result
                    if search_result
                    else "검색 결과 없음"
                )

                messages.append({
                    "role": "tool",
                    "content": tool_content,
                    "toolCallId": tool_id
                })

            second_payload = {
                "messages": messages,
                "topP": 0.8,
                "temperature": 0.7,
                "maxTokens": 1024
            }

            for chunk in stream_response(
                second_payload,
                headers
            ):
                yield chunk

        else:
            final_text = (
                result_message.get("content", "")
                or "무슨 이야기든 편하게 해주세요."
            )

            yield final_text

    except Exception as e:
        print(f"RAG API ERROR: {e}")

        fallback_message = (
            "지금 응답 연결이 잠시 불안정해요. "
            "조금 뒤에 다시 이야기해볼까요?"
        )

        for char in fallback_message:
            yield char
            time.sleep(0.01)