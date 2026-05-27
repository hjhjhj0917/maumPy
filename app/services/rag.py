import os
import json
import time
import requests
from datetime import datetime
from app.services.embedding import generate_hcx_embedding
from app.core.database import db

HCX_API_KEY = os.getenv("HCX_API_KEY")
HCX_RAG_URL = os.getenv("HCX_RAG_URL")

def get_user_context(user_id, user_input=None):
    try:
        # user_id를 정수로 변환하여 조회 시도 (문자열로 들어왔을 가능성 대비)
        try:
            user_no_int = int(user_id)
        except ValueError:
            user_no_int = user_id

        print(f"--- [DB 하이브리드 조회 시작] USER_NO: {user_no_int} (Type: {type(user_no_int)}) ---")

        # 1. 최신 일기 2건 조회
        recent_diaries = list(
            db["DIARY_LOGS"]
            .find({"USER_NO": user_no_int})
            .sort("REG_DT", -1)
            .limit(2)
        )

        # 2. 질문 관련 과거 일기 조회 (Vector Search)
        relevant_diaries = []
        if user_input:
            print(f"--- [기억 소환 시작] 질문: {user_input} ---")
            query_vector = generate_hcx_embedding(user_input)
            if query_vector:
                # Atlas Vector Search Pipeline
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "EMBEDDING",
                            "queryVector": query_vector,
                            "numCandidates": 100,
                            "limit": 2
                        }
                    },
                    {
                        "$match": {"USER_NO": user_no_int} # 반드시 내 일기만 필터링
                    }
                ]
                try:
                    relevant_diaries = list(db["DIARY_LOGS"].aggregate(pipeline))
                except Exception as ve:
                    print(f"Vector Search Index Error (DIARY_LOGS): {ve}")
                    # 인덱스가 없을 경우를 대비해 에러 발생 시 빈 리스트로 유지

        # 3. 중복 제거 및 병합 (_id 기준)
        seen_ids = set()
        combined_diaries = []
        for d in recent_diaries + relevant_diaries:
            d_id = str(d.get("_id"))
            if d_id not in seen_ids:
                combined_diaries.append(d)
                seen_ids.add(d_id)

        if not combined_diaries:
            return "최근 작성된 일기가 없습니다."

        # 날짜순 정렬
        combined_diaries.sort(key=lambda x: x.get("REG_DT") or datetime.min, reverse=True)

        context = "사용자의 기록(최근 및 관련 기억):\n"

        for d in combined_diaries:
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
사용자의 최근 일기 내용을 바탕으로 친구처럼 다정하고 아주 상세하게 대화해 주세요.

[사용자 일기 문맥]
{diary_context}

사용자가 과거의 즐거웠던 기억을 물어보면, 위 [사용자 일기 문맥]에 있는 내용을 아주 자세하고 따뜻하게 이야기해 주세요.
친근한 반말과 존댓말을 적절히 섞거나 다정한 말투를 사용하고, 이모지를 활용하여 따뜻한 분위기를 만들어 주세요.

[중요 지침]
1. 정책이나 복지 서비스(취업, 주거 등)에 대한 정보의 출처는 반드시 **'대한민국 공공서비스 정보 (행정안전부 제공)'**라고 언급해 주세요.
2. 외부 정보를 제공할 때는 사용자의 상황에 맞춰 따뜻한 조언과 함께 전달해 주세요.
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

    headers["Accept"] = "text/event-stream"
    payload["stream"] = True

    print(f"--- HCX 스트리밍 호출 시작 ---")
    print(f"URL: {HCX_RAG_URL}")

    with requests.post(
            HCX_RAG_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=30
    ) as response:

        print(f"HCX 응답 상태: {response.status_code}")

        if response.status_code != 200:
            yield "정보를 가져오는 중 오류가 발생했습니다.\n"
            return

        for line in response.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8")

            if not decoded.startswith("data:"):
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
                    # 💡 핵심: data: 를 빼고 순수 텍스트와 \n만 전송! (스프링이 data:를 붙여줌)
                    formatted_delta = delta.replace('\n', '<br>').replace(' ', '<sp>')
                    yield f"{formatted_delta}\n"

            except Exception as e:
                print(f"파싱 에러: {e}")
                continue


def generate_rag_response_stream(user_id, user_input):
    try:
        diary_context = get_user_context(user_id, user_input)
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
            "maxTokens": 1024,
            "stream": False
        }

        print(f"--- HCX 1차 비스트리밍 호출 시작 ---")

        response = requests.post(HCX_RAG_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            yield "대화 중 오류가 발생했습니다.\n"
            return

        result_json = response.json()
        print(f"--- [HCX 1차 응답 전체 데이터] ---\n{json.dumps(result_json, ensure_ascii=False, indent=2)}\n-----------------------")

        result_obj = result_json.get("result", {})
        
        # [DEBUG] 응답 상태 및 중단 사유 분석
        stop_reason = result_json.get("stopReason") or result_obj.get("stopReason")
        if stop_reason:
            print(f"--- [DEBUG] Stop Reason: {stop_reason} ---")
            if stop_reason == "content_filter":
                print("⚠️ 경고: 세이프티 필터에 의해 답변이 차단되었습니다.")
        
        # 1. 메시지 객체 찾기
        result_message = {}
        if "message" in result_obj:
            result_message = result_obj["message"]
        elif "choices" in result_obj and len(result_obj["choices"]) > 0:
            result_message = result_obj["choices"][0].get("message", {})
        elif "message" in result_json:
            result_message = result_json["message"]

        # 2. 텍스트 내용 추출
        full_content = (
                result_message.get("content", "") or
                result_obj.get("outputText", "") or
                result_obj.get("answer", "") or
                result_obj.get("text", "") or
                ""
        )

        print("full_content:", full_content)

        if not full_content and "choices" in result_obj:
            for choice in result_obj["choices"]:
                full_content = choice.get("text", "") or choice.get("message", {}).get("content", "")
                if full_content: break

        # 3. 도구 호출 추출
        tool_calls = (
                result_message.get("tool_calls") or
                result_message.get("toolCalls") or
                result_obj.get("tool_calls") or
                result_obj.get("toolCalls") or
                []
        )

        print("tool_calls:", tool_calls)

        if tool_calls:
            print(f"--- [TOOL CALLS DETECTED] ---")

            yield "<think>마음이 필요한 정보를 찾고 있어요...</think>\n"

            assistant_msg = {"role": "assistant", "content": full_content, "tool_calls": tool_calls}
            messages.append(assistant_msg)

            for tool in tool_calls:
                tool_name = tool["function"]["name"]
                args = tool["function"]["arguments"]

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                tool_query = args.get("query", "")
                tool_id = tool.get("id") or tool.get("toolCallId")

                collection_name = "MENTAL_INST" if tool_name == "search_hospital" else "PUBLIC_SVC"
                search_result = execute_vector_search(tool_query, collection_name)

                messages.append({
                    "role": "tool",
                    "content": search_result if search_result else "관련 정보를 찾지 못했습니다.",
                    "tool_call_id": tool_id
                })

            second_payload = {
                "messages": messages,
                "topP": 0.8,
                "temperature": 0.7,
                "maxTokens": 1024,
                "stream": False
            }

            second_headers = headers.copy()
            second_headers["Accept"] = "application/json"

            second_res = requests.post(HCX_RAG_URL, headers=second_headers, json=second_payload, timeout=30)

            if second_res.status_code == 200:
                res_data = second_res.json()
                final_content = (
                        res_data.get("result", {}).get("message", {}).get("content", "") or
                        res_data.get("message", {}).get("content", "") or
                        res_data.get("result", {}).get("outputText", "")
                )
                if final_content:
                    for char in final_content:
                        formatted_char = char.replace('\n', '<br>').replace(' ', '<sp>')
                        yield f"{formatted_char}\n"
                        time.sleep(0.005)
            else:
                yield "정보를 정리하는 중 오류가 발생했습니다.\n"
        else:
            if not full_content and result_obj.get("usage", {}).get("completionTokens", 0) > 0:
                full_content = "말씀하신 내용에 대해 정보를 찾아보았지만, 상세한 답변을 구성하는 데 잠시 문제가 생겼어요. 다시 한번 구체적으로 물어봐 주시겠어요?"

            final_text = full_content or "오늘 하루도 정말 고생 많았어요. 더 궁금한 점이 있으시면 언제든 말씀해 주세요."
            for char in final_text:
                formatted_char = char.replace('\n', '<br>').replace(' ', '<sp>')
                yield f"{formatted_char}\n"
                time.sleep(0.01)

    except Exception as e:
        print(f"RAG API ERROR: {e}")
        fallback_message = "지금 응답 연결이 잠시 불안정해요. 조금 뒤에 다시 이야기해볼까요?"
        for char in fallback_message:
            formatted_char = char.replace('\n', '<br>').replace(' ', '<sp>')
            yield f"{formatted_char}\n"
            time.sleep(0.01)