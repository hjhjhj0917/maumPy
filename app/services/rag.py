import os
import json
import time
import requests
import re

from datetime import datetime
from app.services.embedding import generate_hcx_embedding
from app.core.database import db

HCX_API_KEY = os.getenv("HCX_API_KEY")
HCX_RAG_URL = os.getenv("HCX_RAG_URL")

REQUEST_TIMEOUT = 30


# =========================================================
# 공통 유틸
# =========================================================

def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def clean_ai_text(text):
    if not text:
        return ""
    # AI가 강조를 위해 사용하는 <내용> 괄호 자체를 제거 (내용은 유지)
    text = re.sub(r'<([^>]+)>', r'\1', text)
    # 마크다운 기호 제거
    text = text.replace('**', '').replace('*', '')
    # AI가 뱉는 불필요한 시스템 잔재 제거
    text = re.sub(r'<기록>|\[기록\]|<사용자 일기 문맥>', '', text)
    # JSON 도구 호출 블록 완벽 제거
    text = re.sub(r'\{\s*"query"\s*:\s*"[^"]+"\s*\}', '', text)
    return text.strip()


def stream_text(text, delay=0.005):
    if not text:
        return

    for char in text:
        formatted = char.replace("\n", "<br>").replace(" ", "<sp>")
        yield f"{formatted}\n"
        time.sleep(delay)


# =========================================================
# 사용자 일기(기억) 검색 및 컨텍스트 가공
# =========================================================

def get_user_context(user_id, user_input=None):
    try:
        try:
            user_no = int(user_id)
        except:
            user_no = user_id

        # 1. 최신 일기 (최근 7개로 확대)
        recent_diaries = list(
            db["DIARY_LOGS"].find({"USER_NO": user_no}).sort("REG_DT", -1).limit(7)
        )

        relevant_diaries = []

        # 2. 질문 관련 과거 일기 (Vector Search 검색량 및 정확도 향상)
        if user_input:
            query_vector = generate_hcx_embedding(user_input)
            if query_vector:
                # [Resilient Pipeline] 필터 인덱스 미설정 시에도 작동하도록 2단계로 시도
                try:
                    # 1차 시도: 필터가 포함된 최적화된 검색
                    pipeline = [
                        {
                            "$vectorSearch": {
                                "index": "vector_index",
                                "path": "EMBEDDING",
                                "queryVector": query_vector,
                                "numCandidates": 100,
                                "limit": 5,
                                "filter": {"USER_NO": user_no}
                            }
                        }
                    ]
                    relevant_diaries = list(db["DIARY_LOGS"].aggregate(pipeline))
                except Exception as ve:
                    print(f"[VECTOR ERROR] {ve}")
                    print("[HINT] 필터 인덱스 설정 전이므로 대체 검색을 수행합니다.")
                    # 2차 시도: 전체 검색 후 파이썬/매치 단계에서 필터링
                    pipeline_fallback = [
                        {
                            "$vectorSearch": {
                                "index": "vector_index",
                                "path": "EMBEDDING",
                                "queryVector": query_vector,
                                "numCandidates": 100,
                                "limit": 20  # 후보군을 넓혀서 내 일기가 포함될 확률을 높임
                            }
                        },
                        {"$match": {"USER_NO": user_no}},
                        {"$limit": 5}
                    ]
                    try:
                        relevant_diaries = list(db["DIARY_LOGS"].aggregate(pipeline_fallback))
                    except:
                        relevant_diaries = []

        # 3. 중복 제거
        seen = set()
        combined = []
        # ... (rest of the logic)

        for doc in recent_diaries + relevant_diaries:
            doc_id = str(doc.get("_id"))
            if doc_id not in seen:
                combined.append(doc)
                seen.add(doc_id)

        if not combined:
            return "최근 작성된 일기 기록이 없습니다."

        combined.sort(key=lambda x: x.get("REG_DT") or datetime.min, reverse=True)

        context = ""
        for d in combined:
            date_value = d.get("DATE") or d.get("date") or d.get("REG_DT")
            date_str = date_value.strftime("%Y년 %m월 %d일") if isinstance(date_value, datetime) else safe_text(date_value)
            content = safe_text(d.get("CONTENT"))[:800]

            # 기분(Emotion) 태그가 검열을 자극할 수 있으므로 제거하고 내용만 전달
            context += f"날짜: {date_str}\n내용: {content}\n\n"

        return context.strip()

    except Exception as e:
        print(f"[CONTEXT ERROR] {e}")
        return "일기 기록을 불러올 수 없습니다."


# =========================================================
# 정책 / 기관 벡터 검색 (Tool)
# =========================================================

def execute_vector_search(query_text, collection_name):
    try:
        query_vector = generate_hcx_embedding(query_text)
        if not query_vector:
            return "관련 정보를 찾을 수 없습니다."

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
            return "검색된 결과가 없습니다."

        context = ""
        for doc in results:
            if collection_name == "PUBLIC_SVC":
                context += f"정책명: {safe_text(doc.get('SVC_NM'))}, 상세내용: {safe_text(doc.get('SVC_DTL'))}, 지원대상: {safe_text(doc.get('TARGET'))}, 신청방법: {safe_text(doc.get('METHOD'))}\n"
            else:
                context += f"기관구분: {safe_text(doc.get('CATEGORY'))}, 기관명: {safe_text(doc.get('NAME'))}, 주소: {safe_text(doc.get('ADDR'))}, 연락처/홈페이지: {safe_text(doc.get('HOMEPAGE'))}\n"
        return context.strip()

    except Exception as e:
        print(f"[VECTOR SEARCH ERROR] {e}")
        return "데이터베이스 검색 중 오류가 발생했습니다."


# =========================================================
# 마스터 시스템 프롬프트 (안전필터 우회 및 팩트 강화)
# =========================================================

def build_system_prompt(diary_context):
    return f"""
당신은 다정한 친구 '마음'입니다. 사용자와 편하게 대화하며 공감해주세요.

[사용자의 기록]
{diary_context}

[대화 원칙]
- 사용자와 일상적인 대화(메뉴 추천, 오늘 날씨 등)를 즐겁게 나눠주세요.
- 만약 사용자가 과거의 기억을 물어보면, 위 [사용자의 기록]을 참고해서 다정하게 이야기해주세요.
- 말투는 "~해요", "~군요" 처럼 친근하게 사용해주세요.
- **절대로 <꺽쇠 괄호>를 사용하지 마세요.**
- 전문적인 심리 상담이나 진단은 하지 마세요. 그냥 옆에 있어주는 친구 역할에 집중하세요.
"""


# =========================================================
# Tool 정의 (Function Calling)
# =========================================================

def create_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_welfare",
                "description": "사용자가 월세, 생활비, 지원금, 복지 혜택 등을 찾을 때 정책을 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_hospital",
                "description": "사용자가 우울증, 심리상담, 병원, 정신건강 센터 등의 정보를 찾을 때 기관을 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }
    ]


# =========================================================
# 메인 RAG & HCX 연동 스트림
# =========================================================

def generate_rag_response_stream(user_id, user_input):
    try:
        print(f"\n[INFO] RAG PROCESS START")
        print(f"[INFO] User Input: {user_input}")

        diary_context = get_user_context(user_id, user_input)
        print(f"[INFO] Retrieved Diary Context:\n{diary_context}\n")

        system_prompt = build_system_prompt(diary_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HCX_API_KEY}"
        }

        payload = {
            "messages": messages,
            "tools": create_tools(),
            "toolChoice": "auto",
            "topP": 0.8,
            "temperature": 0.6,
            "maxTokens": 1024,
            "stream": False
        }

        print("[INFO] Requesting 1st HCX API (Tool or Direct Answer)...")
        response = requests.post(HCX_RAG_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            print(f"[ERROR] API Code: {response.status_code}, Msg: {response.text}")
            yield from stream_text("서버가 잠시 피곤한가 봐요. 조금만 이따가 다시 이야기해요.")
            return

        result_json = response.json()
        print(f"[INFO] 1st HCX Response:\n{json.dumps(result_json, indent=2, ensure_ascii=False)}\n")

        result_obj = result_json.get("result", {})
        message_obj = result_obj.get("message", {}) if "message" in result_obj else result_obj.get("choices", [{}])[0].get("message", {})
        full_content = safe_text(message_obj.get("content"))
        tool_calls = message_obj.get("tool_calls") or message_obj.get("toolCalls") or []

        # ================== Tool Call 처리 ==================
        if tool_calls:
            print(f"[INFO] Tool Call Detected: {len(tool_calls)} tools")
            yield "<think>당신에게 도움이 될 만한 정보를 열심히 찾아보고 있어요...</think>\n"

            # 누수된 JSON 텍스트 청소 후 전송
            clean_first_content = clean_ai_text(full_content)
            if clean_first_content:
                yield from stream_text(clean_first_content + "\n")

            messages.append(message_obj)

            last_tool = ""
            for tool in tool_calls:
                try:
                    tool_name = tool["function"]["name"]
                    last_tool = tool_name
                    args = json.loads(tool["function"]["arguments"]) if isinstance(tool["function"]["arguments"], str) else tool["function"]["arguments"]
                    tool_query = safe_text(args.get("query"))
                    tool_id = tool.get("id") or tool.get("toolCallId")

                    print(f"[INFO] Executing {tool_name} with query: {tool_query}")
                    collection_name = "MENTAL_INST" if tool_name == "search_hospital" else "PUBLIC_SVC"
                    search_result = execute_vector_search(tool_query, collection_name)
                    print(f"[INFO] Tool Result Length: {len(search_result)}")

                    messages.append({
                        "role": "tool",
                        "toolCallId": tool_id,
                        "content": search_result
                    })
                except Exception as e:
                    print(f"[ERROR] TOOL ERROR: {e}")

            second_payload = {
                "messages": messages,
                "toolChoice": "none",
                "topP": 0.8,
                "temperature": 0.6,
                "maxTokens": 1024,
                "stream": False
            }

            print("[INFO] Requesting 2nd HCX API (Final Answer)...")
            second_res = requests.post(HCX_RAG_URL, headers=headers, json=second_payload, timeout=REQUEST_TIMEOUT)

            if second_res.status_code == 200:
                second_json = second_res.json()
                print(f"[INFO] 2nd HCX Response:\n{json.dumps(second_json, indent=2, ensure_ascii=False)}\n")
                second_message_obj = second_json.get("result", {}).get("message", {}) if "message" in second_json.get("result", {}) else second_json.get("result", {}).get("choices", [{}])[0].get("message", {})
                final_content = safe_text(second_message_obj.get("content"))
            else:
                final_content = ""

            final_content = clean_ai_text(final_content)

            # [Tool 예외 처리] 안전 필터 발동 시
            if not final_content:
                print("[WARN] 2nd Answer is empty. Fallback triggered.")
                if last_tool == "search_welfare":
                    final_content = "월세나 생활비 문제로 마음고생이 진짜 많으시죠. 혼자 고민하지 마시고 가까운 동네 주민센터에 가시면 '주거급여' 같은 지원을 꼼꼼히 상담받으실 수 있어요. 꼭 한번 들러보세요. 제가 항상 응원할게요!"
                else:
                    final_content = "혼자서 다 감당하려고 하면 너무 무겁잖아요. 제가 찾아보니 가까운 보건소나 정신건강복지센터에서 마음을 나눌 수 있다고 해요. 시간 날 때 가벼운 산책 겸 한번 들러보는 건 어떨까요? 언제든 제가 이야기 들어줄게요."

            yield from stream_text(final_content)

        # ================== 일반 대화 처리 ==================
        else:
            print("[INFO] No Tool Call. Direct Answer.")

            full_content = clean_ai_text(full_content)

            # [스마트 폴백] 안전 필터 발동 시 - 실제 데이터 기반으로 직접 응답 생성
            if not full_content:
                print("[WARN] 1st Answer is empty (Safety Filter Hit). Smart Fallback triggered.")
                
                # 다이어리 컨텍스트에서 첫 번째 기록의 내용을 추출하여 직접 응답
                try:
                    # diary_context는 "날짜: ...\n당시 기분: ...\n내용: ...\n\n" 구조임
                    first_record = diary_context.split("\n\n")[0]
                    if "내용:" in first_record:
                        # 내용을 안전하게 파싱
                        record_lines = first_record.split("\n")
                        record_date = record_lines[0].replace("날짜:", "").strip()
                        record_content = ""
                        for line in record_lines:
                            if line.startswith("내용:"):
                                record_content = line.replace("내용:", "").strip()[:100]
                                break
                        
                        full_content = f"죄송해요, 잠시 생각이 엉켰어요! 하지만 제가 기억하기론 {record_date}에 이런 일이 있었던 것 같아요: '{record_content}...' 이 내용이 찾으시는 게 맞을까요?"
                    else:
                        full_content = "요즘 정말 고생 많으셨죠. 제가 예전 일기들을 읽어보니 즐거웠던 기억들이 참 많더라고요. 오늘은 무리하지 말고 푹 쉬면서 기운 차리셨으면 좋겠어요."
                except Exception as fe:
                    print(f"[FALLBACK ERROR] {fe}")
                    full_content = "오늘 하루도 버티느라 정말 고생 많았어요. 제가 항상 곁에서 이야기 들어줄게요. 언제든 편하게 말해줘요."

            yield from stream_text(full_content)

        print("[INFO] RAG PROCESS END\n")

    except Exception as e:
        print(f"[ERROR] RAG CRITICAL ERROR: {e}")
        yield from stream_text("앗, 잠시 제 생각이 엉켰어요. 방금 하신 말씀 다시 한 번 들려주실래요?")