import os
import json
import time

import requests
import re
import base64
from app.services.tts import synthesize_speech

from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from app.services.embedding import generate_embedding
from app.core.database import db

# embedding.py/tts.py/summary.py와 동일한 서비스 계정 재사용
GCP_CREDENTIALS_JSON = os.getenv("GCP_CREDENTIALS_JSON")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

GEMINI_API_URL = (
    f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}"
    f"/locations/{GCP_LOCATION}/publishers/google/models/{GEMINI_CHAT_MODEL}:generateContent"
)

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_credentials_info = json.loads(GCP_CREDENTIALS_JSON)
_credentials = service_account.Credentials.from_service_account_info(_credentials_info, scopes=_SCOPES)


def _get_access_token() -> str:
    if not _credentials.valid:
        _credentials.refresh(Request())
    return _credentials.token


REQUEST_TIMEOUT = 30


# 공통 유틸
def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def clean_ai_text(text):
    if not text:
        return ""
    # AI가 강조를 위해 사용하는 <내용> 괄호 자체를 제거 (내용은 유지)
    text = re.sub(r'<([^>]+)>', r'\1', text)
    # 마크다운(**굵게**, - 목록 등)은 여기서 지우지 않고 그대로 둠 — 화면에서 ReactMarkdown이 렌더링해서 가독성을 살림
    # AI가 뱉는 불필요한 시스템 잔재 제거
    text = re.sub(r'<기록>|\[기록\]|<사용자 일기 문맥>', '', text)
    # JSON 도구 호출 블록 완벽 제거
    text = re.sub(r'\{\s*"query"\s*:\s*"[^"]+"\s*\}', '', text)
    # AI가 목록을 줄바꿈 없이 ". - 항목", "! * 항목"처럼 문장 부호 뒤에 바로 이어 쓰는 경우가 있어서,
    # 문장 부호(.?!) 바로 뒤의 -/*(굵게 표시용 **는 건드리지 않음)를 실제 목록 블록으로 교정.
    # 목록은 줄바꿈 한 번만으로도 새 블록으로 인식되므로(빈 줄까지는 불필요 — 오히려 loose list가 되어 간격만 넓어짐),
    # 줄바꿈 하나만 넣어서 촘촘한 목록으로 만듦
    text = re.sub(r'(?<=[.?!])\s+[-*](?!\*)\s+', '\n- ', text)
    # 화면이 white-space: pre-wrap이라 AI가 가끔 내는 연속 공백(스페이스 2칸 이상)이 그대로 보이는 문제가 있어서,
    # 줄바꿈은 그대로 두고 가로 공백만 한 칸으로 정리함
    text = re.sub(r'[^\S\n]{2,}', ' ', text)
    # AI가 문단 사이에 줄바꿈을 3개 이상 연달아 쓰는 경우가 있어서, 빈 문단이 여러 개 끼어 간격이 과하게 벌어짐
    # → 문단 구분에 필요한 빈 줄 하나(줄바꿈 두 번)까지만 남기고 나머지는 정리
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def stream_text(text, delay=0.005):
    if not text:
        return

    for char in text:
        formatted = char.replace("\n", "<br>").replace(" ", "<sp>")
        yield f"{formatted}\n"
        time.sleep(delay)

# 문장 종결 부호(. ! ?) 또는 줄바꿈 + 그 뒤 공백까지 통째로 한 조각으로 잡아냄.
# re.split은 구분자 자체(특히 줄바꿈)를 결과에서 없애버려서 문단/목록 구분이 사라지는 문제가 있었기 때문에,
# 구분자를 버리지 않고 조각 끝에 그대로 남겨두는 findall 방식으로 바꿈
SENTENCE_CHUNK_PATTERN = re.compile(r'.*?(?:[.!?]+\s*|\n+\s*)|.+$', flags=re.DOTALL)

# TTS로 보내기 전 이모지를 제거하기 위한 패턴 (화면 표시용 텍스트에는 영향 없음)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # 이모지 전체 대역 (표정, 사물, 동물, 기호 등)
    "\U00002600-\U000027BF"  # 기타 기호 및 딩뱃 (☀ ✨ ❤ 등)
    "\U0001F1E0-\U0001F1FF"  # 국기
    "\U00002B00-\U00002BFF"  # 화살표/별 등 잡기호
    "\U0000FE0F"             # variation selector (이모지 렌더링 지시자)
    "]+",
    flags=re.UNICODE
)


def strip_emoji_for_tts(text):
    # TTS 합성용으로만 이모지를 제거하고 앞뒤 공백을 정리함
    return EMOJI_PATTERN.sub('', text).strip()


# 마크다운 기호(굵게, 목록, 헤더)가 그대로 있으면 TTS가 "별표 별표"처럼 읽어버리므로,
# 화면 표시용 텍스트는 그대로 두고 TTS로 보내기 직전에만 이 함수로 제거함
MARKDOWN_PATTERN = re.compile(r'\*\*|\*|^#{1,6}\s*|^[-•]\s*', flags=re.MULTILINE)


def strip_markdown_for_tts(text):
    return MARKDOWN_PATTERN.sub('', text).strip()


def stream_text_with_audio(text):
    """
    텍스트는 문장 단위 TTS 합성(네트워크 호출, 문장당 1~2초)을 기다리지 않고 먼저 전부
    끊김없이 흘려보내고, 그 다음에 문장별 오디오를 순서대로 이어서 전송함.
    (문장마다 텍스트→오디오를 번갈아 보내면 오디오 합성 대기 때문에 텍스트 출력이
    문장 단위로 끊겨 보이는 문제가 있어서, 텍스트 전송과 오디오 합성을 분리함)
    """
    if not text:
        return

    # 조각 끝의 줄바꿈/공백을 그대로 유지 — 화면에는 이 조각을 통째로 흘려보내서 문단/목록 구분이 살아있게 함
    chunks = [c for c in SENTENCE_CHUNK_PATTERN.findall(text) if c.strip()]

    for chunk in chunks:
        yield from stream_text(chunk)

    # 텍스트 전송이 끝났다는 걸 프론트에 알림 — 프론트는 이 시점부터 마크다운(굵게/목록)을
    # 렌더링해도 됨. 이게 없으면 프론트가 "오디오까지 다 끝나야 완료"로 판단해서
    # 오디오 합성 시간만큼 마크다운 적용이 늦어짐
    yield "[[TEXT_DONE]]\n"

    for chunk in chunks:
        # TTS에는 앞뒤 공백/줄바꿈을 정리하고, 이모지와 마크다운 기호를 제거한 버전만 넘김
        tts_text = strip_markdown_for_tts(strip_emoji_for_tts(chunk.strip()))
        if not tts_text:
            continue
        audio_bytes = synthesize_speech(tts_text)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            yield f"[[AUDIO]]{audio_b64}[[/AUDIO]]\n"

# 사용자 일기(기억) 검색 및 컨텍스트 가공
def get_user_context(user_id, user_input=None):
    try:
        try:
            user_no = int(user_id)
        except:
            user_no = user_id

        # 1. 최신 일기 7개
        recent_diaries = list(
            db["DIARY_LOGS"].find({"USER_NO": user_no}).sort("REG_DT", -1).limit(7)
        )

        relevant_diaries = []

        # 2. 질문 관련 과거 일기
        if user_input:
            # 검색 질의이므로 RETRIEVAL_QUERY로 임베딩 (저장된 문서는 RETRIEVAL_DOCUMENT로 임베딩되어 있음)
            query_vector = generate_embedding(user_input, task_type="RETRIEVAL_QUERY") # 사용자 텍스트 임베딩
            if query_vector:
                # [Resilient Pipeline] 필터 인덱스 미설정 시에도 작동하도록 2단계로 시도
                try:
                    # 1차 시도: 필터가 포함된 최적화된 검색 + 유사도 점수 추출
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
                        },
                        {
                            "$project": {
                                "CONTENT": 1,
                                "DATE": 1,
                                "REG_DT": 1,
                                "score": {"$meta": "vectorSearchScore"}
                            }
                        }
                    ]
                    raw_diaries = list(db["DIARY_LOGS"].aggregate(pipeline))
                    # 유사도가 0.6 이상인 의미 있는 데이터만 필터링 (쓰레기 데이터 차단)
                    relevant_diaries = [doc for doc in raw_diaries if doc.get('score', 0) >= 0.60]

                except Exception as ve:
                    print(f"[VECTOR ERROR] {ve}")
                    print("[HINT] 필터 인덱스 설정 전이므로 대체 검색을 수행합니다.")
                    # 2차 시도: 전체 검색 후 파이썬/매치 단계에서 필터링 + 유사도 점수 추출
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
                        {"$limit": 5},
                        {
                            "$project": {
                                "CONTENT": 1,
                                "DATE": 1,
                                "REG_DT": 1,
                                "score": {"$meta": "vectorSearchScore"}
                            }
                        }
                    ]
                    try:
                        raw_fallback = list(db["DIARY_LOGS"].aggregate(pipeline_fallback))
                        relevant_diaries = [doc for doc in raw_fallback if doc.get('score', 0) >= 0.60]
                    except:
                        relevant_diaries = []

        # 3. 중복 제거
        seen = set()
        combined = []

        for doc in recent_diaries + relevant_diaries:
            doc_id = str(doc.get("_id"))
            if doc_id not in seen:
                combined.append(doc)
                seen.add(doc_id)

        if not combined:
            return "최근 작성된 일기 기록이 없습니다."

        combined.sort(key=lambda x: x.get("REG_DT") or datetime.min, reverse=True)

        context = ""
        for d in combined: # 내용 추출 및 용약 날짜 형식 맞춤
            date_value = d.get("DATE") or d.get("date") or d.get("REG_DT")
            date_str = date_value.strftime("%Y년 %m월 %d일") if isinstance(date_value, datetime) else safe_text(date_value)
            content = safe_text(d.get("CONTENT"))[:800]

            # 기분(Emotion) 태그가 검열을 자극할 수 있으므로 제거하고 내용만 전달
            context += f"날짜: {date_str}\n내용: {content}\n\n"

        return context.strip()

    except Exception as e:
        print(f"[CONTEXT ERROR] {e}")
        return "일기 기록을 불러올 수 없습니다."


# 정책 / 기관 벡터 검색 (Tool)
# 반환값: (LLM 프롬프트용 텍스트 컨텍스트, 프론트 카드 렌더링용 구조화 데이터 리스트)
def execute_vector_search(query_text, collection_name):
    try:
        # 검색 질의이므로 RETRIEVAL_QUERY로 임베딩
        query_vector = generate_embedding(query_text, task_type="RETRIEVAL_QUERY") # 핵심 단어를 임베딩
        if not query_vector:
            return "관련 정보를 찾을 수 없습니다.", []

        pipeline = [ # 벡터 서치 요청 파이프 라인에 맞게 구조 생성
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

        results = list(db[collection_name].aggregate(pipeline)) # Vector Search로 받아오 데이터 변수에 저장
        if not results:
            return "검색된 결과가 없습니다.", []

        context = ""
        cards = []
        # 카드는 화면에 너무 많이 뜨지 않도록 상위 3개까지만 노출
        for doc in results[:3]:
            if collection_name == "PUBLIC_SVC": # LLM이 읽기 쉬운 형태로 가공
                context += f"정책명: {safe_text(doc.get('SVC_NM'))}, 상세내용: {safe_text(doc.get('SVC_DTL'))}, 지원대상: {safe_text(doc.get('TARGET'))}, 신청방법: {safe_text(doc.get('METHOD'))}\n"
                cards.append({
                    "type": "welfare",
                    "name": safe_text(doc.get('SVC_NM')),
                    "summary": safe_text(doc.get('SVC_DTL'))[:120],
                    "target": safe_text(doc.get('TARGET')),
                    "method": safe_text(doc.get('METHOD'))
                })
            else:
                context += f"기관구분: {safe_text(doc.get('CATEGORY'))}, 기관명: {safe_text(doc.get('NAME'))}, 주소: {safe_text(doc.get('ADDR'))}, 연락처/홈페이지: {safe_text(doc.get('HOMEPAGE'))}\n"
                cards.append({
                    "type": "hospital",
                    "name": safe_text(doc.get('NAME')),
                    "category": safe_text(doc.get('CATEGORY')),
                    "address": safe_text(doc.get('ADDR')),
                    "contact": safe_text(doc.get('HOMEPAGE'))
                })

        # 나머지(4~5번째) 결과는 카드 없이 텍스트 컨텍스트에만 반영
        for doc in results[3:]:
            if collection_name == "PUBLIC_SVC":
                context += f"정책명: {safe_text(doc.get('SVC_NM'))}, 상세내용: {safe_text(doc.get('SVC_DTL'))}, 지원대상: {safe_text(doc.get('TARGET'))}, 신청방법: {safe_text(doc.get('METHOD'))}\n"
            else:
                context += f"기관구분: {safe_text(doc.get('CATEGORY'))}, 기관명: {safe_text(doc.get('NAME'))}, 주소: {safe_text(doc.get('ADDR'))}, 연락처/홈페이지: {safe_text(doc.get('HOMEPAGE'))}\n"

        return context.strip(), cards # 앞뒤 공백 제거

    except Exception as e:
        print(f"[VECTOR SEARCH ERROR] {e}")
        return "데이터베이스 검색 중 오류가 발생했습니다.", []


# 마스터 시스템 프롬프트 (안전필터 우회 및 팩트 강화)
def build_system_prompt(diary_context, is_daily_talk=False):
    if is_daily_talk:
        return """
당신은 사용자와 편안하게 일상을 나누는 다정하고 따뜻한 챗봇 '마음'입니다.

[대화 원칙]
1. 사용자의 일상적인 질문(메뉴 추천, 날씨, 안부 등)에 상식과 공감 능력을 발휘하여 자연스럽고 친절하게 추천 및 대답해 주세요.
2. 말투는 "~해요", "~군요", "~어떨까요?" 처럼 친근하고 부드럽게 사용하세요.
3. 기계적인 답변을 피하고 친한 친구처럼 대화하세요.
4. 여러 개를 추천하거나 나열할 때는 한 문장에 몰아넣지 말고 줄바꿈(마크다운 목록 "- ")으로 항목을 나눠서 한눈에 보이게 해주세요.
"""
    else:
        return f"""
당신은 사용자의 고민을 공감하고 따뜻하게 대화하는 챗봇 '마음'입니다.

[참고 정보: 사용자의 과거 기록 및 검색된 정책/기관]
{diary_context if diary_context else "참고할 기록이 없습니다."}

[대화 원칙]
1. 사용자의 질문이 정책/기관 정보를 찾는 것이라면 검색된 정보를 바탕으로 바로 답하고, 사용자의 감정이나 최근 일상에 대한 질문이 아니라면 과거 일기 내용을 억지로 끌어와 언급하지 마세요.
2. 정보가 없다면 억지로 지어내지 말고, "해당 내용에 대해서는 찾을 수 없네요"라고 솔직하게 말하며 공감해 주세요.
3. 사용자가 자신의 감정/기분/최근 일상을 묻거나 이야기할 때만, 관련된 과거 일기 내용을 "기록을 보니 ~하셨군요"처럼 자연스럽게 언급해 주세요. 질문과 무관한 일기 내용을 먼저 꺼내지 마세요.
4. 말투는 "~해요", "~군요" 처럼 친근하게 사용하고, 전문적인 심리 상담이나 섣부른 진단은 절대 하지 마세요.
5. 절대로 <꺽쇠 괄호>를 사용하지 마세요.
6. 서로 다른 생각이나 화제로 넘어갈 때는 빈 줄로 문단을 나누고, 핵심 단어는 **굵게** 표시해서 한눈에 들어오게 해주세요. 여러 항목을 나열할 때는 "- "로 시작하는 목록을 쓰세요.
"""


# Tool 정의 (Gemini Function Calling 규격 — JSON Schema 타입은 대문자)
def create_tools():
    return [{
        "functionDeclarations": [
            {
                "name": "search_welfare",
                "description": "사용자가 월세, 생활비, 지원금, 복지 혜택 등을 찾을 때 정책을 검색합니다.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"query": {"type": "STRING"}},
                    "required": ["query"]
                }
            },
            {
                "name": "search_hospital",
                "description": "사용자가 우울증, 심리상담, 병원, 정신건강 센터 등의 정보를 찾을 때 기관을 검색합니다.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"query": {"type": "STRING"}},
                    "required": ["query"]
                }
            }
        ]
    }]


# Spring이 넘겨준 대화 기록(Redis 저장분)을 Gemini contents 형식으로 변환.
# role은 "user"/"bot"으로 오는데 Gemini는 "user"/"model"을 씀. <think>...</think>는
# 스트리밍 중 화면에 잠깐 보여주는 안내 문구라 실제 대화 내용이 아니므로 제거하고 넘김
def build_history_contents(history):
    if not history:
        return []

    contents = []
    for msg in history:
        role = "model" if msg.get("role") == "bot" else "user"
        text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', msg.get("content", "")).strip()
        if not text:
            continue
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents


# 메인 RAG & Gemini 연동 스트림
def generate_rag_response_stream(user_id, user_input, history=None):
    try:
        print(f"\n[INFO] RAG PROCESS START")
        print(f"[INFO] User Input: {user_input}")

        # 일상 대화인지 판별하여 불필요한 일기 로드를 막음 (안전필터 방지)
        daily_keywords = ["메뉴", "저녁", "점심", "아침", "날씨", "추천해", "안녕", "반가워"]
        is_daily_talk = any(keyword in user_input for keyword in daily_keywords)

        diary_context = ""

        if not is_daily_talk: # 일상 대화 판별 해서 일기를 로드할지 말지 정함
            diary_context = get_user_context(user_id, user_input)
            print(f"[INFO] Retrieved Diary Context:\n{diary_context}\n")
        else:
            print("[INFO] Daily talk detected. Skipping diary context retrieval.")

        # 분리된 시스템 프롬프트 적용 (Gemini는 system을 systemInstruction으로 별도 분리)
        system_prompt = build_system_prompt(diary_context, is_daily_talk)

        contents = build_history_contents(history) + [
            {"role": "user", "parts": [{"text": user_input}]}
        ]

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {_get_access_token()}"
        }

        # 기본 페이로드 구성 (Tool 일단 제외)
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {
                "topP": 0.8,
                "temperature": 0.7 if is_daily_talk else 0.4,  # 일상 대화일 때는 창의성을 살짝 높임
                "maxOutputTokens": 1024,
                # gemini-2.5-flash는 기본적으로 답변 전에 내부적으로 "생각(thinking)" 토큰을 쓰는데,
                # 이 토큰이 maxOutputTokens 예산을 같이 잡아먹어서 정작 답변이 잘리는 문제가 있었음.
                # 실시간 채팅은 깊은 추론이 필요 없으므로 thinking을 꺼서 답변 토큰을 온전히 확보함
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }

        # 검색/지원이 필요한 상황에서만 Tool을 주입
        if not is_daily_talk: # Fuction Calling 으로 AI가 자동으로 적절한 도구를 호출
            payload["tools"] = create_tools()
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        print("[INFO] Requesting 1st Gemini API (Tool or Direct Answer)...")
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            print(f"[ERROR] API Code: {response.status_code}, Msg: {response.text}")
            yield from stream_text("서버가 잠시 피곤한가 봐요. 조금만 이따가 다시 이야기해요.")
            return

        result_json = response.json() # 여기서 생성된 답변을 딕셔너리 타입으로 변경
        print(f"[INFO] 1st Gemini Response:\n{json.dumps(result_json, indent=2, ensure_ascii=False)}\n")

        candidates = result_json.get("candidates", [])
        content_obj = candidates[0].get("content", {}) if candidates else {}
        parts = content_obj.get("parts", [])

        # Gemini는 텍스트 파트와 함수 호출 파트가 같은 parts 배열에 섞여서 옴
        full_content = safe_text("".join(p.get("text", "") for p in parts if "text" in p))
        function_calls = [p["functionCall"] for p in parts if "functionCall" in p]

        # Tool Call 처리
        if function_calls:
            print(f"[INFO] Tool Call Detected: {len(function_calls)} tools")
            yield "<think>당신에게 도움이 될 만한 정보를 열심히 찾아보고 있어요...</think>\n"

            contents.append(content_obj)

            function_response_parts = []
            all_cards = [] # 프론트에 카드로 보여줄 구조화된 검색 결과 (정책/기관 상세는 여기서 전달, 답변 텍스트는 짧게)
            for call in function_calls:
                try:
                    tool_name = call.get("name") # AI가 분석한 함수명을 가져옴
                    args = call.get("args", {}) # Gemini는 args를 이미 dict로 줌
                    tool_query = safe_text(args.get("query")) # args에서 query 부분 출력, safe_text로 공백등을 제거, 핵심 단어

                    print(f"[INFO] Executing {tool_name} with query: {tool_query}")
                    collection_name = "MENTAL_INST" if tool_name == "search_hospital" else "PUBLIC_SVC" # 호출한 도구 명과 컬렉션 명을 매핑
                    search_result, cards = execute_vector_search(tool_query, collection_name) # Vector Search 수행
                    all_cards.extend(cards)
                    print(f"[INFO] Tool Result Length: {len(search_result)}, Cards: {len(cards)}")

                    # Gemini API 규격에 맞게 Tool 결과를 functionResponse 파트로 구성
                    function_response_parts.append({
                        "functionResponse": {
                            "name": tool_name,
                            "response": {"content": search_result if search_result else "관련 정보가 없습니다."}
                        }
                    })
                except Exception as e:
                    print(f"[ERROR] TOOL ERROR: {e}")

            contents.append({"role": "user", "parts": function_response_parts})

            # 카드로 상세 정보를 이미 보여주므로, 답변 텍스트는 짧은 안내 멘트 정도로만 작성하도록 지시
            card_aware_prompt = system_prompt + "\n\n[안내]\n방금 찾은 정책/기관의 상세 정보(이름, 대상, 연락처 등)는 화면에 카드로 따로 표시됩니다. 답변에서는 상세 항목을 나열하지 말고, \"이런 것들을 찾았어요\"처럼 1~2문장으로 짧게 안내만 해주세요."

            second_payload = {
                "systemInstruction": {"parts": [{"text": card_aware_prompt}]},
                "contents": contents,
                "generationConfig": {
                    "topP": 0.8,
                    "temperature": 0.4,
                    "maxOutputTokens": 1024,
                    "thinkingConfig": {"thinkingBudget": 0}
                }
            }

            print("[INFO] Requesting 2nd Gemini API (Final Answer)...")
            second_res = requests.post(GEMINI_API_URL, headers=headers, json=second_payload, timeout=REQUEST_TIMEOUT)

            if second_res.status_code == 200:
                second_json = second_res.json() # 텍스트 가공
                print(f"[INFO] 2nd Gemini Response:\n{json.dumps(second_json, indent=2, ensure_ascii=False)}\n")
                second_candidates = second_json.get("candidates", [])
                second_parts = second_candidates[0].get("content", {}).get("parts", []) if second_candidates else []
                final_content = safe_text("".join(p.get("text", "") for p in second_parts if "text" in p))
            else:
                print(f"[ERROR] 2nd API Failed: {second_res.status_code}, {second_res.text}")
                final_content = ""

            final_content = clean_ai_text(final_content) # 불필요한 기호 제거

            # 억지스러운 하드코딩 제거, 자연스러운 에러 핸들링
            if not final_content:
                final_content = "원하시는 정보를 찾는 데 잠시 오류가 있었어요. 다시 한 번 물어봐 주시겠어요?"

            # 카드가 있으면 답변 텍스트보다 먼저 전송해서, 화면에 카드가 먼저 나타나게 함
            if all_cards:
                yield f"[[CARD]]{json.dumps(all_cards, ensure_ascii=False)}[[/CARD]]\n"

            yield from stream_text_with_audio(final_content)

        # ================== 일반 대화 처리 ==================
        else:
            print("[INFO] No Tool Call. Direct Answer.")

            full_content = clean_ai_text(full_content) # 불필요한 기호 제거

            # 자연스러운 폴백
            if not full_content:
                print("[WARN] 1st Answer is empty (Safety Filter Hit). Smart Fallback triggered.")
                full_content = "제가 잠시 딴생각을 하느라 말씀을 놓쳤네요. 방금 하신 말씀 다시 한 번 들려주시겠어요? 아니면 마음이 무거우실 때 언제든 편하게 털어놓아 주세요."

            yield from stream_text_with_audio(full_content)

        print("[INFO] RAG PROCESS END\n")

    except Exception as e:
        print(f"[ERROR] RAG CRITICAL ERROR: {e}")
        yield from stream_text("앗, 잠시 제 생각이 엉켰어요. 방금 하신 말씀 다시 한 번 들려주실래요?")