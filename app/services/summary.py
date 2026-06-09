import os
import re
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

HCX_API_URL = os.getenv("HCX_API_URL")
HCX_API_KEY = os.getenv("HCX_API_KEY")


def generate_hcx_summary(content: str, dep_level: int, raw_emotions: dict) -> str:

    # raw_emotions.items() 딕셔너리 형태의 데이터를 튜플 형태로 변경후, key는 이 수치 데이터만 비교하게 x[1]로 설정
    # sorted(, reverse=True) 를 사용해서 그 값을 역순 정력 가장 큰 값이 가장 먼저 오게
    # [:3] 통해 상위 3가지 감정만 슬라이싱
    top_emotions = sorted(raw_emotions.items(), key=lambda x: x[1], reverse=True)[:3]

    # 상위 3가지 감정을 LLM이 일기 편한 문장 형식으로 변화하는 과정
    # 상위 3가지 감정 튜플을 반복문으로 꺼내와서 감정(85%%) 형태로 변경후 각 감정은 , 로 연결
    emotions_str = ", ".join([f"{emo}({round(prob * 100, 1)}%)" for emo, prob in top_emotions])

    # 분류한 감정 결과를 다시 문장 형식으로 변경
    dep_str_map = {
        0: "정상 범위 내의 정서 상태",
        1: "우울 증상 의심 및 전문가 상담 권장"
    }
    dep_status = dep_str_map.get(dep_level, "알 수 없음")

    # 요청 헤더를 HCX 모델에 맞게 지정
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {HCX_API_KEY}',
        'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4())
    }

    # 요청 바디 부분을 역할을 분리해서 system이면 content에 모델의 역할을 지정, user면 content에 사용자 질문을 작성
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 텍스트의 맥락을 깊이 있게 파악하는 전문적이고 따뜻한 심리 상담가입니다. "
                    "반드시 아래의 [출력 형식]과 [제한 사항]에 맞춰 평문(Plain Text)으로만 답변하세요.\n\n"
                    "[출력 형식]\n"
                    "첫 번째 문단: 일기의 전체적인 흐름과 핵심 사건을 한눈에 파악할 수 있도록 간략하게 요약합니다.\n"
                    "두 번째 문단: 위 요약된 내용과 분석된 주요 감정, 우울증 정도를 자연스럽게 연결하여 심리 상태를 분석하고 따뜻한 위로를 건넵니다.\n\n"
                    "[제한 사항]\n"
                    "1. 두 문단 사이의 연결이 부자연스럽지 않도록, 두 번째 문단의 시작을 딱딱한 결과 보고가 아닌 이야기하듯 부드럽게 이어가세요.\n"
                    "2. 두 문단 사이에는 단 한 번의 줄바꿈(빈 줄 하나)만 허용합니다.\n"
                    "3. '**', '*', '#' 등 마크다운 기호와 이모지, 특수기호는 절대 사용하지 마세요.\n"
                    "4. '일기 내용 요약:', '심리 분석:' 등 어떠한 소제목이나 라벨도 달지 마세요."
                )
            },
            {
                "role": "user",
                "content": f"일기 원문:\n{content}\n\n주요 감정:\n{emotions_str}\n\n우울 증상 단계:\n{dep_status}\n\n위 정보를 바탕으로 부드럽고 자연스럽게 이어지는 심리 분석 요약을 작성해주세요."
            }
        ],
        "maxCompletionTokens": 5120, # 모델 답변 최대길이 지정
        "temperature": 0.5, # 답변의 창의성 설정 0에 가까울 수록 일관되고 보수적, 1에 가까울 수록 변동성 심하고 창의적
        "topP": 0.8 # 생성할 단어 후보군의 상위 누적 합계를 지정 (높을수록 관련성 있는 응답 생성)
    }

    try:
        # 요청 URL과 인증 정보가 든 헤더와 페이로드를 추가해 요청함
        response = requests.post(HCX_API_URL, headers=headers, json=payload)

        # 응답 받은 결과에서 상태코드를 확인해서 에러가 났다면 에러를 발생함
        response.raise_for_status()

        # 응답 받은 JSON 형태의 텍스트를 .json을 통해서 구조에 따라 딕셔너리 형태로 바뀌어 변수에 저장함
        res_data = response.json()

        # 응답 구조에 맞게 result에 message에 content로 접근해서 요약 텍스트를 불러옴
        summary_text = res_data.get('result', {}).get('message', {}).get('content', '')

        if summary_text:
            # 마크다운 및 불필요한 따옴표 제거
            summary_text = summary_text.replace('**', '').replace('*', '').replace('"', '')

            # 혹시라도 생성된 소제목 강제 제거 (re 라이브러리를 사용해 re.sub은 정규식을 이용해 텍스트 치환)
            summary_text = re.sub(r'^(일기 내용 요약|심리 분석과 위로|심리 분석 요약|심리적 평가와 위로|요약|심리 분석)\s*:\s*', '', summary_text,
                                  flags=re.MULTILINE)

            # 과도한 줄바꿈(3번 이상의 \n)을 딱 2번(빈 줄 하나)으로 축소하여 가독성 개선
            summary_text = re.sub(r'\n{3,}', '\n\n', summary_text)

            # 앞뒤 공백 제거
            summary_text = summary_text.strip()

            return summary_text
        else:
            return "일기 내용과 감정을 종합적으로 분석하고 있습니다.\n\n오늘은 스스로를 다독여주는 시간을 가져보는 건 어떨까요?"

    except Exception:
        return "일기 내용과 감정을 종합적으로 분석하고 있습니다.\n\n오늘은 스스로를 다독여주는 시간을 가져보는 건 어떨까요?"