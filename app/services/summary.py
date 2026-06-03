import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

HCX_API_URL = os.getenv("HCX_API_URL")
HCX_API_KEY = os.getenv("HCX_API_KEY")


def generate_hcx_summary(content: str, dep_level: int, raw_emotions: dict) -> str:
    top_emotions = sorted(raw_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    emotions_str = ", ".join([f"{emo}({round(prob * 100, 1)}%)" for emo, prob in top_emotions])

    dep_str_map = {
        0: "정상 범위 내의 정서 상태",
        1: "우울 증상 의심 및 전문가 상담 권장"
    }
    dep_status = dep_str_map.get(dep_level, "알 수 없음")

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {HCX_API_KEY}'
    }

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
        "maxCompletionTokens": 5120,
        "temperature": 0.5,
        "topP": 0.8
    }

    try:
        response = requests.post(HCX_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        res_data = response.json()

        summary_text = res_data.get('result', {}).get('message', {}).get('content', '')

        if summary_text:
            # 1. 마크다운 및 불필요한 따옴표 제거
            summary_text = summary_text.replace('**', '').replace('*', '').replace('"', '')

            # 2. 혹시라도 생성된 소제목 강제 제거
            summary_text = re.sub(r'^(일기 내용 요약|심리 분석과 위로|심리 분석 요약|심리적 평가와 위로|요약|심리 분석)\s*:\s*', '', summary_text,
                                  flags=re.MULTILINE)

            # 3. 과도한 줄바꿈(3번 이상의 \n)을 딱 2번(빈 줄 하나)으로 축소하여 가독성 개선
            summary_text = re.sub(r'\n{3,}', '\n\n', summary_text)

            # 4. 앞뒤 공백 제거
            summary_text = summary_text.strip()

            return summary_text
        else:
            return "일기 내용과 감정을 종합적으로 분석하고 있습니다.\n\n오늘은 스스로를 다독여주는 시간을 가져보는 건 어떨까요?"

    except Exception:
        return "일기 내용과 감정을 종합적으로 분석하고 있습니다.\n\n오늘은 스스로를 다독여주는 시간을 가져보는 건 어떨까요?"