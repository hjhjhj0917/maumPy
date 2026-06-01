import os
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
                    "다음 5가지 원칙을 엄격하게 지켜서 사용자의 심리 상태를 분석하세요.\n"
                    "1. [이모지 금지] 답변 텍스트에 어떠한 이모지나 이모티콘도 절대 사용하지 마세요.\n"
                    "2. [전체 요약] 사용자가 작성한 일기의 전체적인 흐름과 핵심 사건을 먼저 간략하게 요약하세요.\n"
                    "3. [감정의 근거] 제시된 '주요 감정'들이 일기의 어느 구절이나 상황에서 비롯되었는지 구체적으로 연결하여 분석해주세요. (예: '~~라고 하신 부분에서 안도감과 기쁨이 깊게 느껴집니다.')\n"
                    "4. [우울 징후 분석] 만약 '우울 증상 단계'가 '우울 증상 의심'으로 나왔다면, 일기의 어떤 내용이나 표현(단어, 문맥 등)에서 우울감이나 위험 신호가 감지되었는지 조심스럽고 객관적으로 짚어주세요. (정상이라면 이 부분은 긍정적인 지지로 대체하세요.)\n"
                    "5. [어조] 분석은 구체적이고 논리적이어야 하지만, 말투는 사용자가 상처받지 않도록 다정하고 포용적인 존댓말을 유지하세요."
                )
            },
            {
                "role": "user",
                "content": f"[일기 원문]\n{content}\n\n[주요 감정]\n{emotions_str}\n\n[우울 증상 단계]\n{dep_status}\n\n위 정보를 바탕으로 심도 있는 심리 분석 요약을 작성해주세요."
            }
        ],
        "thinking": {
            "effort": "low"
        },
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
            return summary_text
        else:
            return "일기 내용과 감정을 종합적으로 분석하고 있습니다. 오늘은 스스로를 다독여주는 시간을 가져보는 건 어떨까요?"

    except Exception:
        return "일기 내용과 감정을 종합적으로 분석하고 있습니다. 오늘은 스스로를 다독여주는 시간을 가져보는 건 어떨까요?"