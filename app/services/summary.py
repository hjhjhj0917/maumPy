import os
import requests
from dotenv import load_dotenv

load_dotenv()

HCX_API_URL = os.getenv("HCX_API_URL")
HCX_API_KEY = os.getenv("HCX_API_KEY")


def generate_hcx_summary(content: str, dep_level: int, raw_emotions: dict) -> str:
    top_emotions = sorted(raw_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    emotions_str = ", ".join([f"{emo}({round(prob * 100, 1)}%)" for emo, prob in top_emotions])

    dep_str_map = {0: "정상", 1: "경미한 우울", 2: "중간 수준의 우울", 3: "심각한 우울"}
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
                "content": "당신은 따뜻하고 공감 능력이 뛰어난 심리 상담가입니다. 사용자의 일기 내용, 감정 분석 결과, 우울증 모델 분석 결과를 종합하여 사용자의 현재 심리 상태를 3~4문장으로 다정하게 요약하고 짧은 위로/조언을 건네주세요. 전문 용어보다는 일상적이고 부드러운 말투를 사용하세요."
            },
            {
                "role": "user",
                "content": f"[일기 원문]\n{content}\n\n[주요 감정]\n{emotions_str}\n\n[우울 증상 단계]\n{dep_status}\n\n위 정보를 바탕으로 심리 상태를 요약해주세요."
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