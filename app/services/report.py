import re

import requests

from app.services.summary import GEMINI_API_URL, _get_access_token


def generate_weekly_report(diary_entries: list[dict]) -> str:
    """
    최근 일주일 일기 목록(제목/요약/주요감정)을 받아 한 주를 돌아보는 짧은 코멘트를 Gemini로 생성함.
    music.py와 동일하게 실패 시 예외를 올리지 않고 안내 문구로 대체하여
    리포트 생성 실패가 마이페이지 조회 자체를 막지 않도록 함
    """
    if not diary_entries:
        return "최근 일주일 동안 작성된 일기가 없어요. 오늘 하루를 기록해보는 건 어떨까요?"

    entries_str = "\n".join(
        f"- {e.get('title', '')}: {e.get('summary', '')} (주요 감정: {e.get('main_emotion', '')})"
        for e in diary_entries
    )

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {_get_access_token()}'
    }

    payload = {
        "systemInstruction": {
            "parts": [{
                "text": (
                    "당신은 따뜻한 심리 상담가입니다. 사용자의 최근 일주일 일기 요약들을 보고, "
                    "한 주간의 감정 흐름을 짚어주는 짧은 격려 코멘트를 3~4문장으로 작성하세요.\n\n"
                    "[제한 사항]\n"
                    "1. '**', '*', '#' 등 마크다운 기호와 이모지, 특수기호는 절대 사용하지 마세요.\n"
                    "2. '주간 리포트:', '이번 주 요약:' 등 어떠한 소제목이나 라벨도 달지 마세요.\n"
                    "3. 과도하게 단정적인 진단 표현은 피하고, 따뜻하고 담백한 어조를 유지하세요."
                )
            }]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": f"최근 일주일 일기 목록:\n{entries_str}\n\n위 내용을 바탕으로 한 주를 돌아보는 짧은 코멘트를 작성해주세요."
                }]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 1024,
            "temperature": 0.6,
            "topP": 0.8,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_data = response.json()

        candidates = res_data.get('candidates', [])
        parts = candidates[0].get('content', {}).get('parts', []) if candidates else []
        comment = parts[0].get('text', '') if parts else ''

        if comment:
            comment = comment.replace('**', '').replace('*', '').replace('"', '')
            comment = re.sub(r'\n{3,}', '\n\n', comment).strip()
            return comment
        else:
            return "이번 한 주도 스스로를 잘 돌보고 계시네요. 다음 주도 응원할게요."

    except Exception as e:
        print(f"[REPORT ERROR] 주간 리포트 생성 실패: {e}")
        return "이번 한 주도 스스로를 잘 돌보고 계시네요. 다음 주도 응원할게요."
