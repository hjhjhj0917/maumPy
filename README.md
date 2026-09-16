# MAUM (마음) — AI & Data API Server

사용자의 일기 텍스트를 분석해 감정과 우울증 수치를 도출하고, RAG(검색 증강 생성) 기반으로 맞춤형 챗봇 상담·정책/기관 추천·음악 추천까지 제공하는 "MAUM" 서비스의 **Python AI/Data 서버** 리포지토리입니다.

MAUM은 3개 저장소로 구성됩니다.

| 저장소 | 역할 |
|---|---|
| [maumProject](https://github.com/hjhjhj0917/example) | Spring Boot 백엔드 — 인증, 일기/채팅 CRUD, 이 서버로의 프록시 |
| **maumPy (현재 저장소)** | FastAPI AI 서버 — 감정/우울증 분석, RAG 챗봇, 임베딩, STT/TTS, 음악 추천 |
| [maumReact](https://github.com/hjhjhj0917/maumReact) | React 프론트엔드 |

React → Spring → (이 서버)로만 요청이 흐르는 구조이며, 이 서버는 외부에 직접 노출되지 않습니다.

* **개발 기간**: 2026.03 ~ 2026.10
* **개발 인원**: 1인 (개인 프로젝트)

---

## Tech Stack

### Language & Framework
- **Language**: Python 3.10
- **Framework**: FastAPI, Uvicorn
- **AI/ML**: PyTorch, Hugging Face Transformers

### AI Models
- **LLM (RAG·요약·음악 큐레이션)**: Google **Gemini** (`gemini-2.5-flash`, Vertex AI REST API 직접 호출)
- **임베딩**: Google **`gemini-embedding-001`** (Vertex AI)
- **감정 분석**: KoELECTRA (KOTE 데이터셋 파인튜닝)
- **우울증 분석**: `klue/roberta-base` (AI Hub 심리상담 데이터 파인튜닝)

### External APIs
- **음성**: Google Cloud Speech-to-Text / Text-to-Speech
- **음악 추천**: Spotify Web API (Search)
- **위치**: Kakao 주소 검색 API
- **공공데이터**: 정신건강기관·청년정책 등 공공데이터포털 API

### Database
- **Vector DB**: MongoDB Atlas Vector Search

---

## Key Features

### 1. 일기 통합 분석 파이프라인 (`analyze.py`)
일기 저장 시 아래 과정을 **한 번의 API 호출**로 순차 처리하고 결과를 Spring에 반환합니다.
1. **우울증 이진 분류** — `klue/roberta-base` 파인튜닝 모델로 우울 징후 여부·점수 산출 (`prediction.py`)
2. **감정 분석** — KoELECTRA(KOTE)로 40여 개 세부 감정 확률 산출 (`emotion.py`)
3. **AI 요약** — Gemini로 일기 내용 + 감정/우울 분석 결과를 자연스러운 위로 문장으로 요약 (`summary.py`)
4. **임베딩** — 제목+내용+요약을 `gemini-embedding-001`로 벡터화해 MongoDB에 저장 (`embedding.py`)
5. **감정 기반 음악 추천** — 뚜렷한(0.8 이상) 감정 + 일기 원문을 Gemini에게 주어 "지금 상황에 어울리는" 음악 분위기를 판단시키고, Spotify Search API로 실제 곡 5곡을 추천 (`music.py`)

### 2. RAG 기반 챗봇 (`rag.py`, `chat.py`)
- 정신건강기관/청년정책 등 공공데이터를 임베딩해 MongoDB Atlas Vector Search로 유사도 검색
- 검색된 컨텍스트 + 최근 대화 이력(멀티턴)을 Gemini에 함께 전달해 스트리밍(SSE) 응답 생성
- 상담기관/정책 카드, TTS 음성 등 여러 형태의 응답을 한 스트림에 함께 실어 보냄

### 3. 음성 인터페이스 (`stt.py`, `tts.py`)
- **STT**: 브라우저에서 녹음한 오디오(webm/opus)를 Google Cloud Speech-to-Text로 텍스트 변환
- **TTS**: 챗봇 답변을 Google Cloud Text-to-Speech(Chirp3-HD 음성)로 합성해 문장 단위로 스트리밍

### 4. 공공 데이터 수집 파이프라인 (`scripts/`)
- 정신건강기관·청년정책 등 공공 API 데이터 수집 (`fetch_mental_inst.py`, `fetch_public_svc.py`)
- 주소 → 좌표 변환 등 데이터 정제 (`migrate_addresses.py`)
- 우울증 분류 모델 학습 스크립트 (`kluebert_train.py`) — 세션/문서 단위 청크 분할 + 환자 단위 데이터 분리로 데이터 누수 방지

---

## Project Structure

```text
.
├── app/
│    ├── api/                  # API 라우터
│    │    ├── analyze.py       # 일기 통합 분석(감정/우울/요약/임베딩/음악추천) 엔드포인트
│    │    ├── batch.py         # 공공데이터 수집/마이그레이션 배치 트리거
│    │    ├── chat.py          # RAG 챗봇 스트리밍(SSE) 엔드포인트
│    │    └── stt.py           # 음성 인식(STT) 엔드포인트
│    ├── core/
│    │    ├── config.py        # 환경설정 로드
│    │    └── database.py      # MongoDB 연결/컬렉션 정의
│    ├── services/             # AI 모델 추론 및 비즈니스 로직
│    │    ├── embedding.py     # Gemini 임베딩(gemini-embedding-001) 생성
│    │    ├── emotion.py       # KoELECTRA 감정 분석
│    │    ├── music.py         # 감정 기반 Spotify 음악 추천
│    │    ├── prediction.py    # 우울증 예측 모델 추론
│    │    ├── rag.py           # Gemini 기반 RAG 검색·응답 생성
│    │    ├── stt.py           # Google Cloud STT 연동
│    │    ├── summary.py       # Gemini 기반 일기 요약
│    │    └── tts.py           # Google Cloud TTS 연동
│    └── main.py               # 애플리케이션 진입점 및 라우터 등록
├── models/
│    └── trained_model_depression_binary/  # AI Hub 데이터로 파인튜닝된 우울증 분석 모델
└── scripts/                   # 데이터 수집·전처리·모델 학습용 스크립트
     ├── data_extractor.py     # 학습 데이터 추출 및 전처리
     ├── fetch_mental_inst.py  # 정신건강 상담기관 데이터 수집
     ├── fetch_public_svc.py   # 공공 서비스(청년정책 등) 데이터 수집
     ├── kluebert_train.py     # klue/roberta-base 우울증 모델 학습 파이프라인
     ├── migrate_addresses.py  # 주소 → 좌표 마이그레이션
     └── reembed_all.py        # 임베딩 일괄 재생성 스크립트
```

---

## Getting Started

### 요구 사항
- Python 3.10
- Google Cloud 프로젝트 (Vertex AI, Speech-to-Text, Text-to-Speech, Storage API 활성화)
- MongoDB Atlas (Vector Search 인덱스 구성)
- Spotify Developer 앱 (Client Credentials)

### 설치
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install fastapi uvicorn[standard] python-multipart python-dotenv \
    torch transformers pymongo requests \
    google-auth google-cloud-speech google-cloud-texttospeech
```
> `requirements.txt`가 아직 없어 위 목록은 실제 임포트 기준 근사치입니다. 새 환경에서 실행 전 `import` 에러가 나면 해당 패키지를 추가로 설치해주세요.

### 환경변수
GCP(Vertex AI/STT/TTS), MongoDB, Spotify, Kakao, 공공데이터포털 등 외부 서비스 인증 정보가 담긴 `.env` 파일이 필요합니다. 값은 별도로 안전하게 전달받아 프로젝트 루트에 구성해주세요.

### 실행
```bash
uvicorn app.main:app --reload --port 8000
```
기본적으로 `http://localhost:8080`(Spring)에서만 호출되는 내부 서버이므로, 단독 실행 시에도 Spring 백엔드가 함께 떠 있어야 전체 플로우를 테스트할 수 있습니다.
