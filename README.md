# MAUM AI Server (maumPy)

일기 텍스트 한 편으로 **우울 징후 판별 → 감정 분석 → AI 요약 → 임베딩 → 음악 추천**까지 한 번에 처리하고, 공공데이터 기반 RAG 챗봇·음성 인식/합성·주간 리포트를 제공하는 MAUM 서비스의 FastAPI 서버입니다.

핵심 설계 원칙은 "**이 서버는 혼자 죽지 않는다**"입니다. 어떤 AI 호출(Gemini, 분류 모델, 외부 API)이 실패해도 예외를 그대로 터뜨리지 않고 대체 응답(기본값·안내 문구)으로 내려보내, 하나의 기능 장애가 일기 작성이나 마이페이지 조회 같은 핵심 흐름 전체를 막지 않도록 합니다.

* **대상 환경**: React → Spring Boot(maumProject) → 이 서버로만 요청이 들어오는 내부 API 서버 (외부에 직접 노출되지 않음)
* **실행 환경**: Python 3.10 + FastAPI/Uvicorn, 로컬 GPU/CPU 추론 모델 2종 + Google Vertex AI(Gemini) 혼합 구성

---

## 목차

1. [주요 기능](#주요-기능)
2. [핵심 설계 / 안전장치](#핵심-설계--안전장치)
3. [핵심 모델 · 규칙 상세](#핵심-모델--규칙-상세)
4. [시작하기](#시작하기)
5. [외부 서비스 준비 사항](#외부-서비스-준비-사항)
6. [프로젝트 구조](#프로젝트-구조)
7. [기술 스택](#기술-스택)
8. [버전 기록](#버전-기록-주요-변경-이력)

---

## 주요 기능

### 1. 일기 통합 분석 — `POST /api/analyze`
일기 저장 시점에 분석 파이프라인 전체를 한 번의 호출로 순차 처리합니다.
- 우울 징후 이진 분류 (`klue/roberta-base` 파인튜닝 모델)
- KOTE 기반 44개 세부 감정 확률 산출 + 9개 감정 그룹 매핑
- Gemini로 본문 + 분석 결과를 위로형 요약 문장으로 변환
- 제목+본문+요약을 `gemini-embedding-001`로 벡터화해 MongoDB에 저장
- 뚜렷한 감정(확률 0.8 이상)이 있으면 Gemini에게 "지금 분위기"를 묻고 Spotify Search API로 실제 곡 5곡 추천

### 2. RAG 기반 챗봇 — `POST /api/rag-chat`
- 사용자 메시지를 임베딩해 MongoDB Atlas Vector Search로 정신건강기관·청년정책 데이터 중 유사 문서 검색
- 검색된 컨텍스트 + 최근 대화 이력(멀티턴, role은 Spring 쪽 "user"/"bot" → Gemini "user"/"model"로 변환)을 Gemini에 전달
- `StreamingResponse`로 답변 텍스트·상담기관/정책 카드·TTS 음성을 하나의 스트림에 함께 실어 보냄 (SSE 유사 방식)

### 3. 음성 인터페이스 — `POST /api/stt`, `POST /api/tts`
- **STT**: 브라우저에서 녹음한 오디오를 Google Cloud Speech-to-Text로 텍스트 변환
- **TTS**: 텍스트에서 이모지/마크다운을 제거한 뒤 문장 단위 청크로 쪼개 Google Cloud TTS(Chirp3-HD)로 합성, Base64 오디오 배열로 반환
  - 챗봇 스트리밍 응답은 생성과 동시에 합성되지만, 과거 대화 내역은 오디오를 다시 저장해두지 않으므로 재진입 시 텍스트만으로 이 엔드포인트를 다시 호출해 음성을 재생성함

### 4. 주간 리포트 코멘트 — `POST /api/weekly-report`
- Spring이 최근 일주일 치 일기의 제목/요약/주요 감정을 모아 전달하면, Gemini가 한 주를 돌아보는 짧은 격려 코멘트를 생성
- 실패 시에도 예외 대신 안내 문구로 대체해 마이페이지 조회 자체는 막히지 않도록 함

### 5. 공공데이터 배치 업데이트 — `POST /batch/update-data`
- 정신건강 상담기관·청년정책 등 공공데이터포털 데이터를 재수집하고 주소→좌표 변환까지 수행하는 무거운 작업을 `BackgroundTasks`로 비동기 처리, 호출 즉시 202 응답

---

## 핵심 설계 / 안전장치

- **우울 분류 2단계 캐스케이드**: 먼저 범용 감성분석 모델로 전체 청크가 뚜렷하게 긍정인지 빠르게 걸러내고, 애매하거나 부정적인 경우에만 파인튜닝된 우울 분류 모델을 추가로 돌려 추론 비용을 줄임
- **청크 분할 + 겹치기(sliding window)**: 토크나이저 512토큰 제한으로 긴 일기가 잘려 문맥이 손실되던 문제를 해결하기 위해, 3문장 단위로 2문장씩 겹치게 분할해 각 청크를 독립 추론 후 확률을 평균
- **환자(세션) 단위 데이터 분리**: `kluebert_train.py` 학습 파이프라인에서 같은 사람의 청크가 train/validation에 동시에 섞여 데이터 누수가 생기던 문제를 막기 위해, 청크를 만들기 전에 사람 단위로 먼저 분리
- **TTS 문장 단위 스트리밍 청크**: 긴 답변을 한 번에 합성하지 않고 문장 단위로 끊어 순차 합성 → 재생 시작까지의 지연을 줄임
- **감정 기반 음악 추천의 2단계 판단**: 세부 감정 확률을 그대로 Spotify 쿼리로 쓰지 않고, Gemini에게 일기 원문과 함께 "이 상황에 어울리는 분위기"를 자연어로 먼저 판단시킨 뒤 그 결과로 검색 — 한국 곡 편향·중복 결과를 후처리로 제거
- **실패 격리(fail-soft)**: AI 요약, 음악 추천, 주간 리포트 등 보조 기능은 실패해도 핵심 분석 결과 저장/응답을 막지 않도록 각 서비스 함수가 예외를 흡수하고 대체 값을 반환

---

## 핵심 모델 · 규칙 상세

| 구분 | 모델 / 서비스 | 용도 |
|---|---|---|
| 우울 징후 1차 필터 | `nlptown/bert-base-multilingual-uncased-sentiment` | 전체 청크가 뚜렷한 긍정인지 빠르게 판별, 긍정이면 2차 추론 생략 |
| 우울 징후 2차 분류 | `klue/roberta-base` 파인튜닝 (`models/trained_model_depression_binary`) | 청크별 우울 증상 확률 산출 → 평균 확률 0.50 기준 이진 판정 |
| 감정 분석 | `searle-j/kote_for_easygoing_people` (KOTE, KoELECTRA 기반) | 문장을 44개 세부 감정 라벨로 다중 분류 |
| LLM (요약/챗봇/리포트/음악 분위기 판단) | Google Gemini `gemini-2.5-flash` (Vertex AI REST 직접 호출) | 일기 요약, RAG 응답 생성, 주간 리포트 코멘트, 음악 분위기 추론 |
| 임베딩 | Google `gemini-embedding-001` (Vertex AI) | 일기·공공데이터 문서 벡터화 → MongoDB Atlas Vector Search |
| 음성 인식/합성 | Google Cloud Speech-to-Text / Text-to-Speech(Chirp3-HD) | STT 변환, 문장 단위 TTS 스트리밍 합성 |
| 음악 검색 | Spotify Web API (Search) | Gemini가 판단한 분위기 키워드로 실제 곡 5곡 검색 |

**44개 감정 라벨(KOTE)은 9개 그룹**(기쁨/신뢰/공포/놀람/슬픔/혐오/분노/기대/무감정)으로 매핑되어 각 그룹의 대표 색상과 함께 프론트에 전달됩니다.

**우울 분류 청크 분할 규칙**: 문장 단위로 쪼갠 뒤 `window=3, step=2`(3문장씩, 2문장씩 겹치며 이동)로 청크를 생성 — 문장 수가 적은 일기는 전체가 한 청크로 처리됩니다.

---

## 시작하기

### 요구 사항
- Python 3.10
- Google Cloud 프로젝트 (Vertex AI, Speech-to-Text, Text-to-Speech API 활성화)
- MongoDB Atlas (Vector Search 인덱스 구성)
- Spotify Developer 앱 (Client Credentials)
- GPU는 선택 사항 (없으면 CPU로 동작, `torch.cuda.is_available()`로 자동 분기)

### 설치
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install fastapi "uvicorn[standard]" python-multipart python-dotenv \
    torch transformers pymongo requests \
    google-auth google-cloud-speech google-cloud-texttospeech
```
> 저장소에 `requirements.txt`가 없어 위 목록은 실제 `import` 기준 근사치입니다. 새 환경에서 실행 시 `ImportError`가 나면 해당 패키지를 추가로 설치해주세요.

### 실행
```bash
uvicorn app.main:app --reload --port 8000
```
서버 기동 시 `startup` 이벤트에서 우울 분류/감정 분석 모델을 미리 한 번 돌려 메모리에 적재합니다. 이 서버는 Spring 백엔드(`maumProject`)를 통해서만 호출되는 내부 API이므로, 단독 실행만으로는 전체 플로우를 테스트할 수 없고 Spring 서버가 함께 떠 있어야 합니다.

---

## 외부 서비스 준비 사항

실행 전 아래 서비스들의 자격 증명을 환경변수(`.env`)로 구성해야 합니다. 실제 키 이름이나 값은 보안을 위해 별도로 안전하게 전달받아 각자 로컬에 구성해주세요.

- **Google Cloud / Vertex AI**: Gemini 챗 모델·임베딩 모델 호출용 서비스 계정, Vertex AI API 활성화
- **Google Cloud Speech-to-Text / Text-to-Speech**: 음성 인식·합성 API 활성화
- **MongoDB Atlas**: 일기/공공데이터 저장 및 Vector Search 인덱스가 구성된 클러스터
- **Spotify Web API**: Client Credentials Flow용 Client ID/Secret
- **공공데이터포털**: 정신건강기관·청년정책 데이터 수집용 API 키
- (필요 시) **Kakao 주소 검색 API**: 기관 주소 → 좌표 변환용

---

## 프로젝트 구조

```text
.
├── app/
│    ├── api/                  # API 라우터
│    │    ├── analyze.py       # 일기 통합 분석(우울/감정/요약/임베딩/음악추천) 엔드포인트
│    │    ├── batch.py         # 공공데이터 수집/마이그레이션 배치 트리거
│    │    ├── chat.py          # RAG 챗봇 스트리밍 엔드포인트
│    │    ├── report.py        # 주간 리포트 코멘트 생성 엔드포인트
│    │    ├── stt.py           # 음성 인식(STT) 엔드포인트
│    │    └── tts.py           # 음성 합성(TTS) 재생성 엔드포인트
│    ├── core/
│    │    ├── config.py        # 환경설정 로드
│    │    └── database.py      # MongoDB 연결/컬렉션 정의
│    ├── services/             # AI 모델 추론 및 비즈니스 로직
│    │    ├── embedding.py     # Gemini 임베딩(gemini-embedding-001) 생성
│    │    ├── emotion.py       # KOTE 기반 44개 감정 분석 + 9개 그룹 매핑
│    │    ├── music.py         # 감정 기반 Spotify 음악 추천
│    │    ├── prediction.py    # 우울 징후 2단계 캐스케이드 분류
│    │    ├── rag.py           # Gemini 기반 RAG 검색·멀티턴 응답 생성
│    │    ├── report.py        # Gemini 기반 주간 리포트 코멘트 생성
│    │    ├── stt.py           # Google Cloud STT 연동
│    │    ├── summary.py       # Gemini 기반 일기 요약
│    │    └── tts.py           # Google Cloud TTS 연동, 문장 단위 청크 분할
│    └── main.py               # 애플리케이션 진입점, 라우터 등록, 모델 사전 로딩
├── models/
│    └── trained_model_depression_binary/  # AI Hub 심리상담 데이터로 파인튜닝된 우울 분류 모델
└── scripts/                   # 데이터 수집·전처리·모델 학습용 스크립트
     ├── data_extractor.py     # 학습 데이터 추출 및 전처리
     ├── fetch_mental_inst.py  # 정신건강 상담기관 데이터 수집
     ├── fetch_public_svc.py   # 공공 서비스(청년정책 등) 데이터 수집
     ├── kluebert_train.py     # klue/roberta-base 우울 분류 모델 학습(청크분할+환자단위분리)
     ├── migrate_addresses.py  # 주소 → 좌표 마이그레이션
     └── reembed_all.py        # 임베딩 일괄 재생성 스크립트
```

---

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 언어 | Python 3.10 |
| 프레임워크 | FastAPI, Uvicorn |
| AI/ML 런타임 | PyTorch, Hugging Face Transformers |
| LLM / 임베딩 | Google Gemini (`gemini-2.5-flash`), `gemini-embedding-001` — Vertex AI REST API 직접 호출 |
| 감정 분석 모델 | KoELECTRA 기반 KOTE 파인튜닝 모델 |
| 우울 분류 모델 | `klue/roberta-base` 파인튜닝 + `nlptown` 다국어 감성분석 1차 필터 |
| 음성 | Google Cloud Speech-to-Text / Text-to-Speech |
| 음악 추천 | Spotify Web API |
| 벡터 DB | MongoDB Atlas Vector Search |
| 위치/주소 | Kakao 주소 검색 API |
| 공공데이터 | 공공데이터포털 (정신건강기관, 청년정책 등) |

---

## 테스트

현재 별도의 자동화된 테스트 스위트는 포함되어 있지 않습니다. `scripts/` 내 학습 스크립트(`kluebert_train.py`)에 검증셋 기반 성능 평가 로직이 포함되어 있으며, 그 외 기능은 `uvicorn` 로컬 실행 후 Spring 서버를 통한 수동 통합 테스트로 검증합니다.

---

## 버전 기록 (주요 변경 이력)

- **초기 구현**: 일기 저장 시 분석 데이터 + 원본 데이터를 MongoDB에 저장, 전체 결과 임베딩 저장 로직 구현
- **우울 분류 모델 반복 개선**: AI Hub 심리상담 데이터로 다수 회차에 걸쳐 `klue/roberta-base` 파인튜닝 및 성능 개선 진행
- **공공데이터 연동**: 정신건강기관·청년정책 공공데이터 수집 및 임베딩 저장 로직 구현, 주소→좌표(Geocoding) 변환 추가
- **RAG 챗봇 1차 구현**: 공공데이터 임베딩 기반 유사도 검색 + 챗봇 응답 생성 성능 개선 반복
- **TTS 기능 추가**: 챗봇 답변을 음성으로 합성하는 기능 최초 도입
- **우울 분류 모델 핵심 버그 수정**: 학습 시 세션의 약 90%가 512토큰 제한으로 잘려 나가던 문제와, 같은 세션/환자의 청크가 train/validation에 함께 섞이던 데이터 누수 문제를 **청크 분할 + 환자 단위 데이터 분리**로 해결
- **챗봇 LLM 전환**: 챗봇 RAG 응답 생성 모델을 **HyperClova X(Clova) → Google Gemini**로 전환
- **RAG 멀티턴 지원**: 챗봇에 최근 대화 이력을 반영한 멀티턴 대화 기능 추가
- **음악 추천 기능 추가**: 감정 분석 결과 기반 Spotify 음악 추천 기능 도입, 이후 한국 곡 편향·중복 결과 품질 개선
- **STT 기능 추가**: Google Cloud STT 기반 음성 인식 API 추가
- **주간 리포트 기능 추가**: 일주일 치 일기 요약을 바탕으로 Gemini가 격려 코멘트를 생성하는 API 추가
- **README 전면 개정**: Clova 기반 설명을 제거하고 Gemini 기준으로 전면 개정
- **전체 주석 정리**: "What" 중심 주석을 제거하고 설계 의도("Why") 중심으로 재정리
- **TTS 개선**: 문장 단위 청크 합성 방식 개선 및 과거 대화 재진입 시 음성 재생성 API 추가
