# MAUM (마음) - AI & Data API Server

사용자의 일기 텍스트를 분석하여 감정과 우울증 수치를 도출하고, RAG(검색 증강 생성) 기술을 활용해 맞춤형 챗봇 상담을 제공하는 "MAUM" 서비스의 Python AI/Data API 리포지토리입니다.

* **개발 기간**: 2026.03 ~ 2026.06
* **개발 인원**: 1인 (개인 프로젝트)

## Tech Stack

### Language & Framework
- **Language**: Python
- **Framework**: FastAPI
- **AI/ML**: PyTorch, Hugging Face Transformers

### AI Models
- **HyperCLOVA X**: 임베딩 V2, HCX-007, RAG Reasoning
- **LLM / NLP**: KoELECTRA, klue/bert-base

### Database
- **Vector DB**: MongoDB Atlas Vector Search

---

## Key Features (AI & Data)

### 1. AI Hub 심리상담 데이터 기반 우울증 예측 모델
- AI Hub에서 제공하는 양질의 심리상담 데이터를 활용하여 `klue/bert-base` 모델을 파인튜닝(`kluebert_train.py`)했습니다.
- 학습된 모델(`trained_model_depression_binary`)을 통해 사용자가 작성한 일기 텍스트 내의 우울증 징후를 이진 분류 및 수치화하여 분석합니다(`prediction.py`).

### 2. KoELECTRA 기반 정밀 감정 분석
- 한국어 감정 분류 데이터셋(KOTE)으로 파인튜닝된 KoELECTRA 모델을 적용하여, 사용자의 일기 텍스트에서 느껴지는 복합적인 감정 상태를 분석 및 분류합니다(`emotion.py`, `analyze.py`).

### 3. HyperCLOVA X 및 RAG 파이프라인
- **데이터 벡터화**: 수집된 공공 API 데이터 및 시설 정보를 HyperCLOVA X 임베딩 V2 모델을 사용해 벡터화(`embedding.py`)하고 MongoDB에 저장합니다.
- **RAG 챗봇**: MongoDB Vector Search 기반의 유사도 검색과 HyperCLOVA X(HCX-007) 추론 모델, RAG Reasoning 모델을 결합하여 컨텍스트가 풍부한 AI 챗봇 응답을 생성합니다(`rag.py`, `chat.py`).
- **일기 요약**: 긴 일기 내용을 핵심 위주로 요약하는 기능을 제공합니다(`summary.py`).

### 4. 공공 데이터 수집 및 전처리 파이프라인
- Python 스크립트를 통해 공공 API(심리상담기관, 청년정책 등) 데이터를 수집(`fetch_mental_inst.py`, `fetch_public_svc.py`)하고, 주소 데이터 등을 정제(`migrate_addresses.py`)하여 RAG 검색 품질을 높입니다.

---

## Project Structure

```text
.
├── app/
│    ├── api/             # API 라우터 (analyze.py, batch.py, chat.py)
│    ├── core/            # DB 연결(database.py) 및 환경 설정(config.py)
│    ├── services/        # AI 모델 추론 및 RAG 비즈니스 로직
│    │    ├── embedding.py   # 벡터 임베딩 생성
│    │    ├── emotion.py     # KoELECTRA 감정 분석 로직
│    │    ├── prediction.py  # 우울증 예측 모델 로직
│    │    ├── rag.py         # RAG 기반 검색 및 컨텍스트 구성
│    │    └── summary.py     # HyperCLOVA X 텍스트 요약
│    └── main.py          # 애플리케이션 진입점 및 서버 실행
├── models/
│    └── trained_model_depression_binary/  # AI Hub 데이터로 파인튜닝된 우울증 분석 모델
└── scripts/              # 데이터 수집, 추출 및 모델 학습용 스크립트
     ├── data_extractor.py        # 학습 데이터 추출 및 전처리
     ├── fetch_mental_inst.py     # 정신건강 상담 기관 데이터 수집
     ├── fetch_public_svc.py      # 공공 서비스 데이터 수집
     ├── kluebert_train.py        # klue/bert-base 우울증 모델 학습 파이프라인
     └── migrate_addresses.py     # 주소 데이터 마이그레이션
