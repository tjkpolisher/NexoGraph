# Nexograph - Interdisciplinary Scientific Knowledge Graph System

## 프로젝트 개요

Nexograph는 GraphRAG 기반 학제간(Interdisciplinary) 지식 베이스 시스템입니다.

**핵심 목표**:
1. **Phase 1-2**: AI 분야 논문/문서를 수집하여 Q&A 챗봇 서비스 제공
2. **Phase 3-4**: 서로 다른 학문 분야 간 숨겨진 연결고리를 발견하고, 창발적 연구 가설을 자동 생성

**현재 Phase**: **Phase 1 (MVP)**

**타겟 사용자**: AI/ML 연구자, 데이터 사이언티스트, 학제간 연구에 관심 있는 연구자

---

## Phase 1 MVP 범위

### 포함 (In Scope)
- 수동 문서 업로드 (PDF, Markdown)
- Upstage Document Parse API를 통한 문서 파싱
- LightRAG 기반 지식 그래프 구축 (인메모리 모드)
- 기본 Q&A 채팅 인터페이스
- 출처 문서 링크 제공
- Qdrant 벡터 검색

### 제외 (Out of Scope) - 추후 Phase에서 구현
- 자동 문서 수집 (arXiv API, RSS)
- 사용자 인증/로그인
- 그래프 시각화
- 대화 히스토리 영구 저장
- Multi-Agent 가설 생성 시스템
- Neo4j 그래프 데이터베이스

---

## 기술 스택

### 확정된 기술 스택 (Phase 1)

| 구성 요소 | 기술 | 버전 | 비고 |
|----------|------|------|------|
| **Python Environment** | Anaconda | - | conda 환경: `nexograph` |
| **Python** | Python | 3.10 | conda로 관리 |
| **Backend Framework** | FastAPI | 0.109+ | 비동기 API 서버 |
| **GraphRAG Engine** | LightRAG | latest | 인메모리 모드 (NetworkX) |
| **Vector Database** | Qdrant | latest | Docker 로컬 실행 |
| **LLM** | Upstage Solar-Pro2 | - | OpenAI 호환 API |
| **Embedding** | Upstage Embedding API | - | solar-embedding-1-large |
| **Document Parser** | Upstage Document Parse | - | OCR + Layout Analysis |
| **Metadata DB** | SQLite + SQLAlchemy | 2.0+ | 문서 메타데이터 저장 |
| **Frontend** | Streamlit | 1.30+ | MVP UI |
| **Containerization** | Docker Compose | - | Qdrant 실행 |

### Phase 2+ 추가 예정
- Neo4j (그래프 데이터베이스)
- AutoGen/AG2 (Multi-Agent 프레임워크)
- Semantic Scholar API (참신성 평가)

---

## 디렉토리 구조
```
nexograph/
├── .env                          # 환경변수 (Git 제외)
├── .gitignore
├── CLAUDE.md                     # 이 파일
├── README.md
├── docker-compose.yml            # Qdrant 컨테이너
├── environment.yml               # Conda 환경 정의
├── requirements.txt
├── pyproject.toml
│
├── backend/
│   ├── __init__.py
│   ├── main.py                   # FastAPI 앱 엔트리포인트
│   ├── config.py                 # 설정 관리 (pydantic-settings)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py       # 의존성 주입
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py         # GET /api/v1/health
│   │       ├── documents.py      # /api/v1/documents/*
│   │       └── chat.py           # POST /api/v1/chat
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── upstage/
│   │   │   ├── __init__.py
│   │   │   ├── document_parser.py    # Document Parse API
│   │   │   ├── embedding.py          # Embedding API
│   │   │   └── llm.py                # Solar LLM API
│   │   ├── qdrant_service.py         # Qdrant 클라이언트
│   │   ├── lightrag_service.py       # LightRAG 래퍼
│   │   └── document_service.py       # 문서 처리 오케스트레이션
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py            # Pydantic 요청/응답 스키마
│   │   └── database.py           # SQLAlchemy 모델
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── frontend/
│   └── streamlit_app.py          # Streamlit MVP UI
│
├── data/
│   ├── uploads/                  # 업로드된 원본 파일
│   ├── parsed/                   # 파싱된 마크다운
│   ├── db/                       # SQLite 파일
│   └── test_papers/              # 테스트용 논문
│
├── scripts/
│   ├── setup_qdrant.py           # Qdrant 초기 설정
│   ├── ingest_document.py        # CLI 문서 수집
│   └── test_apis.py              # API 연결 테스트
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # pytest fixtures
│   ├── test_upstage_apis.py
│   ├── test_qdrant.py
│   ├── test_lightrag.py
│   └── test_api_endpoints.py
│
├── notebooks/                    # 실험용 (선택사항)
│   └── experiments.ipynb
│
└── docs/
    └── PRD.md                    # 전체 PRD 문서
```

---

## API 설계

### Base URL
```
http://localhost:8000/api/v1
```

### 엔드포인트 명세

#### 1. Health Check
```
GET /health

Response 200:
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "qdrant": "connected",
    "upstage": "connected",
    "lightrag": "initialized"
  }
}
```

#### 2. 문서 업로드
```
POST /documents/upload
Content-Type: multipart/form-data

Request:
- file: PDF 또는 Markdown 파일 (required)
- title: string (optional, 미입력 시 파일명 사용)
- category: "paper" | "blog" | "documentation" (optional, default: "paper")
- tags: string[] (optional)

Response 202:
{
  "document_id": "uuid",
  "status": "processing",
  "message": "Document upload started"
}

Response 201 (처리 완료 시):
{
  "document_id": "uuid",
  "status": "completed",
  "title": "string",
  "chunks_count": 10,
  "entities_extracted": 25,
  "processing_time_ms": 3500
}
```

#### 3. 문서 목록 조회
```
GET /documents
Query params:
- page: int (default: 1)
- limit: int (default: 20, max: 100)
- category: string (optional)
- search: string (optional, 제목 검색)

Response 200:
{
  "documents": [
    {
      "id": "uuid",
      "title": "string",
      "category": "paper",
      "tags": ["AI", "NLP"],
      "chunks_count": 10,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 100,
  "page": 1,
  "limit": 20
}
```

#### 4. 문서 상세 조회
```
GET /documents/{document_id}

Response 200:
{
  "id": "uuid",
  "title": "string",
  "category": "paper",
  "tags": ["AI", "NLP"],
  "original_filename": "paper.pdf",
  "parsed_content_preview": "첫 500자...",
  "chunks_count": 10,
  "entities": ["GPT-4", "Transformer", "Attention"],
  "created_at": "2025-01-15T10:30:00Z",
  "file_size_bytes": 1024000
}
```

#### 5. 문서 삭제
```
DELETE /documents/{document_id}

Response 204: No Content
```

#### 6. 채팅 (Q&A)
```
POST /chat

Request:
{
  "query": "Transformer의 Attention 메커니즘을 설명해줘",
  "mode": "hybrid",  // "local" | "global" | "hybrid"
  "top_k": 5,        // 검색할 청크 수 (optional, default: 5)
  "include_sources": true  // 출처 포함 여부 (optional, default: true)
}

Response 200:
{
  "answer": "Transformer의 Attention 메커니즘은...",
  "sources": [
    {
      "document_id": "uuid",
      "document_title": "Attention Is All You Need",
      "chunk_preview": "관련 청크 미리보기...",
      "relevance_score": 0.95
    }
  ],
  "mode_used": "hybrid",
  "processing_time_ms": 1200
}
```

---

## 환경 변수

### .env 파일 구조
```env
# === Upstage API ===
UPSTAGE_API_KEY=up_xxxxxxxxxxxxxxxxxxxxxxxx

# === Qdrant ===
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=nexograph_documents

# === Application ===
APP_ENV=development  # development | production
APP_DEBUG=true
APP_VERSION=0.1.0

# === Database ===
DATABASE_URL=sqlite:///./data/db/nexograph.db

# === LightRAG ===
LIGHTRAG_WORKING_DIR=./data/lightrag

# === Optional: Backup LLM ===
# OPENAI_API_KEY=sk-xxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

### config.py 구현 참고
```python
import os
import warnings
from pydantic_settings import BaseSettings
from functools import lru_cache

# Conda 환경 확인 (개발 시 도움)
def check_conda_env():
    expected_env = "nexograph"
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != expected_env:
        warnings.warn(
            f"Expected conda env '{expected_env}', but got '{current_env or 'None'}'. "
            f"Run: conda activate {expected_env}",
            UserWarning
        )

check_conda_env()

class Settings(BaseSettings):
    # Upstage
    upstage_api_key: str
    upstage_base_url: str = "https://api.upstage.ai/v1"
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "nexograph_documents"
    
    # App
    app_env: str = "development"
    app_debug: bool = True
    app_version: str = "0.1.0"
    
    # Database
    database_url: str = "sqlite:///./data/db/nexograph.db"
    
    # LightRAG
    lightrag_working_dir: str = "./data/lightrag"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

---

## Upstage API 사용법

### 1. Document Parse API
```python
# POST https://api.upstage.ai/v1/document-ai/document-parse
# Content-Type: multipart/form-data

import httpx

async def parse_document(file_content: bytes, filename: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.upstage.ai/v1/document-ai/document-parse",
            headers={"Authorization": f"Bearer {UPSTAGE_API_KEY}"},
            files={"document": (filename, file_content)},
            data={"output_formats": "markdown"}  # markdown 출력
        )
        return response.json()

# 응답 구조:
# {
#   "content": {
#     "markdown": "# Title\n\n## Section...",
#     "html": "<h1>Title</h1>...",
#     "text": "Title Section..."
#   },
#   "elements": [
#     {"category": "title", "content": "...", "page": 1},
#     {"category": "paragraph", "content": "...", "page": 1},
#     {"category": "figure", "base64_encoding": "...", "page": 2}
#   ]
# }
```

### 2. Embedding API
```python
# POST https://api.upstage.ai/v1/solar/embeddings

async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.upstage.ai/v1/solar/embeddings",
            headers={
                "Authorization": f"Bearer {UPSTAGE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "solar-embedding-1-large-passage",  # 문서용
                # "model": "solar-embedding-1-large-query",  # 쿼리용
                "input": text
            }
        )
        return response.json()["data"][0]["embedding"]

# 벡터 차원: 4096
# 주의: passage 모델과 query 모델이 분리됨 (asymmetric embedding)
```

### 3. Solar LLM API (OpenAI 호환)
```python
# POST https://api.upstage.ai/v1/solar/chat/completions

async def chat_completion(messages: list[dict], temperature: float = 0.1) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.upstage.ai/v1/solar/chat/completions",
            headers={
                "Authorization": f"Bearer {UPSTAGE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "solar-pro",  # 또는 "solar-mini"
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2048
            },
            timeout=60.0
        )
        return response.json()["choices"][0]["message"]["content"]

# 사용 예시:
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is GraphRAG?"}
# ]
```

---

## LightRAG 통합 가이드

### 패키지 정보
- **패키지명**: `lightrag-hku` (주의: `lightrag`가 아님!)
- **버전**: `>=1.4.0,<2.0.0`
- **GitHub**: https://github.com/HKUDS/LightRAG
- **PyPI**: https://pypi.org/project/lightrag-hku/

### 설치
```bash
pip install "lightrag-hku>=1.4.0,<2.0.0"
```

### 지원 스토리지 (1.4.x 기준)
- **Vector**: QdrantVectorDBStorage (우리가 사용)
- **Graph**: NetworkXStorage (MVP), Neo4JStorage (Phase 2)
- **KV**: JsonKVStorage (MVP)

### 기본 사용법 (1.4.x API)
```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

async def initialize_rag():
    rag = LightRAG(
        working_dir="./data/lightrag",
        embedding_func=custom_embedding_func,  # Upstage Embedding
        llm_model_func=custom_llm_func,        # Upstage Solar
        # 스토리지 설정 (선택사항 - 기본값 사용 가능)
        # vector_storage="QdrantVectorDBStorage",
        # graph_storage="NetworkXStorage",
    )
    await initialize_pipeline_status(rag)
    return rag

# 문서 추가
await rag.ainsert(document_text)

# 검색
result = await rag.aquery(
    query="What is Attention mechanism?",
    param=QueryParam(mode="hybrid")  # "local", "global", "hybrid", "naive"
)
```

### 커스텀 LLM/Embedding 함수 시그니처
```python
# LLM 함수 시그니처 (1.4.x)
async def custom_llm_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs
) -> str:
    # Upstage Solar API 호출
    ...

# Embedding 함수 시그니처 (1.4.x)  
async def custom_embedding_func(texts: list[str]) -> list[list[float]]:
    # Upstage Embedding API 호출
    ...
```

### 주의사항
1. 패키지명이 `lightrag-hku`임 (다른 `lightrag` 패키지와 혼동 주의)
2. 1.4.x 버전은 async 패턴 사용 (`ainsert`, `aquery`)
3. `initialize_pipeline_status()` 호출 필요
4. Context7에서 최신 API 문서 확인 권장

### 구현 전 확인 사항
```bash
# LightRAG 최신 문서 확인
# https://github.com/HKUDS/LightRAG

# 특히 확인할 내용:
# 1. 커스텀 LLM 함수 시그니처
# 2. 커스텀 Embedding 함수 시그니처
# 3. QueryParam 옵션들
# 4. Qdrant 백엔드 지원 여부
```

---

## Qdrant 설정

### 컬렉션 생성 스크립트
```python
# scripts/setup_qdrant.py

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def setup_collection():
    client = QdrantClient(host="localhost", port=6333)
    
    # 컬렉션 생성 (없으면)
    collections = [c.name for c in client.get_collections().collections]
    
    if "nexograph_documents" not in collections:
        client.create_collection(
            collection_name="nexograph_documents",
            vectors_config=VectorParams(
                size=4096,  # Upstage Embedding 차원
                distance=Distance.COSINE
            )
        )
        print("✅ Collection 'nexograph_documents' created")
    else:
        print("ℹ️ Collection 'nexograph_documents' already exists")

if __name__ == "__main__":
    setup_collection()
```

### Docker Compose 실행
```bash
# Qdrant 시작
docker-compose up -d

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs qdrant

# 대시보드 접속
# http://localhost:6333/dashboard
```

---

## 코딩 컨벤션

### Python 스타일
- **버전**: Python 3.10+
- **타입 힌트**: 모든 함수에 필수
- **Docstring**: Google style
- **포매터**: Black (line-length=88)
- **Import 정렬**: isort
- **Linter**: Ruff 권장

### 예시
```python
from typing import Optional
from pydantic import BaseModel

class DocumentResponse(BaseModel):
    """문서 응답 스키마.
    
    Attributes:
        id: 문서 고유 식별자
        title: 문서 제목
        status: 처리 상태
    """
    id: str
    title: str
    status: str
    chunks_count: Optional[int] = None


async def process_document(
    file_content: bytes,
    filename: str,
    *,
    category: str = "paper"
) -> DocumentResponse:
    """문서를 처리하고 지식 그래프에 추가합니다.
    
    Args:
        file_content: 파일 바이너리 내용
        filename: 원본 파일명
        category: 문서 카테고리 (기본값: "paper")
        
    Returns:
        처리된 문서 정보
        
    Raises:
        ValueError: 지원하지 않는 파일 형식인 경우
        UpstageAPIError: API 호출 실패 시
    """
    # 구현...
```

### 비동기 패턴
```python
# ✅ 좋은 예: async/await 일관성 유지
async def get_documents() -> list[Document]:
    async with get_db_session() as session:
        result = await session.execute(select(Document))
        return result.scalars().all()

# ❌ 나쁜 예: 동기/비동기 혼합
def get_documents():  # sync 함수에서
    asyncio.run(...)  # async 호출 - 피해야 함
```

### 에러 처리
```python
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential

# API 호출에는 재시도 로직 적용
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def call_upstage_api(...):
    ...

# HTTP 에러는 명확한 메시지와 함께
raise HTTPException(
    status_code=400,
    detail={
        "error": "invalid_file_type",
        "message": "Only PDF and Markdown files are supported",
        "supported_types": [".pdf", ".md"]
    }
)
```

---

## 구현 우선순위

### Phase 1 구현 순서
```
Week 1:
├── Day 1-2: 프로젝트 기본 구조
│   ├── FastAPI 앱 초기화 (main.py)
│   ├── 설정 관리 (config.py)
│   └── Health check 엔드포인트
│
├── Day 3-4: 인프라 연결
│   ├── Qdrant 서비스 (qdrant_service.py)
│   └── 컬렉션 생성 스크립트
│
└── Day 5: Upstage API 연동
    ├── Document Parse 서비스
    ├── Embedding 서비스
    └── LLM 서비스

Week 2:
├── Day 1-2: LightRAG 통합
│   ├── LightRAG 서비스 래퍼
│   └── 커스텀 LLM/Embedding 연결
│
├── Day 3-4: 문서 처리 파이프라인
│   ├── 문서 업로드 API
│   ├── 파싱 → 청킹 → 인덱싱 플로우
│   └── 메타데이터 저장 (SQLite)
│
└── Day 5: 채팅 API
    ├── Q&A 엔드포인트
    └── 출처 추적 로직

Week 3:
├── Day 1-2: Streamlit UI
│   ├── 문서 업로드 UI
│   └── 채팅 인터페이스
│
├── Day 3-4: 통합 테스트
│   ├── API 테스트
│   └── E2E 테스트
│
└── Day 5: 버그 수정 및 문서화
```

---

## 테스트

### 테스트 실행
```bash
# 전체 테스트
pytest

# 특정 파일
pytest tests/test_upstage_apis.py

# 커버리지 포함
pytest --cov=backend --cov-report=html
```

### 테스트 구조
```python
# tests/conftest.py
import pytest
from httpx import AsyncClient
from backend.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

# tests/test_api_endpoints.py
@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

## 주의사항 및 제약조건

### 0. Conda 환경 (가장 중요!)
- **모든 명령어 실행 전** `conda activate nexograph` 필수
- 터미널을 새로 열 때마다 환경 활성화 필요
- VS Code 사용 시: Python 인터프리터를 conda nexograph로 설정
- 패키지 설치는 환경 활성화 후 `pip install` 사용

### 1. API 키 보안
- **절대** 코드에 API 키 하드코딩 금지
- `.env` 파일은 `.gitignore`에 포함
- 커밋 전 `git diff --staged`로 확인

### 2. Upstage API 제한
- Rate Limit 존재 (정확한 수치는 콘솔에서 확인)
- 크레딧 소모 모니터링 필요
- Document Parse: 파일당 최대 크기 제한 있음 (동기 요청 시 최대 100페이지, 비동기 요청 시 최대 1000페이지까지 가능)

### 3. LightRAG 주의점
- 버전별 API 차이 큼 - 공식 문서 반드시 확인
- 인메모리 모드는 서버 재시작 시 데이터 손실 (working_dir 필요)
- 대용량 문서 처리 시 메모리 사용량 주의

### 4. Qdrant 주의점
- Docker 볼륨 마운트로 데이터 영속성 확보
- 컬렉션 삭제 시 복구 불가

### 5. 비동기 처리
- FastAPI는 async 함수 권장
- 동기 라이브러리 사용 시 `run_in_executor` 활용
- DB 세션은 요청별로 생성/종료

---

## 자주 사용하는 명령어
```bash
# === Conda 환경 ===
conda activate nexograph          # 환경 활성화 (필수!)
conda deactivate                  # 환경 비활성화
conda env list                    # 환경 목록 확인
conda env export > environment.yml  # 환경 내보내기

# === 개발 서버 실행 ===
# 반드시 conda activate nexograph 후 실행
uvicorn backend.main:app --reload --port 8000

# === Streamlit 실행 ===
streamlit run frontend/streamlit_app.py

# === Docker (Qdrant) ===
docker-compose up -d      # 시작
docker-compose down       # 종료
docker-compose logs -f    # 로그

# === 테스트 ===
pytest -v
pytest --cov=backend

# === 포맷팅 ===
black backend/
isort backend/

# === 의존성 관리 ===
pip install -r requirements.txt        # 패키지 설치
pip freeze > requirements.txt          # 현재 패키지 저장
conda env create -f environment.yml    # 환경 재현 (다른 PC)
```

---

## 참고 문서

### 공식 문서
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Upstage API Docs](https://developers.upstage.ai/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### 프로젝트 문서
- `docs/PRD.md` - 전체 PRD (상세 기술 스택, 로드맵)

---

## Phase 3-4 미리보기 (참고용)

Phase 1-2 완료 후, 다음 기능들이 추가될 예정:

### Multi-Agent 가설 생성 시스템
- **Path Explorer**: 지식 그래프에서 두 개념 사이의 경로 탐색
- **Analogy Detector**: 분야 간 숨겨진 유비(Analogy) 발견
- **Hypothesis Generator**: 창발적 연구 가설 자동 생성
- **Domain Experts**: 물리학, 천문학, 심리학 등 분야별 전문가 에이전트
- **Novelty Assessor**: Semantic Scholar API로 참신성 평가

### 확장 도메인
- Scientific AI (Physics-Informed ML, Neural Operators)
- 물리학
- 기계공학 (유체역학, 열역학)
- 천문학 (관측천문학, 천체물리학)
- 심리학 (인지심리학, 의사결정론)
- 교육학 (학습과학, 교육공학)

이 기능들은 AutoGen/AG2 프레임워크를 사용하여 구현 예정.

---

*Last Updated: 2025-01-01*
*Version: 0.1.0 (Phase 1 MVP)*