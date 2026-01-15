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
    ├── PRD.md                    # 전체 PRD 문서
    ├── guides/                   # 개발 가이드
    │   ├── api-design.md
    │   ├── development.md
    │   ├── qdrant-setup.md
    │   └── testing.md
    └── reference/                # 참고 자료
        ├── implementation-roadmap.md
        └── future-roadmap.md
```

---

## 📖 기술 문서 활용 가이드

### 🎯 Skills (도메인별 패턴)
코드 작성 시 관련 skill을 직접 참조하세요:

- **`/nexograph-architecture`**: 아키텍처 원칙, 코딩 컨벤션, 에러 처리 패턴
  - 계층화된 아키텍처 (Router → Service → Repository)
  - 비동기 우선 원칙
  - 의존성 주입 패턴
  - 네이밍 컨벤션

- **`/upstage-integration`**: Upstage API (Document Parse, Embedding, LLM) 연동
  - 비대칭 임베딩 모델 사용법 (query/passage 분리)
  - Document Parse API 사용법
  - Solar LLM API 사용법
  - 에러 처리 및 재시도 로직

- **`/graphrag-patterns`**: GraphRAG 엔티티/관계 타입, 검색 모드 선택
  - AI 도메인 엔티티 타입 (MODEL, TECHNIQUE, DATASET 등)
  - 관계 타입 (DEVELOPED_BY, USES_TECHNIQUE 등)
  - LightRAG 검색 모드 선택 가이드
  - 문서 청킹 전략

### 🤖 Agents (작업별 전문가)
복잡한 작업은 전문 agent에게 위임하세요:

- **새 모듈 작성**: "nexograph-module-writer agent를 사용해 DocumentService 작성"
  - FastAPI 라우터, LightRAG 서비스, Qdrant 리포지토리 등
  - 계층화된 아키텍처 준수
  - 타입 힌팅 및 Pydantic 검증

- **LightRAG 설정**: "nexograph-lightrag-specialist agent로 임베딩 설정 최적화"
  - LightRAG 초기화 및 설정
  - Upstage 연동 패턴
  - 쿼리 최적화

- **테스트 실행**: "nexograph-integration-tester agent로 통합 테스트 실행"
  - pytest 실행 및 결과 분석
  - 실패 원인 분석 및 수정 방안 제시

### 📄 상세 가이드 (docs/guides/)
구현 시 필요한 상세 명세:

- **API 엔드포인트**: `docs/guides/api-design.md`
  - Health Check, 문서 업로드, 문서 조회, 채팅 등 모든 엔드포인트 명세

- **개발 환경 설정**: `docs/guides/development.md`
  - 환경 변수 (.env)
  - config.py 구현
  - 자주 사용하는 명령어
  - 핵심 주의사항

- **Qdrant 설정**: `docs/guides/qdrant-setup.md`
  - 컬렉션 생성 스크립트
  - Docker Compose 실행

- **테스트 작성**: `docs/guides/testing.md`
  - 테스트 구조 및 실행
  - Fixtures 활용
  - Mock 사용법

### 📚 참고 자료 (docs/reference/)
필요시 참조하는 문서:

- **구현 로드맵**: `docs/reference/implementation-roadmap.md`
  - Phase 1 구현 순서 (Week 1-3)

- **Phase 3-4 계획**: `docs/reference/future-roadmap.md`
  - Multi-Agent 가설 생성 시스템
  - 확장 도메인

---

## ⚠️ 핵심 주의사항

### 0. Conda 환경 (가장 중요!)
- **모든 명령어 실행 전** `conda activate nexograph` 필수
- 터미널을 새로 열 때마다 환경 활성화 필요
- VS Code 사용 시: Python 인터프리터를 conda nexograph로 설정
- 패키지 설치는 환경 활성화 후 `pip install` 사용

### 1. API 키 보안
- **절대** 코드에 API 키 하드코딩 금지
- `.env` 파일은 `.gitignore`에 포함
- 커밋 전 `git diff --staged`로 확인

### 2. 비동기 처리
- FastAPI는 async 함수 권장
- 동기 라이브러리 사용 시 `run_in_executor` 활용
- DB 세션은 요청별로 생성/종료

### 3. 문서 참조 우선순위
1. **Skills** - 코딩 패턴 및 컨벤션
2. **Agents** - 복잡한 작업 위임
3. **docs/guides/** - 상세 구현 명세
4. **docs/reference/** - 참고 자료

---

## 자주 사용하는 명령어

```bash
# Conda 환경 활성화 (필수!)
conda activate nexograph

# 개발 서버 실행
uvicorn backend.main:app --reload --port 8000

# Streamlit 실행
streamlit run frontend/streamlit_app.py

# Docker (Qdrant)
docker-compose up -d      # 시작
docker-compose down       # 종료

# 테스트
pytest -v
pytest --cov=backend
```

**전체 명령어 목록**: `docs/guides/development.md` 참조

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
- `docs/guides/` - 개발 가이드 모음
- `docs/reference/` - 참고 자료 모음

---

*Last Updated: 2026-01-15*
*Version: 0.1.0 (Phase 1 MVP)*
