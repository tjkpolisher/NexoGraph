# NexoGraph 🔗

**Interdisciplinary Scientific Knowledge Graph System**

GraphRAG 기반 학제간 지식 베이스 시스템으로, AI 분야 논문과 문서를 수집하여 Q&A 챗봇 서비스를 제공합니다.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Current Phase: Phase 2 (Enhanced Q&A)

### Phase 1 완료된 기능

- ✅ **Document Upload**: PDF와 Markdown 파일 업로드 및 파싱
- ✅ **Vector Search**: Qdrant를 이용한 고속 벡터 검색
- ✅ **Knowledge Graph**: LightRAG 기반 지식 그래프 구축
- ✅ **Q&A Chatbot**: Upstage Solar LLM 기반 질문답변
- ✅ **Source Citation**: 답변 출처 표시 및 추적 (relevance score 포함)
- ✅ **Web UI**: Streamlit 기반 사용자 친화적 인터페이스
- ✅ **Resilience**: Circuit Breaker, 재시도 로직, Rate limit 보호
- ✅ **Performance**: 배치 임베딩 처리, Token-aware 청킹

### Phase 2 진행 예정

- 🔄 **Auto Collection**: arXiv API를 통한 자동 문서 수집
- 🔄 **Authentication**: 기본 사용자 인증
- 🔄 **Chat History**: 대화 히스토리 영구 저장
- 🔄 **Visualization**: 기본 그래프 시각화

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Frontend (Streamlit)                │
│  - Document Upload  - Chat Interface            │
└─────────────────┬───────────────────────────────┘
                  │ REST API
┌─────────────────┴───────────────────────────────┐
│            Backend (FastAPI)                     │
│  - Document Processing  - Chat Endpoint         │
└──┬────────────┬──────────────┬──────────────┬───┘
   │            │              │              │
   │            │              │              │
┌──▼───┐  ┌────▼────┐  ┌──────▼─────┐  ┌────▼────┐
│SQLite│  │ Qdrant  │  │  LightRAG  │  │ Upstage │
│  DB  │  │ Vector  │  │   Graph    │  │   API   │
└──────┘  └─────────┘  └────────────┘  └─────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv
- Docker & Docker Compose (for Qdrant)
- Upstage API Key ([Get it here](https://console.upstage.ai/))

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd NexoGraph

# 2. Create conda environment
conda env create -f environment.yml
conda activate nexograph

# OR use pip
conda create -n nexograph python=3.10 -y
conda activate nexograph
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your UPSTAGE_API_KEY

# 4. Start infrastructure
docker-compose up -d  # Start Qdrant

# 5. Initialize database
python scripts/init_db.py
python scripts/setup_qdrant.py

# 6. Start services
# Terminal 1: Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend (optional)
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Quick Test

```bash
# Check health
curl http://localhost:8000/api/v1/health

# Upload test document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/test_papers/test_rag_document.md" \
  -F "title=Test Document" \
  -F "category=paper"

# Ask a question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "mode": "hybrid"}'
```

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - 전체 프로젝트 사양 및 개발 가이드
- **[SETUP_AND_TEST.md](SETUP_AND_TEST.md)** - 상세 설치 및 테스트 가이드
- **[API Documentation](http://localhost:8000/docs)** - Swagger UI (서버 실행 후)

## 🧪 Testing

```bash
# Run integration tests (backend must be running)
python scripts/test_integration.py

# Run unit tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html
```

## 📁 Project Structure

```
nexograph/
├── backend/              # FastAPI backend
│   ├── api/             # API routes
│   ├── models/          # Database & schemas
│   └── services/        # Business logic
├── frontend/            # Streamlit UI
├── data/                # Data storage
│   ├── db/             # SQLite database
│   ├── parsed/         # Parsed documents
│   ├── lightrag/       # LightRAG data
│   └── test_papers/    # Test documents
├── scripts/             # Utility scripts
└── tests/               # Test files
```

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend** | FastAPI | 0.109+ |
| **Frontend** | Streamlit | 1.30+ |
| **Vector DB** | Qdrant | latest |
| **Graph RAG** | LightRAG | 1.4+ |
| **LLM** | Upstage Solar-Pro2 | - |
| **Embedding** | Upstage Embedding | 4096-dim |
| **Parser** | Upstage Document Parse | - |
| **Database** | SQLite + SQLAlchemy | 2.0+ |
| **Containerization** | Docker Compose | - |

## 🎨 Features in Detail

### Document Processing Pipeline

1. **Upload** → PDF/MD file upload
2. **Parse** → Upstage Document Parse (OCR + Layout)
3. **Chunk** → Intelligent text chunking (12K chars, 500 overlap)
4. **Embed** → Vector embeddings (4096-dim)
5. **Store** → Qdrant (vectors) + LightRAG (graph)
6. **Index** → SQLite (metadata)

### Q&A Pipeline

1. **Query** → User question
2. **Embed** → Query embedding
3. **Retrieve** → Vector search (Qdrant) + Graph search (LightRAG)
4. **Combine** → Context from both sources
5. **Generate** → Solar LLM generates answer
6. **Cite** → Source attribution

## 🔮 Roadmap

### Phase 1 (Completed) ✅
- ✅ Basic document upload and Q&A
- ✅ Vector + Graph hybrid search
- ✅ Simple web interface
- ✅ Circuit Breaker & Retry logic
- ✅ Batch embedding processing

### Phase 2 (Current) 🔄
- 🔄 Auto document collection (arXiv API)
- 🔄 User authentication
- 🔄 Graph visualization
- 🔄 Persistent chat history
- 🔄 Entity extraction display

### Phase 3-4 (Future)
- Multi-Agent hypothesis generation
- Cross-domain knowledge discovery
- Advanced graph analytics (Neo4j)
- Multi-domain expansion

## 🐛 Troubleshooting

See [SETUP_AND_TEST.md](SETUP_AND_TEST.md) for detailed troubleshooting guide.

Common issues:
- **ModuleNotFoundError**: Run `pip install -r requirements.txt`
- **Qdrant connection failed**: Run `docker-compose up -d`
- **API key error**: Check `.env` file
- **Database error**: Run `python scripts/init_db.py`

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🤝 Contributing

This is currently a Phase 2 MVP. Contributions will be welcome in future phases.

## 📧 Contact

For questions or issues, please check the documentation or create an issue.

---

**Version**: 0.2.0 (Phase 2)
**Last Updated**: 2026-02-18
