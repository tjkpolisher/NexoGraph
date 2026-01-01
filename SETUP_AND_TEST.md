# Nexograph - Setup and Integration Testing Guide

## 🚀 Quick Start Guide

### Step 1: Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate nexograph

# OR install packages with pip
conda create -n nexograph python=3.10 -y
conda activate nexograph
pip install -r requirements.txt
```

### Step 2: Configuration

1. **Create `.env` file** (if not exists):
```bash
cp .env.example .env
```

2. **Edit `.env` file** with your API keys:
```env
# === Upstage API ===
UPSTAGE_API_KEY=up_xxxxxxxxxxxxxxxxxxxxxxxx

# === Qdrant ===
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=nexograph_documents

# === Application ===
APP_ENV=development
APP_DEBUG=true
APP_VERSION=0.1.0

# === Database ===
DATABASE_URL=sqlite:///./data/db/nexograph.db

# === LightRAG ===
LIGHTRAG_WORKING_DIR=./data/lightrag
```

### Step 3: Infrastructure Setup

```bash
# 1. Start Qdrant (Docker)
docker-compose up -d

# 2. Initialize Database
python scripts/init_db.py

# 3. Setup Qdrant Collection
python scripts/setup_qdrant.py
```

### Step 4: Start Services

```bash
# Terminal 1: Backend API Server
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend UI (optional)
streamlit run frontend/streamlit_app.py --server.port 8501
```

---

## 🧪 Integration Testing

### Test 1: Health Check

```bash
# Check API is running
curl http://localhost:8000/api/v1/health

# Expected output:
# {
#   "status": "healthy",
#   "version": "0.1.0",
#   "services": {
#     "qdrant": "connected",
#     "upstage": "configured",
#     "lightrag": "initialized"
#   }
# }
```

### Test 2: Document Upload

**Option A: Using cURL**

```bash
# Create a test markdown file
echo "# Test Document

## Introduction
This is a test document for Nexograph.

## Content
LightRAG is a graph-based RAG system that combines vector search with knowledge graphs.

## Conclusion
It enables more accurate question-answering." > data/test_papers/test.md

# Upload document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/test_papers/test.md" \
  -F "title=Test Document" \
  -F "category=paper" \
  -F "tags=test,rag,knowledge-graph"

# Expected output:
# {
#   "document_id": "uuid-here",
#   "status": "completed",
#   "title": "Test Document",
#   "chunks_count": 3,
#   "processing_time_ms": 3500
# }
```

**Option B: Using Streamlit UI**

1. Open http://localhost:8501
2. Click "Browse files" in sidebar
3. Select a PDF or Markdown file
4. Fill in title, category, tags
5. Click "Upload"

### Test 3: List Documents

```bash
# Get document list
curl http://localhost:8000/api/v1/documents?page=1&limit=10

# Expected output:
# {
#   "documents": [
#     {
#       "id": "uuid",
#       "title": "Test Document",
#       "category": "paper",
#       "tags": ["test", "rag"],
#       "chunks_count": 3,
#       "created_at": "2025-01-01T10:00:00Z"
#     }
#   ],
#   "total": 1,
#   "page": 1,
#   "limit": 10
# }
```

### Test 4: Get Document Detail

```bash
# Replace {document_id} with actual ID from previous response
curl http://localhost:8000/api/v1/documents/{document_id}

# Expected output:
# {
#   "id": "uuid",
#   "title": "Test Document",
#   "category": "paper",
#   "tags": ["test", "rag"],
#   "original_filename": "test.md",
#   "parsed_content_preview": "# Test Document\n\n## Introduction...",
#   "chunks_count": 3,
#   "entities": [],
#   "created_at": "2025-01-01T10:00:00Z",
#   "file_size_bytes": 256
# }
```

### Test 5: Chat Query

```bash
# Ask a question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LightRAG?",
    "mode": "hybrid",
    "top_k": 5,
    "include_sources": true
  }'

# Expected output:
# {
#   "answer": "LightRAG is a graph-based RAG system...",
#   "sources": [
#     {
#       "document_id": "uuid",
#       "document_title": "Test Document",
#       "chunk_preview": "LightRAG is a graph-based...",
#       "relevance_score": 0.95
#     }
#   ],
#   "mode_used": "hybrid",
#   "processing_time_ms": 1200
# }
```

### Test 6: Delete Document

```bash
# Delete a document
curl -X DELETE http://localhost:8000/api/v1/documents/{document_id}

# Expected: HTTP 204 No Content
```

---

## 🎯 Streamlit UI Testing

### Full Flow Test

1. **Open UI**: Navigate to http://localhost:8501

2. **Upload Document**:
   - Click sidebar "Browse files"
   - Select a test PDF or Markdown file
   - Enter title: "My Test Paper"
   - Select category: "paper"
   - Enter tags: "AI, NLP, Test"
   - Click "Upload"
   - ✅ Should see success message with document ID and chunk count

3. **Verify Document List**:
   - Check sidebar "Documents" section
   - ✅ Should see the uploaded document

4. **Ask Questions**:
   - Select search mode: "hybrid"
   - Set top_k: 5
   - Enable "Show sources"
   - Type question: "What is this document about?"
   - Press Enter
   - ✅ Should see AI-generated answer
   - ✅ Should see source citations in expander

5. **Check Sources**:
   - Expand "Sources" section
   - ✅ Should see document title, relevance score, and preview

6. **Delete Document**:
   - Click trash icon (🗑️) next to document in sidebar
   - ✅ Document should disappear from list

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

**Problem**: Missing Python packages

**Solution**:
```bash
conda activate nexograph
pip install -r requirements.txt
```

### Issue: Qdrant connection failed

**Problem**: Qdrant is not running

**Solution**:
```bash
docker-compose up -d
python scripts/setup_qdrant.py
```

### Issue: Database error

**Problem**: Database not initialized

**Solution**:
```bash
python scripts/init_db.py
```

### Issue: API key error

**Problem**: Missing or invalid Upstage API key

**Solution**:
```bash
# Edit .env file
nano .env

# Add your API key
UPSTAGE_API_KEY=up_your_actual_key_here
```

### Issue: Port already in use

**Problem**: Port 8000 or 8501 is occupied

**Solution**:
```bash
# Use different ports
uvicorn backend.main:app --reload --port 8001
streamlit run frontend/streamlit_app.py --server.port 8502
```

---

## 📊 Expected Results

### Successful Setup Indicators

✅ **Database**: `data/db/nexograph.db` file created
✅ **Qdrant**: `docker ps` shows qdrant container running
✅ **API**: http://localhost:8000/docs shows Swagger UI
✅ **Health**: `/health` endpoint returns `"status": "healthy"`
✅ **UI**: http://localhost:8501 loads Streamlit interface

### Successful Test Indicators

✅ **Document Upload**: Returns 201 with document_id
✅ **Document List**: Returns documents array
✅ **Chat Query**: Returns answer with sources
✅ **Source Citations**: Shows document title and relevance score
✅ **Processing Time**: < 5 seconds for simple queries

---

## 🔧 Advanced Testing

### Test with Real Papers

```bash
# Download a sample paper (example)
# Place it in data/test_papers/

# Upload via API
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/test_papers/attention_paper.pdf" \
  -F "title=Attention Is All You Need" \
  -F "category=paper" \
  -F "tags=NLP,Transformer,Attention"
```

### Test Different Query Modes

```bash
# Local mode (entity-focused)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain self-attention", "mode": "local"}'

# Global mode (community-level)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformer applications?", "mode": "global"}'

# Hybrid mode (recommended)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare BERT and GPT", "mode": "hybrid"}'
```

---

## 📝 Notes

- **First upload**: May take 2-3 minutes as LightRAG initializes
- **Subsequent uploads**: Usually faster (30-60 seconds)
- **Chat queries**: Typically 1-3 seconds with cached embeddings
- **API Credits**: Monitor Upstage API usage to avoid overage

---

## 🎉 Success Criteria

Your system is working correctly if:

1. ✅ Health check returns healthy status
2. ✅ Documents can be uploaded without errors
3. ✅ Document list shows uploaded documents
4. ✅ Chat queries return relevant answers
5. ✅ Source citations match uploaded documents
6. ✅ Streamlit UI shows all features
7. ✅ Document deletion works properly

If all tests pass, congratulations! Your Nexograph system is fully operational. 🚀
