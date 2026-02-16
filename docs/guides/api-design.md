# API 설계

## Base URL
```
http://localhost:8000/api/v1
```

## 엔드포인트 명세

### 1. Health Check
```
GET /health

Response 200:
{
  "status": "healthy",
  "version": "0.2.0",
  "services": {
    "qdrant": "connected",
    "upstage": "connected",
    "lightrag": "initialized"
  }
}
```

### 1-1. Metrics (Phase 1 구현됨)
```
GET /metrics

Response 200:
{
  "total_requests": 150,
  "total_errors": 3,
  "error_rate": 0.02,
  "endpoints": {
    "/api/v1/chat": {
      "requests": 50,
      "errors": 1,
      "avg_response_time_ms": 1200
    }
  },
  "services": {
    "upstage_embedding": {
      "requests": 100,
      "rate_limits": 2
    }
  }
}
```

### 1-2. Circuit Breaker Status (Phase 1 구현됨)
```
GET /circuit-breaker/status

Response 200:
{
  "services": {
    "upstage_embedding": {
      "state": "closed",
      "failure_count": 0,
      "last_failure": null
    },
    "upstage_llm": {
      "state": "half_open",
      "failure_count": 3,
      "last_failure": "2026-02-16T10:30:00Z"
    }
  }
}
```

### 2. 문서 업로드
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

### 3. 문서 목록 조회
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

### 4. 문서 상세 조회
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

### 5. 문서 삭제
```
DELETE /documents/{document_id}

Response 204: No Content
```

### 6. 채팅 (Q&A)
```
POST /chat

Request:
{
  "query": "Transformer의 Attention 메커니즘을 설명해줘",
  "mode": "hybrid",  // "local" | "global" | "hybrid" | "naive" | "mix"
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

Mode 설명:
- hybrid: 벡터 검색 + 그래프 검색 (권장)
- local: 엔티티 중심 검색
- global: 전역 지식 검색
- naive: 단순 벡터 검색만
- mix: 모든 모드 조합
```

---

*Last Updated: 2026-02-16*
