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
    "upstage": "configured",
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

## Phase 2 엔드포인트

> Phase 2 엔드포인트는 별도 명시가 없는 한 인증이 필요합니다.
> 인증 헤더: `Authorization: Bearer <access_token>`
>
> 예외 (인증 불필요):
> - `POST /auth/register`
> - `POST /auth/login`

### 7. 회원가입
```
POST /auth/register
// 인증 불필요

Request:
{
  "email": "user@example.com",
  "password": "securepassword",
  "name": "홍길동"
}

Response 201:
{
  "id": "uuid",
  "email": "user@example.com",
  "name": "홍길동",
  "created_at": "2026-02-18T10:00:00Z"
}

Response 409 (이메일 중복):
{
  "detail": "Email already registered"
}

Response 422 (유효성 실패):
{
  "detail": [{"loc": ["body", "email"], "msg": "invalid email format"}]
}
```

### 8. 로그인
```
POST /auth/login
// 인증 불필요

Request:
{
  "email": "user@example.com",
  "password": "securepassword"
}

Response 200:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}

Response 401 (인증 실패):
{
  "detail": "Invalid email or password"
}
```

### 9. 현재 사용자 정보
```
GET /auth/me
Authorization: Bearer <access_token>

Response 200:
{
  "id": "uuid",
  "email": "user@example.com",
  "name": "홍길동",
  "interests": ["NLP", "Computer Vision"],
  "created_at": "2026-02-18T10:00:00Z"
}

Response 401:
{
  "detail": "Not authenticated"
}
```

### 10. 대화 목록 조회
```
GET /conversations
Authorization: Bearer <access_token>
Query params:
- page: int (default: 1)
- limit: int (default: 20, max: 100)

Response 200:
{
  "conversations": [
    {
      "id": "uuid",
      "title": "Transformer 아키텍처 질문",
      "message_count": 5,
      "created_at": "2026-02-18T10:00:00Z",
      "updated_at": "2026-02-18T11:30:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "limit": 20
}
```
> 사용자별 격리: 현재 인증된 사용자의 대화만 반환

### 11. 대화 메시지 조회
```
GET /conversations/{conversation_id}/messages
Authorization: Bearer <access_token>
Query params:
- page: int (default: 1)
- limit: int (default: 50, max: 200)

Response 200:
{
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "Transformer의 Attention 메커니즘을 설명해줘",
      "sources": null,
      "created_at": "2026-02-18T10:00:00Z"
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "Transformer의 Attention 메커니즘은...",
      "sources": [
        {
          "document_id": "uuid",
          "document_title": "Attention Is All You Need",
          "relevance_score": 0.95
        }
      ],
      "created_at": "2026-02-18T10:00:05Z"
    }
  ],
  "total": 10,
  "page": 1,
  "limit": 50
}

Response 404 (대화 없음 또는 권한 없음):
{
  "detail": "Conversation not found"
}
```

### 12. 대화 삭제
```
DELETE /conversations/{conversation_id}
Authorization: Bearer <access_token>

Response 204: No Content

Response 404:
{
  "detail": "Conversation not found"
}
```

### 13. 채팅 (Phase 2 확장)
```
POST /chat
Authorization: Bearer <access_token>

Request:
{
  "query": "Transformer의 Attention 메커니즘을 설명해줘",
  "mode": "hybrid",
  "top_k": 5,
  "include_sources": true,
  "conversation_id": "uuid"  // optional, 기존 대화 이어하기
}

Response 200:
{
  "answer": "Transformer의 Attention 메커니즘은...",
  "sources": [...],
  "mode_used": "hybrid",
  "processing_time_ms": 1200,
  "conversation_id": "uuid",  // 신규 생성 또는 기존 대화 ID
  "message_id": "uuid"
}
```
> conversation_id 미제공 시 새 대화가 자동 생성됩니다.

### 14. 수집 설정
```
POST /collection/config
Authorization: Bearer <access_token>

Request:
{
  "categories": ["cs.AI", "cs.CL", "cs.LG"],
  "keywords": ["transformer", "large language model"],
  "schedule_interval_hours": 24,
  "max_papers_per_run": 50
}

Response 200:
{
  "config_id": "uuid",
  "categories": ["cs.AI", "cs.CL", "cs.LG"],
  "keywords": ["transformer", "large language model"],
  "schedule_interval_hours": 24,
  "max_papers_per_run": 50,
  "next_run_at": "2026-02-19T10:00:00Z"
}
```

### 15. 수집 현황 조회
```
GET /collection/status
Authorization: Bearer <access_token>

Response 200:
{
  "is_running": false,
  "last_run_at": "2026-02-18T10:00:00Z",
  "last_run_result": {
    "papers_found": 30,
    "papers_new": 12,
    "papers_duplicate": 18,
    "errors": 0
  },
  "total_collected": 150,
  "next_run_at": "2026-02-19T10:00:00Z"
}
```

### 16. 수동 수집 트리거
```
POST /collection/trigger
Authorization: Bearer <access_token>

Request:
{
  "categories": ["cs.AI"],       // optional, 설정 기본값 사용
  "keywords": ["attention"],     // optional
  "max_papers": 10               // optional
}

Response 202:
{
  "job_id": "uuid",
  "status": "started",
  "message": "Collection job started"
}
```

### 17. 엔티티 목록 조회
```
GET /graph/entities
Authorization: Bearer <access_token>
Query params:
- type: string (optional, 예: "MODEL", "TECHNIQUE", "DATASET")
- search: string (optional, 이름 검색)
- page: int (default: 1)
- limit: int (default: 50, max: 100)

Response 200:
{
  "entities": [
    {
      "name": "GPT-4",
      "type": "MODEL",
      "mention_count": 15,
      "source_documents": ["uuid1", "uuid2"]
    }
  ],
  "total": 200,
  "page": 1,
  "limit": 50
}
```

### 18. 관계 목록 조회
```
GET /graph/relations
Authorization: Bearer <access_token>
Query params:
- entity: string (optional, 특정 엔티티 기준 관계)
- type: string (optional, 관계 타입 필터)
- page: int (default: 1)
- limit: int (default: 50, max: 200)

Response 200:
{
  "relations": [
    {
      "source": "GPT-4",
      "target": "Transformer",
      "type": "USES_TECHNIQUE",
      "weight": 0.85
    }
  ],
  "total": 500,
  "page": 1,
  "limit": 50
}
```

### 19. 문서별 엔티티 조회
```
GET /documents/{document_id}/entities
Authorization: Bearer <access_token>

Response 200:
{
  "document_id": "uuid",
  "entities": [
    {
      "name": "GPT-4",
      "type": "MODEL",
      "source_chunk_id": "chunk_uuid"
    },
    {
      "name": "Transformer",
      "type": "TECHNIQUE",
      "source_chunk_id": "chunk_uuid"
    }
  ],
  "total": 25
}
```

---

*Last Updated: 2026-02-18*
