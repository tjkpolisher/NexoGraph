# Phase 2-4 로드맵

## Phase 2: Enhanced Q&A (현재)

### 목표
Phase 1의 수동 업로드 기반 Q&A 시스템을 자동화 및 개인화된 시스템으로 발전

### 주요 기능

#### 1. 자동 문서 수집 (arXiv API)
**목적**: AI/ML 최신 논문 자동 수집

**기술 스택**:
- arXiv API (OAI-PMH 또는 RSS)
- APScheduler 또는 Celery (스케줄링)

**구현 범위**:
```
backend/services/
├── arxiv_service.py          # arXiv API 클라이언트
├── collection_scheduler.py   # 수집 스케줄러
└── collection_job.py         # 수집 작업 정의

backend/api/routes/
└── collection.py             # 수집 관리 API
```

**API 엔드포인트**:
- `POST /api/v1/collection/config` - 수집 설정
- `GET /api/v1/collection/status` - 수집 현황
- `POST /api/v1/collection/trigger` - 수동 수집 트리거

#### 2. 대화 히스토리 영구 저장
**목적**: 사용자별 대화 내역 저장 및 검색

**데이터베이스 스키마**:
```sql
-- conversations 테이블
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT,  -- Phase 2.3 인증 후 활용
    title TEXT,
    created_at DATETIME,
    updated_at DATETIME
);

-- messages 테이블
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    role TEXT,  -- 'user' | 'assistant'
    content TEXT,
    sources JSON,
    created_at DATETIME
);
```

#### 3. 기본 사용자 인증
**목적**: 개인화된 경험 제공

**옵션 비교**:
| 방식 | 장점 | 단점 | 추천 |
|------|------|------|------|
| JWT | 확장성, Stateless | 토큰 관리 복잡 | 확장 계획 시 |
| Session | 간단, 취소 용이 | 서버 상태 유지 | MVP 추천 ✅ |

**구현 범위**:
- 이메일/비밀번호 로그인
- 세션 기반 인증 (Redis 또는 메모리)
- 프로필 관리 (이름, 관심 분야)

#### 4. 기본 그래프 시각화
**목적**: 지식 그래프 탐색 인터페이스 제공

**라이브러리 선택**:
| 라이브러리 | 장점 | 단점 |
|-----------|------|------|
| Pyvis | Python 네이티브, Streamlit 호환 | 대규모 그래프 성능 |
| D3.js | 인터랙티브, 커스터마이징 | JavaScript 필요 |
| vis.js | 사용 편리, 좋은 성능 | 커스터마이징 제한 |

**추천**: Pyvis (Streamlit 호환성)

#### 5. 엔티티 추출 표시
**목적**: 문서에서 추출된 엔티티 시각화

**현재 상태**: `documents.py:383`에 TODO 표시
```python
# TODO: Extract entities from LightRAG in Phase 2
entities = []
```

**구현 방안**:
- LightRAG 내부 NetworkX 그래프에서 엔티티 쿼리
- 문서별 엔티티 목록 API
- UI에서 엔티티 태그 표시

---

## Phase 3: Knowledge Discovery

### 목표
지식 그래프를 활용한 통찰 발견 시스템 구축

### 주요 기능

#### 1. Neo4j 마이그레이션
**목적**: 고급 그래프 쿼리 및 분석

**마이그레이션 전략**:
1. LightRAG 데이터 Neo4j로 동기화
2. 읽기 쿼리를 점진적으로 Neo4j로 이전
3. 최종적으로 LightRAG는 인덱싱 전용

**Neo4j 스키마**:
```cypher
// 노드 타입
(:Document {id, title, category, ...})
(:Entity {id, name, type, ...})
(:Chunk {id, content, embedding, ...})

// 관계 타입
(Document)-[:CONTAINS]->(Chunk)
(Chunk)-[:MENTIONS]->(Entity)
(Entity)-[:RELATED_TO]->(Entity)
```

#### 2. 고급 그래프 분석
- 경로 탐색 (두 개념 간 연결 발견)
- 커뮤니티 탐지 (관련 개념 클러스터)
- 중심성 분석 (핵심 개념 식별)

#### 3. Prometheus 모니터링
**현재 상태**: `metrics_service.py`에 InMemoryMetricsCollector 구현됨
**Phase 3 목표**: Prometheus 통합

---

## Phase 4: Multi-Agent 가설 생성 시스템

### 목표
서로 다른 학문 분야 간 숨겨진 연결고리를 발견하고, 창발적 연구 가설을 자동 생성

### 에이전트 구성

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│              (가설 생성 워크플로우 조율)                        │
└─────────┬───────────────────────────────────────────────────┘
          │
    ┌─────┴─────┬─────────────┬─────────────┬─────────────┐
    │           │             │             │             │
┌───▼───┐ ┌────▼────┐ ┌──────▼──────┐ ┌───▼───┐ ┌───────▼───────┐
│ Path  │ │ Analogy │ │ Hypothesis  │ │Domain │ │   Novelty     │
│Explorer│ │Detector │ │ Generator   │ │Experts│ │   Assessor    │
└───────┘ └─────────┘ └─────────────┘ └───────┘ └───────────────┘
```

#### 에이전트 역할
- **Path Explorer**: 지식 그래프에서 두 개념 사이의 경로 탐색
- **Analogy Detector**: 분야 간 숨겨진 유비(Analogy) 발견
- **Hypothesis Generator**: 창발적 연구 가설 자동 생성
- **Domain Experts**: 물리학, 천문학, 심리학 등 분야별 전문가 에이전트
- **Novelty Assessor**: Semantic Scholar API로 참신성 평가

### 기술 스택
- AutoGen/AG2 (Multi-Agent 프레임워크)
- Semantic Scholar API (참신성 평가)
- Neo4j (그래프 쿼리)

### 확장 도메인
Phase 4에서는 다음 도메인으로 확장 예정:
- Scientific AI (Physics-Informed ML, Neural Operators)
- 물리학
- 기계공학 (유체역학, 열역학)
- 천문학 (관측천문학, 천체물리학)
- 심리학 (인지심리학, 의사결정론)
- 교육학 (학습과학, 교육공학)

---

## 타임라인 예상

| Phase | 기간 | 주요 마일스톤 |
|-------|------|--------------|
| Phase 2 | 4-6주 | arXiv 수집, 인증, 시각화 |
| Phase 3 | 6-8주 | Neo4j, 고급 분석, 모니터링 |
| Phase 4 | 8-12주 | Multi-Agent 시스템, 도메인 확장 |

---

*Last Updated: 2026-02-16*
