# 구현 로드맵

## Phase 1 (완료)

### Week 1: 기초 인프라 ✅
```
Week 1:
├── Day 1-2: 프로젝트 기본 구조 ✅
│   ├── FastAPI 앱 초기화 (main.py)
│   ├── 설정 관리 (config.py)
│   └── Health check 엔드포인트
│
├── Day 3-4: 인프라 연결 ✅
│   ├── Qdrant 서비스 (qdrant_service.py)
│   └── 컬렉션 생성 스크립트
│
└── Day 5: Upstage API 연동 ✅
    ├── Document Parse 서비스
    ├── Embedding 서비스
    └── LLM 서비스
```

### Week 2: 핵심 기능 ✅
```
Week 2:
├── Day 1-2: LightRAG 통합 ✅
│   ├── LightRAG 서비스 래퍼
│   └── 커스텀 LLM/Embedding 연결
│
├── Day 3-4: 문서 처리 파이프라인 ✅
│   ├── 문서 업로드 API
│   ├── 파싱 → 청킹 → 인덱싱 플로우
│   └── 메타데이터 저장 (SQLite)
│
└── Day 5: 채팅 API ✅
    ├── Q&A 엔드포인트
    └── 출처 추적 로직
```

### Week 3: UI 및 테스트 ✅
```
Week 3:
├── Day 1-2: Streamlit UI ✅
│   ├── 문서 업로드 UI
│   └── 채팅 인터페이스
│
├── Day 3-4: 통합 테스트 ✅
│   ├── API 테스트
│   └── E2E 테스트
│
└── Day 5: 추가 구현 ✅
    ├── Circuit Breaker 패턴
    ├── Metrics 수집 서비스
    └── 배치 임베딩 처리
```

### Phase 1 최종 산출물
- **Backend**: ~4,700+ 라인 (프로덕션 품질)
- **Frontend**: 525 라인 (Streamlit)
- **테스트**: 42개 (30 passed, 12 API 테스트 skipped)
- **문서**: CLAUDE.md, README.md, guides/

---

## Phase 2 로드맵 (현재)

### 우선순위 1: 자동 문서 수집 (arXiv API)
```
├── arXiv API 클라이언트 구현
│   ├── 검색 쿼리 (카테고리, 키워드)
│   ├── 결과 파싱 (제목, 저자, 초록, PDF URL)
│   └── Rate limit 처리
│
├── 자동 수집 스케줄러
│   ├── 주기적 수집 (APScheduler 또는 Celery)
│   ├── 중복 체크 (arXiv ID 기반)
│   └── 배치 처리
│
└── UI 확장
    ├── 수집 설정 페이지
    └── 수집 현황 대시보드
```

### 우선순위 2: 대화 히스토리 영구 저장
```
├── 데이터베이스 스키마 확장
│   ├── conversations 테이블
│   └── messages 테이블
│
├── API 엔드포인트
│   ├── GET /conversations
│   ├── GET /conversations/{id}/messages
│   └── DELETE /conversations/{id}
│
└── UI 확장
    ├── 대화 목록 사이드바
    └── 대화 이어하기 기능
```

### 우선순위 3: 기본 사용자 인증
```
├── 인증 방식 선택
│   ├── 옵션 A: JWT 토큰 기반
│   └── 옵션 B: 세션 기반 (간단)
│
├── 사용자 관리
│   ├── 회원가입 / 로그인
│   └── 프로필 관리
│
└── 권한 관리
    ├── 문서 소유권
    └── 공유 설정
```

### 우선순위 4: 그래프 시각화
```
├── 시각화 라이브러리 선택
│   ├── 옵션 A: Pyvis (Python 네이티브)
│   └── 옵션 B: D3.js (인터랙티브)
│
├── 시각화 API
│   ├── GET /graph/nodes
│   └── GET /graph/relations
│
└── UI 컴포넌트
    ├── 그래프 뷰어
    └── 노드 상세 정보 패널
```

### 우선순위 5: LightRAG 엔티티 추출 표시
```
├── 엔티티 추출 API 구현
│   ├── LightRAG 내부 그래프 쿼리
│   └── 엔티티 정규화
│
└── UI 확장
    ├── 문서 상세 페이지에 엔티티 표시
    └── 엔티티 검색 기능
```

---

## Phase 3-4 예정 기능

- Neo4j 그래프 데이터베이스 마이그레이션
- Multi-Agent 가설 생성 시스템 (AutoGen/AG2)
- Semantic Scholar API 연동 (참신성 평가)
- 고급 그래프 분석
- 다학문 도메인 확장

자세한 내용: `docs/reference/future-roadmap.md` 참조

---

*Last Updated: 2026-02-16*
