# Phase 1 구현 로드맵

## 구현 순서

### Week 1: 기초 인프라
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
```

### Week 2: 핵심 기능
```
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
```

### Week 3: UI 및 테스트
```
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

*Last Updated: 2026-01-15*
