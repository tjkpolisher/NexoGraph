# Rate Limit E2E Tests - Implementation Summary

## 개요

Rate Limit 처리 강화를 위한 **E2E 테스트 스위트**가 구현되었습니다.

### 구현 파일

| 파일 | 설명 |
|------|------|
| `pytest.ini` | Pytest 설정 (마커, asyncio 모드) |
| `tests/conftest.py` | 공유 fixtures 및 pytest 훅 |
| `tests/e2e/__init__.py` | E2E 테스트 패키지 |
| `tests/e2e/test_rate_limit_handling.py` | Mock API 기반 테스트 (권장) |
| `tests/e2e/test_real_api_rate_limits.py` | 실제 API 테스트 (선택적) |
| `requirements.txt` | pytest-mock >= 3.12.0 추가 |
| `docs/guides/testing.md` | 테스트 실행 가이드 업데이트 |

---

## 주요 기능

### 1. Mock API 테스트 (권장)
**파일**: `tests/e2e/test_rate_limit_handling.py`

#### 테스트 클래스 및 주요 테스트:
- **TestMock429Responses**
  - `test_429_response_raises_rate_limit_error()`: 429 응답 처리
  - `test_429_with_retry_after_header()`: Retry-After 헤더 추출
  - `test_consecutive_429_responses()`: 연속 429 응답 처리

- **TestRetryLogicValidation**
  - `test_retry_succeeds_after_rate_limits()`: Retry 로직 검증
  - `test_max_retries_exceeded()`: 최대 재시도 초과 처리
  - `test_retry_with_exponential_backoff()`: 지수 백오프 타이밍
  - `test_non_rate_limit_errors_not_retried()`: 비-429 에러 미재시도

- **TestCircuitBreakerStateTransition**
  - `test_circuit_starts_closed()`: 초기 상태 검증
  - `test_circuit_opens_after_threshold()`: 실패 임계값 도달 시 OPEN
  - `test_open_circuit_raises_error()`: OPEN 상태 에러 발생
  - `test_circuit_half_open_after_recovery_timeout()`: HALF_OPEN 전이
  - `test_different_services_independent_circuits()`: 서비스별 독립 Circuit

- **TestMetricsCollection**
  - `test_metrics_increment_on_rate_limit()`: 메트릭 증가 검증
  - `test_metrics_record_request()`: 요청 기록 테스트
  - `test_metrics_endpoint_stats()`: 엔드포인트 메트릭 추적
  - `test_metrics_per_service_stats()`: 서비스별 통계
  - `test_metrics_error_rate_edge_cases()`: 에러율 계산 엣지 케이스

- **TestIntegratedRateLimitHandling**
  - `test_full_flow_rate_limit_retry_recovery()`: 전체 흐름 통합 테스트
  - `test_circuit_breaker_prevents_cascading_failures()`: Circuit 연쇄 장애 차단

#### 실행 명령어:
```bash
# 모든 Mock API 테스트 실행 (권장)
pytest tests/e2e/test_rate_limit_handling.py -v

# 특정 테스트 클래스만 실행
pytest tests/e2e/test_rate_limit_handling.py::TestCircuitBreakerStateTransition -v

# 특정 테스트만 실행
pytest tests/e2e/test_rate_limit_handling.py::TestMetricsCollection::test_metrics_increment_on_rate_limit -v
```

### 2. 실제 API 테스트 (선택적)
**파일**: `tests/e2e/test_real_api_rate_limits.py`

#### 테스트 클래스 및 주요 테스트:
- **TestRealUpstageEmbeddingAPI**
  - `test_embedding_basic_call()`: 기본 임베딩 호출
  - `test_embedding_with_custom_text()`: 긴 텍스트 임베딩
  - `test_embedding_query_vs_passage_types()`: Query vs Passage 타입
  - `test_embedding_rate_limit_handling()`: Rate limit 처리

- **TestRealUpstageLLMAPI**
  - `test_llm_basic_completion()`: 기본 LLM 완성
  - `test_llm_with_system_prompt()`: 시스템 프롬프트
  - `test_llm_rate_limit_handling()`: Rate limit 처리

- **TestRealUpstageDocumentParserAPI**
  - `test_document_parser_basic()`: 기본 문서 파싱
  - `test_document_parser_rate_limit_handling()`: Parser rate limit

- **TestRealAPICircuitBreakerIntegration**
  - `test_circuit_breaker_with_real_embedding_api()`: Circuit Breaker 통합
  - `test_metrics_collection_with_real_api()`: 메트릭 수집 검증

#### 실행 명령어:
```bash
# API 키 설정 (Windows)
set UPSTAGE_API_KEY=your_api_key

# 또는 (Linux/Mac)
export UPSTAGE_API_KEY=your_api_key

# 실제 API 테스트 실행
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v

# 느린 테스트 제외
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v -m "not slow"

# 특정 API 테스트만
pytest tests/e2e/test_real_api_rate_limits.py::TestRealUpstageEmbeddingAPI --run-real-api -v
```

---

## 공유 Fixtures (conftest.py)

### Mock 클라이언트 Fixtures

```python
# 기본 성공 응답
mock_httpx_client

# 429 rate limit 응답 (처음 2회) → 성공
mock_rate_limited_client  # Returns (mock_client, call_count_dict)

# 항상 실패 (429)
mock_always_fail_client

# 타임아웃 시뮬레이션
mock_timeout_client

# 연결 오류 시뮬레이션
mock_connection_error_client

# Document Parser 성공 응답
mock_successful_parser_response

# Document Parser rate limit
mock_rate_limited_parser_response
```

### 사용 예시:
```python
@pytest.mark.asyncio
async def test_something(mock_rate_limited_client):
    mock_client, call_count = mock_rate_limited_client
    # mock_client 사용
```

---

## pytest 설정 (pytest.ini)

### Markers (마커)
```bash
# async 테스트
@pytest.mark.asyncio

# 실제 API 테스트 (--run-real-api 플래그 필요)
@pytest.mark.real_api

# 통합 테스트
@pytest.mark.integration

# 단위 테스트
@pytest.mark.unit

# 느린 테스트 (60초 이상)
@pytest.mark.slow
```

### 마커를 사용한 필터링:
```bash
# 느린 테스트 제외
pytest -v -m "not slow"

# 실제 API 테스트만
pytest -v -m real_api --run-real-api

# 통합 테스트만
pytest -v -m integration
```

### Command Line Options
```bash
# 실제 API 테스트 활성화
--run-real-api
```

---

## 테스트 커버리지

### 커버되는 시나리오

| 시나리오 | 테스트 클래스 | 테스트 메서드 |
|----------|---------------|--------------|
| **Mock 429 처리** | TestMock429Responses | 3개 |
| **Retry 로직** | TestRetryLogicValidation | 4개 |
| **Circuit Breaker** | TestCircuitBreakerStateTransition | 5개 |
| **메트릭 수집** | TestMetricsCollection | 5개 |
| **통합 테스트** | TestIntegratedRateLimitHandling | 2개 |
| **실제 API - Embedding** | TestRealUpstageEmbeddingAPI | 4개 |
| **실제 API - LLM** | TestRealUpstageLLMAPI | 3개 |
| **실제 API - Parser** | TestRealUpstageDocumentParserAPI | 2개 |
| **실제 API - Circuit** | TestRealAPICircuitBreakerIntegration | 2개 |

**총 테스트 수**: Mock 19개 + Real 13개 = 32개

---

## 실행 예시

### 시나리오 1: Mock 테스트 (CI/CD 권장)
```bash
# 전체 Mock 테스트 실행 (빠름, 네트워크 불필요)
pytest tests/e2e/test_rate_limit_handling.py -v

# 결과 예시
tests/e2e/test_rate_limit_handling.py::TestMock429Responses::test_429_response_raises_rate_limit_error PASSED
tests/e2e/test_rate_limit_handling.py::TestRetryLogicValidation::test_retry_succeeds_after_rate_limits PASSED
tests/e2e/test_rate_limit_handling.py::TestCircuitBreakerStateTransition::test_circuit_starts_closed PASSED
...
```

### 시나리오 2: 전체 테스트 (Mock + Real, API 키 필요)
```bash
# API 키 설정
export UPSTAGE_API_KEY=your_key

# Mock + Real 테스트 실행
pytest tests/e2e/ --run-real-api -v

# Mock만 실행 (Real API 제외)
pytest tests/e2e/test_rate_limit_handling.py -v
```

### 시나리오 3: 커버리지 분석
```bash
# Rate Limit 관련 커버리지 생성
pytest tests/e2e/ \
  --cov=backend.services.circuit_breaker \
  --cov=backend.services.metrics_service \
  --cov-report=html \
  --cov-report=term-missing

# HTML 리포트 확인
start htmlcov/index.html
```

---

## 주요 설계 결정

### 1. Mock vs Real API 분리
- **Mock 테스트** (기본): 빠르고 반복 가능, CI/CD 친화적
- **Real API 테스트** (선택적): 실제 동작 검증, 수동 실행

### 2. pytest-asyncio 자동 모드
```ini
asyncio_mode = auto
```
- async 함수 자동 감지
- `@pytest.mark.asyncio` 필수 (명시적)

### 3. Custom Marker for Real API
```bash
@pytest.mark.real_api  # --run-real-api 플래그 필요
```
- CI/CD에서 기본적으로 skip
- 의도적 실행만 가능

### 4. 구조화된 Fixtures
- `conftest.py`에 모든 공유 fixtures 중앙화
- 테스트별 의존성 주입으로 테스트 격리
- 상태 관리 (call_count, metrics)

---

## 문제 해결 (Troubleshooting)

### Q: "UPSTAGE_API_KEY not set" 에러
**A**: Real API 테스트를 실행하려면 환경변수 설정 필요
```bash
# Windows
set UPSTAGE_API_KEY=your_key
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api

# Linux/Mac
export UPSTAGE_API_KEY=your_key
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api
```

### Q: "Module not found" 에러
**A**: tests 디렉토리 경로 확인
```bash
# 올바른 실행 위치
cd C:\Users\jktak\NexoGraph
pytest tests/e2e/ -v
```

### Q: 타임아웃 에러
**A**: pytest.ini의 timeout 설정 확인
```ini
[pytest]
timeout = 60  # 초 단위
```

### Q: Async 테스트가 skip됨
**A**: `@pytest.mark.asyncio` 마커 확인
```python
# 필수
@pytest.mark.asyncio
async def test_async():
    pass
```

---

## 다음 단계

### Phase 2 개선 사항
1. **Prometheus 통합**: `InMemoryMetricsCollector` → Prometheus
2. **GraphQL 쿼리 테스트**: LightRAG 그래프 검색 테스트
3. **부하 테스트**: Locust를 사용한 동시성 테스트
4. **성능 벤치마크**: Rate limit 처리의 성능 측정

---

## 참고 자료

- **pytest 공식 문서**: https://docs.pytest.org/
- **pytest-asyncio**: https://pytest-asyncio.readthedocs.io/
- **Upstage API**: https://developers.upstage.ai/docs/
- **circuitbreaker 라이브러리**: https://github.com/fabfuel/python-circuitbreaker

---

*Created: 2026-01-29*
*Version: 1.0 (Initial E2E Test Suite)*
