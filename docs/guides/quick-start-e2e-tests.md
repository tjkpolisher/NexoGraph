# Quick Start - E2E Tests 실행 가이드

## ⚡ 빠른 시작

### 1. Mock API 테스트 실행 (권장)
```bash
# 폴더 이동
cd C:\Users\jktak\NexoGraph

# 테스트 실행
pytest tests/e2e/test_rate_limit_handling.py -v
```

**예상 결과**:
```
tests/e2e/test_rate_limit_handling.py::TestMock429Responses::test_429_response_raises_rate_limit_error PASSED
tests/e2e/test_rate_limit_handling.py::TestMock429Responses::test_429_with_retry_after_header PASSED
...
======================== 19 passed in 2.34s ========================
```

---

## 2. 실제 API 테스트 (선택적)

### Windows
```bash
# API 키 설정
set UPSTAGE_API_KEY=your_actual_api_key

# 테스트 실행
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v
```

### Linux / Mac
```bash
# API 키 설정
export UPSTAGE_API_KEY=your_actual_api_key

# 테스트 실행
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v
```

---

## 3. 특정 테스트만 실행

### Circuit Breaker 테스트만
```bash
pytest tests/e2e/test_rate_limit_handling.py::TestCircuitBreakerStateTransition -v
```

### Metrics 테스트만
```bash
pytest tests/e2e/test_rate_limit_handling.py::TestMetricsCollection -v
```

### 한 개의 테스트만
```bash
pytest "tests/e2e/test_rate_limit_handling.py::TestMock429Responses::test_429_response_raises_rate_limit_error" -v
```

---

## 4. 커버리지 생성

```bash
# 커버리지 리포트 생성
pytest tests/e2e/ --cov=backend.services.circuit_breaker --cov=backend.services.metrics_service --cov-report=html

# HTML 리포트 열기 (Windows)
start htmlcov/index.html

# HTML 리포트 열기 (Linux/Mac)
open htmlcov/index.html
```

---

## 5. 마커를 사용한 필터링

```bash
# 느린 테스트 제외
pytest tests/e2e/ -v -m "not slow"

# Real API 테스트만 (flag 필요)
pytest tests/e2e/ -v -m real_api --run-real-api

# 비동기 테스트만
pytest tests/e2e/ -v -m asyncio
```

---

## 📋 생성된 파일 확인

### 테스트 파일
- ✅ `tests/conftest.py` - 공유 fixtures
- ✅ `tests/e2e/__init__.py` - 패키지 초기화
- ✅ `tests/e2e/test_rate_limit_handling.py` - Mock 테스트 (19개)
- ✅ `tests/e2e/test_real_api_rate_limits.py` - Real API 테스트 (13개)

### 설정 파일
- ✅ `pytest.ini` - Pytest 구성
- ✅ `requirements.txt` - 의존성 추가

### 문서
- ✅ `docs/guides/testing.md` - 테스트 실행 가이드
- ✅ `RATE_LIMIT_E2E_TESTS.md` - 상세 설명
- ✅ `E2E_TESTS_IMPLEMENTATION_COMPLETE.md` - 구현 완료 보고서
- ✅ `QUICK_START_E2E_TESTS.md` - 이 파일

---

## 🎯 테스트 내용

### Mock API 테스트 (test_rate_limit_handling.py) - 19개
| 클래스 | 테스트 항목 |
|--------|-----------|
| TestMock429Responses | 429 응답 처리 (3개) |
| TestRetryLogicValidation | Retry 로직 (4개) |
| TestCircuitBreakerStateTransition | Circuit Breaker 상태 (5개) |
| TestMetricsCollection | 메트릭 수집 (5개) |
| TestIntegratedRateLimitHandling | 통합 시나리오 (2개) |

### 실제 API 테스트 (test_real_api_rate_limits.py) - 13개
| 클래스 | 테스트 항목 |
|--------|-----------|
| TestRealUpstageEmbeddingAPI | Embedding API (4개) |
| TestRealUpstageLLMAPI | LLM API (3개) |
| TestRealUpstageDocumentParserAPI | Parser API (2개) |
| TestRealAPICircuitBreakerIntegration | Circuit 통합 (2개) |
| TestRealAPIWithMetrics | 메트릭 통합 (1개) |

---

## 🔍 테스트 검증 항목

### ✓ Rate Limit 처리 (429)
- 429 상태 코드 인식
- Retry-After 헤더 추출
- 연속 429 응답 처리

### ✓ Retry 로직
- 실패 후 자동 재시도
- 지수 백오프 (1s, 2s, 4s, 8s...)
- 최대 재시도 초과 처리

### ✓ Circuit Breaker
- CLOSED → OPEN 전이
- OPEN 상태에서 요청 차단
- HALF_OPEN 상태 전이
- 서비스별 독립 관리

### ✓ 메트릭 수집
- 요청/응답 시간 추적
- Rate limit hits 카운팅
- Retry 시도 횟수
- 에러율 계산

---

## 🐛 문제 해결

### Q: "pytest: command not found"
**A**: Python이 설치되지 않았거나 PATH에 없음
```bash
# Python 확인
python --version

# pip 설치 확인
pip --version

# pytest 설치
pip install pytest pytest-asyncio pytest-mock
```

### Q: "ModuleNotFoundError: No module named 'backend'"
**A**: 프로젝트 루트에서 실행
```bash
# ✓ 올바른 위치
cd C:\Users\jktak\NexoGraph
pytest tests/e2e/

# ✗ 잘못된 위치
cd tests
pytest e2e/
```

### Q: "UPSTAGE_API_KEY not set"
**A**: Real API 테스트는 API 키 필요
```bash
# 선택 1: Mock 테스트만 사용
pytest tests/e2e/test_rate_limit_handling.py -v

# 선택 2: API 키 설정 후 실제 테스트
set UPSTAGE_API_KEY=your_key  # Windows
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v
```

### Q: "Tests skipped with reason: Real API tests disabled"
**A**: --run-real-api 플래그 추가
```bash
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v
```

### Q: async 테스트가 실행되지 않음
**A**: @pytest.mark.asyncio 마커 확인 (pytest.ini의 asyncio_mode = auto)
```bash
# 마커가 없으면 실행 안 됨
# tests/e2e/test_rate_limit_handling.py 파일 확인
```

---

## 📊 실행 결과 예시

### Mock 테스트 결과
```
$ pytest tests/e2e/test_rate_limit_handling.py -v

tests/e2e/test_rate_limit_handling.py::TestMock429Responses::test_429_response_raises_rate_limit_error PASSED
tests/e2e/test_rate_limit_handling.py::TestMock429Responses::test_429_with_retry_after_header PASSED
tests/e2e/test_rate_limit_handling.py::TestMock429Responses::test_consecutive_429_responses PASSED
tests/e2e/test_rate_limit_handling.py::TestRetryLogicValidation::test_retry_succeeds_after_rate_limits PASSED
tests/e2e/test_rate_limit_handling.py::TestRetryLogicValidation::test_max_retries_exceeded PASSED
...
======================== 19 passed in 2.34s ========================
```

### 커버리지 결과
```
$ pytest tests/e2e/ --cov=backend.services.circuit_breaker --cov-report=term-missing

backend/services/circuit_breaker.py   85    7    92%

======================== 19 passed in 2.45s ========================
```

---

## 🎓 학습 예시

### 1. 특정 Fixture 사용
```python
# conftest.py의 mock_rate_limited_client 사용
@pytest.mark.asyncio
async def test_example(mock_rate_limited_client):
    mock_client, call_count = mock_rate_limited_client
    # mock_client 사용
    # call_count["value"] 확인
```

### 2. Circuit Breaker 테스트
```python
from backend.services.circuit_breaker import CircuitBreakerManager

@pytest.mark.asyncio
async def test_circuit():
    manager = CircuitBreakerManager()
    breaker = manager.get_breaker("embedding")

    # Circuit 상태 확인
    assert manager.get_state("embedding") == "closed"
```

### 3. 메트릭 검증
```python
from backend.services.metrics_service import InMemoryMetricsCollector

@pytest.mark.asyncio
async def test_metrics():
    metrics = InMemoryMetricsCollector()
    metrics.increment("upstage_embedding_calls", 5)

    summary = metrics.get_summary()
    assert summary["upstage_services"]["embedding"]["calls"] == 5
```

---

## 📞 추가 정보

- **상세 가이드**: `RATE_LIMIT_E2E_TESTS.md` 참고
- **구현 보고서**: `E2E_TESTS_IMPLEMENTATION_COMPLETE.md` 참고
- **테스트 가이드**: `docs/guides/testing.md` 참고

---

*Quick Start Guide v1.0*
*Last Updated: 2026-01-29*
