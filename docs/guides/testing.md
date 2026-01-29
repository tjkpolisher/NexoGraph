# 테스트 가이드

## 테스트 실행

### 기본 명령어

```bash
# 전체 테스트
pytest

# 특정 파일
pytest tests/test_upstage_apis.py

# 커버리지 포함
pytest --cov=backend --cov-report=html

# Verbose 모드 (더 자세한 출력)
pytest -v

# 특정 테스트만 실행
pytest tests/e2e/test_rate_limit_handling.py -v
```

### Rate Limit 테스트 실행

```bash
# Mock API를 사용한 테스트 (기본)
pytest tests/e2e/ -v

# 실제 API 테스트 활성화 (UPSTAGE_API_KEY 필요)
UPSTAGE_API_KEY=your_key pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v

# Mock 테스트만 (실제 API 제외)
pytest tests/e2e/test_rate_limit_handling.py -v

# 느린 테스트 제외
pytest -v -m "not slow"
```

## 테스트 구조

### pytest.ini - 구성 파일
```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py

markers =
    asyncio: async 테스트
    real_api: 실제 API 테스트 (--run-real-api 플래그 필요)
    integration: 통합 테스트
    unit: 단위 테스트
    slow: 느린 테스트 (시간이 오래 걸림)
```

### conftest.py - 공유 Fixtures
테스트 전역 설정 및 공통 fixtures 제공:
- `mock_httpx_client`: 기본 Mock HTTP 클라이언트
- `mock_rate_limited_client`: 429 응답 시뮬레이션
- `mock_always_fail_client`: 항상 실패하는 클라이언트
- `mock_timeout_client`: 타임아웃 시뮬레이션
- `mock_connection_error_client`: 연결 오류 시뮬레이션

### E2E 테스트 디렉토리 구조
```
tests/
├── conftest.py                        # 공유 fixtures 및 설정
├── __init__.py
├── e2e/                               # E2E 테스트
│   ├── __init__.py
│   ├── test_rate_limit_handling.py    # Mock API 기반 테스트
│   └── test_real_api_rate_limits.py   # 실제 API 테스트
├── test_upstage_services.py           # 기존 서비스 테스트
├── test_lightrag_service.py
├── test_qdrant_service.py
└── ...
```

### 테스트 예시

#### Mock API를 사용한 Rate Limit 테스트
```python
# tests/e2e/test_rate_limit_handling.py
@pytest.mark.asyncio
async def test_429_response_raises_rate_limit_error(self, mock_rate_limited_client):
    """429 응답이 올바르게 처리되는지 확인"""
    mock_client, call_count = mock_rate_limited_client

    # Mock 클라이언트는 처음 2회 429 반환, 3회째부터 성공
    assert call_count["value"] == 0
```

#### 실제 API를 사용한 테스트 (선택적)
```python
# tests/e2e/test_real_api_rate_limits.py
@pytest.mark.real_api
async def test_embedding_basic_call(self, api_key):
    """실제 Upstage API 호출 테스트"""
    from backend.services.upstage.embedding import UpstageEmbeddingService

    service = UpstageEmbeddingService(api_key=api_key)
    result = await service.get_embedding("Hello world")

    assert len(result) == 4096
```

## 테스트 작성 가이드

### 1. 비동기 테스트
```python
# @pytest.mark.asyncio 데코레이터 필수
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### 2. Fixtures 활용
```python
# conftest.py에서 정의한 fixture 사용
@pytest.mark.asyncio
async def test_with_mock_client(mock_rate_limited_client):
    mock_client, call_count = mock_rate_limited_client
    # mock_client 사용
```

### 3. Mock 사용
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    with patch('backend.services.upstage.llm.httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "test"}}]}
        )
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

        # 테스트 로직
```

### 4. Circuit Breaker 테스트
```python
@pytest.mark.asyncio
async def test_circuit_breaker_opens():
    """Circuit Breaker가 올바르게 열리는지 확인"""
    from backend.services.circuit_breaker import CircuitBreakerManager

    manager = CircuitBreakerManager()
    breaker = manager.get_breaker("embedding")

    # 5번 실패 시뮬레이션
    for _ in range(5):
        try:
            await breaker.call_async(failing_function)
        except Exception:
            pass

    # Circuit이 열려야 함
    assert manager.get_state("embedding") == "open"
```

### 5. 메트릭 테스트
```python
@pytest.mark.asyncio
async def test_metrics_collection():
    """메트릭이 올바르게 수집되는지 확인"""
    from backend.services.metrics_service import InMemoryMetricsCollector

    metrics = InMemoryMetricsCollector()
    metrics.increment("upstage_embedding_calls", 2)
    metrics.increment("upstage_embedding_rate_limits", 1)

    summary = metrics.get_summary()
    assert summary["upstage_services"]["embedding"]["calls"] == 2
    assert summary["upstage_services"]["embedding"]["rate_limit_hits"] == 1
```

### 6. 테스트 카테고리
- **단위 테스트**: 개별 함수/메서드 테스트
- **통합 테스트**: 여러 컴포넌트 간 상호작용 테스트
- **E2E 테스트**: `tests/e2e/` - Mock 및 실제 API를 통한 전체 플로우 테스트

---

## Rate Limit 테스트 시나리오

### 시나리오 1: Mock API를 통한 429 처리 (권장)

```bash
# 실행 명령어
pytest tests/e2e/test_rate_limit_handling.py -v

# 테스트되는 항목
✓ 429 응답 처리
✓ Retry 로직 (지수 백오프)
✓ Circuit Breaker 상태 전환 (CLOSED → OPEN → HALF_OPEN)
✓ 메트릭 수집
```

**장점**:
- 실제 API 호출 비용 없음
- 빠른 실행 (네트워크 지연 없음)
- 반복 가능한 결과
- CI/CD 파이프라인에 최적

### 시나리오 2: 실제 API를 통한 테스트 (선택적)

```bash
# API 키 설정 후 실행
export UPSTAGE_API_KEY=your_key
pytest tests/e2e/test_real_api_rate_limits.py --run-real-api -v

# 테스트되는 항목
✓ 실제 API 연결성
✓ 실제 Rate Limit 동작
✓ 실제 응답 포맷 검증
✓ 엔드-투-엔드 메트릭 수집
```

**주의사항**:
- API 호출 비용 발생
- 네트워크 지연으로 인한 테스트 시간 증가
- 실제 API 상태에 의존
- 개발 중에는 제한적 사용 권장

---

## 테스트 마커 (Markers)

### 마커 종류
```bash
# 모든 async 테스트 실행
pytest -m asyncio

# 실제 API 테스트만 실행 (--run-real-api 플래그 필요)
pytest -m real_api --run-real-api

# 느린 테스트 제외
pytest -m "not slow"

# 통합 테스트만 실행
pytest -m integration
```

---

## 커버리지 분석

```bash
# 커버리지 생성
pytest --cov=backend --cov-report=html tests/

# HTML 리포트 열기 (Windows)
start htmlcov/index.html

# Rate Limit 관련 코드 커버리지
pytest --cov=backend.services.circuit_breaker \
       --cov=backend.services.metrics_service \
       --cov-report=html tests/e2e/
```

---

## 자주 하는 실수

### ❌ 잘못된 예
```python
# async 함수인데 @pytest.mark.asyncio 없음
async def test_something():
    result = await some_async_call()

# Mock이 AsyncMock이 아님
with patch('module.function') as mock:
    mock.return_value = value  # ← 동기 Mock
```

### ✅ 올바른 예
```python
# async 함수에 마커 추가
@pytest.mark.asyncio
async def test_something():
    result = await some_async_call()

# AsyncMock 사용
with patch('module.function') as mock:
    mock = AsyncMock(return_value=value)
```

---

*Last Updated: 2026-01-29*
*Version: 1.1 (Rate Limit E2E 테스트 추가)*
