# 테스트 가이드

## 테스트 실행

```bash
# 전체 테스트
pytest

# 특정 파일
pytest tests/test_upstage_apis.py

# 커버리지 포함
pytest --cov=backend --cov-report=html
```

## 테스트 구조

### conftest.py - 공유 Fixtures
```python
# tests/conftest.py
import pytest
from httpx import AsyncClient
from backend.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
```

### 테스트 예시
```python
# tests/test_api_endpoints.py
@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## 테스트 작성 가이드

### 1. 비동기 테스트
- `@pytest.mark.asyncio` 데코레이터 사용
- async 함수로 작성
- await 키워드로 비동기 호출

### 2. Fixtures 활용
- 공통 설정은 conftest.py에 fixture로 정의
- 테스트 함수 매개변수로 fixture 주입

### 3. Mock 사용
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    with patch('backend.services.upstage.llm.httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "test"}}]}
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

        # 테스트 로직
```

### 4. 테스트 카테고리
- **단위 테스트**: 개별 함수/메서드 테스트
- **통합 테스트**: 여러 컴포넌트 간 상호작용 테스트
- **E2E 테스트**: 전체 플로우 테스트

---

*Last Updated: 2026-01-15*
