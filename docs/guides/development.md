# 개발 환경 가이드

## 환경 변수

### .env 파일 구조
```env
# === Upstage API ===
UPSTAGE_API_KEY=up_xxxxxxxxxxxxxxxxxxxxxxxx

# === Qdrant ===
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=nexograph_documents

# === Application ===
APP_ENV=development  # development | production
APP_DEBUG=true
APP_VERSION=0.1.0

# === Database ===
DATABASE_URL=sqlite:///./data/db/nexograph.db

# === LightRAG ===
LIGHTRAG_WORKING_DIR=./data/lightrag

# === Optional: Backup LLM ===
# OPENAI_API_KEY=sk-xxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

### config.py 구현 참고
```python
import os
import warnings
from pydantic_settings import BaseSettings
from functools import lru_cache

# Conda 환경 확인 (개발 시 도움)
def check_conda_env():
    expected_env = "nexograph"
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != expected_env:
        warnings.warn(
            f"Expected conda env '{expected_env}', but got '{current_env or 'None'}'. "
            f"Run: conda activate {expected_env}",
            UserWarning
        )

check_conda_env()

class Settings(BaseSettings):
    # Upstage
    upstage_api_key: str
    upstage_base_url: str = "https://api.upstage.ai/v1"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "nexograph_documents"

    # App
    app_env: str = "development"
    app_debug: bool = True
    app_version: str = "0.1.0"

    # Database
    database_url: str = "sqlite:///./data/db/nexograph.db"

    # LightRAG
    lightrag_working_dir: str = "./data/lightrag"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

---

## 자주 사용하는 명령어

```bash
# === Conda 환경 ===
conda activate nexograph          # 환경 활성화 (필수!)
conda deactivate                  # 환경 비활성화
conda env list                    # 환경 목록 확인
conda env export > environment.yml  # 환경 내보내기

# === 개발 서버 실행 ===
# 반드시 conda activate nexograph 후 실행
uvicorn backend.main:app --reload --port 8000

# === Streamlit 실행 ===
streamlit run frontend/streamlit_app.py

# === Docker (Qdrant) ===
docker-compose up -d      # 시작
docker-compose down       # 종료
docker-compose logs -f    # 로그

# === 테스트 ===
pytest -v
pytest --cov=backend

# === 포맷팅 ===
black backend/
isort backend/

# === 의존성 관리 ===
pip install -r requirements.txt        # 패키지 설치
pip freeze > requirements.txt          # 현재 패키지 저장
conda env create -f environment.yml    # 환경 재현 (다른 PC)
```

---

## 주의사항 및 제약조건

### 0. Conda 환경 (가장 중요!)
- **모든 명령어 실행 전** `conda activate nexograph` 필수
- 터미널을 새로 열 때마다 환경 활성화 필요
- VS Code 사용 시: Python 인터프리터를 conda nexograph로 설정
- 패키지 설치는 환경 활성화 후 `pip install` 사용

### 1. API 키 보안
- **절대** 코드에 API 키 하드코딩 금지
- `.env` 파일은 `.gitignore`에 포함
- 커밋 전 `git diff --staged`로 확인

### 2. Upstage API 제한
- Rate Limit 존재 (정확한 수치는 콘솔에서 확인)
- 크레딧 소모 모니터링 필요
- Document Parse: 파일당 최대 크기 제한 있음 (동기 요청 시 최대 100페이지, 비동기 요청 시 최대 1000페이지까지 가능)

### 3. LightRAG 주의점
- 버전별 API 차이 큼 - 공식 문서 반드시 확인
- 인메모리 모드는 서버 재시작 시 데이터 손실 (working_dir 필요)
- 대용량 문서 처리 시 메모리 사용량 주의

### 4. Qdrant 주의점
- Docker 볼륨 마운트로 데이터 영속성 확보
- 컬렉션 삭제 시 복구 불가

### 5. 비동기 처리
- FastAPI는 async 함수 권장
- 동기 라이브러리 사용 시 `run_in_executor` 활용
- DB 세션은 요청별로 생성/종료

---

*Last Updated: 2026-01-15*
