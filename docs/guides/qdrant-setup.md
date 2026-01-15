# Qdrant 설정 가이드

## 컬렉션 생성 스크립트

```python
# scripts/setup_qdrant.py

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def setup_collection():
    client = QdrantClient(host="localhost", port=6333)

    # 컬렉션 생성 (없으면)
    collections = [c.name for c in client.get_collections().collections]

    if "nexograph_documents" not in collections:
        client.create_collection(
            collection_name="nexograph_documents",
            vectors_config=VectorParams(
                size=4096,  # Upstage Embedding 차원
                distance=Distance.COSINE
            )
        )
        print("✅ Collection 'nexograph_documents' created")
    else:
        print("ℹ️ Collection 'nexograph_documents' already exists")

if __name__ == "__main__":
    setup_collection()
```

## Docker Compose 실행

```bash
# Qdrant 시작
docker-compose up -d

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs qdrant

# 대시보드 접속
# http://localhost:6333/dashboard
```

## 주요 설정

### 벡터 차원
- **Size**: 4096 (Upstage solar-embedding-1-large)
- **Distance**: COSINE (코사인 유사도)

### 컬렉션 이름
- **기본값**: `nexograph_documents`
- `.env` 파일의 `QDRANT_COLLECTION_NAME`으로 변경 가능

### 볼륨 마운트
- 데이터 영속성을 위해 Docker 볼륨 사용
- `docker-compose.yml`에서 볼륨 경로 확인

---

*Last Updated: 2026-01-15*
