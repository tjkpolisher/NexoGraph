"""
API 연결 테스트 스크립트
conda activate nexograph 후 실행하세요.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_conda_environment():
    """Conda 환경 확인"""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "nexograph":
        print(f"✅ Conda environment: {conda_env}")
        return True
    else:
        print(f"⚠️ Conda environment: {conda_env or 'None'}")
        print("   → conda activate nexograph 실행 필요")
        return False

def test_python_packages():
    """필수 패키지 설치 확인"""
    print(f"\nPython version: {sys.version}")

    required_packages = [
        "fastapi", "uvicorn", "pydantic", 
        "httpx", "qdrant_client", "streamlit",
        "sqlalchemy", "tenacity"
    ]

    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} (not installed)")
            missing.append(pkg)

    if missing:
        print(f"\n→ pip install {' '.join(missing)}")
        return False
    return True

def test_qdrant():
    """Qdrant 연결 테스트"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"✅ Qdrant: Connected (Collections: {len(collections.collections)})")
        return True
    except Exception as e:
        print(f"❌ Qdrant: {e}")
        print("   → docker-compose up -d 실행 필요")
        return False

def test_upstage_api():
    """Upstage API 연결 테스트"""
    import httpx

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        print("❌ UPSTAGE_API_KEY not found in .env")
        return False

    if api_key == "up_xxxxxxxxxxxxxxxxxxxxxxxx":
        print("⚠️ UPSTAGE_API_KEY is placeholder. Please set real key in .env")
        return False

    url = "https://api.upstage.ai/v1/solar/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "solar-pro",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }

    try:
        response = httpx.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            print("✅ Upstage Solar LLM API: Connected")
            return True
        else:
            print(f"❌ Upstage API: {response.status_code} - {response.text[:100]}")
            return False
    except Exception as e:
        print(f"❌ Upstage API: {e}")
        return False

def test_env_file():
    """환경변수 파일 확인"""
    if os.path.exists(".env"):
        print("✅ .env file exists")
        return True
    else:
        print("❌ .env file not found")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Nexograph Environment Test")
    print("=" * 50)

    print("\n[1/5] Conda Environment")
    conda_ok = test_conda_environment()

    print("\n[2/5] Environment File")
    env_ok = test_env_file()

    print("\n[3/5] Python Packages")
    packages_ok = test_python_packages()

    print("\n[4/5] Qdrant Connection")
    qdrant_ok = test_qdrant()

    print("\n[5/5] Upstage API Connection")
    upstage_ok = test_upstage_api()

    print("\n" + "=" * 50)
    all_ok = conda_ok and env_ok and packages_ok and qdrant_ok and upstage_ok
    if all_ok:
        print("✅ All checks passed! Ready for development.")
    else:
        print("⚠️ Some checks failed. Please fix before proceeding.")
    print("=" * 50)
