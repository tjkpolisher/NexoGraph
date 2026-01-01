"""Quick test to verify all imports work correctly."""

import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print()

try:
    print("Testing imports...")

    # Test config
    from backend.config import get_settings
    print("✓ backend.config imported successfully")

    settings = get_settings()
    print(f"✓ Settings loaded: version {settings.app_version}")

    # Test dependencies
    from backend.api.dependencies import get_settings as dep_get_settings
    print("✓ backend.api.dependencies imported successfully")

    # Test health route
    from backend.api.routes.health import router
    print("✓ backend.api.routes.health imported successfully")

    # Test main app
    from backend.main import app
    print("✓ backend.main imported successfully")
    print(f"✓ FastAPI app created: {app.title} v{app.version}")

    print()
    print("=" * 60)
    print("All imports successful!")
    print("=" * 60)
    print()
    print("To start the server manually, run:")
    print("  conda activate nexograph")
    print("  uvicorn backend.main:app --reload --port 8000")
    print()
    print("Then test the health endpoint:")
    print("  curl http://localhost:8000/api/v1/health")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
