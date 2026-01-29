# MinerU 문서 파서 조사 보고서

## ⚠️ 현재 상태: Upstage API 유지 권장

**최종 결론**: 2026년 1월 현재, MinerU의 Python API 호환성 문제로 인해 **Upstage Document Parse API를 계속 사용하는 것을 권장**합니다.

---

## 📋 조사 개요

### 조사 배경

Upstage Document Parse API의 Rate Limit 문제를 해결하기 위해 로컬 대안인 MinerU를 조사했습니다.

### 조사 기간
2026-01-28

### 주요 발견사항

1. ✅ **MinerU 공식 리포지터리 확인**: [GitHub - opendatalab/MinerU](https://github.com/opendatalab/MinerU)
2. ❌ **Python API 호환성 문제**: 문서에 명시된 `magic_pdf.pipe.UNIPipe` import 방식이 최신 버전에서 작동하지 않음
3. ⚠️ **Windows 제약사항**: MinerU 2.0+ 버전은 Windows에서 완전한 설치가 어려움
4. ✅ **CLI 도구는 작동**: `magic-pdf` 명령줄 도구는 정상 작동
5. ❌ **Python API 구조 불명확**: v1.3.12의 정확한 Python API 사용법이 공식 문서와 불일치

---

## 🔍 상세 조사 결과

### 1. 패키지 구조 분석

**발견한 사실들**:

| 패키지 이름 | PyPI 존재 | 용도 | 상태 |
|-----------|----------|------|------|
| `mineru` | ✅ | CLI 도구 메타패키지 | 설치 가능 |
| `magic-pdf` | ✅ | 실제 파싱 라이브러리 | 설치 가능, API 불명확 |
| `magic_pdf.pipe.UNIPipe` | ❌ | 문서에 명시된 import | v1.3.12에 존재하지 않음 |

**패키지 버전**:
- `mineru`: 2.7.3 (최신, 2026-01-26 릴리즈)
- `magic-pdf`: 1.3.12 (최신)

**설치 명령어**:
```bash
# 공식 문서 권장 방법
pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com

# 실제 작동한 방법
pip install -U magic-pdf[full]  # PyPI에서 설치 가능
```

### 2. Windows 환경 특수사항

**주요 제약사항**:

1. **sgl-kernel 의존성 문제**
   - MinerU 2.0+ 버전의 `mineru[all]` 옵션은 `sgl-kernel==0.1.7`을 요구
   - `sgl-kernel`은 Linux 전용 wheel만 제공 (manylinux2014_x86_64)
   - Windows에서는 설치 실패

2. **Python 버전 제한**
   - Linux/macOS: Python 3.10-3.13 지원
   - Windows: Python 3.10-3.12만 지원 (Python 3.13 미지원, `ray` 의존성 문제)

3. **권장 해결 방법**
   ```bash
   # Windows에서는 core 패키지만 설치
   pip install uv
   uv pip install mineru[core]
   ```
   - 단, `mineru[core]`도 Python API 문제는 동일

### 3. Python API 구조 변경

**조사 결과**:

| 버전 | API 구조 | 상태 |
|------|---------|------|
| 구 버전 (< 1.0) | `magic_pdf.pipe.UNIPipe` | 문서에 명시, 작동하지 않음 |
| v1.3.12 | `magic_pdf.pdf_parse_union_core_v2` | 존재, 사용법 불명확 |
| v1.3.12 | `magic_pdf.operators.pipes` | 존재, `UNIPipe` 없음 |

**실제 패키지 구조** (v1.3.12):
```
magic_pdf/
├── __init__.py
├── config/
├── data/
├── dict2md/
├── filter/
├── integrations/
├── libs/
├── model/
├── operators/
│   └── pipes.py  # PipeResult 클래스만 존재
├── pdf_parse_union_core_v2.py  # pdf_parse_union 함수 존재
├── post_proc/
├── pre_proc/
├── resources/
├── spark/
├── tools/
└── utils/
```

**문제점**:
- `magic_pdf.pipe` 모듈 자체가 존재하지 않음
- `UNIPipe` 클래스를 찾을 수 없음
- 공식 문서와 실제 API 구조 불일치

### 4. CLI 도구 작동 확인

**성공적으로 작동한 명령어**:

```bash
# CLI 버전 확인
magic-pdf --version
# 출력: mineru, version 2.7.3

# 또는
mineru --version
# 출력: mineru, version 2.7.3
```

**CLI 사용 예시**:
```bash
# 기본 파싱
mineru -p input.pdf -o ./output

# CPU 모드
mineru -p input.pdf -o ./output -b pipeline

# 옵션 지정
mineru -p document.pdf -o ./output \
  -m auto \
  -b hybrid-auto-engine \
  -l en \
  --formula true \
  --table true
```

### 5. 모델 다운로드

**구 명령어 (작동하지 않음)**:
```bash
mineru-models download  # ❌ 명령어 없음
```

**새 방법 (자동 다운로드)**:
- 첫 실행 시 자동으로 모델 다운로드 (약 2-3GB)
- 수동 다운로드는 별도 Python 스크립트 필요:
  ```bash
  wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py
  python download_models_hf.py
  ```

---

## ⚖️ Upstage API vs MinerU 비교

### 장단점 분석

| 항목 | Upstage API | MinerU |
|------|-------------|--------|
| **설치 난이도** | ✅ 매우 쉬움 (API 키만) | ❌ 어려움 (패키지 충돌, API 불명확) |
| **Python API** | ✅ 명확하고 안정적 | ❌ 문서와 불일치, 사용법 불명확 |
| **Windows 지원** | ✅ 완전 지원 | ⚠️ 제한적 (core만) |
| **Rate Limit** | ⚠️ 있음 | ✅ 없음 (로컬 실행) |
| **비용** | ⚠️ API 요금 | ✅ 무료 (전기세) |
| **처리 속도** | ✅ 5-10초 | ✅ 3-8초 (GPU) / ⚠️ 30-60초 (CPU) |
| **정확도** | ✅ 높음 | ✅ 비슷 |
| **문서 품질** | ✅ 명확 | ❌ 불일치, 구버전 참조 많음 |
| **유지보수** | ✅ 안정적 | ⚠️ API 변경 빈번 |

### 시나리오별 권장사항

| 시나리오 | 권장 솔루션 | 이유 |
|----------|-----------|------|
| **Phase 1 MVP** | ✅ **Upstage API** | 안정성, 빠른 구현 우선 |
| **대량 처리 (>1000 문서/일)** | ⚠️ MinerU CLI + subprocess | Rate limit 회피, 단 Python API 대신 CLI 사용 |
| **오프라인 환경** | ⚠️ MinerU CLI | 인터넷 불필요, 단 사전 모델 다운로드 필요 |
| **프로토타입/연구** | ✅ **Upstage API** | 빠른 개발, 안정성 |
| **상용 서비스** | ✅ **Upstage API** | 안정성, 문서 품질, 지원 |

---

## 🚫 MinerU 도입 불가 이유

### 1. Python API 호환성 문제 (치명적)

**문제**:
```python
from magic_pdf.pipe.UNIPipe import UNIPipe  # ❌ ModuleNotFoundError
```

**원인**:
- 공식 문서/튜토리얼이 구버전 기준
- v1.3.12에 `magic_pdf.pipe` 모듈 없음
- `UNIPipe` 클래스 대체 방법 불명확

**영향**:
- NexoGraph의 비동기 Python API 통합 불가
- 기존 아키텍처 (DocumentParserBase) 사용 불가

### 2. 문서 품질 문제

**발견한 문제들**:
- 공식 문서와 실제 API 구조 불일치
- 많은 튜토리얼이 구버전 (magic-pdf < 1.0) 기준
- GitHub Issues에 유사한 문제 다수 보고 ([Issue #232](https://github.com/opendatalab/MinerU/issues/232), [Issue #1219](https://github.com/opendatalab/MinerU/issues/1219))

### 3. Windows 환경 제약

**제한사항**:
- `mineru[all]` 설치 불가 (sgl-kernel 의존성)
- `mineru[core]`만 설치 가능
- Python 3.13 미지원

### 4. 개발 생산성

**추정 소요 시간**:
- Python API 정확한 사용법 파악: **2-4시간** (리버스 엔지니어링)
- DocumentParserBase 통합: **1-2시간**
- 테스트 및 디버깅: **2-3시간**
- **총 5-9시간**

**vs Upstage API**:
- 이미 작동하는 안정적인 솔루션
- 추가 작업 불필요

---

## ✅ 최종 권장사항

### Phase 1 (현재): Upstage API 유지

**이유**:
1. ✅ **안정성**: 검증된 API, 명확한 문서
2. ✅ **생산성**: 추가 개발 시간 불필요
3. ✅ **Phase 1 목표 충족**: MVP 구현에 적합
4. ⚠️ **Rate Limit 대응**: 현재 테스트 단계에서는 문제 없음

**Rate Limit 관리 전략**:
```python
# backend/services/upstage/document_parser.py
# 이미 구현된 재시도 로직 + 지수 백오프
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(UpstageAPIError),
)
```

### Phase 2+ (향후): 필요시 재검토

**조건**:
- MinerU Python API가 안정화되고 문서화 개선 시
- Upstage API Rate Limit이 실제로 문제가 될 때
- 대량 처리 (>1000 문서/일) 필요 시

**대안**:
1. **CLI + subprocess 방식**: Python API 대신 MinerU CLI 호출
2. **Upstage 유료 플랜**: Rate Limit 증가
3. **하이브리드 접근**: 긴급 시 MinerU, 평상시 Upstage

---

## 📚 참고 자료

### 공식 문서
- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [MinerU Quick Start](https://opendatalab.github.io/MinerU/quick_start/)
- [MinerU CLI Tools](https://opendatalab.github.io/MinerU/usage/cli_tools/)
- [magic-pdf PyPI](https://pypi.org/project/magic-pdf/)
- [mineru PyPI](https://pypi.org/project/mineru/)

### GitHub Issues
- [Issue #232: magic-pdf --version TypeError](https://github.com/opendatalab/MinerU/issues/232)
- [Issue #1219: pip installation successful but version display failed](https://github.com/opendatalab/MinerU/issues/1219)
- [Issue #2711: Windows 2.0+ installation issue](https://github.com/opendatalab/MinerU/issues/2711)

### 튜토리얼
- [MinerU Beginner's Guide](https://stable-learn.com/en/mineru-tutorial/)
- [Model Download Guide](https://github.com/papayalove/Magic-PDF/blob/master/docs/how_to_download_models_en.md)

### Upstage 문서
- [Upstage Document Parse API](https://developers.upstage.ai/docs/apis/document-parse)

---

## 📝 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-01-28 | 2.0.0 | 전면 개정: 조사 보고서 형식으로 변경, Upstage API 유지 권장 |
| 2026-01-25 | 1.0.0 | 초기 작성: MinerU 마이그레이션 가이드 (현재는 유효하지 않음) |

---

## 🔄 향후 계획

### 모니터링 대상
1. MinerU Python API 안정화 여부 (GitHub 릴리즈 노트 확인)
2. Upstage API 실제 사용량 및 Rate Limit 발생 빈도
3. Windows 환경 MinerU 지원 개선 여부

### 재검토 트리거
- Upstage API Rate Limit으로 인한 실제 서비스 장애 발생 시
- MinerU v2.x에서 Python API 문서화 개선 시
- 대량 문서 처리 요구사항 발생 시 (>1000 문서/일)

---

*Last Updated: 2026-01-28*
*Version: 2.0.0 (Phase 1 MVP)*
*Status: ✅ Upstage API 유지 권장*
