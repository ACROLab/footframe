# FootFrame — A4 기준 2D 발 치수 측정 (OpenCV 전용)

스마트폰에서 촬영한 **위쪽 사진 1~4장**과 **측면 사진 1~4장**을 자동 분석하여 발볼과 발 길이의 **A4 평면 기준 투영 추정값**을 구하는 연구/수업 프로젝트입니다.

> 배포판 범위: **v33 순수 2D OpenCV 측정 파이프라인**. v34 이후의 U-Net, DQN, PyTorch 및 학습 데이터는 의도적으로 포함하지 않았습니다. v29 원본 전체를 병합한 것은 아니며, 기존 v33 독립 실행형 재구축 소스를 유지합니다.

## 핵심 기능

- 일반 흰색 A4(210 × 297mm)를 기준으로 자동 네 모서리 탐색(수동 클릭 없음)
- 흰색 영역/Contour, LSD 선분, 경계 대비 검증 및 가려진 A4 보조 후보
- RANSAC 기반 선 정밀화 → `getPerspectiveTransform` / `warpPerspective`
- HSV/Lab 색상 및 Morphology 기반 발 마스크 → 측정값 환산
- 사진별 관측 검증 → 중앙값/MAD 이상치 검사 → 유효 사진 결과 융합
- 반응형 모바일 웹에서 업로드 및 사진별 검출 경계 확인
- 개인 사진이나 업로드 원본을 파일로 저장하지 않고 메모리에서 분석

**주의:** Homography는 A4가 놓인 **평면**을 보정합니다. 발은 높이가 있으므로 실제 발 길이·발볼의 정확성이 보장되지 않습니다. 신발 구매용 확정 치수나 의료용 측정기로 사용하지 마세요. 자동 신발 모델 추천 기능은 이 소스에 포함되어 있지 않습니다.

## GitHub 업로드

1. 이 ZIP을 **압축 해제**합니다.
2. GitHub의 **기존 FootFrame 저장소 루트**에 압축 해제한 **파일 및 폴더 내용물 전체**를 업로드/교체합니다. 상위 폴더 `FootFrame_CVOnly_GitHub_Ready` 자체를 또 하나의 하위 폴더로 만들지 마세요.
3. `core/`와 `tests/` 폴더의 상대 경로를 그대로 유지한 상태로 **Commit changes**를 선택합니다.
4. 기존 연결된 클라우드 서비스가 있으면 해당 저장소의 최신 커밋을 배포합니다. `render.yaml`은 Render Blueprint용 설정입니다.

**GitHub에 업로드하면 안 되는 자료:** 개인 발 사진, 강의 PDF, 실측 개인정보, 가중치 파일, `.env` 등. 이 ZIP에는 이러한 자료가 들어 있지 않습니다.

## 빠른 실행 / 배포 설정

Python 3.11 이상 권장. Python 패키지로는 NumPy와 OpenCV headless만 필요합니다.

```bash
python -m pip install -r requirements.txt
python app.py
```

- 로컬 주소: `http://127.0.0.1:8000`
- 기본 웹 화면: `/`
- 상태 검사: `/health`
- 사진 분석 API: `POST /analyze` (multipart/form-data; `top`, `side` 항목)
- 서버는 배포 환경의 `PORT`와 `HOST` 환경변수를 읽습니다.

Render의 일반 **Web Service**로 수동 연결할 경우:

- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`
- Health Check Path: `/health`
- Root Directory: 저장소 루트

무료 서비스의 절전·사용량 제한과 업로드 데이터의 처리·개인정보 정책은 서비스별로 확인해야 합니다. 업로드 파일은 서버 디스크에 저장하지 않지만 메모리와 전송 과정에서 처리됩니다.

## 소스 구성

| 경로 | 역할 |
|---|---|
| `app.py` | 사진 업로드, 서버 API, 결과 모바일 화면 |
| `core/geometry.py` | 종이 후보 검증과 원근 보정 |
| `core/detect_a4_experimental.py` | LSD 선분 기반 A4 검출 |
| `core/occluded_paper.py` | 부분적으로 가려진 종이의 보조 검출 |
| `core/refine_a4.py` | RANSAC 기반 A4 경계 정밀화 |
| `core/measurement.py` | HSV/Lab 발 마스크, 발 길이·발볼 측정 |
| `core/fuse_2d.py` | 사진별 이상 관측 검사와 융합 |
| `evaluate.py` | 지정 사진 측정, 선택적 실측 오차 **사후 검증** |
| `audit_v33.py` | 검출선·보정 영상·마스크 이미지 생성 |
| `benchmark_v33.py` | 합성 종이 장면의 모서리 정확도 시험 |
| `tests/` | 기본 CV 및 v33 회귀 테스트 |
| `render.yaml` | Render 배포 설정 |

## 테스트

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

`tests/test_v33.py`의 일부 테스트는 비공개 발 사진이 있을 때만 실행되며, 사진이 없는 공개 GitHub 사본에서는 정상적으로 건너뜁니다.

### 내 사진으로 검증하기

사진은 저장소에 올리지 말고 로컬 별도 폴더에 저장한 뒤 다음 명령을 사용합니다.

```bash
python evaluate.py --top path/to/top1.jpg path/to/top2.jpg \
  --side path/to/side1.jpg path/to/side2.jpg --out evaluation.json
```

실측이 있으면 `--reference-length`와 `--reference-width` 옵션으로 **분석 후 결과 비교**에만 추가할 수 있습니다. 이 숫자는 `core/`의 측정 알고리즘에 사용되지 않습니다.

A4 경계·발 마스크 시각화:

```bash
python audit_v33.py --samples path/to/samples --out local_diagnostics
```

여기서 `path/to/samples`에는 `top/`, `side/` 하위 폴더가 있어야 합니다. 산출된 그림/JSON 역시 공개 저장소에 실수로 추가하지 마세요.

## 검증 범위

이전 개발에서 제공된 **동일 피험자의 8장**으로 소스에 변경 없이 재현한 기본 측정:

- 발 길이 `247.75 mm` (유효 측면 2장, `provisional`)
- 발볼 `97.25 mm` (유효 위쪽 4장)
- 별도의 실측 기준 250 mm / 100 mm와의 차이는 각각 −2.25 mm / −2.75 mm
- 다른 사람의 발·다른 스마트폰·조명에 대한 일반화 성능은 검증하지 않았습니다.

**실측 250/100mm는 검증 보고용 숫자일 뿐 알고리즘의 보정 상수가 아닙니다.**
