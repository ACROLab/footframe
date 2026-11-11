# FootFrame — Version 1 (V1)

**강의자료 용어로 파일명을 통일한 순수 2D OpenCV 발 치수 측정 프로젝트**

- 기준물: **A4 210 × 297 mm**
- 입력: 웹에서는 위쪽(top) 사진 2~4장, 측면(side) 사진 2~4장씩 업로드 (API는 종류별 최소 1장을 받지만 최종 치수 통합에는 유효 관측 2장 이상 필요)
- 출력: A4 평면을 기준으로 환산한 **발볼**(top)과 **발 길이**(side), 사용/제외 사진 및 그 이유
- 구현: OpenCV + NumPy. **딥러닝, 강화학습, 3D 복원, 수동 꼭짓점 선택 없음**
- 중요한 한계: Homography는 **A4 평면의 원근만 보정**합니다. 발 높이와 렌즈 왜곡에 의한 오차는 남습니다.

## V1 의미

**V1은 새 GitHub 저장소를 위한 이름/폴더 구조 버전입니다.**
측정 로직은 기존에 검증했던 CV-only(v33 알고리즘) 버전과 동일하게 보존하는 것을 목표로 분리/이름 변경하였습니다.
새 데이터로 새 모델을 개발하거나 정확도를 개선했다고 주장하지 않습니다.

## 실행 순서와 파일 위치

| 순서 | 강의 개념 | 위치 | 실제 역할 |
|---:|---|---|---|
| 1 | L02 Image Processing | `core/image_processing.py` | `resize_image()` 입력 이미지 축소 |
| 2 | L03 Color Segmentation, L02 Morphology | `core/color_segmentation.py` | `_candidate_contours()` A4 밝은 영역 후보, `foot_mask()` 피부색 발 마스크 |
| 3 | L02 Edge Detection, L06 Linear Algebra | `core/edge_detection.py` | `detect_a4()` LSD 직선 검출 및 교점 사각형 후보 |
| 4 | L02 Canny / Hough Transform | `core/hough_transform.py` | `occluded_paper_proposals()` 가려진 A4 후보 |
| 5 | L06 Least Squares / L07 RANSAC | `core/ransac.py` | `refine_quad()` 경계 직선 보정, `order_quad()` 순서 정렬 |
| 6 | L04 Image Formation / L05 Image Geometry | `core/image_geometry.py` | `_paper_proof()` A4 후보 검증, `detect_paper()` 최종 선택 |
| 7 | L05 Homography | `core/homography.py` | `rectify()` A4 210×297mm를 2px/mm 좌표계로 변환 |
| 8 | L02 Segmentation / L05 Coordinate Transformation | `core/measurement_2d.py` | `top_width()`, `side_length()`, `analyze_frame()`, `analyze_batch()` |
| 9 | L07 Outlier Rejection + MAD | `core/outlier_rejection.py` | `fuse()` 사진별 측정치 이상치 제거 및 통합 |

A4 후보는 **Contour + LSD 직선 검출**을 병렬로 만들고, `image_geometry.py`가 이를 평가합니다. `ransac.py`로 경계를 보정한 뒤, `homography.py`가 평면으로 펼칩니다. `measurement_2d.py`는 `color_segmentation.py`의 발 마스크를 이용합니다.

## 전체 폴더

```text
FootFrame_V1/
├── app.py                         # 웹 서버 및 사진 분석 API (/analyze, /health)
├── web/                           # 실제 서비스 화면 (웹 프런트엔드)
│   ├── index.html                 # 업로드 및 결과 페이지
│   ├── styles.css                 # 반응형 화면 디자인
│   ├── app.js                     # 파일 미리보기, 업로드, 결과 표시
│   └── favicon.svg                # 브라우저 아이콘
├── evaluate.py                    # 사진별 측정 및 실측 오차 검증(실측은 알고리즘에 미사용)
├── visualize_results.py           # 원본/A4 검출/마스크 이미지 생성
├── benchmark_homography.py        # 합성 A4 네 모서리 위치 오차 테스트
├── core/
│   ├── __init__.py
│   ├── image_processing.py
│   ├── color_segmentation.py
│   ├── edge_detection.py
│   ├── hough_transform.py
│   ├── ransac.py
│   ├── image_geometry.py
│   ├── homography.py
│   ├── measurement_2d.py
│   └── outlier_rejection.py
├── tests/
│   ├── conftest.py
│   ├── test_image_processing.py
│   └── test_image_geometry.py
├── COURSE_CODE_GUIDE.md           # 상세한 강의 단원 ↔ 함수 ↔ 코드 위치
├── requirements.txt
├── requirements-dev.txt
├── render.yaml                    # ACROLab/footframe, branch main
├── start_windows.bat
├── start_ubuntu.sh
├── VERSION                        # 1.0.0
└── .gitignore
```

## 설치 및 테스트

Python **3.11 이상** 권장. Windows / Ubuntu 동일한 핵심 명령입니다.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 실측 사진 분석 (웹 화면 외 별도 CLI)

```bash
python evaluate.py --top top1.jpg top2.jpg --side side1.jpg side2.jpg --out result.json
```

오차 비교만 수행하려면 `--reference-length` 및 `--reference-width`를 추가합니다. 실제 길이는 검출기에서 사용하지 않습니다.

이미지 시각화는 다음 폴더 형식으로 사진을 정리한 후 실행합니다.

```text
samples/
├── top/   # *.jpg
└── side/  # *.jpg
```

```bash
python visualize_results.py --samples samples --out reports/v1_visuals
python benchmark_homography.py --out reports/v1_synthetic
```

## 웹/Render

이 문서의 주요 목적은 **컴퓨터비전 학습용 코드 설명**입니다. `app.py`는 API와 정적 파일을 제공하며 웹 화면은 `web/index.html`, `web/styles.css`, `web/app.js`에 분리했습니다. 실제 페이지는 업로드와 결과에만 집중하고, 사진별 검출 로그는 접을 수 있는 상세 결과 안에 배치했습니다.
기존 Render의 Source를 `ACROLab/footframe`, Branch를 `main`, Build Command를 `pip install -r requirements.txt`, Start Command를 `python app.py`로 설정하면 됩니다. 기존 서비스는 Render Dashboard에서 Source 연결을 변경한 경우 Blueprint와 일치 여부를 따로 확인하세요.

## 결과 해석

기존 8장의 동일 피험자 사진에서는 **발 길이 247.75mm**, **발볼 97.25mm**가 산출되었습니다.
실측 **250mm / 100mm**는 테스트 결과의 사후 비교에만 사용한 값이며 보정 상수는 아닙니다. 이 수치는 다른 사람·조명·카메라에 대한 정확도를 보증하지 않습니다.
