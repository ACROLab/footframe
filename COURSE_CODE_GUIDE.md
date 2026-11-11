# FootFrame V1 — 강의 내용 / 파일 / 코드 찾기

V1에서 사용자에게 보여 주는 코드와 **실제 구현 파일**의 연결입니다.

## 1. Image Processing — `core/image_processing.py`

L02, L07 Review: 이미지 해상도 조절 및 보간.

```python
im = resize_image(bgr)  # 긴 변 최대 950 px
```

## 2. Color Space & Color Segmentation — `core/color_segmentation.py`

L03 HSV/Lab, L02 Gaussian Blur/Thresholding/Morphology:

```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
sm = cv2.GaussianBlur(lab[:, :, 0], (13, 13), 0)
mask = np.uint8((sm > local + 7) & (hsv[:, :, 1] < 83)) * 255
```

- `_candidate_contours(image)`: 밝은 A4의 외곽선 후보 4점을 찾습니다.
- `foot_mask(rect)`: 피부색 범위의 이진 마스크를 만들고 가장 큰 연결 성분을 선택합니다.

## 3. Edge Detection — `core/edge_detection.py`

L02 Edge Detection, L06 Linear Algebra:

```python
raw = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
```

- `line_geom()`: 선분 길이·방향·법선.
- `intersect()`: 선형방정식 `np.linalg.solve`로 두 직선의 교점 계산.
- `detect_a4()`: 직선 네 개를 조합해 A4 사각형 후보 생성.

## 4. Canny & Hough Transform — `core/hough_transform.py`

L02 Edge Detection 응용:

```python
edges = cv2.Canny(component_mask, 30, 90)
lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, ...)
```

- `occluded_paper_proposals()`: 밝은 종이 영역의 3개 선분과 독립적인 네 번째 LSD 선분으로 후보 생성.
- 직접 관측하지 않은 모서리는 치수 융합에서 **진단용으로만** 사용합니다.

## 5. RANSAC & Least Squares — `core/ransac.py`

L06 Solving Problems, L07 RANSAC:

```python
res = np.abs((p - p[i]) @ n)
inliers = res < 2.3
```

- `refine_quad()`: 노이즈를 제외하고 네 종이 경계 정밀 피팅.
- `sensitivity_mm()`: A4 모서리 위치 흔들림이 환산 길이에 주는 영향(전체 발 길이 신뢰구간 아님).

## 6. Image Geometry — `core/image_geometry.py`

L04 Image Formation 및 L05 Image Geometry:

- `_paper_proof()`: 사각형의 면적, 볼록성, 흰색/밝기 대비, 매끄러움 검증.
- `detect_paper()`: Contour, LSD, 가려진 종이 후보를 통합하고 A4 네 점 확정.

## 7. Homography — `core/homography.py`

L05 Homography/Perspective Warping:

```python
Hmat = cv2.getPerspectiveTransform(q, dst)
rectified = cv2.warpPerspective(image, Hmat, (W, H))
```

- `rectify()`: 검출된 A4를 210×297mm(가로 방향이면 297×210mm)로 펼칩니다.
- 출력 스케일: 기본 2 pixel/mm.
- **발의 3차원 높이로 발생하는 시차는 보정하지 않습니다.**

## 8. 2D Measurement — `core/measurement_2d.py`

L02 Segmentation + L05 Coordinate Transformation:

```python
width = float(np.percentile(widths, 95))
length = (right - left) / px_per_mm
```

- `top_width()`: 앞발 영역의 행별 가로 폭 95백분위.
- `side_length()`: 측면 발 마스크의 유효 가로 범위.
- `analyze_frame()`: 사진 한 장 처리 및 실패 이유 저장.
- `analyze_batch()`: 여러 top/side 사진 분석 후 융합기 호출.

## 9. Outlier Rejection — `core/outlier_rejection.py`

L07 Outlier Rejection의 목적과 같은 통계적 검증. *RANSAC 자체는 아닙니다.*

```python
centre = float(np.median(values))
mad = float(np.median(np.abs(values - centre)))
threshold = max(10., 3. * 1.4826 * mad)
```

- `fuse()`: 두 장 이상 유효한 사진에 대해 MAD 이상치 검사 후 가중평균.
- 기본 `analyze_batch()`는 기하 오차값 0을 넘겨 **동일 가중치**로 통합합니다.

## 보조 파일

| 파일 | 실제 역할 |
|---|---|
| `evaluate.py` | 실제 사진 측정값을 JSON으로 저장하고 실측값과 비교(사후 검증) |
| `visualize_results.py` | A4 모서리, 평면 보정 영상, 발 마스크를 파일로 저장 |
| `benchmark_homography.py` | 합성 장면에서 A4 코너 위치 오차를 측정 |
| `app.py` | 웹 사진 업로드를 `analyze_batch()`로 연결하는 진입점 (강의 설명 대상에서는 제외) |
| `tests/*.py` | 정상 동작을 회귀 테스트 |
| `render.yaml` | GitHub `ACROLab/footframe` 배포 설정 |

## V1의 기술적 범위

- 고정된 실측 발 길이/발볼로 보정하지 않습니다.
- 2D 마스크로 평면 투영 길이를 구하는 방식입니다.
- 개별 사진 추정 오차를 줄이기 위해 여러 사진을 활용하지만 3D 복원 또는 해부학적 치수의 완전 검증은 아닙니다.
- 파일명과 모듈 경계는 학습 목적에 맞게 바꾸었지만 수치 알고리즘은 기존 CV-only 베이스라인을 보존했습니다.

## 기존 파일 이름과 V1 파일 이름 비교

| 기존 CV-only 소스 | V1 강의 용어 파일 | 변경 내용 |
|---|---|---|
| `core/geometry.py` | `core/image_processing.py`, `core/image_geometry.py`, `core/homography.py`, `core/color_segmentation.py` | 이미지 축소, 종이 검증, 원근 보정, 흰 영역 후보를 각 강의 단원별로 분리 |
| `core/detect_a4_experimental.py` | `core/edge_detection.py` | LSD 기반 직선 후보 찾기 |
| `core/refine_a4.py` | `core/ransac.py` | RANSAC/최소제곱 모서리 정밀화 |
| `core/occluded_paper.py` | `core/hough_transform.py` | Canny/Hough 기반 가려진 종이 후보 |
| `core/measurement.py` | `core/color_segmentation.py`, `core/measurement_2d.py` | 피부색 분할과 실제 길이 계산 분리 |
| `core/fuse_2d.py` | `core/outlier_rejection.py` | 중앙값/MAD 및 관측 융합 |
| `audit_v33.py` | `visualize_results.py` | 영상 진단의 목적이 드러나도록 이름 변경 |
| `benchmark_v33.py` | `benchmark_homography.py` | 합성 A4 경계 평가 목적이 드러나도록 이름 변경 |

**주의:** V1에는 기존 이름의 파일을 중복해서 넣지 않았습니다. 모든 `import`는 새로운 파일 경로를 바라봅니다.


## 10. 웹 화면 / 서비스 (컴퓨터비전 강의 코드와 별도)

- `web/index.html`: 사용자가 보는 한 페이지(사진 업로드, 측정 결과)
- `web/styles.css`: 데스크톱과 모바일 반응형 디자인
- `web/app.js`: 업로드 사진 미리보기, 분석 요청, 결과 표시 및 상세 진단 토글
- `app.py`: Python 표준 라이브러리 HTTP 서버. `/analyze` 호출 시 `core/measurement_2d.py`의 `analyze_batch()`를 실행합니다.

**컴퓨터비전 학습 파일은 모두 `core/` 안에 있고, 웹 디자인 변경은 측정 로직에 영향을 주지 않습니다.**
