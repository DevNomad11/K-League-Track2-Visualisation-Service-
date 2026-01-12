# K리그 AI 시각화 서비스 - 경기 영상 기반 최적 패스 예측 서비스

실시간 축구 영상 시각화를 위한 서비스입니다. 선수 및 공 객체 추적, 팀 분류, 피치 컨트롤 시각화, 최적의 패스 경로 제안 기능을 제공합니다.

## 주요 기능

- **선수 및 공 감지**: YOLO 기반 선수, 골키퍼, 심판, 공 감지
- **다중 객체 추적**: Re-ID를 활용한 BoT-SORT 트래커로 일관된 선수 식별
- **팀 분류**: SigLIP + UMAP + KMeans 클러스터링을 통한 자동 팀 분류
- **피치 키포인트 감지**: 호모그래피 기반 좌표 변환
- **피치 컨트롤 시각화**:
  - 정적 보로노이 다이어그램
  - 동적 (속도 가중) 보로노이 다이어그램
- **패스 분석**:
  - 패스 성공 확률 예측
  - Expected Threat (xT) 시각화
  - 최적 수신 지점을 포함한 쓰루패스 제안
- **2D 미니맵**: 실시간 전술적 탑뷰

## 프로젝트 구조

```
final/
├── inference.py              # 메인 실행 파일
├── config.py                 # 설정, 경로, 색상
├── view_transformer.py       # 호모그래피 변환
├── pitch_drawing.py          # 피치 시각화
├── minimap.py                # 2D 전술 미니맵
├── voronoi.py                # 정적 및 동적 보로노이
├── velocity_tracker.py       # 선수 속도 추정
├── dynamic_pitch_control.py  # 도달 시간 계산
├── pass_visualization.py     # 패스 성공률/xT 시각화
├── tracking.py               # 트랙 관리
├── annotators.py             # 팀별 색상 주석
├── utils.py                  # 모델 로딩 유틸리티
├── ball_tracker.py           # 칼만 필터 기반 공 추적
├── team_classifier.py        # SigLIP 기반 팀 분류
├── pass_success_predictor.py # 패스 성공 ML 모델
├── calculate_xt_grid.py      # Expected Threat 계산기
├── requirements.txt          # Python 의존성
└── README.md                 # 이 파일
```

## 설치 방법

###  패키지 설치
```bash
pip install -r requirements.txt
```

###  모델 가중치 다운로드

https://drive.google.com/drive/folders/1V9teQbgAZoVSZyxCAjj55wy8ZgsNbFyS?usp=drive_link
다음 모델 파일들을 `models/` 디렉토리에 배치하세요:

| 모델 | 파일명 | 설명 |
|------|--------|------|
| 공 감지 | `ball_detect.pt` | 공 감지용 YOLO 모델 |
| 선수 감지 | `player_detect.pt` | 선수/골키퍼/심판 감지용 YOLO 모델 |
| 피치 키포인트 | `pitch_detect.pt` | 피치 키포인트 감지용 YOLO 모델 |
| Re-ID | `osnet_x0_25_msmt17.pt` | 선수 재식별용 OSNet 모델 |
| 패스 성공 | `pass_success_model.pkl` | 학습된 패스 성공 예측기 |


## 사용법

### 기본 사용법
```bash
python inference.py -i input/video.mp4
```


### 명령줄 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `-i, --input` | str | 필수 | 입력 영상 경로 |
| `-o, --output` | str | 자동 | 출력 영상 경로 |
| `--device` | int | 0 | CUDA 디바이스 ID |
| `--voronoi` | flag | False | 미니맵에 보로노이 활성화 |
| `--voronoi-frame` | flag | False | 영상 프레임에 보로노이 오버레이 |
| `--dynamic-voronoi` | flag | False | 속도 가중 동적 보로노이 활성화 |
| `--pass-success` | flag | False | 패스 성공 확률 선 표시 |
| `--pass-xt` | flag | False | xT 가중 패스 경로 표시 |
| `--pass-xt-v11` | flag | False | 쓰루패스 제안 표시 (v11) |
| `--no-minimap` | flag | False | 2D 전술 미니맵 비활성화 |
| `--no-cmc` | flag | False | 카메라 모션 보정 비활성화 |
| `--no-reid` | flag | False | Re-ID 기능 비활성화 |
| `--show-traces` 
| `--robust-ball-tracking` | flag | False | 칼만 필터 공 추적 활성화 |
| `--conf-player` | float | 0.25 | 선수 감지 신뢰도 |
| `--conf-ball` | float | 0.07 | 공 감지 신뢰도 |
| `--team-0-attacks-left` | flag | False | 팀 0이 왼쪽→오른쪽 공격 |
| `--team-0-attacks-right` | flag | False | 팀 0이 오른쪽→완쪽 공격 |



## 예제

### 데모 버전) 모든 기능 활성화
```bash
python inference.py \
    --input sample_1.mp4 \
    --dynamic-voronoi \
    --pass-xt-v11 \
    --team-0-attacks-left
```

팀 공격 방향은 영상별로 수동으로 설정해야 합니다.
- Sample_1.mp4: --team-0-attacks-left
- Sample_2.mp4: --team-0-attacks-right
- Sample_3.mp4: --team-0-attacks-left

## 시연 영상
- Sample_1.mp4: https://www.youtube.com/watch?v=v_p0nuirmdA
- Sample_2.mp4: https://www.youtube.com/watch?v=oGbYmQ4Kr9s
- Sample_3.mp4: https://www.youtube.com/watch?v=uSIDer_HNzM


## 기술 세부사항

### 좌표계
- **픽셀 좌표**: 원본 영상 프레임 (픽셀)
- **월드 좌표**: 실제 피치 (센티미터)
- 피치 크기: 10500cm x 6800cm (105m x 68m)

### 피치 키포인트
호모그래피 계산을 위해 32개의 사전 정의된 피치 키포인트 사용:
- 코너 포인트, 페널티 에어리어 코너, 골 에어리어 코너
- 센터 서클 포인트, 페널티 스팟, 센터 스팟

### 동적 보로노이 (Dynamic Pitch Control)
정적 보로노이(거리 기반)와 달리 동적 보로노이는 다음을 고려:
- 선수 현재 위치
- 선수 속도 벡터
- 최대 스프린트 속도 (7 m/s)
- 도달 시간 계산

### 패스 성공 모델
K리그 이벤트 데이터로 학습:
- 시작/종료 좌표
- 패스 거리 및 각도
- XGBoost

### Expected Threat (xT)
각 위치에서 득점 확률을 계산하는 그리드 기반 모델 (16x12 존).

### 라이센스 정보

## AGPL-3.0 Licensed
- **Ultralytics YOLO** - https://github.com/ultralytics/ultralytics
- **BoxMOT** - https://github.com/mikel-brostrom/boxmot

## Apache-2.0 Licensed
- **SigLIP (google/siglip-base-patch16-224)** - https://huggingface.co/google/siglip-base-patch16-224
- **Transformers** - https://github.com/huggingface/transformers

## MIT Licensed
- **Supervision** - https://github.com/roboflow/supervision
