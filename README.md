# 구미 버스 실시간 추적 및 예측 시스템

API 끊김 현상에 대응하여 머신러닝 기반 예측으로 버스 위치를 계속 추적하는 시스템입니다.

## 🎯 주요 기능

1. **실시간 버스 추적**: 정류장 변경 시 시간 갱신, 앱 내에서 초 단위 카운트다운
2. **API 끊김 감지**: 3분간 업데이트 없으면 자동으로 예측 모드 전환
3. **머신러닝 예측**: 과거 패턴 학습으로 버스 도착 시간 예측
4. **자동 갱신**: 예측 모드에서도 1분마다 새로운 예측값 출력

## 📁 프로젝트 구조

```
.
├── bus_arrivals.csv              # 원본 데이터 (루트에 위치, 1분 간격 수집)
├── data/
│   └── processed_bus_arrivals.csv # 전처리된 데이터 (자동 생성)
├── models/
│   └── bus_predictor.pkl         # 학습된 예측 모델 (자동 생성)
├── data_preprocessing.py         # 데이터 전처리
├── bus_predictor.py              # 예측 모델
├── bus_tracker.py                # 버스 추적 및 상태 관리
├── main.py                       # FastAPI 서버
├── data_collector.py             # 데이터 수집 스크립트
├── train_model.py                # 모델 학습 스크립트
└── requirements.txt              # 의존성 패키지
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

`bus_arrivals.csv` 파일을 프로젝트 루트 디렉토리에 준비합니다. 필수 컬럼:
- collection_time, weekday, time_slot, weather, temp, humidity
- rain_mm, snow_mm, nodeid, nodenm, routeid, routeno
- routetp, arrprevstationcnt, arrtime, vehicletp

**중요**: 제공해주신 수집 스크립트를 사용하면 자동으로 올바른 형식의 CSV가 생성됩니다.

### 3. 모델 학습

```bash
python train_model.py
```

이 스크립트는:
- 데이터 전처리 (날씨 보간, 중복 버스 분리 등)
- 피처 엔지니어링
- 모델 학습 및 검증
- 모델 저장 (`models/bus_predictor.pkl`)

### 4. 서버 실행

```bash
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

### 5. 데이터 수집 시작

**옵션 1: 실시간 수집 (기존 수집 스크립트 사용)**

제공해주신 수집 스크립트를 그대로 실행하고, 별도 터미널에서 data_collector를 실행:

```bash
# 터미널 1: 버스 데이터 수집 (기존 스크립트)
python your_bus_collection_script.py

# 터미널 2: FastAPI로 데이터 전송
python data_collector.py --csv bus_arrivals.csv --interval 60
```

**옵션 2: 시뮬레이션 모드 (테스트용, 60배속 재생)**

```bash
python data_collector.py --csv bus_arrivals.csv --simulate --speed 60
```

## 🔌 API 엔드포인트

### 데이터 수신
- `POST /api/bus-arrival` - 단일 버스 데이터 수신
- `POST /api/bus-arrival/batch` - 일괄 버스 데이터 수신

### 버스 정보 조회
- `GET /api/buses` - 모든 버스 정보
- `GET /api/buses/route/{route_id}` - 특정 노선 버스
- `GET /api/buses/station/{node_id}` - 특정 정류장 버스
- `GET /api/buses/route/{route_id}/station/{node_id}` - 노선+정류장

### 시스템 상태
- `GET /api/status` - 서버 상태 및 통계

### API 문서
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📊 응답 예시

```json
{
  "routeid": "GMB19010",
  "routeno": "190",
  "nodeid": "GMB132",
  "nodenm": "금오공대종점",
  "arrprevstationcnt": 8,
  "arrtime": 626,
  "display_minutes": 10,
  "display_seconds": 26,
  "vehicletp": "일반차량",
  "routetp": "일반버스",
  "status": "active",
  "last_update": "2025-11-13T12:46:35.686139+00:00"
}
```

- `status`: `active` (실시간 데이터) 또는 `predicted` (예측 모드)
- `arrtime`: 현재 남은 시간 (초)
- `display_minutes`, `display_seconds`: 화면 표시용

## 🔄 작동 원리

### 1. 실시간 모드 (Active)

```
새 데이터 수신 → 정류장 변경 감지 → 초기 시간 설정 → 앱에서 초 단위 감소
```

- 정류장이 변경되면(`arrprevstationcnt` 변경) 그 시점의 시간 기준으로 카운트다운
- 정류장이 변경되지 않으면 시간 갱신 안 됨
- 앱은 `last_station_change_time`부터 경과 시간을 계산하여 실시간 표시

### 2. 예측 모드 (Predicted)

```
3분간 업데이트 없음 → 예측 모드 전환 → 1분마다 새 예측 생성
```

- 마지막 업데이트로부터 3분 경과 시 자동 전환
- 머신러닝 모델로 도착 시간 예측
- 1분마다 새로운 예측값 갱신
- 실시간 데이터 재수신 시 자동으로 Active 모드 복귀

## 🛠️ 데이터 전처리

### 1. 날씨 데이터 보간
- 결측치를 전후 20분(±20 rows) 데이터로 보간

### 2. 중복 버스 분리
- 같은 노선에 여러 대가 있을 때 시간 차이로 클러스터링
- 예: `[320, 570, 320, 480]` → `[320, 320]`, `[570, 480]` 그룹으로 분리

### 3. 피처 생성
- 시간대 피처: hour, minute, day_of_week, is_weekend, is_rush_hour
- 속도 피처: avg_time_per_station (정류장당 평균 시간)
- 궤적 추적: 이전 상태와 비교하여 변화 감지

## 🎓 모델 정보

- **알고리즘**: Gradient Boosting Regressor
- **학습 피처**: 
  - 정류장 정보 (routeid, nodeid)
  - 시간 정보 (hour, minute, weekday, is_rush_hour)
  - 날씨 정보 (temp, humidity, rain_mm, snow_mm)
  - 버스 정보 (routetp, vehicletp, arrprevstationcnt)
  - 속도 정보 (avg_time_per_station)

- **타겟**: arrtime (도착까지 남은 시간, 초)

## 🔧 커스터마이징

### 타임아웃 설정 변경
`main.py`의 `background_tasks()` 함수에서:
```python
tracker.check_lost_buses(timeout_seconds=180)  # 3분 → 원하는 시간으로 변경
```

### 예측 갱신 간격 변경
`bus_tracker.py`의 `BusTracker` 클래스에서:
```python
self.prediction_interval = 60  # 60초 → 원하는 간격으로 변경
```

### 모델 파라미터 조정
`bus_predictor.py`의 `train()` 메서드에서 모델 파라미터 수정 가능

## 📝 주의사항

1. **정류장 점프 없음**: 버스가 정류장을 건너뛰지 않는다고 가정
2. **시간 순서**: 데이터는 시간순으로 수집되어야 함
3. **메모리**: 많은 버스를 추적할 경우 메모리 사용량 증가
4. **도착 버스 정리**: 30초 이하 남은 버스는 자동 제거

## 🐛 문제 해결

### 모델 로드 실패
```bash
# 모델 재학습
python train_model.py
```

### API 연결 오류
```bash
# 서버 상태 확인
curl http://localhost:8000/api/status
```

### 데이터 수집 안 됨
- CSV 파일 경로 확인
- 파일 권한 확인
- 데이터 형식 확인

## 📄 라이센스

MIT License

## 👥 기여

이슈와 PR을 환영합니다!