# 구미 버스 실시간 추적 시스템

버스 도착 시간을 예측하고 실시간으로 추적하는 시스템입니다.


## 설치

```bash
pip install -r requirements.txt
```

---

## 사용 방법

### 1. 모델 학습

```bash
# bus_arrivals.csv 파일이 있어야 합니다
python train.py
```

**출력 파일:**
- `models/bus_predictor.pkl` - 학습된 모델
- `data/processed_bus_arrivals.csv` - 전처리된 데이터

---

### 2. 서버 실행

#### 실시간 모드 (프로덕션)
```bash
python main.py
```

- 현재 시각(`datetime.now()`)을 기준으로 동작
- 실제 버스 데이터 수신 시 사용

#### 시뮬레이션 모드 (테스트/개발)
```bash
python main.py --simulation
```

- `collection_time`을 기준으로 동작
- 과거 데이터를 재생하여 테스트 가능

---

### 3. 데이터 수집

#### 실시간 모드
```bash
python data_collector.py --csv bus_arrivals.csv --interval 60
```

- CSV 파일을 60초마다 확인
- 새로운 데이터만 서버로 전송

#### 시뮬레이션 모드
```bash
python data_collector.py --csv bus_arrivals.csv --simulate --speed 60
```

- `--speed 60`: 실제 시간의 60배 속도로 재생
- 예: 1시간 데이터를 1분 만에 재생

**중요**: 서버도 `--simulation` 플래그로 실행되어야 합니다!

---

## API 사용법

### 전체 버스 조회
```bash
GET http://localhost:8000/api/buses
```

### 노선별 조회
```bash
GET http://localhost:8000/api/buses/route/GMB19010
```

### 정류장별 조회
```bash
GET http://localhost:8000/api/buses/station/GMB132
```

### 노선+정류장 조회
```bash
GET http://localhost:8000/api/buses/route/GMB19010/station/GMB132
```

### 서버 상태 확인
```bash
GET http://localhost:8000/api/status
```

---

## 디렉토리 구조

```
.
├── bus_predictor.py           # ML 예측 모델
├── bus_tracker.py             # 버스 추적 로직
├── data_preprocessing.py      # 데이터 전처리
├── data_collector.py          # 데이터 수집
├── main.py                    # FastAPI 서버
├── train.py                   # 모델 학습 스크립트
├── requirements.txt           # 패키지 의존성
├── bus_arrivals.csv           # 원본 데이터 (직접 준비)
├── models/
│   └── bus_predictor.pkl      # 학습된 모델
└── data/
    └── processed_bus_arrivals.csv  # 전처리된 데이터
```

---

## 주요 클래스 및 메서드

### BusPredictor
- `train(df)`: 모델 학습
- `predict(features)`: 도착 시간 예측
- `predict_by_historical_pattern()`: Fallback 예측

### BusTracker
- `process_new_data(data)`: 새 데이터 처리
- `check_lost_buses()`: 오래된 버스 감지
- `update_predictions()`: 예측 업데이트
- `get_bus_info()`: 버스 정보 조회

### BusDataPreprocessor
- `load_data()`: 데이터 로드
- `prepare_training_data(df)`: 전처리 파이프라인

---

## 버스 상태

- **ACTIVE**: 실시간 데이터 수신 중
- **PREDICTED**: 180초 이상 데이터 없음, ML 예측 사용

---

## Fallback 계층

1. **ML 예측**: GradientBoostingRegressor
2. **Historical Pattern**: 
   - 노선 + 정류장 + 시간대
   - 노선 + 정류장
   - 노선 + 남은 정류장 수
3. **기본값**: 정류장당 70초

---

## 시뮬레이션 vs 실시간 모드

| 항목 | 실시간 모드 | 시뮬레이션 모드 |
|------|------------|----------------|
| 시간 기준 | `datetime.now()` | `collection_time` |
| 사용 목적 | 프로덕션 | 테스트/개발 |
| 데이터 소스 | 실시간 API | 과거 CSV |
| Countdown | 실제 시간 경과 | 가상 시간 경과 |

---

## 문제 해결

### 모델 로드 실패
```
⚠ 모델 로드 실패: [Errno 2] No such file or directory
```
→ `train.py`를 먼저 실행하여 모델을 학습하세요.

### collection_time 파싱 실패
```
⚠ collection_time 파싱 실패 → now()로 대체
```
→ 실시간 모드에서는 정상입니다. 시뮬레이션 모드에서 발생하면 데이터 형식을 확인하세요.

### 버스가 표시되지 않음
1. 서버 상태 확인: `GET /api/status`
2. 데이터 수집기가 실행 중인지 확인
3. CSV 파일에 새 데이터가 있는지 확인

---

## 성능 지표 예시

```
R² Train=0.8532, Val=0.8234
MAE=45.23s (0.75m), RMSE=67.89s
```

- **R²**: 설명력 (높을수록 좋음, 0~1)
- **MAE**: 평균 절대 오차
- **RMSE**: 평균 제곱근 오차

---

## 라이선스

MIT

---

## 기여

이슈와 PR을 환영합니다!