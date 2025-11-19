#!/usr/bin/env python3
"""
버스 도착 예측 모델 학습 스크립트
"""

import os
import sys
from datetime import datetime

from data_preprocessing import BusDataPreprocessor
from bus_predictor import BusArrivalPredictor


def main():
    print("=" * 60)
    print("구미 버스 도착 예측 모델 학습")
    print("=" * 60)
    
    # 경로 설정
    data_path = "bus_arrivals.csv"
    model_path = "models/bus_predictor.pkl"
    
    # 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. 데이터 확인
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일이 없습니다: {data_path}")
        print("   bus_arrivals.csv 파일을 먼저 준비해주세요.")
        sys.exit(1)
    
    print(f"\n✓ 데이터 파일 확인: {data_path}")
    
    # 2. 데이터 전처리
    print("\n" + "=" * 60)
    print("STEP 1: 데이터 전처리")
    print("=" * 60)
    
    preprocessor = BusDataPreprocessor(data_path)
    
    print("데이터 로딩 중...")
    df = preprocessor.load_data()
    print(f"✓ 원본 데이터: {len(df):,} rows")
    
    print("\n전처리 실행 중...")
    print("  - 날씨 결측치 보간")
    print("  - 중복 버스 분리")
    print("  - 피처 생성")
    print("  - 버스 궤적 추적")
    
    processed_df = preprocessor.prepare_training_data(df)
    print(f"✓ 전처리 완료: {len(processed_df):,} rows")
    
    # 전처리된 데이터 저장
    processed_path = "data/processed_bus_arrivals.csv"
    processed_df.to_csv(processed_path, index=False)
    print(f"✓ 전처리 데이터 저장: {processed_path}")
    
    # 3. 모델 학습
    print("\n" + "=" * 60)
    print("STEP 2: 모델 학습")
    print("=" * 60)
    
    predictor = BusArrivalPredictor(model_path)
    predictor.train(processed_df)
    
    # 4. 모델 저장
    print("\n" + "=" * 60)
    print("STEP 3: 모델 저장")
    print("=" * 60)
    
    predictor.save(model_path)
    
    # 5. 모델 테스트
    print("\n" + "=" * 60)
    print("STEP 4: 모델 테스트")
    print("=" * 60)
    
    # 샘플 데이터로 예측 테스트
    test_cases = [
        {
            'routeid': 'GMB19010',
            'nodeid': 'GMB132',
            'routetp': '일반버스',
            'vehicletp': '일반차량',
            'arrprevstationcnt': 8,
            'weekday': 'Thu',
            'time_slot': 'afternoon',
            'weather': 'Clear',
            'temp': 15.2,
            'humidity': 49.0,
            'rain_mm': 0.0,
            'snow_mm': 0.0,
            'hour': 12,
            'minute': 46,
            'day_of_week': 3,
            'is_weekend': 0,
            'is_rush_hour': 0,
            'avg_time_per_station': 78.25
        },
        {
            'routeid': 'GMB19210',
            'nodeid': 'GMB130',
            'routetp': '좌석버스',
            'vehicletp': '일반차량',
            'arrprevstationcnt': 6,
            'weekday': 'Mon',
            'time_slot': 'morning',
            'weather': 'Clear',
            'temp': 18.0,
            'humidity': 45.0,
            'rain_mm': 0.0,
            'snow_mm': 0.0,
            'hour': 8,
            'minute': 30,
            'day_of_week': 0,
            'is_weekend': 0,
            'is_rush_hour': 1,
            'avg_time_per_station': 112.0
        }
    ]
    
    print("\n테스트 케이스 예측:")
    for i, test in enumerate(test_cases, 1):
        try:
            prediction = predictor.predict(test)
            print(f"\n테스트 {i}:")
            print(f"  노선: {test['routeid']} ({test['routetp']})")
            print(f"  정류장: {test['nodeid']}")
            print(f"  남은 정류장: {test['arrprevstationcnt']}개")
            print(f"  시간대: {test['weekday']} {test['hour']}:{test['minute']:02d}")
            print(f"  예측 도착 시간: {prediction:.0f}초 ({prediction/60:.1f}분)")
        except Exception as e:
            print(f"\n테스트 {i} 실패: {e}")
    
    # 6. 완료
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"모델 파일: {model_path}")
    print(f"처리된 데이터: {processed_path}")
    print(f"\n서버 실행 명령:")
    print("  python main.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()