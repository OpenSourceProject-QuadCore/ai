#!/usr/bin/env python3
"""
버스 도착 예측 모델 학습 스크립트 (수정 버전)

주요 변경사항:
- 수정된 BusArrivalPredictor 사용
- Training/Inference 분리 확인
- Feature 검증 추가
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

from data_preprocessing import BusDataPreprocessor
from bus_predictor import BusArrivalPredictor


def main():
    parser = argparse.ArgumentParser(description="버스 도착 예측 모델 학습 (수정 버전)")
    parser.add_argument("--data", default="bus_arrivals.csv", help="원본 데이터 경로")
    parser.add_argument("--model", default="models/bus_predictor.pkl", help="모델 저장 경로")
    parser.add_argument("--cv", action="store_true", help="Cross-Validation 수행")
    parser.add_argument("--tune", action="store_true", help="Hyperparameter Tuning")
    parser.add_argument("--no-preprocess", action="store_true", help="전처리된 데이터 사용")
    args = parser.parse_args()
    
    print("=" * 80)
    print("구미 버스 도착 예측 모델 학습 (수정 버전)")
    print("=" * 80)
    print(f"데이터: {args.data}")
    print(f"모델 저장: {args.model}")
    print(f"Cross-Validation: {'Yes' if args.cv else 'No'}")
    print(f"Hyperparameter Tuning: {'Yes' if args.tune else 'No'}")
    print()
    print("🔧 주요 개선 사항:")
    print("  ✓ Training/Inference Feature 완전 분리")
    print("  ✓ Target Leakage 제거 (arrtime 기반 feature 제거)")
    print("  ✓ Categorical Features 직접 사용 (OneHotEncoding)")
    print("=" * 80)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(args.data):
        print(f"❌ 데이터 파일이 없습니다: {args.data}")
        sys.exit(1)
    
    # ======================================================================
    # STEP 1: 데이터 전처리
    # ======================================================================
    processed_path = "data/processed_bus_arrivals.csv"
    
    if args.no_preprocess and os.path.exists(processed_path):
        print(f"\n전처리된 데이터 로드: {processed_path}")
        processed_df = pd.read_csv(processed_path)
        processed_df['collection_time'] = pd.to_datetime(processed_df['collection_time'])
        print(f"✓ 데이터 로드: {len(processed_df):,} rows")
    else:
        print("\n" + "=" * 80)
        print("STEP 1: 데이터 전처리")
        print("=" * 80)
        
        preprocessor = BusDataPreprocessor(args.data)
        
        print("데이터 로딩 중...")
        df = preprocessor.load_data()
        print(f"✓ 원본 데이터: {len(df):,} rows")
        
        print("\n전처리 실행 중...")
        processed_df = preprocessor.prepare_training_data(df, verbose=True)
        
        if len(processed_df) == 0:
            print("❌ 전처리 후 데이터가 없습니다.")
            sys.exit(1)
        
        processed_df.to_csv(processed_path, index=False)
        print(f"✓ 전처리 데이터 저장: {processed_path}")
    
    # ======================================================================
    # STEP 2: 모델 학습
    # ======================================================================
    print("\n" + "=" * 80)
    print("STEP 2: 모델 학습")
    print("=" * 80)
    
    predictor = BusArrivalPredictor(args.model)
    
    try:
        predictor.train(
            processed_df,
            use_cv=args.cv,
            use_tuning=args.tune,
            verbose=True
        )
    except Exception as e:
        print(f"❌ 모델 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ======================================================================
    # STEP 3: 모델 저장
    # ======================================================================
    print("\n" + "=" * 80)
    print("STEP 3: 모델 저장")
    print("=" * 80)
    
    predictor.save(args.model)
    
    # ======================================================================
    # STEP 4: Feature Importance
    # ======================================================================
    if predictor.feature_importance_ is not None:
        print("\n" + "=" * 80)
        print("Feature Importance (Top 15)")
        print("=" * 80)
        
        for _, row in predictor.feature_importance_.head(15).iterrows():
            bar_length = int(row['importance'] * 50 / predictor.feature_importance_['importance'].max())
            bar = '█' * bar_length
            print(f"{row['feature']:40s} {bar} {row['importance']:.4f}")
    
    # ======================================================================
    # STEP 5: 모델 테스트
    # ======================================================================
    print("\n" + "=" * 80)
    print("STEP 5: 모델 테스트 (KeyError 검증)")
    print("=" * 80)
    
    test_samples = processed_df.sample(n=min(5, len(processed_df)), random_state=42)
    
    print(f"\n실제 데이터에서 {len(test_samples)}개 샘플 테스트:")
    
    errors = []
    success_count = 0
    
    for idx, (_, row) in enumerate(test_samples.iterrows(), 1):
        try:
            # Feature 준비 (arrtime 제외!)
            features = {
                'routeid': row['routeid'],
                'nodeid': row.get('nodeid', ''),
                'routetp': row['routetp'],
                'vehicletp': row['vehicletp'],
                'arrprevstationcnt': int(row['arrprevstationcnt']),
                'weekday': row['weekday'],
                'time_slot': row.get('time_slot', 'afternoon'),
                'weather': row.get('weather', 'Unknown'),
                'temp': float(row['temp']) if pd.notna(row['temp']) else 15.0,
                'humidity': float(row['humidity']) if pd.notna(row['humidity']) else 50.0,
                'rain_mm': float(row['rain_mm']) if pd.notna(row['rain_mm']) else 0.0,
                'snow_mm': float(row['snow_mm']) if pd.notna(row['snow_mm']) else 0.0,
                'hour': int(row['hour']),
                'minute': int(row['minute']),
                'day_of_week': int(row['day_of_week']),
                'is_weekend': int(row['is_weekend']),
                'is_rush_hour': int(row['is_rush_hour']),
                'avg_time_per_station': float(row.get('avg_time_per_station', 60.0))
            }
            
            # 예측 (여기서 KeyError가 발생하면 안 됨!)
            prediction = predictor.predict(features)
            actual = row['arrtime']
            error = abs(prediction - actual)
            error_pct = (error / actual * 100) if actual > 0 else 0
            
            errors.append(error)
            success_count += 1
            
            print(f"\n✓ 테스트 {idx}: 성공")
            print(f"  노선: {row['routeid']} ({row['routetp']})")
            print(f"  정류장: {row.get('nodeid', 'Unknown')}")
            print(f"  남은 정류장: {row['arrprevstationcnt']}개")
            print(f"  실제: {actual:.0f}초 ({actual/60:.1f}분)")
            print(f"  예측: {prediction:.0f}초 ({prediction/60:.1f}분)")
            print(f"  오차: {error:.0f}초 ({error_pct:.1f}%)")
            
        except KeyError as e:
            print(f"\n❌ 테스트 {idx}: KeyError 발생 - {e}")
            print("  → 학습/추론 비대칭 문제 발생!")
            
        except Exception as e:
            print(f"\n❌ 테스트 {idx}: 실패 - {e}")
    
    # ======================================================================
    # STEP 6: 결과 요약
    # ======================================================================
    print("\n" + "=" * 80)
    print("학습 완료!")
    print("=" * 80)
    print(f"모델 파일: {args.model}")
    print(f"처리된 데이터: {processed_path}")
    print(f"\n테스트 결과: {success_count}/{len(test_samples)} 성공")
    
    if success_count == len(test_samples):
        print("✓ KeyError 없이 모든 테스트 성공! 👍")
    else:
        print("⚠️  일부 테스트 실패. 로그를 확인하세요.")
    
    if errors:
        print(f"\n평균 오차: {np.mean(errors):.2f}초 ({np.mean(errors)/60:.2f}분)")
    
    if predictor.cv_results_ is not None:
        print(f"\nCross-Validation MAE: "
              f"{-predictor.cv_results_['test_neg_mean_absolute_error'].mean():.2f}초")
    
    print(f"\n서버 실행 명령:")
    print("  # 실시간 모드:")
    print("  python main.py")
    print("\n  # 시뮬레이션 모드:")
    print("  python main.py --simulation")
    print("=" * 80)


if __name__ == "__main__":
    main()