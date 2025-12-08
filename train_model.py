import argparse
import os
from data_preprocessing import BusDataPreprocessor
from bus_predictor import BusArrivalPredictor


def main():
    parser = argparse.ArgumentParser(description="버스 도착 예측 모델 학습")
    parser.add_argument("--data", default="bus_arrivals.csv", help="원본 데이터 파일")
    parser.add_argument("--cv", action="store_true", help="교차 검증 수행")
    parser.add_argument("--ensemble", action="store_true", help="앙상블 모델 사용")
    parser.add_argument("--no-preprocess", action="store_true", 
                       help="전처리 건너뛰기 (이미 전처리된 데이터 사용)")
    parser.add_argument("--use-actual", action="store_true",
                       help="실제 도착 시간 라벨 사용 (기본: API 라벨)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("버스 도착 예측 모델 학습")
    print("=" * 70)
    
    # ============================================================
    # 1. 데이터 전처리
    # ============================================================
    processed_path = "data/processed_bus_arrivals.csv"
    
    if args.no_preprocess and os.path.exists(processed_path):
        print(f"\n📂 전처리된 데이터 로드: {processed_path}")
        import pandas as pd
        processed_df = pd.read_csv(processed_path)
        print(f"  → 사용 가능한 Feature 수: {processed_df.shape[1]}개")
        print(f"  → Feature 목록 일부: {processed_df.columns[:12].tolist()} ...")
        print(f"  → {len(processed_df):,} rows")
        
        # 라벨 타입 확인
        if 'actual_arrtime' in processed_df.columns:
            print(f"  → 라벨 타입: 실제 도착 시간")
            use_actual_labels = True
        else:
            print(f"  → 라벨 타입: API 라벨")
            use_actual_labels = False
    else:
        print(f"\n🔧 데이터 전처리 시작...")
        use_actual_labels = args.use_actual
        
        preprocessor = BusDataPreprocessor(args.data)
        
        df = preprocessor.load_data()
        print(f"  → 원본 데이터: {len(df):,} rows")
        
        # 전처리 실행
        processed_df = preprocessor.prepare_training_data(
            df,
            use_actual_labels=use_actual_labels,
            validate=True,
            verbose=True
        )
        
        if processed_df is None or len(processed_df) == 0:
            print("❌ 전처리 실패!")
            return
        
        # 통일된 파일명으로 저장
        os.makedirs("data", exist_ok=True)
        processed_df.to_csv(processed_path, index=False)
        print(f"\n✓ 전처리 완료: {processed_path}")
        print(f"  → {len(processed_df):,} rows")
        
        if use_actual_labels:
            print(f"  → 라벨: 실제 도착 시간 (actual_arrtime)")
        else:
            print(f"  → 라벨: API 라벨 (arrtime)")
    
    # ============================================================
    # 2. 모델 학습
    # ============================================================
    print(f"\n🤖 모델 학습 시작...")
    print(f"  교차 검증: {'Yes' if args.cv else 'No'}")
    print(f"  앙상블: {'Yes' if args.ensemble else 'No'}")
    
    # 타겟 컬럼 명시적 지정
    if 'actual_arrtime' in processed_df.columns:
        target_col = 'actual_arrtime'
        print(f"  타겟 컬럼: {target_col} (실제 도착 시간)")
    else:
        target_col = 'arrtime'
        print(f"  타겟 컬럼: {target_col} (API 예측)")
    
    predictor = BusArrivalPredictor()
    
    # 학습 (target_col 명시적 전달)
    predictor.train(
        processed_df,
        target_col=target_col,  # ★ EXPLICIT
        use_cv=args.cv,
        use_ensemble=args.ensemble
    )

    # 1) route별 평균 sec_per_station
    predictor.statistics['route_sec_per_station'] = (
        processed_df.groupby('routeid')['sec_per_station']
        .mean()
        .dropna()
        .to_dict()
    )

    # 2) node별 평균 sec_per_station
    predictor.statistics['node_sec_per_station'] = (
        processed_df.groupby('nodeid')['sec_per_station']
        .mean()
        .dropna()
        .to_dict()
    )

    # 3) route + hour 평균 sec_per_station
    route_hour = (
        processed_df.groupby(['routeid', 'hour'])['sec_per_station']
        .mean()
        .dropna()
    )
    predictor.statistics['route_hour_sec_per_station'] = {
        (r, int(h)): v for (r, h), v in route_hour.items()
    }

    # 4) route별 max_station_route (station_progress_ratio 계산용)
    predictor.statistics['route_max_station'] = (
        processed_df.groupby('routeid')['arrprevstationcnt']
        .max()
        .to_dict()
    )
    
    # ============================================================
    # 3. 모델 저장
    # ============================================================
    os.makedirs("models", exist_ok=True)
    model_path = "models/bus_predictor.pkl"
    
    predictor.save(model_path)
    print(f"\n✓ 모델 저장 완료: {model_path}")
    
    # ============================================================
    # 4. 최종 안내
    # ============================================================
    print("\n" + "=" * 70)
    print("학습 완료!")
    print("=" * 70)
    print(f"\n저장된 파일:")
    print(f"  - 전처리 데이터: {processed_path}")
    print(f"  - 모델: {model_path}")
    
    print(f"\n서버 실행 명령:")
    print(f"  # 실시간 모드:")
    print(f"  python main.py")
    print(f"\n  # 시뮬레이션 모드:")
    print(f"  python main.py --simulation")
    
    print(f"\n재학습 명령:")
    print(f"  # 전처리 재사용:")
    print(f"  python train_model.py --no-preprocess --cv --ensemble")
    print(f"\n  # 전체 재실행:")
    print(f"  python train_model.py --cv --ensemble")
    
    if use_actual_labels:
        print(f"\n⚠️  주의: 실제 도착 시간 라벨 사용")
        print(f"  → 더 정확하지만, 데이터 수집 기간 필요")
    


if __name__ == "__main__":
    main()