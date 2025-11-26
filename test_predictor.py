#!/usr/bin/env python3
"""
BusArrivalPredictor 검증 스크립트

피드백에서 지적된 3가지 문제 해결 확인:
1. KeyError 발생 여부
2. Target Leakage 제거 확인
3. Categorical Features 사용 확인
"""

import pandas as pd
import numpy as np
from bus_predictor import BusArrivalPredictor


def test_1_keyerror_check():
    """테스트 1: Inference에서 KeyError 발생 여부"""
    print("=" * 80)
    print("테스트 1: KeyError 검사")
    print("=" * 80)
    
    # 간단한 학습 데이터 생성
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'routeid': np.random.choice(['GMB101', 'GMB102', 'GMB103'], n),
        'nodeid': np.random.choice(['NODE1', 'NODE2'], n),
        'routetp': np.random.choice(['일반버스', '좌석버스'], n),
        'vehicletp': np.random.choice(['일반차량', '저상버스'], n),
        'weekday': np.random.choice(['Mon', 'Tue', 'Wed'], n),
        'arrprevstationcnt': np.random.randint(1, 20, n),
        'arrtime': np.random.randint(100, 1000, n),
        'hour': np.random.randint(6, 22, n),
        'minute': np.random.randint(0, 60, n),
        'day_of_week': np.random.randint(0, 7, n),
        'is_weekend': np.random.randint(0, 2, n),
        'is_rush_hour': np.random.randint(0, 2, n),
        'temp': np.random.uniform(5, 30, n),
        'humidity': np.random.uniform(30, 80, n),
        'rain_mm': np.random.uniform(0, 10, n),
        'snow_mm': np.zeros(n),
        'weather': np.random.choice(['Clear', 'Rain', 'Cloudy'], n),
        'time_slot': np.random.choice(['morning', 'afternoon', 'evening'], n),
        'avg_time_per_station': np.random.uniform(40, 80, n)
    })
    
    predictor = BusArrivalPredictor()
    
    try:
        print("\n학습 중...")
        predictor.train(df, use_cv=False, use_tuning=False, verbose=False)
        print("✓ 학습 성공")
    except Exception as e:
        print(f"❌ 학습 실패: {e}")
        return False
    
    # Inference 테스트 (arrtime 없이!)
    print("\nInference 테스트 (arrtime 제외)...")
    
    test_features = {
        'routeid': 'GMB101',
        'nodeid': 'NODE1',
        'routetp': '일반버스',
        'vehicletp': '일반차량',
        'weekday': 'Mon',
        'arrprevstationcnt': 10,
        # arrtime 없음!
        'hour': 14,
        'minute': 30,
        'day_of_week': 0,
        'is_weekend': 0,
        'is_rush_hour': 0,
        'temp': 20.0,
        'humidity': 50.0,
        'rain_mm': 0.0,
        'snow_mm': 0.0,
        'weather': 'Clear',
        'time_slot': 'afternoon',
        'avg_time_per_station': 60.0
    }
    
    try:
        prediction = predictor.predict(test_features)
        print(f"✓ 예측 성공: {prediction:.2f}초")
        print("✓ KeyError 없음!")
        return True
    except KeyError as e:
        print(f"❌ KeyError 발생: {e}")
        print("→ 학습/추론 비대칭 문제 존재!")
        return False
    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        return False


def test_2_target_leakage_check():
    """테스트 2: Target Leakage 제거 확인"""
    print("\n" + "=" * 80)
    print("테스트 2: Target Leakage 검사")
    print("=" * 80)
    
    # 학습 데이터
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'routeid': np.random.choice(['GMB101', 'GMB102', 'GMB103'], n),
        'nodeid': np.random.choice(['NODE1', 'NODE2'], n),
        'routetp': np.random.choice(['일반버스', '좌석버스'], n),
        'vehicletp': np.random.choice(['일반차량', '저상버스'], n),
        'weekday': np.random.choice(['Mon', 'Tue', 'Wed'], n),
        'arrprevstationcnt': np.random.randint(1, 20, n),
        'arrtime': np.random.randint(100, 1000, n),
        'hour': np.random.randint(6, 22, n),
        'minute': np.random.randint(0, 60, n),
        'day_of_week': np.random.randint(0, 7, n),
        'is_weekend': np.random.randint(0, 2, n),
        'is_rush_hour': np.random.randint(0, 2, n),
        'temp': np.random.uniform(5, 30, n),
        'humidity': np.random.uniform(30, 80, n),
        'rain_mm': np.random.uniform(0, 10, n),
        'snow_mm': np.zeros(n),
        'weather': np.random.choice(['Clear', 'Rain', 'Cloudy'], n),
        'time_slot': np.random.choice(['morning', 'afternoon', 'evening'], n),
        'avg_time_per_station': np.random.uniform(40, 80, n)
    })
    
    predictor = BusArrivalPredictor()
    predictor.train(df, use_cv=False, use_tuning=False, verbose=False)
    
    # Feature columns 확인
    print("\nFeature columns 검사:")
    print(f"Total features: {len(predictor.feature_columns)}")
    
    # arrtime 직접 사용 여부 확인
    arrtime_features = [f for f in predictor.feature_columns if 'arrtime' in f.lower()]
    
    if arrtime_features:
        print(f"⚠️  arrtime 관련 feature 발견:")
        for f in arrtime_features:
            print(f"  - {f}")
        print("→ Target Leakage 위험!")
        return False
    else:
        print("✓ arrtime 직접 사용 없음")
        
    # time_efficiency 제거 확인
    if 'time_efficiency' in predictor.feature_columns:
        print("⚠️  time_efficiency feature 발견")
        print("→ arrtime / arrprevstationcnt 사용 → Leakage!")
        return False
    else:
        print("✓ time_efficiency 제거됨")
    
    print("✓ Target Leakage 제거 확인!")
    return True


def test_3_categorical_features_check():
    """테스트 3: Categorical Features 사용 확인"""
    print("\n" + "=" * 80)
    print("테스트 3: Categorical Features 검사")
    print("=" * 80)
    
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'routeid': np.random.choice(['GMB101', 'GMB102', 'GMB103'], n),
        'nodeid': np.random.choice(['NODE1', 'NODE2'], n),
        'routetp': np.random.choice(['일반버스', '좌석버스'], n),
        'vehicletp': np.random.choice(['일반차량', '저상버스'], n),
        'weekday': np.random.choice(['Mon', 'Tue', 'Wed'], n),
        'arrprevstationcnt': np.random.randint(1, 20, n),
        'arrtime': np.random.randint(100, 1000, n),
        'hour': np.random.randint(6, 22, n),
        'minute': np.random.randint(0, 60, n),
        'day_of_week': np.random.randint(0, 7, n),
        'is_weekend': np.random.randint(0, 2, n),
        'is_rush_hour': np.random.randint(0, 2, n),
        'temp': np.random.uniform(5, 30, n),
        'humidity': np.random.uniform(30, 80, n),
        'rain_mm': np.random.uniform(0, 10, n),
        'snow_mm': np.zeros(n),
        'weather': np.random.choice(['Clear', 'Rain', 'Cloudy'], n),
        'time_slot': np.random.choice(['morning', 'afternoon', 'evening'], n),
        'avg_time_per_station': np.random.uniform(40, 80, n)
    })
    
    predictor = BusArrivalPredictor()
    predictor.train(df, use_cv=False, use_tuning=False, verbose=False)
    
    # Categorical features 확인
    print("\nCategorical features 검사:")
    
    expected_cats = ['routeid', 'routetp', 'vehicletp', 'weather', 'weekday']
    found_cats = []
    
    for cat in expected_cats:
        cat_features = [f for f in predictor.feature_columns if f.startswith(f'{cat}_')]
        if cat_features:
            found_cats.append(cat)
            print(f"✓ {cat}: {len(cat_features)}개 OneHot features")
    
    if len(found_cats) >= 3:  # 최소 3개 이상
        print(f"\n✓ Categorical features 사용 확인! ({len(found_cats)}/5)")
        return True
    else:
        print(f"\n⚠️  Categorical features 부족: {len(found_cats)}/5")
        print("→ 노선/차종별 패턴을 제대로 학습 못 함!")
        return False


def run_all_tests():
    """전체 테스트 실행"""
    print("\n")
    print("█" * 80)
    print(" " * 20 + "BusArrivalPredictor 검증 테스트")
    print("█" * 80)
    print()
    
    results = []
    
    # Test 1: KeyError
    result1 = test_1_keyerror_check()
    results.append(("KeyError 검사", result1))
    
    # Test 2: Target Leakage
    result2 = test_2_target_leakage_check()
    results.append(("Target Leakage 검사", result2))
    
    # Test 3: Categorical Features
    result3 = test_3_categorical_features_check()
    results.append(("Categorical Features 검사", result3))
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✓ 통과" if result else "❌ 실패"
        print(f"{test_name:30s}: {status}")
    
    total_passed = sum(r for _, r in results)
    print(f"\n총 {total_passed}/{len(results)} 통과")
    
    if total_passed == len(results):
        print("\n🎉 모든 테스트 통과! 수정 완료!")
        return True
    else:
        print("\n⚠️  일부 테스트 실패. 추가 수정 필요.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)