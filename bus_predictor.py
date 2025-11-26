import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import xgboost as xgb
import joblib
import warnings
from typing import Dict, Any, Optional, Tuple, List
warnings.filterwarnings('ignore')


class BusArrivalPredictor:
    """
    버스 도착 시간 예측 모델 (최종 버전)
    
    개선 사항:
    1. CV-aware Target Encoding (Data Leakage 완전 방지)
    2. Feature Interaction 추가 (routeid×hour 등)
    3. Ensemble 모델 (GBM + RF + XGBoost)
    4. Training/Inference 완전 분리
    5. Categorical Features 직접 사용
    """

    def __init__(self, model_path: str = "models/bus_predictor.pkl"):
        self.model = None
        self.model_path = model_path
        
        # Feature 관련
        self.feature_columns = None
        self.numeric_features = None
        self.categorical_features = None
        
        # Target Encoding (CV-aware)
        self.target_encoders = {}  # {feature_name: TargetEncoder}
        self.target_encoding_stats = {}  # Inference용 fallback
        
        # OneHot Encoding
        self.onehot_encoders = {}
        self.top_categories = {}
        
        # Statistics
        self.statistics = {
            'route_stats': {},
            'station_stats': {},
            'route_hour_stats': {},
            'route_station_stats': {}
        }
        
        # CV & Feature Importance
        self.cv_results_ = None
        self.feature_importance_ = None
        
        # Ensemble 여부
        self.use_ensemble = False

    # ==================================================================
    # 1. Statistics 계산
    # ==================================================================
    def _compute_statistics(self, df: pd.DataFrame) -> None:
        """통계 계산 및 저장"""
        # Route 통계
        route_stats = df.groupby('routeid')['arrtime'].agg(['mean', 'std', 'count']).reset_index()
        route_stats.columns = ['routeid', 'avg_time', 'std_time', 'count']
        route_stats['std_time'] = route_stats['std_time'].fillna(0)
        self.statistics['route_stats'] = route_stats.set_index('routeid').to_dict('index')
        
        # Station 통계
        station_stats = df.groupby('nodeid')['arrtime'].mean().reset_index()
        station_stats.columns = ['nodeid', 'avg_time']
        self.statistics['station_stats'] = station_stats.set_index('nodeid').to_dict('index')
        
        # Route + Hour 통계
        route_hour_stats = df.groupby(['routeid', 'hour'])['arrtime'].mean().reset_index()
        for _, row in route_hour_stats.iterrows():
            key = (row['routeid'], int(row['hour']))
            self.statistics['route_hour_stats'][key] = row['arrtime']
        
        # Route + Station Count 통계
        route_station_stats = df.groupby(['routeid', 'arrprevstationcnt'])['arrtime'].mean().reset_index()
        for _, row in route_station_stats.iterrows():
            key = (row['routeid'], int(row['arrprevstationcnt']))
            self.statistics['route_station_stats'][key] = row['arrtime']

    # ==================================================================
    # 2. CV-aware Target Encoding
    # ==================================================================
    def _apply_target_encoding_cv(self, X: pd.DataFrame, y: pd.Series, 
                                   cv_folds: int = 5) -> pd.DataFrame:
        """
        CV-aware Target Encoding
        
        각 fold에서 train 데이터의 통계만 사용하여 validation 인코딩
        → Data Leakage 완전 방지
        """
        X_encoded = X.copy()
        
        # Target Encoding 대상 features
        target_encode_cols = ['routeid', 'nodeid']  # 카디널리티 높은 것들
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for col in target_encode_cols:
            if col not in X.columns:
                continue
            
            # 각 fold별로 encoding
            encoded_col = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                # Train fold에서 통계 계산
                encoder = TargetEncoder(cols=[col], smoothing=1.0)
                encoder.fit(X_train_fold[[col]], y_train_fold)
                
                # Validation fold 인코딩
                encoded_val = encoder.transform(X_val_fold[[col]])[col].values
                encoded_col[val_idx] = encoded_val
            
            X_encoded[f'{col}_target_encoded'] = encoded_col
            
            # 전체 데이터로 encoder 저장 (inference용)
            final_encoder = TargetEncoder(cols=[col], smoothing=1.0)
            final_encoder.fit(X[[col]], y)
            self.target_encoders[col] = final_encoder
            
            # Fallback 통계 저장
            self.target_encoding_stats[col] = X_encoded.groupby(col)[f'{col}_target_encoded'].mean().to_dict()
        
        return X_encoded

    def _apply_target_encoding_inference(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inference용 Target Encoding (저장된 encoder 사용)"""
        X_encoded = X.copy()
        
        for col, encoder in self.target_encoders.items():
            if col in X.columns:
                try:
                    encoded = encoder.transform(X[[col]])[col]
                    X_encoded[f'{col}_target_encoded'] = encoded
                except:
                    # Fallback: 저장된 통계 사용
                    X_encoded[f'{col}_target_encoded'] = X[col].map(
                        self.target_encoding_stats[col]
                    ).fillna(300)  # Unknown은 평균값으로
        
        return X_encoded

    # ==================================================================
    # 3. Feature Interaction
    # ==================================================================
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Interaction 추가
        
        중요한 feature 조합:
        - routeid × hour
        - vehicletp × is_rush_hour
        - weather × temp
        - routeid × arrprevstationcnt
        """
        df = df.copy()
        
        # 1. routeid × hour (노선별 시간대 패턴)
        if 'routeid' in df.columns and 'hour' in df.columns:
            df['routeid_hour'] = df['routeid'].astype(str) + '_H' + df['hour'].astype(str)
        
        # 2. routeid × arrprevstationcnt (노선별 거리 패턴)
        if 'routeid' in df.columns and 'arrprevstationcnt' in df.columns:
            df['routeid_station'] = df['routeid'].astype(str) + '_S' + df['arrprevstationcnt'].astype(str)
        
        # 3. vehicletp × is_rush_hour (차종별 출퇴근 시간 성능)
        if 'vehicletp' in df.columns and 'is_rush_hour' in df.columns:
            df['vehicletp_rush'] = df['vehicletp'].astype(str) + '_R' + df['is_rush_hour'].astype(str)
        
        # 4. weather × temp (날씨×온도 조합)
        if 'weather' in df.columns and 'temp' in df.columns:
            # 온도 범주화
            temp_category = pd.cut(df['temp'], bins=[-np.inf, 0, 10, 20, 30, np.inf], 
                                   labels=['very_cold', 'cold', 'mild', 'warm', 'hot'])
            df['weather_temp'] = df['weather'].astype(str) + '_' + temp_category.astype(str)
        
        # 5. routetp × arrprevstationcnt (버스 타입별 거리)
        if 'routetp' in df.columns and 'arrprevstationcnt' in df.columns:
            station_category = pd.cut(df['arrprevstationcnt'], 
                                     bins=[-np.inf, 3, 10, np.inf],
                                     labels=['short', 'medium', 'long'])
            df['routetp_distance'] = df['routetp'].astype(str) + '_' + station_category.astype(str)
        
        # 6. 수치형 interaction (곱셈)
        df['station_hour_mult'] = df['arrprevstationcnt'] * df['hour']
        df['station_temp_mult'] = df['arrprevstationcnt'] * df['temp']
        df['rush_station_mult'] = df['is_rush_hour'] * df['arrprevstationcnt']
        
        return df

    # ==================================================================
    # 4. Feature Engineering
    # ==================================================================
    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 feature 생성"""
        df = df.copy()
        
        # Aggregation (lookup-based)
        df['route_avg_time'] = df['routeid'].map(
            lambda x: self.statistics['route_stats'].get(x, {}).get('avg_time', 300)
        )
        df['route_std_time'] = df['routeid'].map(
            lambda x: self.statistics['route_stats'].get(x, {}).get('std_time', 0)
        )
        df['station_avg_time'] = df['nodeid'].map(
            lambda x: self.statistics['station_stats'].get(x, {}).get('avg_time', 300)
        )
        df['route_hour_avg_time'] = df.apply(
            lambda row: self.statistics['route_hour_stats'].get(
                (row['routeid'], int(row['hour'])), 300
            ), axis=1
        )
        
        # Interaction (기존)
        df['station_hour_interaction'] = df['arrprevstationcnt'] * df['hour']
        df['rush_station_interaction'] = df['is_rush_hour'] * df['arrprevstationcnt']
        df['temp_station_interaction'] = df['temp'] * df['arrprevstationcnt']
        
        # 날씨
        df['bad_weather'] = ((df['rain_mm'] > 0) | (df['snow_mm'] > 0)).astype(int)
        df['weather_rush_interaction'] = df['bad_weather'] * df['is_rush_hour']
        
        # Domain-specific
        df['time_of_day'] = pd.cut(
            df['hour'], bins=[0, 6, 12, 18, 24],
            labels=[0, 1, 2, 3], include_lowest=True
        ).astype(int)
        
        df['extreme_cold'] = (df['temp'] < 0).astype(int)
        df['extreme_hot'] = (df['temp'] > 30).astype(int)
        
        df['rain_level'] = pd.cut(
            df['rain_mm'], bins=[-np.inf, 0, 5, 20, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        df['station_distance_category'] = pd.cut(
            df['arrprevstationcnt'], bins=[-np.inf, 3, 10, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        
        df['avg_time_per_station'] = df['route_avg_time'] / np.maximum(df['arrprevstationcnt'], 1)
        
        return df

    # ==================================================================
    # 5. Categorical Encoding (OneHot)
    # ==================================================================
    def _prepare_onehot_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """OneHot Encoding (interaction features 포함)"""
        df = df.copy()
        
        # OneHot 대상 (카디널리티 낮은 것들 + interaction)
        onehot_cols = [
            'routetp', 'vehicletp', 'weather', 'weekday',
            'routeid_hour', 'vehicletp_rush', 'weather_temp', 'routetp_distance'
        ]
        
        available_cols = [c for c in onehot_cols if c in df.columns]
        
        if fit:
            for col in available_cols:
                # Top-N
                top_n = 30 if col.endswith('_hour') else 20
                value_counts = df[col].value_counts()
                self.top_categories[col] = value_counts.head(top_n).index.tolist()
                
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: x if x in self.top_categories[col] else 'OTHER'
                )
                
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoder.fit(df[[f'{col}_encoded']])
                self.onehot_encoders[col] = encoder
        
        else:
            for col in available_cols:
                if col in self.onehot_encoders:
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: x if x in self.top_categories.get(col, []) else 'OTHER'
                    )
        
        # OneHot 적용
        encoded_dfs = []
        for col in available_cols:
            if col in self.onehot_encoders:
                encoded = self.onehot_encoders[col].transform(df[[f'{col}_encoded']])
                feature_names = [f'{col}_{cat}' for cat in self.onehot_encoders[col].categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                encoded_dfs.append(encoded_df)
        
        if encoded_dfs:
            df = pd.concat([df] + encoded_dfs, axis=1)
        
        return df

    # ==================================================================
    # 6. Ensemble Model
    # ==================================================================
    def _create_ensemble_model(self, best_params_gbm: dict = None) -> VotingRegressor:
        """
        Ensemble 모델 생성 (GBM + RF + XGBoost)
        
        각 모델의 장점:
        - GBM: 순차적 학습, 에러 보정
        - RF: 병렬 학습, 과적합 방지
        - XGBoost: 최적화된 GBM, 빠른 속도
        """
        # GradientBoosting
        if best_params_gbm:
            gbm = GradientBoostingRegressor(**best_params_gbm, random_state=42)
        else:
            gbm = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            )
        
        # RandomForest
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Voting Ensemble (평균)
        ensemble = VotingRegressor([
            ('gbm', gbm),
            ('rf', rf),
            ('xgb', xgb_model)
        ])
        
        return ensemble

    # ==================================================================
    # 7. Training
    # ==================================================================
    def train(self, df: pd.DataFrame, target_col: str = 'arrtime',
              use_cv: bool = True, use_tuning: bool = False,
              use_ensemble: bool = True, verbose: bool = True) -> None:
        """모델 학습 (최종 버전)"""
        
        if verbose:
            print("=" * 80)
            print("Feature Engineering 시작 (최종 버전)")
            print("=" * 80)
        
        self.use_ensemble = use_ensemble
        
        # 기본 전처리
        df = df.copy()
        df['weather'] = df['weather'].fillna('Unknown')
        df['temp'] = df['temp'].fillna(df['temp'].median())
        df['humidity'] = df['humidity'].fillna(df['humidity'].median())
        df['rain_mm'] = df['rain_mm'].fillna(0.0)
        df['snow_mm'] = df['snow_mm'].fillna(0.0)
        
        # 1. Statistics 계산
        if verbose:
            print("\n1. Statistics 계산...")
        self._compute_statistics(df)
        
        # 2. Base Features
        if verbose:
            print("\n2. Base Features 생성...")
        df = self._add_base_features(df)
        
        # 3. Feature Interactions
        if verbose:
            print("\n3. Feature Interactions 생성...")
        df = self._add_feature_interactions(df)
        
        # 4. Target Encoding (CV-aware)
        if verbose:
            print("\n4. Target Encoding (CV-aware)...")
        df = self._apply_target_encoding_cv(df, df[target_col])
        
        # 5. OneHot Encoding
        if verbose:
            print("\n5. OneHot Encoding...")
        df = self._prepare_onehot_features(df, fit=True)
        
        # 6. Feature Columns 정의
        numeric_features = [
            'arrprevstationcnt', 'hour', 'minute', 'day_of_week',
            'is_weekend', 'is_rush_hour', 'temp', 'humidity',
            'rain_mm', 'snow_mm', 'avg_time_per_station',
            'station_hour_interaction', 'rush_station_interaction',
            'temp_station_interaction', 'weather_rush_interaction',
            'route_avg_time', 'route_std_time', 'station_avg_time',
            'route_hour_avg_time',
            'time_of_day', 'extreme_cold', 'extreme_hot',
            'rain_level', 'station_distance_category', 'bad_weather',
            'station_hour_mult', 'station_temp_mult', 'rush_station_mult',
            'routeid_target_encoded', 'nodeid_target_encoded'
        ]
        
        onehot_features = []
        for col in self.onehot_encoders.keys():
            cats = self.onehot_encoders[col].categories_[0]
            onehot_features.extend([f'{col}_{cat}' for cat in cats])
        
        self.feature_columns = [f for f in numeric_features if f in df.columns]
        self.feature_columns += [f for f in onehot_features if f in df.columns]
        self.numeric_features = numeric_features
        
        if verbose:
            print(f"\n✓ Total Features: {len(self.feature_columns)}")
            print(f"  - Numeric: {len([f for f in self.feature_columns if f not in onehot_features])}")
            print(f"  - OneHot: {len([f for f in self.feature_columns if f in onehot_features])}")
            print(f"  - Target Encoded: 2 (routeid, nodeid)")
            print(f"  - Interactions: 8")
        
        # 7. 데이터 준비
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        X = df[self.feature_columns]
        y = df[target_col]
        
        if verbose:
            print(f"\n학습 데이터: {len(X):,} rows × {len(self.feature_columns)} features")
        
        # 8. 모델 생성
        if use_tuning:
            if verbose:
                print("\n" + "=" * 80)
                print("Hyperparameter Tuning...")
            best_params = self._tune_hyperparameters(X, y, verbose)
            
            if use_ensemble:
                self.model = self._create_ensemble_model(best_params)
            else:
                self.model = GradientBoostingRegressor(**best_params, random_state=42)
        else:
            if use_ensemble:
                if verbose:
                    print("\n" + "=" * 80)
                    print("Ensemble 모델 생성 (GBM + RF + XGBoost)...")
                self.model = self._create_ensemble_model()
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=7,
                    min_samples_split=10, min_samples_leaf=4, subsample=0.8,
                    random_state=42
                )
        
        # 9. Cross-Validation
        if use_cv:
            if verbose:
                print("\n" + "=" * 80)
                print("Cross-Validation 수행...")
            self._perform_cross_validation(X, y, verbose)
        
        # 10. 최종 학습
        if verbose:
            print("\n" + "=" * 80)
            print("최종 모델 학습...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # 평가
        train_r2 = self.model.score(X_train, y_train)
        val_r2 = self.model.score(X_val, y_val)
        
        y_pred = self.model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        
        if verbose:
            print(f"\n최종 성능:")
            print(f"  R² Train: {train_r2:.4f}")
            print(f"  R² Val:   {val_r2:.4f}")
            print(f"  MAE:      {mae:.2f}초 ({mae/60:.2f}분)")
            print(f"  RMSE:     {rmse:.2f}초")
        
        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            if verbose:
                print(f"\nTop 10 Important Features:")
                for _, row in self.feature_importance_.head(10).iterrows():
                    print(f"  {row['feature']:40s}: {row['importance']:.4f}")
        elif use_ensemble:
            # Ensemble의 경우 개별 모델 중 하나의 importance 사용
            if hasattr(self.model.estimators_[0], 'feature_importances_'):
                self.feature_importance_ = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.estimators_[0].feature_importances_
                }).sort_values('importance', ascending=False)

    def _tune_hyperparameters(self, X, y, verbose):
        """Hyperparameter Tuning"""
        param_dist = {
            'n_estimators': [100, 150, 200, 250, 300],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'max_depth': [5, 7, 9, 11],
            'min_samples_split': [5, 10, 15, 20],
            'min_samples_leaf': [2, 4, 6],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 0.5, 0.7]
        }
        
        random_search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_dist, n_iter=30, cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1, random_state=42,
            verbose=1 if verbose else 0
        )
        
        random_search.fit(X, y)
        
        if verbose:
            print(f"\n✓ Best Parameters:")
            for key, value in random_search.best_params_.items():
                print(f"  {key}: {value}")
            print(f"\n✓ Best MAE: {-random_search.best_score_:.2f}초")
        
        return random_search.best_params_

    def _perform_cross_validation(self, X, y, verbose):
        """Cross-Validation"""
        cv_results = cross_validate(
            self.model, X, y, cv=5,
            scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'],
            return_train_score=True, n_jobs=-1
        )
        
        self.cv_results_ = cv_results
        
        if verbose:
            print(f"\n✓ Cross-Validation Results (5-Fold):")
            print(f"  MAE:  {-cv_results['test_neg_mean_absolute_error'].mean():.2f} "
                  f"(±{cv_results['test_neg_mean_absolute_error'].std():.2f})초")
            print(f"  RMSE: {-cv_results['test_neg_root_mean_squared_error'].mean():.2f} "
                  f"(±{cv_results['test_neg_root_mean_squared_error'].std():.2f})초")
            print(f"  R²:   {cv_results['test_r2'].mean():.4f} "
                  f"(±{cv_results['test_r2'].std():.4f})")

    # ==================================================================
    # 8. Prediction
    # ==================================================================
    def predict(self, features: Dict[str, Any]) -> float:
        """예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        try:
            df = pd.DataFrame([features])
            
            # Feature engineering
            df = self._add_base_features(df)
            df = self._add_feature_interactions(df)
            df = self._apply_target_encoding_inference(df)
            df = self._prepare_onehot_features(df, fit=False)
            
            # 결측치 처리
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(0)
            
            pred = float(self.model.predict(df[self.feature_columns])[0])
            return max(pred, 0)

        except Exception as e:
            print(f"[예측 오류] {e}")
            import traceback
            traceback.print_exc()
            raise e

    # ==================================================================
    # 9. Historical Fallback
    # ==================================================================
    def predict_by_historical_pattern(self, df: pd.DataFrame,
                                      routeid: str, nodeid: str,
                                      prev_station_cnt: int,
                                      weekday: str, hour: int) -> float:
        """Historical fallback"""
        df2 = df.copy()

        mask = (
            (df2['routeid'].astype(str) == str(routeid)) &
            (df2['nodeid'].astype(str) == str(nodeid)) &
            (df2['arrprevstationcnt'] == prev_station_cnt) &
            (df2['weekday'].astype(str) == str(weekday)) &
            (df2['hour'] == hour)
        )
        if len(df2[mask]) > 0:
            return float(df2[mask]['arrtime'].median())

        mask = (
            (df2['routeid'].astype(str) == str(routeid)) &
            (df2['nodeid'].astype(str) == str(nodeid)) &
            (df2['arrprevstationcnt'] == prev_station_cnt)
        )
        if len(df2[mask]) > 0:
            return float(df2[mask]['arrtime'].median())

        mask = (
            (df2['routeid'].astype(str) == str(routeid)) &
            (df2['arrprevstationcnt'] == prev_station_cnt)
        )
        if len(df2[mask]) > 0:
            return float(df2[mask]['arrtime'].median())

        return float(prev_station_cnt * 70)

    # ==================================================================
    # 10. Save / Load
    # ==================================================================
    def save(self, path: Optional[str] = None) -> None:
        save_path = path or self.model_path
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_encoders": self.target_encoders,
            "target_encoding_stats": self.target_encoding_stats,
            "onehot_encoders": self.onehot_encoders,
            "top_categories": self.top_categories,
            "statistics": self.statistics,
            "feature_importance": self.feature_importance_,
            "cv_results": self.cv_results_,
            "use_ensemble": self.use_ensemble
        }, save_path)
        print(f"모델 저장 완료: {save_path}")

    def load(self, path: Optional[str] = None) -> 'BusArrivalPredictor':
        load_path = path or self.model_path
        data = joblib.load(load_path)

        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.numeric_features = data.get('numeric_features')
        self.categorical_features = data.get('categorical_features')
        self.target_encoders = data.get('target_encoders', {})
        self.target_encoding_stats = data.get('target_encoding_stats', {})
        self.onehot_encoders = data.get('onehot_encoders', {})
        self.top_categories = data.get('top_categories', {})
        self.statistics = data.get('statistics', {})
        self.feature_importance_ = data.get('feature_importance')
        self.cv_results_ = data.get('cv_results')
        self.use_ensemble = data.get('use_ensemble', False)

        print(f"모델 로드 완료: {load_path}")
        return self