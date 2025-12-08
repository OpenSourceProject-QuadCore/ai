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
    버스 도착 시간 예측 모델 (성능 개선 버전)
    
    주요 개선사항:
    1. ✅ route_based_features 활용 추가
    2. ✅ Ensemble 가중치 CV 기반 자동화
    3. 출퇴근 시간 세밀화 (분 단위 혼잡도)
    4. 요일×시간 interaction 강화
    5. 거리별 Feature 강화
    """

    def __init__(self, model_path: str = "models/bus_predictor.pkl"):
        self.model = None
        self.model_path = model_path
        
        # Feature 관련
        self.feature_columns = None
        self.numeric_features = None
        self.categorical_features = None
        
        # Target Encoding (CV-aware)
        self.target_encoders = {}
        self.target_encoding_stats = {}
        
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
        
        # Ensemble 여부 & 가중치
        self.use_ensemble = False
        self.ensemble_weights_ = None  # ★ CV 기반 가중치 저장

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
        """CV-aware Target Encoding"""
        X_encoded = X.copy()
        
        target_encode_cols = ['routeid', 'nodeid']
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for col in target_encode_cols:
            if col not in X.columns:
                continue
            
            encoded_col = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                encoder = TargetEncoder(cols=[col], smoothing=1.0)
                encoder.fit(X_train_fold[[col]], y_train_fold)
                
                encoded_val = encoder.transform(X_val_fold[[col]])[col].values
                encoded_col[val_idx] = encoded_val
            
            X_encoded[f'{col}_target_encoded'] = encoded_col
            
            final_encoder = TargetEncoder(cols=[col], smoothing=1.0)
            final_encoder.fit(X[[col]], y)
            self.target_encoders[col] = final_encoder
            
            self.target_encoding_stats[col] = X_encoded.groupby(col)[f'{col}_target_encoded'].mean().to_dict()
        
        return X_encoded

    def _apply_target_encoding_inference(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inference용 Target Encoding"""
        X_encoded = X.copy()
        
        for col, encoder in self.target_encoders.items():
            if col in X.columns:
                try:
                    encoded = encoder.transform(X[[col]])[col]
                    X_encoded[f'{col}_target_encoded'] = encoded
                except:
                    X_encoded[f'{col}_target_encoded'] = X[col].map(
                        self.target_encoding_stats[col]
                    ).fillna(300)
        
        return X_encoded

    # ==================================================================
    # 3. Feature Interaction (개선!)
    # ==================================================================
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature Interaction 추가 (개선 버전)"""
        df = df.copy()
        
        # 출퇴근 시간 세밀화 (분 단위)
        df['time_in_minutes'] = df['hour'] * 60 + df['minute']
        
        # 러시아워 레벨 (7단계)
        rush_bins = [0, 420, 480, 540, 1020, 1080, 1140, 1440]
        rush_labels = [0, 1, 2, 3, 4, 5, 6]
        
        df['rush_level'] = pd.cut(
            df['time_in_minutes'],
            bins=rush_bins,
            labels=rush_labels,
            include_lowest=True
        ).astype(int)
        
        # 출퇴근 피크 여부
        df['is_peak_rush'] = (
            ((df['hour'] == 8) & (df['minute'] >= 0) & (df['minute'] < 30)) |
            ((df['hour'] == 17) & (df['minute'] >= 30)) |
            ((df['hour'] == 18) & (df['minute'] < 30))
        ).astype(int)
        
        # 요일×시간 강화
        df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
        
        df['weekday_category'] = df['day_of_week'].apply(
            lambda x: 'mon_thu' if x < 4 else ('fri' if x == 4 else ('sat' if x == 5 else 'sun'))
        )
        
        df['weekday_hour'] = df['weekday_category'] + '_H' + df['hour'].astype(str)
        df['weekday_rush'] = df['weekday_category'] + '_R' + df['rush_level'].astype(str)
        
        # 거리 카테고리 (5단계)
        distance_bins = [-np.inf, 2, 5, 10, 20, np.inf]
        distance_labels = ['very_short', 'short', 'medium', 'long', 'very_long']
        
        df['distance_category'] = pd.cut(
            df['arrprevstationcnt'],
            bins=distance_bins,
            labels=distance_labels
        )
        
        # 온도 카테고리
        temp_bins = [-np.inf, -5, 0, 10, 20, 25, 30, np.inf]
        temp_labels = ['freezing', 'very_cold', 'cold', 'mild', 'warm', 'hot', 'very_hot']
        
        df['temp_category'] = pd.cut(
            df['temp'],
            bins=temp_bins,
            labels=temp_labels
        )
        
        # 날씨×러시레벨
        df['weather_rush'] = df['weather'].astype(str) + '_R' + df['rush_level'].astype(str)
        
        # 비/눈×러시레벨
        df['bad_weather'] = ((df['rain_mm'] > 0) | (df['snow_mm'] > 0)).astype(int)
        df['bad_weather_rush'] = df['bad_weather'].astype(str) + '_R' + df['rush_level'].astype(str)
        
        # 기존 Interaction
        if 'routeid' in df.columns and 'hour' in df.columns:
            df['routeid_hour'] = df['routeid'].astype(str) + '_H' + df['hour'].astype(str)
        
        if 'routeid' in df.columns and 'arrprevstationcnt' in df.columns:
            df['routeid_station'] = df['routeid'].astype(str) + '_S' + df['arrprevstationcnt'].astype(str)
        
        if 'routeid' in df.columns:
            df['routeid_rush'] = df['routeid'].astype(str) + '_R' + df['rush_level'].astype(str)
        
        if 'vehicletp' in df.columns:
            df['vehicletp_rush'] = df['vehicletp'].astype(str) + '_R' + df['rush_level'].astype(str)
        
        if 'routetp' in df.columns:
            df['routetp_distance'] = df['routetp'].astype(str) + '_' + df['distance_category'].astype(str)
        
        # 수치형 interaction
        df['station_hour_mult'] = df['arrprevstationcnt'] * df['hour']
        df['station_rush_mult'] = df['arrprevstationcnt'] * df['rush_level']
        df['station_temp_mult'] = df['arrprevstationcnt'] * df['temp']
        df['peak_station_mult'] = df['is_peak_rush'] * df['arrprevstationcnt']
        
        return df

    # ==================================================================
    # 4. Feature Engineering
    # ==================================================================
    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 feature 생성"""
        df = df.copy()
        
        # Aggregation
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
        """OneHot Encoding"""
        df = df.copy()
        
        onehot_cols = [
            'routetp', 'vehicletp', 'weather', 'weekday',
            'routeid_hour', 'routeid_rush', 'vehicletp_rush',
            'weather_rush', 'bad_weather_rush',
            'weekday_hour', 'weekday_rush',
            'routetp_distance'
        ]
        
        available_cols = [c for c in onehot_cols if c in df.columns]
        
        if fit:
            for col in available_cols:
                if '_' in col and col not in ['routetp', 'vehicletp']:
                    top_n = 50
                else:
                    top_n = 20
                
                value_counts = df[col].value_counts()
                self.top_categories[col] = value_counts.head(top_n).index.tolist()
                
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: x if x in self.top_categories[col] else 'OTHER'
                )
                
                try:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                except TypeError:
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
    # 6. Ensemble Model with CV-based Weights (★ 개선!)
    # ==================================================================
    def _create_ensemble_model(self, X, y, best_params_gbm: dict = None, verbose: bool = True) -> VotingRegressor:
        """
        Ensemble 모델 생성 (★★★ CV 기반 가중치 자동화! ★★★)
        """
        
        # ============================================================
        # ★★★ 각 모델별 CV 수행하여 성능 측정! ★★★
        # ============================================================
        
        # GradientBoosting
        if best_params_gbm:
            gbm = GradientBoostingRegressor(**best_params_gbm, random_state=42)
        else:
            gbm = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=5,
                subsample=0.85,
                max_features='sqrt',
                random_state=42
            )
        
        # RandomForest
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=15,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=8,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("개별 모델 CV 성능 측정 (가중치 계산용)")
            print("=" * 60)
        
        # ============================================================
        # ★★★ CV로 각 모델 성능 측정! ★★★
        # ============================================================
        models = {
            'GradientBoosting': gbm,
            'RandomForest': rf,
            'XGBoost': xgb_model
        }
        
        scores = {}
        
        for name, model in models.items():
            cv_result = cross_validate(
                model, X, y, cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            mae = -cv_result['test_score'].mean()
            scores[name] = mae
            
            if verbose:
                print(f"{name:20s}: MAE = {mae:.2f}초 (± {cv_result['test_score'].std():.2f})")
        
        # ============================================================
        # ★★★ 성능 기반 가중치 계산! ★★★
        # ============================================================
        # 오차가 작을수록 가중치 높게
        # 방법: 1/MAE로 계산 후 정규화
        
        inv_scores = {name: 1.0 / mae for name, mae in scores.items()}
        total_inv = sum(inv_scores.values())
        weights = [inv_scores['GradientBoosting'] / total_inv,
                   inv_scores['RandomForest'] / total_inv,
                   inv_scores['XGBoost'] / total_inv]
        
        # 가중치 저장
        self.ensemble_weights_ = {
            'GradientBoosting': weights[0],
            'RandomForest': weights[1],
            'XGBoost': weights[2]
        }
        
        if verbose:
            print(f"\n계산된 가중치:")
            print(f"  GradientBoosting: {weights[0]:.3f}")
            print(f"  RandomForest:     {weights[1]:.3f}")
            print(f"  XGBoost:          {weights[2]:.3f}")
            print("=" * 60)
        
        # Voting Ensemble 생성
        ensemble = VotingRegressor([
            ('gbm', gbm),
            ('rf', rf),
            ('xgb', xgb_model)
        ], weights=weights)
        
        return ensemble

    # ==================================================================
    # 7. Training
    # ==================================================================
    def train(self, df: pd.DataFrame, target_col: str = 'arrtime',
              use_cv: bool = True, use_tuning: bool = False,
              use_ensemble: bool = True, verbose: bool = True) -> None:
        """모델 학습 (개선 버전)"""
        
        if verbose:
            print("=" * 80)
            print("Feature Engineering 시작 (성능 개선 버전)")
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
            print("\n3. Feature Interactions 생성 (개선!)...")
        df = self._add_feature_interactions(df)
        
        # 4. Target Encoding (CV-aware)
        if verbose:
            print("\n4. Target Encoding (CV-aware)...")
        df = self._apply_target_encoding_cv(df, df[target_col])
        
        # 5. OneHot Encoding
        if verbose:
            print("\n5. OneHot Encoding...")
        df = self._prepare_onehot_features(df, fit=True)
        
        # ============================================================
        # ★★★ 6. Feature Columns 정의 (route_based_features 추가!) ★★★
        # ============================================================
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
            'station_hour_mult', 'station_temp_mult',
            'routeid_target_encoded', 'nodeid_target_encoded',
            # 신규 개선 Feature
            'time_in_minutes', 'rush_level', 'is_peak_rush',
            'is_weekday', 'station_rush_mult', 'peak_station_mult',
            # ============================================================
            # ★★★ route_based_features 추가! ★★★
            # ============================================================
            'sec_per_station',
            'route_avg_sec',
            'node_avg_sec',
            'route_hour_avg_sec',
            'station_progress_ratio',
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
            print(f"  - ★ route_based_features: 5개 추가!")
        
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
                self.model = self._create_ensemble_model(X, y, best_params, verbose)
            else:
                self.model = GradientBoostingRegressor(**best_params, random_state=42)
        else:
            if use_ensemble:
                if verbose:
                    print("\n" + "=" * 80)
                    print("Ensemble 모델 생성 (★ CV 기반 가중치 자동화!)")
                self.model = self._create_ensemble_model(
                    X, y,
                    best_params_gbm=None,   # ← 여기
                    verbose=verbose
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.08, max_depth=8,
                    min_samples_split=15, min_samples_leaf=5, subsample=0.85,
                    max_features='sqrt', random_state=42
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
                print(f"\nTop 20 Important Features:")
                for _, row in self.feature_importance_.head(20).iterrows():
                    print(f"  {row['feature']:45s}: {row['importance']:.4f}")
        elif use_ensemble:
            if hasattr(self.model.estimators_[0], 'feature_importances_'):
                self.feature_importance_ = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.estimators_[0].feature_importances_
                }).sort_values('importance', ascending=False)

    def _tune_hyperparameters(self, X, y, verbose):
        """Hyperparameter Tuning"""
        param_dist = {
            'n_estimators': [200, 250, 300, 350],
            'learning_rate': [0.05, 0.08, 0.1, 0.12],
            'max_depth': [7, 8, 9, 10],
            'min_samples_split': [10, 15, 20],
            'min_samples_leaf': [4, 5, 6],
            'subsample': [0.8, 0.85, 0.9],
            'max_features': ['sqrt', 0.6, 0.7, 0.8]
        }
        
        random_search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_dist, n_iter=40, cv=3,
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
    def predict_by_historical_pattern(
        self,
        historical_data,
        routeid: str,
        nodeid: str,
        prev_station_cnt: int,
        weekday: str,
        hour: int
    ) -> float:
        """과거 데이터 기반 패턴 예측"""
        import pandas as pd
        
        df2 = historical_data.copy()
        
        if 'hour' not in df2.columns:
            if 'collection_time' in df2.columns:
                df2['hour'] = pd.to_datetime(df2['collection_time']).dt.hour
            else:
                hour = None
        
        mask = (
            (df2['routeid'].astype(str) == str(routeid)) &
            (df2['nodeid'].astype(str) == str(nodeid)) &
            (df2['arrprevstationcnt'] == prev_station_cnt) &
            (df2['weekday'].astype(str) == str(weekday))
        )
        
        if 'hour' not in df2.columns:
            if 'collection_time' in df2.columns:
                df2['hour'] = pd.to_datetime(df2['collection_time']).dt.hour
        
        matched = df2[mask]
        
        if len(matched) > 0:
            if 'actual_arrtime' in matched.columns:
                target_col = 'actual_arrtime'
            else:
                target_col = 'arrtime'
            
            avg_time = matched[target_col].mean()
            print(f"  [Historical Pattern] {len(matched)} matches, avg={avg_time:.0f}s")
            return avg_time
        else:
            mask_broad = (
                (df2['routeid'].astype(str) == str(routeid)) &
                (df2['nodeid'].astype(str) == str(nodeid)) &
                (df2['arrprevstationcnt'] == prev_station_cnt)
            )
            
            matched_broad = df2[mask_broad]
            
            if len(matched_broad) > 0:
                if 'actual_arrtime' in matched_broad.columns:
                    target_col = 'actual_arrtime'
                else:
                    target_col = 'arrtime'
                
                avg_time = matched_broad[target_col].mean()
                print(f"  [Historical Pattern] {len(matched_broad)} broader matches, avg={avg_time:.0f}s")
                return avg_time
            else:
                default_time = prev_station_cnt * 60
                print(f"  [Historical Pattern] No matches, default: {default_time}s")
                return default_time

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
            "use_ensemble": self.use_ensemble,
            "ensemble_weights": self.ensemble_weights_  # ★ 가중치 저장!
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
        self.ensemble_weights_ = data.get('ensemble_weights')  # ★ 가중치 로드!

        print(f"모델 로드 완료: {load_path}")
        if self.ensemble_weights_:
            print(f"  Ensemble 가중치:")
            for name, weight in self.ensemble_weights_.items():
                print(f"    {name}: {weight:.3f}")
        
        return self