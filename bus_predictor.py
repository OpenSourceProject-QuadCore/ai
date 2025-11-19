import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


class BusArrivalPredictor:
    """
    버스 도착 시간 예측 모델
    개선 사항:
    1) LabelEncoder UNK 토큰 추가
    2) 예측 실패 시 historical pattern → 기본 fallback 계층화
    """

    def __init__(self, model_path: str = "models/bus_predictor.pkl"):
        self.model = None
        self.model_path = model_path
        self.feature_columns = None
        self.label_encoders = {}
        self.unk_token = "UNK"

    # ----------------------------------------------------------------------
    # 1. LabelEncoder 개선: UNK 지원
    # ----------------------------------------------------------------------
    def _fit_label_encoder(self, series: pd.Series, col: str) -> None:
        """
        학습 시 UNK 토큰 추가하여 LabelEncoder를 fit
        """
        le = LabelEncoder()
        values = series.astype(str).fillna("NA").unique()
        # UNK 클래스 추가
        le.fit(np.append(values, self.unk_token))
        self.label_encoders[col] = le

    def _transform_label_encoder(self, series: pd.Series, col: str) -> np.ndarray:
        """
        예측 시 unseen 값은 UNK로 치환 후 transform
        """
        le = self.label_encoders[col]
        values = series.astype(str).fillna("NA")
        allowed = set(le.classes_)

        # unseen → UNK
        values = values.apply(lambda x: x if x in allowed else self.unk_token)

        return le.transform(values)

    # ----------------------------------------------------------------------
    # 2. Feature Processing
    # ----------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = df.copy()

        categorical_cols = ['routeid', 'nodeid', 'routetp', 'vehicletp',
                            'weekday', 'time_slot', 'weather']

        # 학습 단계: fit + transform
        if training:
            for col in categorical_cols:
                if col in df.columns:
                    self._fit_label_encoder(df[col], col)
                    df[f"{col}_encoded"] = self._transform_label_encoder(df[col], col)
        else:
            # 예측 단계: transform only
            for col in categorical_cols:
                if col in df.columns and col in self.label_encoders:
                    df[f"{col}_encoded"] = self._transform_label_encoder(df[col], col)

        return df

    # ----------------------------------------------------------------------
    # 3. Training
    # ----------------------------------------------------------------------
    def train(self, df: pd.DataFrame, target_col: str = 'arrtime'):
        print("피처 인코딩 준비 중...")
        df = self.prepare_features(df, training=True)

        feature_cols = [
            'arrprevstationcnt', 'hour', 'minute', 'day_of_week',
            'is_weekend', 'is_rush_hour', 'temp', 'humidity',
            'rain_mm', 'snow_mm', 'avg_time_per_station'
        ]

        # 인코딩된 컬럼 추가
        encoded_cols = [col for col in df.columns if col.endswith("_encoded")]
        feature_cols.extend(encoded_cols)

        # 실제 존재하는 컬럼만 사용
        self.feature_columns = [c for c in feature_cols if c in df.columns]

        df[self.feature_columns] = df[self.feature_columns].fillna(0)

        X = df[self.feature_columns]
        y = df[target_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"학습 데이터: {len(X_train)} rows")
        print(f"검증 데이터: {len(X_val)} rows")

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # 평가
        train_r2 = self.model.score(X_train, y_train)
        val_r2 = self.model.score(X_val, y_val)

        y_pred = self.model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

        print(f"R² Train={train_r2:.4f}, Val={val_r2:.4f}")
        print(f"MAE={mae:.2f}s ({mae/60:.2f}m), RMSE={rmse:.2f}s")

    # ----------------------------------------------------------------------
    # 4. Prediction
    # ----------------------------------------------------------------------
    def predict(self, features: dict) -> float:
        """
        예측 + fallback 포함
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        try:
            df = pd.DataFrame([features])
            df = self.prepare_features(df, training=False)
            df[self.feature_columns] = df[self.feature_columns].fillna(0)

            pred = float(self.model.predict(df[self.feature_columns])[0])
            return max(pred, 0)

        except Exception as e:
            print(f"[예측 오류] {e}")
            raise e

    # ----------------------------------------------------------------------
    # 5. Historical Pattern Fallback
    # ----------------------------------------------------------------------
    def predict_by_historical_pattern(self, df: pd.DataFrame,
                                      routeid: str, nodeid: str,
                                      prev_station_cnt: int,
                                      weekday: str, hour: int):
        """
        여러 조건 → 약한 조건 순서로 fallback
        """

        df2 = df.copy()

        # 1) 가장 강한 조건
        mask = (
            (df2['routeid'] == routeid) &
            (df2['nodeid'] == nodeid) &
            (df2['arrprevstationcnt'] == prev_station_cnt) &
            (df2['weekday'] == weekday) &
            (df2['hour'] == hour)
        )
        if len(df2[mask]) > 0:
            return df2[mask]['arrtime'].median()

        # 2) 시간 조건 제외
        mask = (
            (df2['routeid'] == routeid) &
            (df2['nodeid'] == nodeid) &
            (df2['arrprevstationcnt'] == prev_station_cnt)
        )
        if len(df2[mask]) > 0:
            return df2[mask]['arrtime'].median()

        # 3) 마지막 fallback — 정류장당 60초
        return prev_station_cnt * 60

    # ----------------------------------------------------------------------
    # 6. Save / Load
    # ----------------------------------------------------------------------
    def save(self, path: str = None):
        save_path = path or self.model_path
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "label_encoders": self.label_encoders,
            "unk_token": self.unk_token
        }, save_path)
        print(f"모델 저장 완료: {save_path}")

    def load(self, path: str = None):
        load_path = path or self.model_path
        data = joblib.load(load_path)

        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.label_encoders = data['label_encoders']
        self.unk_token = data.get('unk_token', "UNK")

        print(f"모델 로드 완료: {load_path}")
        return self
