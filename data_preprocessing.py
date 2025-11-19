import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple

class BusDataPreprocessor:
    """버스 도착 데이터 전처리 클래스"""

    def __init__(self, csv_path: str = "data/bus_arrivals.csv"):
        self.csv_path = csv_path

    def load_data(self) -> pd.DataFrame:
        """CSV 데이터 로드"""
        df = pd.read_csv(self.csv_path)
        df['collection_time'] = pd.to_datetime(df['collection_time'])
        return df

    # ------------------------------------------------------
    # 1. 날씨 보간
    # ------------------------------------------------------
    def fill_weather_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """날씨 데이터 결측치를 ±20분 내 forward/backward fill"""
        df = df.sort_values('collection_time').copy()
        weather_cols = ['weather', 'temp', 'humidity', 'rain_mm', 'snow_mm']

        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].ffill(limit=20)
                df[col] = df[col].bfill(limit=20)

        return df

    # ------------------------------------------------------
    # 2. 버스 여러 대 분리 — 개선된 clustering
    # ------------------------------------------------------
    def _cluster_by_time_diff(self, arrtimes: List[int], threshold: int = 180) -> List[List[int]]:
        """
        arrtime이 threshold 이상 떨어지면 다른 버스로 간주.
        → 그룹 시작값 대비 차이를 기준으로 분리.
        """
        if not arrtimes:
            return []

        clusters = [[0]]
        base_value = arrtimes[0]

        for i in range(1, len(arrtimes)):
            if arrtimes[i] - base_value >= threshold:
                clusters.append([i])
                base_value = arrtimes[i]
            else:
                clusters[-1].append(i)

        return clusters

    # ------------------------------------------------------
    # 3. 여러 대 버스 분리를 정보 손실 없이 적용
    # ------------------------------------------------------
    def separate_multiple_buses(self, df: pd.DataFrame) -> pd.DataFrame:
        """같은 시간/정류장/노선에 여러 대 버스가 있을 때 분리 + bus_index 생성"""
        result_rows = []

        grouped = df.groupby(['collection_time', 'nodeid', 'routeid'])

        for (_, _nodeid, _routeid), group in grouped:
            if len(group) == 1:
                row = group.iloc[0].copy()
                row['bus_index'] = 0
                result_rows.append(row)
                continue

            # arrtime 기준으로 정렬
            sorted_group = group.sort_values('arrtime').reset_index(drop=True)
            arr_list = sorted_group['arrtime'].tolist()

            # 개선된 cluster 적용
            buses = self._cluster_by_time_diff(arr_list)

            for bus_idx, cluster in enumerate(buses):
                for i in cluster:
                    row = sorted_group.loc[i].copy()
                    row['bus_index'] = bus_idx  # 새로운 surrogate id 역할
                    result_rows.append(row)

        return pd.DataFrame(result_rows).reset_index(drop=True)

    # ------------------------------------------------------
    # 4. 추가 피처 생성
    # ------------------------------------------------------
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df['collection_time'].dt.hour
        df['minute'] = df['collection_time'].dt.minute
        df['day_of_week'] = df['collection_time'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df['avg_time_per_station'] = df['arrtime'] / (df['arrprevstationcnt'] + 1e-5)
        return df

    # ------------------------------------------------------
    # 5. 버스 trajectory 생성 — route + node + bus_index 단위
    # ------------------------------------------------------
    def create_bus_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['routeid', 'nodeid', 'bus_index', 'collection_time']).copy()

        group_key = ['routeid', 'nodeid', 'bus_index']

        df['prev_station'] = df.groupby(group_key)['arrprevstationcnt'].shift(1)
        df['prev_arrtime'] = df.groupby(group_key)['arrtime'].shift(1)
        df['prev_time'] = df.groupby(group_key)['collection_time'].shift(1)

        df['station_changed'] = (df['arrprevstationcnt'] != df['prev_station']).astype(int)
        df['time_elapsed'] = (df['collection_time'] - df['prev_time']).dt.total_seconds()

        return df

    # ------------------------------------------------------
    # 6. 전체 파이프라인
    # ------------------------------------------------------
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fill_weather_missing(df)
        df = self.separate_multiple_buses(df)
        df = self.add_features(df)
        df = self.create_bus_trajectory(df)

        # 결측 제거
        df = df.dropna(subset=['arrtime', 'arrprevstationcnt'])

        return df
