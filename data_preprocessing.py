import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

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
        """
        날씨 데이터 결측치를 시간 기반으로 보간
        
        실제 데이터 특성:
        - 날씨 API 호출 실패 시 결측 발생 (16:16~16:18 구간 등)
        - 보통 수 분 이내로 다시 복구
        - ±10분 이내 값으로 보간
        
        예시:
        16:14:54 - Clear, 13.9, 36.0 ✓
        16:16:29 - (결측)           → 16:14:54 값 사용 (forward fill)
        16:20:11 - Clear, 12.7, 38.0 ✓
        """
        df = df.sort_values('collection_time').copy()
        weather_cols = ['weather', 'temp', 'humidity', 'rain_mm', 'snow_mm']
        
        max_gap_seconds = 1200  # 10분
        
        for col in weather_cols:
            if col not in df.columns:
                continue
            
            # 결측치 인덱스 찾기
            missing_mask = df[col].isna()
            
            if not missing_mask.any():
                continue
            
            # Forward fill (시간 제한 적용)
            last_valid_idx = None
            last_valid_time = None
            
            for idx in df.index:
                if not missing_mask.loc[idx]:
                    # 유효한 값
                    last_valid_idx = idx
                    last_valid_time = df.loc[idx, 'collection_time']
                elif last_valid_idx is not None:
                    # 결측치
                    time_diff = (df.loc[idx, 'collection_time'] - last_valid_time).total_seconds()
                    if time_diff <= max_gap_seconds:
                        df.loc[idx, col] = df.loc[last_valid_idx, col]
            
            # Backward fill (아직 결측인 것만)
            missing_mask = df[col].isna()
            next_valid_idx = None
            next_valid_time = None
            
            for idx in reversed(df.index):
                if not missing_mask.loc[idx]:
                    # 유효한 값
                    next_valid_idx = idx
                    next_valid_time = df.loc[idx, 'collection_time']
                elif next_valid_idx is not None:
                    # 결측치
                    time_diff = (next_valid_time - df.loc[idx, 'collection_time']).total_seconds()
                    if time_diff <= max_gap_seconds:
                        df.loc[idx, col] = df.loc[next_valid_idx, col]

        return df

    # ------------------------------------------------------
    # 2. 버스 여러 대 분리 - 최종 수정
    # ------------------------------------------------------
    def separate_multiple_buses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간대를 넘나들며 같은 버스 추적
        
        핵심 규칙:
        1. 같은 collection_time의 버스는 모두 다른 버스
        2. arrprevstationcnt가 같으면 arrtime도 같아야 함
        3. arrprevstationcnt가 줄면 arrtime도 감소
        4. 추월 없음 (순서 유지)
        """
        df = df.sort_values(['routeid', 'nodeid', 'collection_time', 'arrtime']).copy()
        
        result_rows = []
        
        # 노선+정류장별로 처리
        for (routeid, nodeid), route_group in df.groupby(['routeid', 'nodeid']):
            
            route_group = route_group.sort_values('collection_time')
            
            # 버스 추적 state
            active_buses = []  # [(bus_index, last_arrtime, last_arrprevstationcnt, last_time), ...]
            next_bus_index = 0
            
            # 시각별로 처리
            for time, time_group in route_group.groupby('collection_time'):
                time_group = time_group.sort_values('arrtime').reset_index(drop=True)
                
                current_buses = [
                    (row['arrtime'], row['arrprevstationcnt'], idx)
                    for idx, row in time_group.iterrows()
                ]
                
                # 매칭
                matched_indices = self._match_buses_with_exact_rules(
                    active_buses, 
                    current_buses,
                    time
                )
                
                # 결과 적용
                new_active_buses = []
                
                for curr_idx, bus_index in enumerate(matched_indices):
                    arrtime, arrprevstationcnt, df_idx = current_buses[curr_idx]
                    
                    if bus_index is None:
                        bus_index = next_bus_index
                        next_bus_index += 1
                    
                    row = time_group.loc[df_idx].copy()
                    row['bus_index'] = bus_index
                    result_rows.append(row)
                    
                    new_active_buses.append((
                        bus_index, 
                        arrtime, 
                        arrprevstationcnt,
                        time
                    ))
                
                active_buses = new_active_buses
        
        return pd.DataFrame(result_rows).reset_index(drop=True)

    def _match_buses_with_exact_rules(self, 
                                      active_buses: List[Tuple[int, int, int, pd.Timestamp]],
                                      current_buses: List[Tuple[int, int, int]],
                                      current_time: pd.Timestamp) -> List[Optional[int]]:
        """
        실제 데이터 특성 반영한 매칭
        
        핵심: arrprevstationcnt가 변하지 않으면 arrtime도 변하지 않음!
        
        Args:
            active_buses: [(bus_index, arrtime, arrprevstationcnt, time), ...]
            current_buses: [(arrtime, arrprevstationcnt, df_idx), ...]
            current_time: 현재 시각
            
        Returns:
            매칭 결과 [bus_index or None, ...]
        """
        if not active_buses:
            return [None] * len(current_buses)
        
        n_prev = len(active_buses)
        n_curr = len(current_buses)
        
        # 비용 행렬
        cost_matrix = np.full((n_prev, n_curr), np.inf)
        
        for i, (bus_idx, prev_arrtime, prev_station_cnt, prev_time) in enumerate(active_buses):
            for j, (curr_arrtime, curr_station_cnt, df_idx) in enumerate(current_buses):
                
                # === 매칭 조건 ===
                
                # 조건 1: 추월 불가
                if j < i:
                    continue
                
                # 조건 2: 정류장은 유지 또는 감소만 가능
                if curr_station_cnt > prev_station_cnt:
                    continue
                
                station_diff = prev_station_cnt - curr_station_cnt
                time_diff = prev_arrtime - curr_arrtime
                
                # 조건 3: 핵심 규칙!
                if station_diff == 0:
                    # 정류장 같으면 → arrtime도 같아야 함 (±10초 허용)
                    if abs(time_diff) > 10:
                        continue  # 정류장 안 변했는데 시간이 많이 다름 = 다른 버스
                    
                    # 비용: 시간 차이
                    cost = abs(time_diff)
                
                elif station_diff > 0:
                    # 정류장 줄었으면 → arrtime도 감소해야 함
                    if time_diff <= 0:
                        continue  # 정류장 줄었는데 시간이 안 줄거나 증가? 불가능
                    
                    # 정류장당 시간 체크 (30~300초)
                    avg_time_per_station = time_diff / station_diff
                    if avg_time_per_station < 30 or avg_time_per_station > 300:
                        continue
                    
                    # 비용: 정류장 변화는 정상 동작이므로 낮은 비용
                    cost = station_diff * 5
                
                else:
                    continue
                
                # 조건 4: 순서 유지 페널티
                order_penalty = (j - i) * 20
                
                cost_matrix[i, j] = cost + order_penalty
        
        # === Greedy 매칭 ===
        matched = [None] * n_curr
        used_prev = set()
        
        pairs = []
        for i in range(n_prev):
            for j in range(n_curr):
                if cost_matrix[i, j] < np.inf:
                    pairs.append((cost_matrix[i, j], i, j))
        
        pairs.sort()
        
        for cost, i, j in pairs:
            if i not in used_prev and matched[j] is None:
                matched[j] = active_buses[i][0]
                used_prev.add(i)
        
        return matched

    # ------------------------------------------------------
    # 3. 추가 피처 생성
    # ------------------------------------------------------
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df['collection_time'].dt.hour
        df['minute'] = df['collection_time'].dt.minute
        df['day_of_week'] = df['collection_time'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df['avg_time_per_station'] = df['arrtime'] / np.maximum(df['arrprevstationcnt'], 1)
        return df

    # ------------------------------------------------------
    # 4. 버스 trajectory 생성
    # ------------------------------------------------------
    def create_bus_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['routeid', 'nodeid', 'bus_index', 'collection_time']).copy()

        group_key = ['routeid', 'nodeid', 'bus_index']

        df['prev_station'] = df.groupby(group_key)['arrprevstationcnt'].shift(1)
        df['prev_arrtime'] = df.groupby(group_key)['arrtime'].shift(1)
        df['prev_time'] = df.groupby(group_key)['collection_time'].shift(1)

        df['station_changed'] = (
            (df['arrprevstationcnt'] != df['prev_station']) | 
            df['prev_station'].isna()
        ).astype(int)
        
        df['time_elapsed'] = (df['collection_time'] - df['prev_time']).dt.total_seconds()
        
        df.loc[df['time_elapsed'] < 0, 'time_elapsed'] = np.nan
        df.loc[df['time_elapsed'] > 3600, 'time_elapsed'] = np.nan

        return df

    # ------------------------------------------------------
    # 5. 이상치 제거
    # ------------------------------------------------------
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        original_len = len(df)
        
        df = df[df['arrtime'] <= 7200]
        df = df[df['arrtime'] >= 0]
        df = df[df['arrprevstationcnt'] <= 50]
        df = df[df['arrprevstationcnt'] >= 0]
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"이상치 제거: {removed}개 ({removed/original_len*100:.2f}%)")
        
        return df

    # ------------------------------------------------------
    # 6. 전체 파이프라인
    # ------------------------------------------------------
    def prepare_training_data(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print(f"원본 데이터: {len(df):,} rows")
        
        df = self.fill_weather_missing(df)
        if verbose:
            print("✓ 날씨 보간 완료")
        
        df = self.separate_multiple_buses(df)
        if verbose:
            print(f"✓ 버스 분리 완료: {len(df):,} rows")
            unique_buses = df.groupby(['routeid', 'nodeid'])['bus_index'].nunique()
            print(f"  평균 버스 수/정류장: {unique_buses.mean():.1f}대")
        
        df = self.add_features(df)
        if verbose:
            print("✓ 피처 생성 완료")
        
        df = self.create_bus_trajectory(df)
        if verbose:
            print("✓ 궤적 생성 완료")
        
        df = self.remove_outliers(df)
        
        essential_cols = ['arrtime', 'arrprevstationcnt', 'routeid', 'nodeid']
        df = df.dropna(subset=essential_cols)
        
        if verbose:
            print(f"최종 데이터: {len(df):,} rows")

        return df