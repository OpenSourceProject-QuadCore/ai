import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


class BusDataPreprocessor:
    """버스 도착 데이터 전처리 클래스 (안정화 버전)"""

    def __init__(self, csv_path: str = "data/bus_arrivals.csv"):
        self.csv_path = csv_path

    # ------------------------------------------------------
    # 0. 기본 로드
    # ------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """CSV 데이터 로드 + collection_time 안전 파싱"""
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"[BusDataPreprocessor] CSV 파일을 찾을 수 없습니다: {self.csv_path}")

        if "collection_time" not in df.columns:
            raise ValueError("[BusDataPreprocessor] 'collection_time' 컬럼이 없습니다.")

        # 안전한 datetime 파싱
        df["collection_time"] = pd.to_datetime(df["collection_time"], errors="coerce")
        invalid = df["collection_time"].isna().sum()
        if invalid > 0:
            print(f"[WARN] collection_time 파싱 실패 행 {invalid}개 제거")
            df = df[~df["collection_time"].isna()].copy()

        if len(df) == 0:
            raise ValueError("[BusDataPreprocessor] collection_time 유효 행이 없습니다.")

        return df

    # ------------------------------------------------------
    # 1. 날씨 보간
    # ------------------------------------------------------
    def fill_weather_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        날씨 데이터 결측치를 시간 기반으로 보간 (안정화 버전)

        - numeric 컬럼: ±10분 이내 forward/backward fill
        - weather(문자열): forward/backward fill 후 남은 결측은 'Unknown'
        """
        if "collection_time" not in df.columns:
            raise ValueError("[fill_weather_missing] 'collection_time' 컬럼이 없습니다.")

        if len(df) == 0:
            return df

        df = df.sort_values("collection_time").copy()
        weather_cols = ["weather", "temp", "humidity", "rain_mm", "snow_mm"]

        # 없는 컬럼은 생성해서 NaN 채워두면 이후 로직이 더 안전
        for col in weather_cols:
            if col not in df.columns:
                df[col] = np.nan

        max_gap_seconds = 1200  # 10분

        for col in weather_cols:
            missing_mask = df[col].isna()
            if not missing_mask.any():
                continue

            # Forward fill (시간 제한 적용)
            last_valid_idx = None
            last_valid_time = None

            for idx in df.index:
                if not missing_mask.loc[idx]:
                    last_valid_idx = idx
                    last_valid_time = df.loc[idx, "collection_time"]
                elif last_valid_idx is not None:
                    time_diff = (df.loc[idx, "collection_time"] - last_valid_time).total_seconds()
                    if time_diff <= max_gap_seconds:
                        df.loc[idx, col] = df.loc[last_valid_idx, col]

            # Backward fill (아직 결측인 것만)
            missing_mask = df[col].isna()
            next_valid_idx = None
            next_valid_time = None

            for idx in reversed(df.index):
                if not missing_mask.loc[idx]:
                    next_valid_idx = idx
                    next_valid_time = df.loc[idx, "collection_time"]
                elif next_valid_idx is not None:
                    time_diff = (next_valid_time - df.loc[idx, "collection_time"]).total_seconds()
                    if time_diff <= max_gap_seconds:
                        df.loc[idx, col] = df.loc[next_valid_idx, col]

        # 남은 결측에 대한 안전 처리
        # weather: 문자열
        if "weather" in df.columns:
            df["weather"] = df["weather"].fillna("Unknown").astype(str)

        # numeric 컬럼: median → 없으면 0
        for col in ["temp", "humidity", "rain_mm", "snow_mm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().any():
                    median_val = float(df[col].median())
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(0.0)

        return df

    # ------------------------------------------------------
    # 2. 버스 여러 대 분리
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
        required_cols = ["routeid", "nodeid", "collection_time", "arrtime", "arrprevstationcnt"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"[separate_multiple_buses] 필수 컬럼 누락: {c}")

        if len(df) == 0:
            # 최소한 schema는 유지
            empty = df.copy()
            empty["bus_index"] = pd.Series(dtype="Int64")
            return empty

        df = df.sort_values(["routeid", "nodeid", "collection_time", "arrtime"]).copy()

        result_rows = []

        # 노선+정류장별로 처리
        for (routeid, nodeid), route_group in df.groupby(["routeid", "nodeid"], sort=False):
            route_group = route_group.sort_values("collection_time")

            # 버스 추적 state
            # (bus_index, last_arrtime, last_arrprevstationcnt, last_time)
            active_buses: List[Tuple[int, float, float, pd.Timestamp]] = []
            next_bus_index = 0

            # 시각별로 처리
            for time, time_group in route_group.groupby("collection_time"):
                time_group = time_group.sort_values("arrtime").reset_index(drop=True)

                current_buses = []
                for idx_row, row in time_group.iterrows():
                    # NaN 방어
                    arrtime = float(row.get("arrtime", np.nan))
                    station_cnt = float(row.get("arrprevstationcnt", np.nan))
                    if arrtime is None or station_cnt is None:
                        continue
                    try:
                        arrtime = float(arrtime)
                        station_cnt = float(station_cnt)
                    except:
                        continue
                    current_buses.append((arrtime, station_cnt, idx_row))

                if not current_buses:
                    continue

                # 매칭
                matched_indices = self._match_buses_with_exact_rules(
                    active_buses,
                    current_buses,
                    time,
                )

                # 결과 적용
                new_active_buses = []

                for curr_idx, bus_index in enumerate(matched_indices):
                    arrtime, arrprevstationcnt, df_idx = current_buses[curr_idx]

                    if bus_index is None:
                        bus_index = next_bus_index
                        next_bus_index += 1

                    row = time_group.loc[df_idx].copy()
                    row["bus_index"] = bus_index
                    result_rows.append(row)

                    new_active_buses.append(
                        (
                            bus_index,
                            arrtime,
                            arrprevstationcnt,
                            time,
                        )
                    )

                active_buses = new_active_buses

        if not result_rows:
            # 모든 행이 NaN 등으로 걸러진 경우
            empty = df.copy()
            empty["bus_index"] = pd.Series(dtype="Int64")
            print("[WARN] separate_multiple_buses 결과가 비어 있습니다.")
            return empty.reset_index(drop=True)

        return pd.DataFrame(result_rows).reset_index(drop=True)

    def _match_buses_with_exact_rules(
        self,
        active_buses: List[Tuple[int, float, float, pd.Timestamp]],
        current_buses: List[Tuple[float, float, int]],
        current_time: pd.Timestamp,
    ) -> List[Optional[int]]:
        """버스 매칭 알고리즘 (안정화 버전)"""
        if not active_buses:
            return [None] * len(current_buses)

        n_prev = len(active_buses)
        n_curr = len(current_buses)

        cost_matrix = np.full((n_prev, n_curr), np.inf)

        for i, (bus_idx, prev_arrtime, prev_station_cnt, prev_time) in enumerate(active_buses):
            for j, (curr_arrtime, curr_station_cnt, df_idx) in enumerate(current_buses):
                # NaN 방어
                if np.isnan(prev_arrtime) or np.isnan(prev_station_cnt):
                    continue
                if np.isnan(curr_arrtime) or np.isnan(curr_station_cnt):
                    continue

                # 조건 1: 추월 불가 (순서 보존)
                if j < i:
                    continue

                # 조건 2: 정류장은 유지 또는 감소만 가능
                if curr_station_cnt > prev_station_cnt:
                    continue

                station_diff = prev_station_cnt - curr_station_cnt
                time_diff = prev_arrtime - curr_arrtime

                if station_diff == 0:
                    # 정류장 같으면 → arrtime도 거의 같아야 함
                    if abs(time_diff) > 10:
                        continue
                    cost = abs(time_diff)

                elif station_diff > 0:
                    # 정류장 줄었으면 → arrtime도 감소해야 함
                    if time_diff <= 0:
                        continue

                    avg_time_per_station = time_diff / max(station_diff, 1e-6)
                    # 너무 비현실적인 속도는 제외
                    if avg_time_per_station < 10 or avg_time_per_station > 600:
                        continue

                    cost = station_diff * 5

                else:
                    continue

                # 순서 유지 페널티
                order_penalty = max(j - i, 0) * 20
                cost_matrix[i, j] = cost + order_penalty

        matched: List[Optional[int]] = [None] * n_curr
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
    # 3. 실제 도착 시간 라벨 생성 (핵심 개선 + 안정화)
    # ------------------------------------------------------
    def create_actual_arrival_labels_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        개선된 실제 도착 시간 라벨 생성 (안정화 버전)

        전략:
        1. 마지막 정류장 ≤2개: API arrtime * 1.1 사용 (높은 신뢰도)
        2. 마지막 정류장 3~5개: API arrtime과 수집 주기 중 작은 값 (중간 신뢰도)
        3. 마지막 정류장 >5개: 제외 (불확실성 너무 큼)
        """
        if "bus_index" not in df.columns:
            raise ValueError("[create_actual_arrival_labels_improved] 'bus_index' 컬럼이 없습니다. 먼저 separate_multiple_buses를 호출하세요.")

        if len(df) == 0:
            print("[INFO] create_actual_arrival_labels_improved: 입력 데이터가 비어 있습니다.")
            return df.copy()

        print("\n" + "=" * 60)
        print("실제 도착 시간 라벨 생성 (개선 버전)")
        print("=" * 60)

        df = df.sort_values(["routeid", "nodeid", "bus_index", "collection_time"]).copy()

        result_rows = []
        stats = {
            "total_buses": 0,
            "station_1_2": 0,
            "station_3_5": 0,
            "excluded": 0,
        }

        for (routeid, nodeid, bus_idx), bus_group in df.groupby(
            ["routeid", "nodeid", "bus_index"], sort=False
        ):
            stats["total_buses"] += 1
            bus_group = bus_group.sort_values("collection_time").reset_index(drop=True)

            if len(bus_group) < 2:
                # 데이터가 1개뿐이면 도착 추정이 의미 없음
                stats["excluded"] += 1
                continue

            last_obs = bus_group.iloc[-1]
            last_time = last_obs["collection_time"]
            last_station_cnt = last_obs.get("arrprevstationcnt", np.nan)
            last_arrtime = last_obs.get("arrtime", np.nan)

            if pd.isna(last_time) or pd.isna(last_station_cnt) or pd.isna(last_arrtime):
                stats["excluded"] += 1
                continue

            # 신뢰도별 처리
            if last_station_cnt <= 2:
                estimated_arrival = last_time + timedelta(seconds=float(last_arrtime) * 1.1)
                confidence = "high"
                stats["station_1_2"] += 1

            elif last_station_cnt <= 5:
                api_estimate = last_time + timedelta(seconds=float(last_arrtime))
                collection_estimate = last_time + timedelta(seconds=90)  # 1.5 수집주기
                estimated_arrival = min(api_estimate, collection_estimate)
                confidence = "medium"
                stats["station_3_5"] += 1

            else:
                stats["excluded"] += 1
                continue

            # 마지막 관측 이전의 데이터에 라벨 부여
            for idx, row in bus_group.iloc[:-1].iterrows():
                obs_time = row["collection_time"]

                if pd.isna(obs_time):
                    continue

                # 실제 소요 시간 계산
                actual_time = (estimated_arrival - obs_time).total_seconds()

                # 이상치 필터
                if actual_time <= 0 or actual_time > 7200:
                    continue

                # API 예측과 너무 차이나면 제외 (데이터 품질 향상)
                api_arr = row.get("arrtime", np.nan)
                if pd.isna(api_arr):
                    continue

                api_prediction_error = abs(actual_time - float(api_arr))
                # 너무 빡세게 걸지 말고, 일정 이상만 필터
                if api_prediction_error > 1200:  # 20분 이상 차이
                    continue

                row_copy = row.copy()
                row_copy["actual_arrtime"] = float(actual_time)
                row_copy["api_arrtime"] = float(api_arr)
                row_copy["estimated_arrival"] = estimated_arrival.isoformat()
                row_copy["last_station_cnt"] = float(last_station_cnt)
                row_copy["confidence"] = confidence
                result_rows.append(row_copy)

        if not result_rows:
            print("[WARN] create_actual_arrival_labels_improved: 유효한 라벨을 생성하지 못했습니다.")
            return pd.DataFrame(columns=list(df.columns) + [
                "actual_arrtime",
                "api_arrtime",
                "estimated_arrival",
                "last_station_cnt",
                "confidence",
            ])

        result_df = pd.DataFrame(result_rows).reset_index(drop=True)

        # 통계 출력
        print(f"\n버스 추적 통계:")
        print(f"  총 버스: {stats['total_buses']:,}대")
        print(f"  높은 신뢰도 (≤2개 정류장): {stats['station_1_2']:,}대")
        print(f"  중간 신뢰도 (3~5개): {stats['station_3_5']:,}대")
        print(f"  제외됨 (>5개 또는 데이터 부족): {stats['excluded']:,}대")
        print(f"\n생성된 학습 샘플: {len(result_df):,}개")
        print(f"  원본 대비: {len(result_df) / len(df) * 100:.1f}%")

        if len(result_df) > 0:
            print(f"\n라벨 품질 분석:")
            error = result_df["actual_arrtime"] - result_df["api_arrtime"]
            print(f"  실제 vs API 평균 차이: {error.mean():.0f}초")
            print(f"  절대 오차: {abs(error).mean():.0f}초")
            print(f"  API가 낙관적 (더 빠름): {(error > 0).sum():,}건 ({(error > 0).mean() * 100:.1f}%)")
            print(f"  API가 비관적 (더 느림): {(error < 0).sum():,}건 ({(error < 0).mean() * 100:.1f}%)")

            high_conf = result_df[result_df["confidence"] == "high"]
            medium_conf = result_df[result_df["confidence"] == "medium"]

            print(f"\n신뢰도별 통계:")
            if len(high_conf) > 0:
                print(
                    f"  높음: {len(high_conf):,}개, "
                    f"오차 {abs(high_conf['actual_arrtime'] - high_conf['api_arrtime']).mean():.0f}초"
                )
            if len(medium_conf) > 0:
                print(
                    f"  중간: {len(medium_conf):,}개, "
                    f"오차 {abs(medium_conf['actual_arrtime'] - medium_conf['api_arrtime']).mean():.0f}초"
                )

        return result_df

    # ------------------------------------------------------
    # 4. 추가 피처 생성
    # ------------------------------------------------------
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 피처 생성"""
        df = df.copy()
        if "collection_time" not in df.columns:
            raise ValueError("[add_features] 'collection_time' 컬럼이 없습니다.")

        df["collection_time"] = pd.to_datetime(df["collection_time"], errors="coerce")
        df = df[~df["collection_time"].isna()].copy()

        if len(df) == 0:
            return df

        df["hour"] = df["collection_time"].dt.hour
        df["minute"] = df["collection_time"].dt.minute
        df["day_of_week"] = df["collection_time"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        return df

    def add_route_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df

        df = df.copy()

        required_cols = [
            "routeid", "nodeid", "bus_index", "collection_time",
            "arrprevstationcnt", "arrtime",
            "prev_station", "prev_arrtime", "prev_time", "time_elapsed",
        ]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"[add_route_based_features] 필수 컬럼 누락: {c}")

        # --------------------------------------------------------------
        # 1) 정류장 이동 시간 계산: sec_per_station
        # --------------------------------------------------------------
        df["sec_per_station"] = np.nan

        valid_move = (
            (df["prev_station"].notna()) &
            (df["arrprevstationcnt"] < df["prev_station"]) &
            (df["time_elapsed"].notna()) &
            (df["time_elapsed"] > 0)
        )

        station_delta = (df["prev_station"] - df["arrprevstationcnt"]).clip(lower=1)

        df.loc[valid_move, "sec_per_station"] = (
            df.loc[valid_move, "time_elapsed"] / station_delta.loc[valid_move]
        )

        # 값이 너무 큰 경우 제거
        df.loc[df["sec_per_station"] > 600, "sec_per_station"] = np.nan
        df.loc[df["sec_per_station"] < 5, "sec_per_station"] = np.nan

        # --------------------------------------------------------------
        # 2) 노선별 평균 속도 Feature
        # --------------------------------------------------------------
        df["route_avg_sec"] = (
            df.groupby("routeid")["sec_per_station"]
            .transform("mean")
            .fillna(df["sec_per_station"].median())
        )

        # --------------------------------------------------------------
        # 3) 정류장별 평균 속도
        # --------------------------------------------------------------
        df["node_avg_sec"] = (
            df.groupby("nodeid")["sec_per_station"]
            .transform("mean")
            .fillna(df["sec_per_station"].median())
        )

        # --------------------------------------------------------------
        # 4) 노선 + 시간대별 평균 속도
        # --------------------------------------------------------------
        if "hour" not in df.columns:
            df["hour"] = pd.to_datetime(df["collection_time"]).dt.hour

        df["route_hour_avg_sec"] = (
            df.groupby(["routeid", "hour"])["sec_per_station"]
            .transform("mean")
            .fillna(df["route_avg_sec"])
        )

        # --------------------------------------------------------------
        # 5) 남은 정류장 비율
        # (모든 노선은 station count가 다르므로 상대값을 줘야 모델이 비교 가능)
        # --------------------------------------------------------------
        df["max_station_route"] = df.groupby("routeid")["arrprevstationcnt"].transform("max")
        df["station_progress_ratio"] = (
            df["arrprevstationcnt"] / df["max_station_route"].replace(0, np.nan)
        ).fillna(0)

        # --------------------------------------------------------------
        # 6) Category Feature 처리 (모델 입력용)
        # --------------------------------------------------------------
        df["routeid"] = df["routeid"].astype(str)
        df["nodeid"] = df["nodeid"].astype(str)
        df["routeno"] = df.get("routeno", "").astype(str)

        df["hour"] = df["hour"].astype(int)
        df["day_of_week"] = df["collection_time"].dt.dayofweek.astype(int)

        return df

    # ------------------------------------------------------
    # 5. 버스 trajectory 생성
    # ------------------------------------------------------
    def create_bus_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df

        required_cols = ["routeid", "nodeid", "bus_index", "collection_time", "arrprevstationcnt", "arrtime"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"[create_bus_trajectory] 필수 컬럼 누락: {c}")

        df = df.sort_values(["routeid", "nodeid", "bus_index", "collection_time"]).copy()

        group_key = ["routeid", "nodeid", "bus_index"]

        df["prev_station"] = df.groupby(group_key)["arrprevstationcnt"].shift(1)
        df["prev_arrtime"] = df.groupby(group_key)["arrtime"].shift(1)
        df["prev_time"] = df.groupby(group_key)["collection_time"].shift(1)

        df["station_changed"] = (
            (df["arrprevstationcnt"] != df["prev_station"]) | df["prev_station"].isna()
        ).astype(int)

        df["time_elapsed"] = (df["collection_time"] - df["prev_time"]).dt.total_seconds()

        df.loc[df["time_elapsed"] < 0, "time_elapsed"] = np.nan
        df.loc[df["time_elapsed"] > 3600, "time_elapsed"] = np.nan

        return df

    # ------------------------------------------------------
    # 6. 이상치 제거
    # ------------------------------------------------------
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df

        df = df.copy()
        original_len = len(df)

        df = df[df["arrtime"].between(0, 7200, inclusive="both")]
        df = df[df["arrprevstationcnt"].between(0, 50, inclusive="both")]

        removed = original_len - len(df)
        if removed > 0:
            print(f"이상치 제거: {removed}개 ({removed / original_len * 100:.2f}%)")

        return df

    # ------------------------------------------------------
    # 7. 라벨 품질 검증
    # ------------------------------------------------------
    def validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """생성된 라벨의 품질 검증 및 요약 출력 (안정화 버전)"""
        required_cols = ["actual_arrtime", "api_arrtime", "arrprevstationcnt"]
        for c in required_cols:
            if c not in df.columns:
                print(f"[WARN] validate_labels: '{c}' 컬럼이 없어 검증을 건너뜁니다.")
                return df

        if len(df) == 0:
            print("[INFO] validate_labels: 입력 데이터가 비어 있어 검증을 건너뜁니다.")
            return df

        print("\n" + "=" * 60)
        print("라벨 품질 상세 검증")
        print("=" * 60)

        df = df.copy()

        df["error"] = df["actual_arrtime"] - df["api_arrtime"]
        df["abs_error"] = df["error"].abs()
        df["error_pct"] = df["abs_error"] / df["actual_arrtime"].clip(lower=1e-6) * 100

        print(f"\n전체 통계:")
        print(f"  샘플 수: {len(df):,}개")
        print(
            f"  평균 실제 도착 시간: {df['actual_arrtime'].mean():.0f}초 "
            f"({df['actual_arrtime'].mean() / 60:.1f}분)"
        )
        print(
            f"  평균 API 예측: {df['api_arrtime'].mean():.0f}초 "
            f"({df['api_arrtime'].mean() / 60:.1f}분)"
        )
        print(
            f"  평균 절대 오차: {df['abs_error'].mean():.0f}초 "
            f"({df['error_pct'].mean():.1f}%)"
        )
        print(f"  중앙값 오차: {df['abs_error'].median():.0f}초")

        if "arrprevstationcnt" in df.columns:
            print(f"\n정류장 수별 오차 분석 (상위 10개):")
            for station_cnt in sorted(df["arrprevstationcnt"].dropna().unique())[:10]:
                subset = df[df["arrprevstationcnt"] == station_cnt]
                if len(subset) > 10:
                    print(
                        f"  {int(station_cnt):2d}개: 평균 {subset['abs_error'].mean():4.0f}초, "
                        f"중앙값 {subset['abs_error'].median():4.0f}초 (n={len(subset):,})"
                    )

        if "hour" in df.columns:
            print(f"\n시간대별 오차 분석:")
            for hour in sorted(df["hour"].dropna().unique()):
                subset = df[df["hour"] == hour]
                if len(subset) > 10:
                    print(
                        f"  {int(hour):2d}시: 평균 {subset['abs_error'].mean():4.0f}초 "
                        f"(n={len(subset):,})"
                    )

        outliers = df[df["error_pct"] > 50]
        print(f"\n이상치 (오차 >50%):")
        print(f"  개수: {len(outliers):,}건 ({len(outliers) / len(df) * 100:.1f}%)")
        if len(outliers) > 0:
            print(f"  평균 오차: {outliers['abs_error'].mean():.0f}초")
            if "arrprevstationcnt" in outliers.columns:
                print(
                    f"  정류장 분포: "
                    f"{outliers['arrprevstationcnt'].value_counts().head(3).to_dict()}"
                )

        return df

    # ------------------------------------------------------
    # 8. 전체 파이프라인
    # ------------------------------------------------------
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        use_actual_labels: bool = True,
        validate: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("[prepare_training_data] 입력 데이터프레임이 비어 있습니다.")

        df = df.copy()

        if verbose:
            print(f"원본 데이터: {len(df):,} rows")

        # 1. 날씨 보간
        df = self.fill_weather_missing(df)
        if verbose:
            print("✓ 날씨 보간 완료")

        # 2. 버스 분리
        df = self.separate_multiple_buses(df)
        if verbose:
            print(f"✓ 버스 분리 완료: {len(df):,} rows")
            if len(df) > 0:
                unique_buses = df.groupby(["routeid", "nodeid"])["bus_index"].nunique()
                print(f"  평균 버스 수/정류장: {unique_buses.mean():.1f}대")

        # 3. 피처 생성 (hour 컬럼 필요)
        df = self.add_features(df)
        if verbose:
            print("✓ 피처 생성 완료 (hour 컬럼 포함)")

        # 4. 실제 도착 라벨 생성
        if use_actual_labels:
            labeled_df = self.create_actual_arrival_labels_improved(df)
            if len(labeled_df) == 0:
                raise ValueError("실제 도착 라벨을 생성할 수 없습니다. 수집 기간/데이터를 확인하세요.")

            # arrtime을 actual_arrtime으로 교체
            labeled_df["arrtime"] = labeled_df["actual_arrtime"]
            df = labeled_df

            if verbose:
                print("✓ 실제 도착 라벨 생성 완료")

            # 검증
            if validate:
                df = self.validate_labels(df)

        # 5. 궤적 생성
        df = self.create_bus_trajectory(df)

        if verbose:
            print("✓ 궤적 생성 완료")

        df = self.add_route_based_features(df)
        if verbose:
            print("✓ 노선기반 Feature 생성 완료")

        # 6. 이상치 제거
        df = self.remove_outliers(df)

        # 7. 필수 컬럼 체크
        essential_cols = ["arrtime", "arrprevstationcnt", "routeid", "nodeid"]
        for c in essential_cols:
            if c not in df.columns:
                raise ValueError(f"[prepare_training_data] 최종 데이터에 필수 컬럼 누락: {c}")

        df = df.dropna(subset=essential_cols)

        if len(df) == 0:
            raise ValueError("[prepare_training_data] 모든 행이 필터링되어 최종 데이터가 비었습니다.")

        if verbose:
            print(f"최종 데이터: {len(df):,} rows")
            print("=" * 60)

        return df

