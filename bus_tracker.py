from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ============================================================
# 버스 모드
# ============================================================
class BusMode(Enum):
    API = "api"           # API 데이터 사용 중
    PREDICTED = "predicted"  # ML 예측 사용 중


# ============================================================
# BusInfo
# ============================================================
@dataclass
class BusInfo:
    routeid: str
    routeno: str
    nodeid: str
    nodenm: str
    slot: int
    arrprevstationcnt: int
    arrtime: int
    vehicletp: str
    routetp: str

    # 모드 및 countdown
    mode: BusMode = BusMode.API
    initial_arrtime: int = 0
    prediction_time: datetime = field(default_factory=datetime.now)
    
    # 메타 정보
    last_update: datetime = field(default_factory=datetime.now)
    weekday: str = ""
    time_slot: str = ""
    weather: str = ""
    temp: float = 0.0
    humidity: float = 0.0
    rain_mm: float = 0.0
    snow_mm: float = 0.0
    
    tracking_id: str = ""

    # trajectory용 상태
    prev_station: Optional[int] = None
    prev_arrtime: Optional[int] = None
    prev_time: Optional[datetime] = None

    # 실시간 속도 feature
    sec_per_station: Optional[float] = None
    time_elapsed: Optional[float] = None

    def __post_init__(self):
        if self.initial_arrtime == 0:
            self.initial_arrtime = self.arrtime
        
        if not self.tracking_id:
            self.tracking_id = f"{self.routeid}_{self.nodeid}_{self.slot}_{int(self.prediction_time.timestamp())}"

    def get_current_arrtime(self, current_time: datetime = None) -> int:
        """
        countdown 계산 (★★★ API 모드도 적용! ★★★)
        
        mode에 관계없이:
        - last_update 이후 경과 시간만큼 차감
        - 실시간 countdown 구현
        """
        if current_time is None:
            current_time = datetime.now()
        
        if self.mode == BusMode.API:
            # ============================================================
            # ★★★ API 모드도 countdown 적용! ★★★
            # ============================================================
            # last_update 이후 경과 시간 계산
            elapsed = (current_time - self.last_update).total_seconds()
            
            # arrtime에서 경과 시간 차감
            remaining = max(0, self.arrtime - int(elapsed))
            
            return remaining
        else:
            # PREDICTED 모드: 기존 방식
            elapsed = (current_time - self.prediction_time).total_seconds()
            remaining = max(0, self.initial_arrtime - int(elapsed))
            
            return remaining


# ============================================================
# BusTracker - 하이브리드 모드
# ============================================================
class BusTracker:
    """
    하이브리드 모드 버스 추적기
    
    핵심 전략:
    1. API 있을 때: API 값 사용 (ACTIVE)
    2. API 끊기면: ML 예측 전환 (1회만!)
    3. 이후: countdown만 (재예측 없음!)
    
    장점:
    - API 정확도 + ML 안정성
    - 최소 예측 (API 끊길 때만)
    - CPU 효율적
    """

    def __init__(self, predictor=None, historical_data=None, 
                 simulation_mode=False,
                 api_timeout_seconds=100):  # 1분 40초
        self.predictor = predictor
        self.historical_data = historical_data
        self.buses: Dict[str, BusInfo] = {}
        self.simulation_mode = simulation_mode
        self.current_time = datetime.now()
        self.api_timeout_seconds = api_timeout_seconds
        # 실시간 속도 통계 저장 (EMA)
        self.route_speed_stats = {}   # routeid -> EMA(sec_per_station)
        self.node_speed_stats = {}    # nodeid -> EMA(sec_per_station)
        
        # 통계
        self.stats = {
            'total_predictions': 0,
            'buses_tracked': 0,
            'buses_arrived': 0,
            'buses_disappeared': 0,
            'api_to_ml_transitions': 0  # API → ML 전환 횟수
        }

    def _key(self, routeid: str, nodeid: str, slot: int):
        return f"{routeid}_{nodeid}_{slot}"

    def _get_current_time(self) -> datetime:
        if self.simulation_mode:
            return self.current_time
        else:
            return datetime.now()

    def update_simulation_time(self, time: datetime):
        if self.simulation_mode:
            self.current_time = time

    # ============================================================
    # ML 예측 (★★★ 완전 개선 버전 ★★★)
    # ============================================================
    def _predict_arrival_time(self, bus: BusInfo, current_time: datetime) -> int:
        """
        ML로 도착 시간 예측 (경과 시간 보정 + 완전 개선!)
        
        개선 사항:
        1. ✅ 경과 시간 보정
        2. ✅ 이미 도착한 버스 체크
        3. ✅ 상세 로그
        4. ✅ 예측값 검증
        5. ✅ 안전한 Fallback
        """
        if not self.predictor:
            return bus.arrtime

        # ============================================================
        # ★★★ 경과 시간 보정 ★★★
        # ============================================================
        elapsed_seconds = (current_time - bus.last_update).total_seconds()
        
        # 1. 평균 정류장당 시간 계산
        if bus.arrprevstationcnt > 0 and bus.arrtime > 0:
            avg_time_per_station = bus.arrtime / bus.arrprevstationcnt
        else:
            avg_time_per_station = 60  # 기본값
        
        # 2. 경과 시간 동안 지나간 정류장 수 추정
        estimated_stations_passed = int(elapsed_seconds / avg_time_per_station)
        
        # 3. 현재 상태 추정
        current_station_cnt = max(0, bus.arrprevstationcnt - estimated_stations_passed)
        current_arrtime = max(0, bus.arrtime - elapsed_seconds)
        
        # ============================================================
        # ★★★ 안전 장치: 이미 도착한 버스 체크 ★★★
        # ============================================================
        if current_arrtime <= 0 or current_station_cnt <= 0:
            print(f"  [ML 예측 스킵] {bus.routeid} #{bus.slot}: 이미 도착 추정")
            print(f"    📍 경과: {elapsed_seconds:.0f}초 ({elapsed_seconds/60:.1f}분)")
            print(f"    🚫 보정 arrtime: {current_arrtime:.0f}초, 정류장: {current_station_cnt}개")
            return 0
        
        # ============================================================
        # ★★★ 보정된 Feature로 예측 ★★★
        # ============================================================
        stats = getattr(self.predictor, "statistics", {}) or {}
        rt_dict = stats.get('route_sec_per_station', {})
        nd_dict = stats.get('node_sec_per_station', {})
        rth_dict = stats.get('route_hour_sec_per_station', {})
        route_max_dict = stats.get('route_max_station', {})

        # --- 1) sec_per_station (버스 개별) ---
        sec_per_station = bus.sec_per_station

        if sec_per_station is None:
            r = bus.routeid
            n = bus.nodeid
            h = current_time.hour

            # 실시간 EMA에서 먼저 찾기
            sec_from_rt = self.route_speed_stats.get(r)
            sec_from_nd = self.node_speed_stats.get(n)

            candidates = []

            if sec_from_rt is not None:
                candidates.append(sec_from_rt)
            if sec_from_nd is not None:
                candidates.append(sec_from_nd)
            if (r, h) in rth_dict:
                candidates.append(rth_dict[(r, h)])
            if r in rt_dict:
                candidates.append(rt_dict[r])
            if n in nd_dict:
                candidates.append(nd_dict[n])

            if candidates:
                sec_per_station = float(np.median(candidates))
            else:
                sec_per_station = 60.0  # 최종 fallback

        # --- 2) route_avg_sec / node_avg_sec / route_hour_avg_sec ---
        route_avg_sec = rt_dict.get(bus.routeid, sec_per_station)
        node_avg_sec = nd_dict.get(bus.nodeid, sec_per_station)
        route_hour_avg_sec = rth_dict.get(
            (bus.routeid, current_time.hour), route_avg_sec
        )

        # --- 3) station_progress_ratio ---
        route_max_station = route_max_dict.get(bus.routeid, max(bus.arrprevstationcnt, 1))
        station_progress_ratio = current_station_cnt / max(route_max_station, 1)
        
        features = {
            "routeid": bus.routeid,
            "nodeid": bus.nodeid,
            "routetp": bus.routetp,
            "vehicletp": bus.vehicletp,
            "arrprevstationcnt": current_station_cnt,  # ★ 보정!
            "weekday": bus.weekday,
            "time_slot": bus.time_slot,
            "weather": bus.weather,
            "temp": bus.temp,
            "humidity": bus.humidity,
            "rain_mm": bus.rain_mm,
            "snow_mm": bus.snow_mm,
            "hour": current_time.hour,
            "minute": current_time.minute,
            "day_of_week": current_time.weekday(),
            "is_weekend": 1 if current_time.weekday() >= 5 else 0,
            "is_rush_hour": 1 if current_time.hour in [7,8,9,17,18,19] else 0,
            "avg_time_per_station": avg_time_per_station,
            "sec_per_station": sec_per_station,
            "route_avg_sec": route_avg_sec,
            "node_avg_sec": node_avg_sec,
            "route_hour_avg_sec": route_hour_avg_sec,
            "station_progress_ratio": station_progress_ratio,
        }

        try:
            predicted = self.predictor.predict(features)
            self.stats['total_predictions'] += 1
            
            # ============================================================
            # ★★★ 상세 로그 (디버깅용) ★★★
            # ============================================================
            print(f"  [ML 예측] {bus.routeid} #{bus.slot}")
            print(f"    📍 경과: {elapsed_seconds:.0f}초 ({elapsed_seconds/60:.1f}분)")
            print(f"    📊 원본: {bus.arrprevstationcnt}개 정류장, {bus.arrtime}초 ({bus.arrtime/60:.1f}분)")
            print(f"    🔧 보정: {current_station_cnt}개 정류장, {current_arrtime:.0f}초 ({current_arrtime/60:.1f}분)")
            print(f"    🎯 예측: {predicted:.0f}초 ({predicted/60:.1f}분)")
            
            # ============================================================
            # ★★★ 예측값 검증 ★★★
            # ============================================================
            if predicted < 0:
                print(f"    ⚠️  음수 예측 감지 → 0으로 보정")
                return 0
            elif predicted > 3600:  # 1시간 이상
                print(f"    ⚠️  과도한 예측 ({predicted/60:.1f}분) → 보정값 사용")
                return int(current_arrtime)
            
            return int(predicted)
        
        except Exception as e:
            print(f"  [ML 예측 실패 → Fallback] {e}")
            
            # ============================================================
            # ★★★ Fallback 체계 ★★★
            # ============================================================
            
            # Fallback 1: Historical Pattern (있으면)
            if self.historical_data is not None:
                try:
                    historical = self.predictor.predict_by_historical_pattern(
                        self.historical_data,
                        bus.routeid, bus.nodeid, 
                        current_station_cnt,  # ★ 보정된 값 사용
                        bus.weekday, current_time.hour
                    )
                    print(f"  [Historical Pattern] {historical:.0f}초 ({historical/60:.1f}분)")
                    return int(historical)
                except Exception as e2:
                    print(f"  [Historical 실패] {e2}")
            
            # Fallback 2: 보정된 arrtime 직접 사용
            print(f"  [최종 Fallback] 보정된 arrtime: {current_arrtime:.0f}초 ({current_arrtime/60:.1f}분)")
            return int(current_arrtime)

    # ============================================================
    # 버스 매칭
    # ============================================================
    def _match_buses(self, existing_buses: List[BusInfo], new_data_list: List[dict]) -> List[tuple]:
        """기존 버스와 신규 데이터 매칭 (실시간용 간소화 버전)"""
        if not existing_buses or not new_data_list:
            return []
        
        n_exist = len(existing_buses)
        n_curr = len(new_data_list)
        cost_matrix = np.full((n_exist, n_curr), np.inf)
        
        for i, bus in enumerate(existing_buses):
            for j, new_data in enumerate(new_data_list):
                prev_station = float(bus.arrprevstationcnt)
                prev_arrtime = float(bus.arrtime)
                curr_station = float(new_data['arrprevstationcnt'])
                curr_arrtime = float(new_data['arrtime'])
                
                # 1) 추월 불가 (앞에 있던 버스가 뒤로 가지 않음)
                if j < i:
                    continue
                
                # 2) 정류장은 유지 또는 감소만 가능
                if curr_station > prev_station:
                    continue
                
                station_diff = prev_station - curr_station
                time_diff = prev_arrtime - curr_arrtime  # 줄어들어야 정상
                
                if station_diff == 0:
                    # 정류장 같으면 arrtime도 거의 같아야 함 (60초 이내)
                    if abs(time_diff) > 60:
                        continue
                    cost = abs(time_diff)
                else:
                    # 정류장 줄었으면 arrtime도 줄어야 함
                    if time_diff <= 0:
                        continue
                    avg_time_per_station = time_diff / max(station_diff, 1e-6)
                    if avg_time_per_station < 10 or avg_time_per_station > 600:
                        continue
                    cost = station_diff * 5  # 정류장 수를 더 강하게 반영
                
                order_penalty = max(j - i, 0) * 20
                cost_matrix[i, j] = cost + order_penalty
        
        # 최소 비용 매칭 (Greedy)
        pairs = []
        for i in range(n_exist):
            for j in range(n_curr):
                if cost_matrix[i, j] < np.inf:
                    pairs.append((cost_matrix[i, j], i, j))
        pairs.sort()
        
        matches = []
        used_exist = set()
        used_new = set()
        for cost, i, j in pairs:
            if i not in used_exist and j not in used_new and cost < 500:
                matches.append((i, j))
                used_exist.add(i)
                used_new.add(j)
        
        return matches


    # ============================================================
    # 데이터 처리 (하이브리드!)
    # ============================================================
    def process_new_data(self, data: dict):
        """단일 데이터 처리"""
        if 'collection_time' in data and isinstance(data['collection_time'], datetime):
            if self.simulation_mode:
                self.current_time = data['collection_time']
        
        self.process_batch([data])

    def process_batch(self, batch: List[dict]):
        """
        배치 처리 (하이브리드 모드)
        
        핵심:
        1. ★★★ API 데이터 중복 제거! ★★★
        2. 매칭된 버스: arrtime 변경 시에만 갱신
        3. 매칭 안 된 버스: cleanup에서 API → ML 전환
        4. 새 버스: API 모드로 시작
        """
        current_time = self._get_current_time()
        
        # ============================================================
        # ★★★ STEP 1: 중복 제거 ★★★
        # ============================================================
        # 같은 (routeid, nodeid, arrtime, arrprevstationcnt)는 하나만 유지
        seen = set()
        deduplicated_batch = []
        
        for d in batch:
            # 고유 키 생성
            key = (d["routeid"], d["nodeid"], d["arrtime"], d["arrprevstationcnt"])
            
            if key not in seen:
                seen.add(key)
                deduplicated_batch.append(d)
            # else:
            #     print(f"🔄 중복 제거: {d['routeid']} @ {d['nodenm']} "
            #           f"(arrtime: {d['arrtime']}초, 정류장: {d['arrprevstationcnt']}개)")
        
        # 중복 제거 통계 (선택: 주석 처리 가능)
        removed = len(batch) - len(deduplicated_batch)
        if removed > 0:
            print(f"📊 중복 제거: {removed}개 (원본: {len(batch)}개 → 처리: {len(deduplicated_batch)}개)")
        
        # ============================================================
        # ★★★ STEP 2: 그룹화 (중복 제거된 데이터로!) ★★★
        # ============================================================
        groups = {}
        
        for d in deduplicated_batch:
            key = (d["routeid"], d["nodeid"])
            groups.setdefault(key, []).append(d)

        for (routeid, nodeid), bus_list in groups.items():
            bus_list.sort(key=lambda x: x["arrtime"])

            existing_keys = [
                k for k in self.buses.keys()
                if k.startswith(f"{routeid}_{nodeid}_")
            ]
            existing_buses = [self.buses[k] for k in existing_keys]

            matches = self._match_buses(existing_buses, bus_list)
            
            matched_existing_idx = set()
            matched_new_idx = set()
            
            # ============================================================
            # ★★★ 매칭된 버스 업데이트 (arrtime 변경 시에만) ★★★
            # ============================================================
            for exist_idx, new_idx in matches:
                existing_bus = existing_buses[exist_idx]
                new_data = bus_list[new_idx]
                
                # --- 1) 이전 상태를 trajectory에 기록 ---
                prev_time = existing_bus.last_update
                curr_time = current_time
                time_elapsed = (curr_time - prev_time).total_seconds()

                prev_station = existing_bus.arrprevstationcnt
                curr_station = new_data['arrprevstationcnt']

                # station 감소 & 시간 정상 경과일 때만 이동으로 간주
                if (
                    prev_station is not None
                    and curr_station is not None
                    and curr_station < prev_station
                    and time_elapsed > 0
                    and time_elapsed < 3600  # 과도한 gap 방지 (preprocessor와 동일)
                ):
                    station_delta = max(prev_station - curr_station, 1)
                    sec_per_station = time_elapsed / station_delta

                    # 비현실적인 속도 필터
                    if 5 <= sec_per_station <= 600:
                        existing_bus.sec_per_station = sec_per_station
                        existing_bus.time_elapsed = time_elapsed
                    else:
                        # 이상치면 그냥 무시
                        existing_bus.sec_per_station = None
                        existing_bus.time_elapsed = None

                # --- 실시간 route/node 속도 EMA 업데이트 ---
                if existing_bus.sec_per_station is not None:
                    r = existing_bus.routeid
                    n = existing_bus.nodeid
                    s = existing_bus.sec_per_station

                    alpha = 0.2  # EMA 계수

                    prev_r = self.route_speed_stats.get(r)
                    if prev_r is None:
                        self.route_speed_stats[r] = s
                    else:
                        self.route_speed_stats[r] = (1 - alpha) * prev_r + alpha * s

                    prev_n = self.node_speed_stats.get(n)
                    if prev_n is None:
                        self.node_speed_stats[n] = s
                    else:
                        self.node_speed_stats[n] = (1 - alpha) * prev_n + alpha * s
                
                # ============================================================
                # ★★★ arrtime 또는 arrprevstationcnt가 변경되었는지 확인 ★★★
                # ============================================================
                arrtime_changed = (existing_bus.arrtime != new_data['arrtime'])
                station_changed = (existing_bus.arrprevstationcnt != new_data['arrprevstationcnt'])
                
                if arrtime_changed or station_changed:
                    # 값이 실제로 변경됨 → 갱신!
                    old_arrtime = existing_bus.arrtime
                    old_station = existing_bus.arrprevstationcnt
                    
                    existing_bus.arrtime = new_data['arrtime']
                    existing_bus.arrprevstationcnt = new_data['arrprevstationcnt']
                    
                    # 선택: 변경 로그 (디버깅용, 필요시 주석 해제)
                    # print(f"  ✏️  갱신: {existing_bus.routeid} #{existing_bus.slot} @ {new_data['nodenm']}")
                    # if arrtime_changed:
                    #     print(f"      arrtime: {old_arrtime}초 → {new_data['arrtime']}초")
                    # if station_changed:
                    #     print(f"      정류장: {old_station}개 → {new_data['arrprevstationcnt']}개")
                # else:
                #     # 값이 같음 → 갱신 안 함
                #     print(f"  ➖ 유지: {existing_bus.routeid} #{existing_bus.slot}: "
                #           f"arrtime {existing_bus.arrtime}초 (변화 없음)")
                
                # ============================================================
                # ★★★ CRITICAL: last_update는 항상 갱신! ★★★
                # ============================================================
                # 이유: countdown 계산에 사용되므로 항상 최신 시간이어야 함!
                # arrtime이 같아도 last_update가 갱신되어야 countdown이 정확함!
                existing_bus.last_update = current_time
                
                # 메타 정보 업데이트
                for meta in ["weekday", "time_slot", "weather",
                           "temp", "humidity", "rain_mm", "snow_mm"]:
                    if meta in new_data:
                        setattr(existing_bus, meta, new_data[meta])
                
                matched_existing_idx.add(exist_idx)
                matched_new_idx.add(new_idx)
            
            # ★ 새 버스 생성 (API 모드로 시작)
            for j, new_data in enumerate(bus_list):
                if j not in matched_new_idx:
                    used_slots = {int(k.split("_")[-1]) for k in existing_keys}
                    new_slot = 0
                    while new_slot in used_slots:
                        new_slot += 1
                    
                    key = self._key(routeid, nodeid, new_slot)
                    self.buses[key] = BusInfo(
                        routeid=routeid,
                        routeno=new_data["routeno"],
                        nodeid=nodeid,
                        nodenm=new_data["nodenm"],
                        slot=new_slot,
                        arrprevstationcnt=new_data["arrprevstationcnt"],
                        arrtime=new_data["arrtime"],
                        initial_arrtime=new_data["arrtime"],
                        vehicletp=new_data["vehicletp"],
                        routetp=new_data["routetp"],
                        mode=BusMode.API,  # ★ API 모드로 시작
                        weekday=new_data["weekday"],
                        time_slot=new_data["time_slot"],
                        weather=new_data["weather"],
                        temp=new_data["temp"],
                        humidity=new_data["humidity"],
                        rain_mm=new_data["rain_mm"],
                        snow_mm=new_data["snow_mm"],
                        last_update=current_time,
                        prediction_time=current_time
                    )
                    
                    self.stats['buses_tracked'] += 1
                    print(f"🆕 새 버스 (API): {routeid} #{new_slot} @ {new_data['nodenm']} "
                          f"(arrtime: {new_data['arrtime']}초)")

    # ============================================================
    # 정리 작업 (핵심!)
    # ============================================================
    def cleanup(self):
        """
        정리 작업 (하이브리드 모드)
        
        1. API → ML 전환 (100초 타임아웃)
        2. countdown 기반 도착 제거
        3. 완전히 사라진 버스 제거 (10분)
        """
        current_time = self._get_current_time()
        
        # 1. API 끊긴 버스 → ML 전환
        self._check_api_timeout(current_time)
        
        # 2. 도착 버스 제거
        arrived = self._remove_arrived_buses(threshold_seconds=30)
        
        # 3. 사라진 버스 제거
        disappeared = self._remove_disappeared_buses(timeout_seconds=600)
        
        # 통계 출력
        total = len(self.buses)
        if arrived > 0 or disappeared > 0 or total > 0:
            api_count = sum(1 for b in self.buses.values() if b.mode == BusMode.API)
            ml_count = sum(1 for b in self.buses.values() if b.mode == BusMode.PREDICTED)
            
            print(f"📊 현재 추적: {total}대 (API: {api_count}, ML: {ml_count}) | "
                  f"예측: {self.stats['total_predictions']}회, "
                  f"도착: {self.stats['buses_arrived']}대, "
                  f"API→ML: {self.stats['api_to_ml_transitions']}회")

    def _check_api_timeout(self, current_time: datetime):
        """
        API 끊긴 버스 감지 및 ML 전환 (★★★ 상세 로그 추가 ★★★)
        
        핵심: 100초간 매칭 안 되면 API 끊긴 것으로 간주
        """
        transitioned = []
        
        for key, bus in self.buses.items():
            # API 모드이면서 오래 업데이트 안 됨
            if bus.mode == BusMode.API:
                elapsed = (current_time - bus.last_update).total_seconds()
                
                if elapsed >= self.api_timeout_seconds:
                    # ============================================================
                    # ★★★ 보정 정보 미리 계산 (로그용) ★★★
                    # ============================================================
                    if bus.arrprevstationcnt > 0 and bus.arrtime > 0:
                        avg_time = bus.arrtime / bus.arrprevstationcnt
                        stations_passed = int(elapsed / avg_time)
                        current_stations = max(0, bus.arrprevstationcnt - stations_passed)
                    else:
                        avg_time = 60
                        stations_passed = 0
                        current_stations = bus.arrprevstationcnt
                    
                    # ============================================================
                    # ★★★ API → ML 전환 (상세 로그!) ★★★
                    # ============================================================
                    print(f"⚠️  API 타임아웃 → ML 전환: {bus.routeid} #{bus.slot}")
                    print(f"   📍 {elapsed/60:.1f}분 전 마지막 API")
                    print(f"   📊 원본: {bus.arrprevstationcnt}개 정류장, {bus.arrtime}초")
                    print(f"   🔧 예상: {current_stations}개 정류장 (약 {stations_passed}개 지남)")
                    
                    # ML 예측 수행 (딱 1회!)
                    predicted_time = self._predict_arrival_time(bus, current_time)
                    
                    # ============================================================
                    # ★★★ PREDICTED 모드로 전환 + 필드 업데이트 ★★★
                    # ============================================================
                    bus.mode = BusMode.PREDICTED
                    bus.initial_arrtime = predicted_time
                    bus.prediction_time = current_time
                    bus.arrtime = predicted_time  # ★ arrtime도 업데이트!
                    
                    self.stats['api_to_ml_transitions'] += 1
                    transitioned.append(bus)
        
        if transitioned:
            print(f"🔄 API → ML 전환: {len(transitioned)}대")

    def _remove_arrived_buses(self, threshold_seconds=30):
        """countdown 기반 도착 버스 제거"""
        current_time = self._get_current_time()
        to_delete = []
        
        for key, bus in self.buses.items():
            remaining = bus.get_current_arrtime(current_time)
            if remaining <= threshold_seconds:
                to_delete.append((key, bus))

        if to_delete:
            print(f"🚏 도착 제거: {len(to_delete)}대")
            for key, bus in to_delete:
                print(f"  - {bus.routeid} #{bus.slot} ({bus.mode.value})")
                del self.buses[key]
                self.stats['buses_arrived'] += 1
        
        return len(to_delete)

    def _remove_disappeared_buses(self, timeout_seconds=600):
        """완전히 사라진 버스 제거 (10분)"""
        current_time = self._get_current_time()
        to_delete = []
        
        for key, bus in self.buses.items():
            elapsed = (current_time - bus.last_update).total_seconds()
            if elapsed >= timeout_seconds:
                to_delete.append((key, bus, elapsed))
        
        if to_delete:
            print(f"🗑️  사라진 버스 제거: {len(to_delete)}대")
            for key, bus, elapsed in to_delete:
                print(f"  - {bus.routeid} #{bus.slot}: {elapsed/60:.1f}분 전")
                del self.buses[key]
                self.stats['buses_disappeared'] += 1
        
        return len(to_delete)

    # ============================================================
    # 조회 API
    # ============================================================
    def get_bus_info(self, routeid: str, nodeid: str) -> List[dict]:
        """노선 + 정류장 조회"""
        current_time = self._get_current_time()
        result = []
        
        for bus in self.buses.values():
            if bus.routeid == routeid and bus.nodeid == nodeid:
                remaining = bus.get_current_arrtime(current_time)
                result.append({
                    "routeid": bus.routeid,
                    "routeno": bus.routeno,
                    "nodeid": bus.nodeid,
                    "nodenm": bus.nodenm,
                    "slot": bus.slot,
                    "arrprevstationcnt": bus.arrprevstationcnt,
                    "arrtime": remaining,
                    "display_minutes": remaining // 60,
                    "display_seconds": remaining % 60,
                    "vehicletp": bus.vehicletp,
                    "routetp": bus.routetp,
                    "mode": bus.mode.value,  # api or predicted
                    "last_update": bus.last_update.isoformat(),
                })

        result.sort(key=lambda x: x["arrtime"])
        return result

    def get_all_buses(self) -> List[dict]:
        """전체 버스 조회"""
        current_time = self._get_current_time()
        result = []
        
        for bus in self.buses.values():
            remaining = bus.get_current_arrtime(current_time)
            result.append({
                "routeid": bus.routeid,
                "routeno": bus.routeno,
                "nodeid": bus.nodeid,
                "nodenm": bus.nodenm,
                "slot": bus.slot,
                "arrprevstationcnt": bus.arrprevstationcnt,
                "arrtime": remaining,
                "display_minutes": remaining // 60,
                "display_seconds": remaining % 60,
                "vehicletp": bus.vehicletp,
                "routetp": bus.routetp,
                "mode": bus.mode.value,
                "last_update": bus.last_update.isoformat(),
            })

        return result

    def get_stats(self) -> dict:
        """통계 정보"""
        api_count = sum(1 for b in self.buses.values() if b.mode == BusMode.API)
        ml_count = sum(1 for b in self.buses.values() if b.mode == BusMode.PREDICTED)
        
        return {
            'total_buses': len(self.buses),
            'api_buses': api_count,
            'ml_buses': ml_count,
            'total_predictions': self.stats['total_predictions'],
            'buses_tracked': self.stats['buses_tracked'],
            'buses_arrived': self.stats['buses_arrived'],
            'buses_disappeared': self.stats['buses_disappeared'],
            'api_to_ml_transitions': self.stats['api_to_ml_transitions'],
            'avg_predictions_per_bus': (
                self.stats['total_predictions'] / max(1, self.stats['buses_tracked'])
            )
        }