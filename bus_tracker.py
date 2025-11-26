from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ============================================================
# 상태 정의
# ============================================================
class BusStatus(Enum):
    ACTIVE = "active"
    PREDICTED = "predicted"


# ============================================================
# BusInfo
# ============================================================
@dataclass
class BusInfo:
    routeid: str
    routeno: str
    nodeid: str
    nodenm: str
    slot: int                   # 앞차 = 0 / 뒤차 = 1 / ...
    arrprevstationcnt: int
    arrtime: int
    vehicletp: str
    routetp: str

    # 상태
    status: BusStatus = BusStatus.ACTIVE
    last_update: datetime = field(default_factory=datetime.now)

    # countdown용
    initial_arrtime: int = 0
    last_station_change_time: datetime = field(default_factory=datetime.now)

    # meta
    weekday: str = ""
    time_slot: str = ""
    weather: str = ""
    temp: float = 0.0
    humidity: float = 0.0
    rain_mm: float = 0.0
    snow_mm: float = 0.0

    # 버스 식별을 위한 추가 정보
    tracking_id: str = ""  # 내부 추적 ID

    def __post_init__(self):
        # 초기 countdown 기준 부여
        if self.initial_arrtime == 0:
            self.initial_arrtime = self.arrtime
        
        # tracking_id 생성
        if not self.tracking_id:
            self.tracking_id = f"{self.routeid}_{self.nodeid}_{self.slot}_{int(self.last_update.timestamp())}"

    # ========================================================
    # 남은 시간 계산 (현재 시간 기준)
    # ========================================================
    def get_current_arrtime(self, current_time: datetime = None) -> int:
        """
        current_time: 기준 시각 (None이면 실시간, 주어지면 시뮬레이션)
        """
        if current_time is None:
            current_time = datetime.now()
        
        elapsed = (current_time - self.last_station_change_time).total_seconds()
        return max(0, self.initial_arrtime - int(elapsed))

    # ========================================================
    # ACTIVE 모드 업데이트
    # ========================================================
    def update(self, new: dict, current_time: datetime = None):
        if current_time is None:
            current_time = datetime.now()

        # 정류장 변화 → countdown 리셋
        if new["arrprevstationcnt"] != self.arrprevstationcnt:
            self.last_station_change_time = current_time
            self.initial_arrtime = new["arrtime"]

        # 필드 업데이트
        self.arrprevstationcnt = new["arrprevstationcnt"]
        self.arrtime = new["arrtime"]
        self.vehicletp = new.get("vehicletp", self.vehicletp)
        self.status = BusStatus.ACTIVE
        self.last_update = current_time

        # meta
        for meta in ["weekday", "time_slot", "weather",
                     "temp", "humidity", "rain_mm", "snow_mm"]:
            if meta in new:
                setattr(self, meta, new[meta])


# ============================================================
# BusTracker (슬롯 기반 버스 다중 추적)
# ============================================================
class BusTracker:

    def __init__(self, predictor=None, historical_data=None, simulation_mode=False):
        self.predictor = predictor
        self.historical_data = historical_data
        self.buses: Dict[str, BusInfo] = {}  # key: "routeid_nodeid_slot"
        self.prediction_interval = 60        # 예측 업데이트 간격(초)
        self.simulation_mode = simulation_mode  # 시뮬레이션 모드 플래그
        self.current_time = datetime.now()   # 현재 시각 (시뮬레이션용)

    # key 생성
    def _key(self, routeid: str, nodeid: str, slot: int):
        return f"{routeid}_{nodeid}_{slot}"

    # ============================================================
    # 시간 처리 통일 - 현재 시각 반환
    # ============================================================
    def _get_current_time(self) -> datetime:
        """시뮬레이션 모드에서는 내부 시각, 실시간 모드에서는 현재 시각"""
        if self.simulation_mode:
            return self.current_time
        else:
            return datetime.now()

    def update_simulation_time(self, time: datetime):
        """시뮬레이션 모드에서 시각 업데이트"""
        if self.simulation_mode:
            self.current_time = time

    # ============================================================
    # 단일 row 입력 (main.py와 호환)
    # ============================================================
    def process_new_data(self, data: dict):
        """
        main.py에서 단일 row를 넣어도, 내부에서는 batch처럼 처리
        """
        # collection_time 업데이트
        if 'collection_time' in data and isinstance(data['collection_time'], datetime):
            if self.simulation_mode:
                self.current_time = data['collection_time']
        
        self.process_batch([data])

    # ============================================================
    # 버스 매칭 알고리즘 - 최소 비용 매칭
    # ============================================================
    def _match_buses(self, existing_buses: List[BusInfo], new_data_list: List[dict]) -> List[tuple]:
        """
        기존 버스와 신규 데이터를 매칭
        Returns: [(existing_bus_idx, new_data_idx), ...]
        """
        if not existing_buses:
            return []
        
        if not new_data_list:
            return []
        
        n_exist = len(existing_buses)
        n_new = len(new_data_list)
        
        # 비용 행렬 생성
        cost_matrix = np.zeros((n_exist, n_new))
        
        for i, bus in enumerate(existing_buses):
            for j, new_data in enumerate(new_data_list):
                # arrtime 차이를 비용으로 사용
                # 정류장 수가 감소한 경우 보너스
                station_diff = bus.arrprevstationcnt - new_data['arrprevstationcnt']
                time_diff = abs(bus.arrtime - new_data['arrtime'])
                
                # 정류장이 진행된 경우(감소) 매칭 가능성 높임
                if station_diff > 0:
                    cost = time_diff * 0.5
                elif station_diff == 0:
                    cost = time_diff
                else:
                    # 정류장이 늘어난 경우 - 다른 버스일 가능성
                    cost = time_diff * 2.0 + 1000
                
                cost_matrix[i, j] = cost
        
        # 간단한 greedy 매칭 (Hungarian algorithm 대신)
        matches = []
        used_new = set()
        
        # 비용이 낮은 순으로 정렬
        pairs = []
        for i in range(n_exist):
            for j in range(n_new):
                pairs.append((cost_matrix[i, j], i, j))
        pairs.sort()
        
        # 각 버스는 한 번씩만 매칭
        used_exist = set()
        for cost, i, j in pairs:
            if i not in used_exist and j not in used_new:
                # 비용이 너무 크면 매칭하지 않음 (새 버스로 간주)
                if cost < 500:  # threshold
                    matches.append((i, j))
                    used_exist.add(i)
                    used_new.add(j)
        
        return matches

    # ============================================================
    # batch 단위 처리: 개선된 슬롯 재배치
    # ============================================================
    def process_batch(self, batch: List[dict]):
        current_time = self._get_current_time()
        
        # 1) routeid + nodeid 그룹핑
        groups = {}
        for d in batch:
            key = (d["routeid"], d["nodeid"])
            groups.setdefault(key, []).append(d)

        # 2) 각 그룹에서 버스 매칭 및 업데이트
        for (routeid, nodeid), bus_list in groups.items():

            # arrtime 기준 정렬
            bus_list.sort(key=lambda x: x["arrtime"])

            # 기존 버스 찾기
            existing_keys = [
                k for k in self.buses.keys()
                if k.startswith(f"{routeid}_{nodeid}_")
            ]
            existing_buses = [self.buses[k] for k in existing_keys]

            # 버스 매칭
            matches = self._match_buses(existing_buses, bus_list)
            
            # 매칭된 버스 업데이트
            matched_existing_idx = set()
            matched_new_idx = set()
            
            for exist_idx, new_idx in matches:
                existing_bus = existing_buses[exist_idx]
                new_data = bus_list[new_idx]
                existing_bus.update(new_data, current_time)
                matched_existing_idx.add(exist_idx)
                matched_new_idx.add(new_idx)
            
            # 매칭 안 된 신규 데이터 → 새 버스 생성
            for j, new_data in enumerate(bus_list):
                if j not in matched_new_idx:
                    # 빈 슬롯 찾기
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
                        vehicletp=new_data["vehicletp"],
                        routetp=new_data["routetp"],
                        weekday=new_data["weekday"],
                        time_slot=new_data["time_slot"],
                        weather=new_data["weather"],
                        temp=new_data["temp"],
                        humidity=new_data["humidity"],
                        rain_mm=new_data["rain_mm"],
                        snow_mm=new_data["snow_mm"],
                        last_update=current_time,
                        last_station_change_time=current_time
                    )
            
            # 매칭 안 된 기존 버스는 유지 (ACTIVE → PREDICTED로 전환 대기)

    # ============================================================
    # 오래된 ACTIVE 버스를 → PREDICTED 모드로 전환
    # ============================================================
    def check_lost_buses(self, timeout_seconds=180):
        current_time = self._get_current_time()

        for key, bus in list(self.buses.items()):
            if bus.status == BusStatus.ACTIVE:
                elapsed = (current_time - bus.last_update).total_seconds()
                if elapsed >= timeout_seconds:
                    bus.status = BusStatus.PREDICTED
                    self._predict_bus(bus, current_time)

    # ============================================================
    # 개별 예측 수행
    # ============================================================
    def _predict_bus(self, bus: BusInfo, current_time: datetime):
        # predictor 없는 경우 → countdown만 갱신
        if not self.predictor:
            bus.initial_arrtime = bus.get_current_arrtime(current_time)
            bus.last_station_change_time = current_time
            return

        # ML 입력 feature 구성
        features = {
            "routeid": bus.routeid,
            "nodeid": bus.nodeid,
            "routetp": bus.routetp,
            "vehicletp": bus.vehicletp,
            "arrprevstationcnt": bus.arrprevstationcnt,
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
            "avg_time_per_station": max(1, bus.arrtime) / max(1, bus.arrprevstationcnt)
        }

        # ML 예측 + fallback
        try:
            predicted = self.predictor.predict(features)
        except Exception as e:
            print(f"[ML 예측 실패 → fallback] {e}")

            if self.historical_data is not None:
                predicted = self.predictor.predict_by_historical_pattern(
                    self.historical_data,
                    bus.routeid, bus.nodeid, bus.arrprevstationcnt,
                    bus.weekday, current_time.hour
                )
            else:
                predicted = bus.arrprevstationcnt * 70

        # countdown 기준 갱신
        bus.arrtime = int(predicted)
        bus.initial_arrtime = bus.arrtime
        bus.last_station_change_time = current_time

    # ============================================================
    # PREDICTED 모드 버스 → 정해진 주기마다 재예측
    # ============================================================
    def update_predictions(self):
        current_time = self._get_current_time()

        for key, bus in self.buses.items():
            if bus.status == BusStatus.PREDICTED:
                elapsed = (current_time - bus.last_station_change_time).total_seconds()
                if elapsed >= self.prediction_interval:
                    self._predict_bus(bus, current_time)

    # ============================================================
    # 도착한 버스 제거 (arrtime <= threshold)
    # ============================================================
    def remove_arrived_buses(self, threshold_seconds=10):
        current_time = self._get_current_time()
        to_delete = []
        
        for key, bus in self.buses.items():
            if bus.get_current_arrtime(current_time) <= threshold_seconds:
                to_delete.append(key)

        for k in to_delete:
            del self.buses[k]

    # ============================================================
    # 노선 + 정류장 조회 API와 연동
    # ============================================================
    def get_bus_info(self, routeid: str, nodeid: str) -> List[dict]:
        current_time = self._get_current_time()
        result = []
        
        for bus in self.buses.values():
            if bus.routeid == routeid and bus.nodeid == nodeid:
                cur = bus.get_current_arrtime(current_time)
                result.append({
                    "routeid": bus.routeid,
                    "routeno": bus.routeno,
                    "nodeid": bus.nodeid,
                    "nodenm": bus.nodenm,
                    "slot": bus.slot,
                    "arrprevstationcnt": bus.arrprevstationcnt,
                    "arrtime": cur,
                    "display_minutes": cur // 60,
                    "display_seconds": cur % 60,
                    "vehicletp": bus.vehicletp,
                    "routetp": bus.routetp,
                    "status": bus.status.value,
                    "last_update": bus.last_update.isoformat(),
                })

        # arrtime 오름차순 정렬
        result.sort(key=lambda x: x["arrtime"])
        return result

    # ============================================================
    # 전체 버스 조회
    # ============================================================
    def get_all_buses(self):
        current_time = self._get_current_time()
        result = []
        
        for bus in self.buses.values():
            cur = bus.get_current_arrtime(current_time)
            result.append({
                "routeid": bus.routeid,
                "routeno": bus.routeno,
                "nodeid": bus.nodeid,
                "nodenm": bus.nodenm,
                "slot": bus.slot,
                "arrprevstationcnt": bus.arrprevstationcnt,
                "arrtime": cur,
                "display_minutes": cur // 60,
                "display_seconds": cur % 60,
                "vehicletp": bus.vehicletp,
                "routetp": bus.routetp,
                "status": bus.status.value,
                "last_update": bus.last_update.isoformat(),
            })

        return result