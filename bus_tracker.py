from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, field
from enum import Enum


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

    def __post_init__(self):
        # 초기 countdown 기준 부여
        if self.initial_arrtime == 0:
            self.initial_arrtime = self.arrtime

    # ========================================================
    # 남은 시간 계산 (ACTIVE / PREDICTED 공통)
    # ========================================================
    def get_current_arrtime(self) -> int:
        elapsed = (datetime.now() - self.last_station_change_time).total_seconds()
        return max(0, self.initial_arrtime - int(elapsed))

    # ========================================================
    # ACTIVE 모드 업데이트
    # ========================================================
    def update(self, new: dict):
        now = datetime.now()

        # 정류장 변화 → countdown 리셋
        if new["arrprevstationcnt"] != self.arrprevstationcnt:
            self.last_station_change_time = now
            self.initial_arrtime = new["arrtime"]

        # 필드 업데이트
        self.arrprevstationcnt = new["arrprevstationcnt"]
        self.arrtime = new["arrtime"]
        self.vehicletp = new.get("vehicletp", self.vehicletp)
        self.status = BusStatus.ACTIVE
        self.last_update = now

        # meta
        for meta in ["weekday", "time_slot", "weather",
                     "temp", "humidity", "rain_mm", "snow_mm"]:
            if meta in new:
                setattr(self, meta, new[meta])


# ============================================================
# BusTracker (슬롯 기반 버스 다중 추적)
# ============================================================
class BusTracker:

    def __init__(self, predictor=None, historical_data=None):
        self.predictor = predictor
        self.historical_data = historical_data
        self.buses: Dict[str, BusInfo] = {}  # key: "routeid_nodeid_slot"
        self.prediction_interval = 60        # 예측 업데이트 간격(초)

    # key 생성
    def _key(self, routeid: str, nodeid: str, slot: int):
        return f"{routeid}_{nodeid}_{slot}"

    # ============================================================
    # 단일 row 입력 (main.py와 호환)
    # ============================================================
    def process_new_data(self, data: dict):
        """
        main.py에서 단일 row를 넣어도, 내부에서는 batch처럼 처리
        """
        self.process_batch([data])

    # ============================================================
    # batch 단위 처리: 동일 timestamp 기준 slot 재배치
    # ============================================================
    def process_batch(self, batch: List[dict]):
        # 1) routeid + nodeid 그룹핑
        groups = {}
        for d in batch:
            key = (d["routeid"], d["nodeid"])
            groups.setdefault(key, []).append(d)

        # 2) 각 그룹에서 arrtime 오름차순으로 slot 재배치
        for (routeid, nodeid), bus_list in groups.items():

            # arrtime 기준 정렬 → slot = index
            bus_list.sort(key=lambda x: x["arrtime"])

            # 기존 슬롯들
            existing_slots = [
                k for k in self.buses.keys()
                if k.startswith(f"{routeid}_{nodeid}_")
            ]
            existing_count = len(existing_slots)

            # 신규 개수
            new_count = len(bus_list)

            # ---------- slot 0..N-1 업데이트 ----------
            for slot, new_data in enumerate(bus_list):
                key = self._key(routeid, nodeid, slot)

                if key in self.buses:
                    # 기존 slot → ACTIVE 업데이트
                    self.buses[key].update(new_data)
                else:
                    # 새 BusInfo 생성
                    self.buses[key] = BusInfo(
                        routeid=routeid,
                        routeno=new_data["routeno"],
                        nodeid=nodeid,
                        nodenm=new_data["nodenm"],
                        slot=slot,
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
                        snow_mm=new_data["snow_mm"]
                    )

            # ---------- slot이 줄어든 경우 → 기존 슬롯 삭제 ----------
            for old_key in existing_slots:
                old_slot = int(old_key.split("_")[-1])
                if old_slot >= new_count:
                    del self.buses[old_key]

    # ============================================================
    # 오래된 ACTIVE 버스를 → PREDICTED 모드로 전환
    # ============================================================
    def check_lost_buses(self, timeout_seconds=180):
        now = datetime.now()

        for key, bus in list(self.buses.items()):
            if (now - bus.last_update).total_seconds() >= timeout_seconds:
                bus.status = BusStatus.PREDICTED
                self._predict_bus(bus)

    # ============================================================
    # 개별 예측 수행
    # ============================================================
    def _predict_bus(self, bus: BusInfo):
        now = datetime.now()

        # predictor 없는 경우 → countdown만 갱신
        if not self.predictor:
            bus.initial_arrtime = bus.get_current_arrtime()
            bus.last_station_change_time = now
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
            "hour": now.hour,
            "minute": now.minute,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "is_rush_hour": 1 if now.hour in [7,8,9,17,18,19] else 0,
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
                    bus.weekday, now.hour
                )
            else:
                predicted = bus.arrprevstationcnt * 60

        # countdown 기준 갱신
        bus.arrtime = int(predicted)
        bus.initial_arrtime = bus.arrtime
        bus.last_station_change_time = now

    # ============================================================
    # PREDICTED 모드 버스 → 정해진 주기마다 재예측
    # ============================================================
    def update_predictions(self):
        now = datetime.now()

        for key, bus in self.buses.items():
            if bus.status == BusStatus.PREDICTED:
                elapsed = (now - bus.last_station_change_time).total_seconds()
                if elapsed >= self.prediction_interval:
                    self._predict_bus(bus)

    # ============================================================
    # 도착한 버스 제거 (arrtime <= threshold)
    # ============================================================
    def remove_arrived_buses(self, threshold_seconds=10):
        to_delete = []
        for key, bus in self.buses.items():
            if bus.get_current_arrtime() <= threshold_seconds:
                to_delete.append(key)

        for k in to_delete:
            del self.buses[k]

    # ============================================================
    # 노선 + 정류장 조회 API와 연동
    # ============================================================
    def get_bus_info(self, routeid: str, nodeid: str) -> List[dict]:
        result = []
        for bus in self.buses.values():
            if bus.routeid == routeid and bus.nodeid == nodeid:
                cur = bus.get_current_arrtime()
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
        result = []
        for bus in self.buses.values():
            cur = bus.get_current_arrtime()
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
