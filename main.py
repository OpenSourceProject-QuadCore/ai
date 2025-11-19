from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import asyncio
from datetime import datetime
import uvicorn

from bus_tracker import BusTracker
from bus_predictor import BusArrivalPredictor
from data_preprocessing import BusDataPreprocessor

app = FastAPI(title="구미 버스 실시간 추적 API")

# --------------------------------------------------------
# CORS
# --------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------------------------------------------
# 글로벌 객체
# --------------------------------------------------------
tracker: Optional[BusTracker] = None
predictor: Optional[BusArrivalPredictor] = None
historical_data: Optional[pd.DataFrame] = None


# --------------------------------------------------------
# Request Models
# --------------------------------------------------------
class BusArrivalData(BaseModel):
    collection_time: str   # 반드시 서버에 전달됨
    weekday: str
    time_slot: str
    weather: str
    temp: float
    humidity: float
    rain_mm: float
    snow_mm: float
    nodeid: str
    nodenm: str
    routeid: str
    routeno: str
    routetp: str
    arrprevstationcnt: int
    arrtime: int
    vehicletp: str


# --------------------------------------------------------
# Response Model
# --------------------------------------------------------
class BusInfoResponse(BaseModel):
    routeid: str
    routeno: str
    nodeid: str
    nodenm: str
    arrprevstationcnt: int
    arrtime: int
    display_minutes: int
    display_seconds: int
    vehicletp: str
    routetp: str
    status: str
    last_update: str


# --------------------------------------------------------
# 서버 시작 시 초기화
# --------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global tracker, predictor, historical_data

    print("=== 서버 초기화 시작 ===")

    # -------------------------
    # 모델 로드
    # -------------------------
    try:
        predictor = BusArrivalPredictor()
        predictor.load("models/bus_predictor.pkl")
        print("✓ 예측 모델 로드 성공")
    except Exception as e:
        print(f"⚠ 모델 로드 실패: {e}")
        predictor = None

    # -------------------------
    # 과거 데이터 로드
    # -------------------------
    try:
        pre = BusDataPreprocessor("bus_arrivals.csv")
        historical_data = pre.load_data()
        print(f"✓ 과거 데이터 로드 ({len(historical_data)} rows)")
    except Exception as e:
        print(f"⚠ 과거 데이터 로드 실패: {e}")
        historical_data = None

    # -------------------------
    # BusTracker 초기화
    # -------------------------
    tracker = BusTracker(predictor=predictor, historical_data=historical_data)
    print("✓ BusTracker 초기화 완료")

    # -------------------------
    # 백그라운드 작업 실행
    # -------------------------
    asyncio.create_task(background_task_loop())
    print("✓ 백그라운드 작업 시작")

    print("=== 초기화 완료 ===")


# --------------------------------------------------------
# 백그라운드 작업
# collection_time 기준을 사용하든 현재 시각(now) 기준을 사용하든
# 설계에 맞게 BusTracker 내부 메서드가 이미 정리되어 있음.
# --------------------------------------------------------
async def background_task_loop():
    while True:

        if tracker is None:
            print("⚠ tracker=None, 초기화 대기 중")
            await asyncio.sleep(2)
            continue

        try:
            tracker.check_lost_buses(timeout_seconds=180)
            tracker.update_predictions()
            tracker.remove_arrived_buses(threshold_seconds=10)

        except Exception as e:
            print(f"⚠ 백그라운드 오류: {e}")

        await asyncio.sleep(10)


# --------------------------------------------------------
# POST: 1개 버스 데이터 수신
# --------------------------------------------------------
@app.post("/api/bus-arrival")
async def receive_bus_data(data: BusArrivalData):
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")

    bus = data.dict()

    # collection_time → datetime 변환
    try:
        bus["collection_time"] = datetime.fromisoformat(bus["collection_time"])
    except Exception:
        print("⚠ collection_time 파싱 실패 → now()로 대체")
        bus["collection_time"] = datetime.now()

    tracker.process_new_data(bus)
    return {"status": "success", "message": "ok"}


# --------------------------------------------------------
# POST: 여러 개 수신
# --------------------------------------------------------
@app.post("/api/bus-arrival/batch")
async def receive_bus_data_batch(data_list: List[BusArrivalData]):
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")

    for d in data_list:
        bus = d.dict()
        try:
            bus["collection_time"] = datetime.fromisoformat(bus["collection_time"])
        except:
            bus["collection_time"] = datetime.now()

        tracker.process_new_data(bus)

    return {"status": "success", "message": f"{len(data_list)} processed"}


# --------------------------------------------------------
# GET: 전체 버스 조회
# --------------------------------------------------------
@app.get("/api/buses", response_model=List[BusInfoResponse])
async def get_all_buses():
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")
    return tracker.get_all_buses()


# --------------------------------------------------------
# GET: 노선별 조회
# --------------------------------------------------------
@app.get("/api/buses/route/{route_id}", response_model=List[BusInfoResponse])
async def get_by_route(route_id: str):
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")

    buses = tracker.get_all_buses()
    return [b for b in buses if b["routeid"] == route_id]


# --------------------------------------------------------
# GET: 정류장별 조회
# --------------------------------------------------------
@app.get("/api/buses/station/{node_id}", response_model=List[BusInfoResponse])
async def get_by_station(node_id: str):
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")

    buses = tracker.get_all_buses()
    buses = [b for b in buses if b["nodeid"] == node_id]
    buses.sort(key=lambda x: x["arrtime"])
    return buses


# --------------------------------------------------------
# GET: 노선+정류장 조회
# --------------------------------------------------------
@app.get("/api/buses/route/{route_id}/station/{node_id}",
         response_model=List[BusInfoResponse])
async def get_by_route_and_station(route_id: str, node_id: str):
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")
    return tracker.get_bus_info(route_id, node_id)


# --------------------------------------------------------
# GET: 서버 상태
# --------------------------------------------------------
@app.get("/api/status")
async def get_status():
    if tracker is None:
        return {"status": "error", "message": "tracker not initialized"}

    buses = tracker.get_all_buses()
    active = sum(1 for b in buses if b["status"] == "active")
    predicted = sum(1 for b in buses if b["status"] == "predicted")

    return {
        "status": "running",
        "total_buses": len(buses),
        "active_buses": active,
        "predicted_buses": predicted,
        "predictor_loaded": predictor is not None,
        "historical_data_loaded": historical_data is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    return {"message": "Gumi Real-time Bus API", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
