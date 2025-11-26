from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import asyncio
from datetime import datetime
import uvicorn
import argparse

from bus_tracker import BusTracker
from bus_predictor import BusArrivalPredictor
from data_preprocessing import BusDataPreprocessor

# --------------------------------------------------------
# 전역 변수
# --------------------------------------------------------
app = FastAPI(title="구미 버스 실시간 추적 API")
tracker: Optional[BusTracker] = None
predictor: Optional[BusArrivalPredictor] = None
historical_data: Optional[pd.DataFrame] = None
SIMULATION_MODE = False  # 시뮬레이션 모드 플래그


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
# Request Models
# --------------------------------------------------------
class BusArrivalData(BaseModel):
    collection_time: str   # ISO format string
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
    slot: int
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

    print("=" * 60)
    print("=== 서버 초기화 시작 ===")
    print(f"모드: {'시뮬레이션' if SIMULATION_MODE else '실시간'}")
    print("=" * 60)

    # -------------------------
    # 모델 로드
    # -------------------------
    try:
        predictor = BusArrivalPredictor()
        predictor.load("models/bus_predictor.pkl")
        print("✓ 예측 모델 로드 성공")
    except Exception as e:
        print(f"⚠ 모델 로드 실패: {e}")
        print("  예측 기능 없이 실행됩니다.")
        predictor = None

    # -------------------------
    # 과거 데이터 로드 (fallback용)
    # -------------------------
    try:
        # 전처리된 데이터가 있으면 사용
        processed_path = "data/processed_bus_arrivals.csv"
        if pd.io.common.file_exists(processed_path):
            historical_data = pd.read_csv(processed_path)
            print(f"✓ 전처리된 과거 데이터 로드 ({len(historical_data):,} rows)")
        else:
            # 없으면 원본 데이터 로드
            pre = BusDataPreprocessor("bus_arrivals.csv")
            historical_data = pre.load_data()
            print(f"✓ 원본 과거 데이터 로드 ({len(historical_data):,} rows)")
    except Exception as e:
        print(f"⚠ 과거 데이터 로드 실패: {e}")
        print("  Fallback 기능이 제한됩니다.")
        historical_data = None

    # -------------------------
    # BusTracker 초기화
    # -------------------------
    tracker = BusTracker(
        predictor=predictor, 
        historical_data=historical_data,
        simulation_mode=SIMULATION_MODE
    )
    print("✓ BusTracker 초기화 완료")

    # -------------------------
    # 백그라운드 작업 실행
    # -------------------------
    asyncio.create_task(background_task_loop())
    print("✓ 백그라운드 작업 시작")

    print("=" * 60)
    print("=== 초기화 완료 ===")
    print("=" * 60)


# --------------------------------------------------------
# 백그라운드 작업
# --------------------------------------------------------
async def background_task_loop():
    """
    주기적으로 버스 상태 업데이트
    - 오래된 버스 PREDICTED 전환
    - PREDICTED 버스 재예측
    - 도착한 버스 제거
    """
    while True:
        if tracker is None:
            await asyncio.sleep(2)
            continue

        try:
            # 180초 동안 업데이트 없으면 PREDICTED로 전환
            tracker.check_lost_buses(timeout_seconds=180)
            
            # PREDICTED 버스 재예측
            tracker.update_predictions()
            
            # 도착 임박 버스 제거 (10초 이하)
            tracker.remove_arrived_buses(threshold_seconds=10)

        except Exception as e:
            print(f"⚠ 백그라운드 작업 오류: {e}")

        await asyncio.sleep(10)


# --------------------------------------------------------
# POST: 1개 버스 데이터 수신
# --------------------------------------------------------
@app.post("/api/bus-arrival")
async def receive_bus_data(data: BusArrivalData):
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")

    bus = data.dict()

    # collection_time 파싱
    try:
        bus["collection_time"] = datetime.fromisoformat(bus["collection_time"])
    except Exception as e:
        if SIMULATION_MODE:
            # 시뮬레이션 모드에서는 반드시 파싱되어야 함
            raise HTTPException(400, f"collection_time 파싱 실패: {e}")
        else:
            # 실시간 모드에서는 현재 시간 사용
            print(f"⚠ collection_time 파싱 실패 → now()로 대체: {e}")
            bus["collection_time"] = datetime.now()

    # 데이터 처리
    tracker.process_new_data(bus)
    return {"status": "success", "message": "ok"}


# --------------------------------------------------------
# POST: 여러 개 수신 (배치)
# --------------------------------------------------------
@app.post("/api/bus-arrival/batch")
async def receive_bus_data_batch(data_list: List[BusArrivalData]):
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")

    processed_count = 0
    error_count = 0

    for d in data_list:
        try:
            bus = d.dict()
            
            # collection_time 파싱
            try:
                bus["collection_time"] = datetime.fromisoformat(bus["collection_time"])
            except Exception as e:
                if SIMULATION_MODE:
                    print(f"⚠ Batch 내 collection_time 파싱 실패 (스킵): {e}")
                    error_count += 1
                    continue
                else:
                    bus["collection_time"] = datetime.now()

            tracker.process_new_data(bus)
            processed_count += 1
            
        except Exception as e:
            print(f"⚠ Batch 처리 오류: {e}")
            error_count += 1

    return {
        "status": "success", 
        "message": f"{processed_count} processed, {error_count} errors"
    }


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
        "mode": "simulation" if SIMULATION_MODE else "realtime",
        "total_buses": len(buses),
        "active_buses": active,
        "predicted_buses": predicted,
        "predictor_loaded": predictor is not None,
        "historical_data_loaded": historical_data is not None,
        "current_time": tracker._get_current_time().isoformat() if tracker else None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    return {
        "message": "Gumi Real-time Bus Tracking API",
        "mode": "simulation" if SIMULATION_MODE else "realtime",
        "docs": "/docs"
    }


# --------------------------------------------------------
# CLI 진입점
# --------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="구미 버스 추적 서버")
    parser.add_argument("--simulation", action="store_true",
                       help="시뮬레이션 모드로 실행 (collection_time 사용)")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 (개발용)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SIMULATION_MODE = args.simulation
    
    print("\n" + "=" * 60)
    if SIMULATION_MODE:
        print("🎬 시뮬레이션 모드로 서버 시작")
        print("   collection_time을 기준으로 동작합니다")
    else:
        print("🔴 실시간 모드로 서버 시작")
        print("   현재 시각을 기준으로 동작합니다")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )