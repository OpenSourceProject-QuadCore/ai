from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import asyncio
from datetime import datetime
import uvicorn
import argparse
import os

from bus_tracker import BusTracker
from bus_predictor import BusArrivalPredictor
from data_preprocessing import BusDataPreprocessor

# --------------------------------------------------------
# 전역 변수
# --------------------------------------------------------
app = FastAPI(title="구미 버스 실시간 추적 API (하이브리드 모드)")
tracker: Optional[BusTracker] = None
predictor: Optional[BusArrivalPredictor] = None
historical_data: Optional[pd.DataFrame] = None
SIMULATION_MODE = False


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
    collection_time: str
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
    mode: str
    last_update: str


# --------------------------------------------------------
# 서버 시작 시 초기화 (수정됨!)
# --------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global tracker, predictor, historical_data

    print("=" * 60)
    print("=== 서버 초기화 시작 (하이브리드 모드) ===")
    print(f"모드: {'시뮬레이션' if SIMULATION_MODE else '실시간'}")
    print("=" * 60)

    # 모델 로드
    try:
        predictor = BusArrivalPredictor()
        predictor.load("models/bus_predictor.pkl")
        print("✓ 예측 모델 로드 성공")
    except Exception as e:
        print(f"⚠ 모델 로드 실패: {e}")
        print("  예측 기능 없이 실행됩니다.")
        predictor = None

    # 과거 데이터 로드
    processed_path = "data/processed_bus_arrivals.csv"
    
    if os.path.exists(processed_path):
        try:
            historical_data = pd.read_csv(processed_path)
            print(f"✓ 전처리된 데이터 로드: {processed_path}")
            print(f"  → {len(historical_data):,} rows")
            
            if 'hour' not in historical_data.columns:
                print(f"  ⚠ hour 컬럼 없음 → 생성 중...")
                if 'collection_time' in historical_data.columns:
                    historical_data['hour'] = pd.to_datetime(
                        historical_data['collection_time']
                    ).dt.hour
                    print(f"  ✓ hour 컬럼 생성 완료")
                else:
                    print(f"  ⚠ collection_time 없음 → hour 조건 비활성화")
            
        except Exception as e:
            print(f"⚠ 전처리 데이터 로드 실패: {e}")
            historical_data = None
    
    else:
        print(f"⚠ 전처리 데이터 없음: {processed_path}")
        print(f"  → Historical pattern fallback 비활성화")
        print(f"  → 먼저 train_model.py를 실행하세요")  # ★ 수정됨
        historical_data = None

    # BusTracker 초기화
    tracker = BusTracker(
        predictor=predictor, 
        historical_data=historical_data,
        simulation_mode=SIMULATION_MODE,
        api_timeout_seconds=100
    )
    print("✓ BusTracker 초기화 완료 (하이브리드 모드)")

    # 백그라운드 작업
    asyncio.create_task(background_cleanup_loop())
    print("✓ 백그라운드 정리 작업 시작 (60초 주기)")

    print("=" * 60)
    print("=== 초기화 완료 ===")
    print("=" * 60)


# --------------------------------------------------------
# 백그라운드 정리 작업
# --------------------------------------------------------
async def background_cleanup_loop():
    """주기적으로 버스 정리 작업 수행 (60초마다)"""
    await asyncio.sleep(5)
    
    cycle = 0
    
    while True:
        if tracker is None:
            await asyncio.sleep(5)
            continue

        try:
            cycle += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if cycle % 5 == 0:
                print(f"\n{'='*60}")
                print(f"[{current_time}] 정리 작업 #{cycle} (하이브리드 모드)")
                print(f"{'='*60}")
            
            tracker.cleanup()
            
            if cycle % 5 == 0:
                stats = tracker.get_stats()
                print(f"통계:")
                print(f"  추적 중: {stats['total_buses']}대 "
                      f"(API: {stats['api_buses']}, ML: {stats['ml_buses']})")
                print(f"  총 예측: {stats['total_predictions']}회")
                print(f"  API→ML 전환: {stats['api_to_ml_transitions']}회")
                print(f"  버스당 평균: {stats['avg_predictions_per_bus']:.1f}회")
                print(f"{'='*60}\n")

        except Exception as e:
            print(f"\n❌ [{datetime.now().strftime('%H:%M:%S')}] 백그라운드 작업 오류: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(60)


# --------------------------------------------------------
# POST: 1개 버스 데이터 수신
# --------------------------------------------------------
@app.post("/api/bus-arrival")
async def receive_bus_data(data: BusArrivalData):
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")

    bus = data.dict()

    try:
        bus["collection_time"] = datetime.fromisoformat(bus["collection_time"])
    except Exception as e:
        if SIMULATION_MODE:
            raise HTTPException(400, f"collection_time 파싱 실패: {e}")
        else:
            bus["collection_time"] = datetime.now()

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
            
            try:
                bus["collection_time"] = datetime.fromisoformat(bus["collection_time"])
            except Exception as e:
                if SIMULATION_MODE:
                    error_count += 1
                    continue
                else:
                    bus["collection_time"] = datetime.now()

            tracker.process_new_data(bus)
            processed_count += 1
            
        except Exception as e:
            print(f"⚠ Batch 처리 오류: {e}")
            error_count += 1

    message_parts = [f"{processed_count} processed"]
    if error_count > 0:
        message_parts.append(f"{error_count} errors")
    
    return {
        "status": "success", 
        "message": ", ".join(message_parts),
        "processed": processed_count,
        "errors": error_count
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

    stats = tracker.get_stats()

    return {
        "status": "running",
        "mode": "하이브리드 모드 (API 우선 + ML 백업)",
        "simulation": SIMULATION_MODE,
        "total_buses": stats['total_buses'],
        "api_buses": stats['api_buses'],
        "ml_buses": stats['ml_buses'],
        "total_predictions": stats['total_predictions'],
        "api_to_ml_transitions": stats['api_to_ml_transitions'],
        "avg_predictions_per_bus": stats['avg_predictions_per_bus'],
        "predictor_loaded": predictor is not None,
        "historical_data_loaded": historical_data is not None,
        "current_time": tracker._get_current_time().isoformat() if tracker else None,
        "timestamp": datetime.now().isoformat()
    }


# --------------------------------------------------------
# GET: 통계 정보
# --------------------------------------------------------
@app.get("/api/stats")
async def get_stats():
    if tracker is None:
        raise HTTPException(500, "트래커 미초기화")
    
    buses = tracker.get_all_buses()
    stats = tracker.get_stats()
    
    routes = {}
    for bus in buses:
        rid = bus["routeid"]
        if rid not in routes:
            routes[rid] = {"api": 0, "ml": 0, "total": 0}
        routes[rid]["total"] += 1
        if bus["mode"] == "api":
            routes[rid]["api"] += 1
        else:
            routes[rid]["ml"] += 1
    
    stations = {}
    for bus in buses:
        nid = bus["nodeid"]
        if nid not in stations:
            stations[nid] = {"count": 0, "name": bus["nodenm"]}
        stations[nid]["count"] += 1
    
    return {
        "total_buses": stats['total_buses'],
        "api_buses": stats['api_buses'],
        "ml_buses": stats['ml_buses'],
        "total_predictions": stats['total_predictions'],
        "api_to_ml_transitions": stats['api_to_ml_transitions'],
        "avg_predictions_per_bus": stats['avg_predictions_per_bus'],
        "buses_tracked": stats['buses_tracked'],
        "buses_arrived": stats['buses_arrived'],
        "buses_disappeared": stats['buses_disappeared'],
        "routes": routes,
        "stations": stations,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    return {
        "message": "Gumi Bus Tracking API - 하이브리드 모드",
        "description": "API 우선 사용, 끊기면 ML 백업 (재예측 없음)",
        "mode": "하이브리드",
        "simulation": SIMULATION_MODE,
        "version": "6.1-fixed",
        "docs": "/docs",
        "endpoints": {
            "status": "/api/status",
            "stats": "/api/stats",
            "all_buses": "/api/buses",
            "by_route": "/api/buses/route/{route_id}",
            "by_station": "/api/buses/station/{node_id}"
        }
    }


# --------------------------------------------------------
# CLI 진입점
# --------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="구미 버스 추적 서버 (하이브리드 모드)")
    parser.add_argument("--simulation", action="store_true",
                       help="시뮬레이션 모드로 실행")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SIMULATION_MODE = args.simulation
    
    print("\n" + "=" * 60)
    if SIMULATION_MODE:
        print("🎬 시뮬레이션 모드로 서버 시작")
    else:
        print("🔴 실시간 모드로 서버 시작")
    print("🔄 하이브리드 모드: API 우선 + ML 백업")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )