import pandas as pd
import asyncio
import aiohttp
from datetime import datetime
import os
from typing import List, Dict
import time


class BusDataCollector:
    """
    버스 데이터 실시간 수집 → FastAPI 전송
    CSV 파일을 tailing 하며 새로운 row만 API로 보냄
    """

    def __init__(self, csv_path: str = "bus_arrivals.csv",
                 api_url: str = "http://localhost:8000"):
        self.csv_path = csv_path
        self.api_url = api_url
        self.last_position = 0            # 마지막으로 읽은 CSV row index
        self.last_mtime = None            # 수정 시간
        

    # ------------------------------------------------------
    # CSV → 새 rows 읽기
    # ------------------------------------------------------
    async def read_new_data(self) -> List[Dict]:
        if not os.path.exists(self.csv_path):
            print(f"CSV 파일 없음: {self.csv_path}")
            return []

        try:
            mtime = os.path.getmtime(self.csv_path)

            # 파일이 변경되지 않았다면 읽지 않음
            if self.last_mtime is not None and mtime == self.last_mtime:
                return []

            self.last_mtime = mtime

            # 전체 CSV 읽기
            df = pd.read_csv(self.csv_path)

            # 새 데이터만 추출
            if self.last_position < len(df):
                new_df = df.iloc[self.last_position:].copy()
                self.last_position = len(df)
            else:
                return []

            # ======== 중요! 서버와 일관성 있게 JSON 직렬화 적용 ========
            clean_rows = []
            for _, row in new_df.iterrows():
                r = row.copy()

                # Timestamp 직렬화 (collection_time)
                try:
                    r['collection_time'] = pd.to_datetime(r['collection_time']).isoformat()
                except Exception:
                    r['collection_time'] = datetime.now().isoformat()

                # NaN → None (JSON serializable)
                r = r.where(pd.notnull(r), None)

                clean_rows.append(r.to_dict())

            return clean_rows

        except Exception as e:
            print(f"데이터 읽기 오류: {e}")
            return []


    # ------------------------------------------------------
    # API로 전송
    # ------------------------------------------------------
    async def send_to_api(self, data: List[Dict]):
        if not data:
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/bus-arrival/batch",
                    json=data
                ) as resp:

                    if resp.status == 200:
                        result = await resp.json()
                        print(f"✓ {len(data)}건 전송: {result.get('message')}")
                    else:
                        print(f"✗ API 상태코드 오류: {resp.status}")

        except aiohttp.ClientError as e:
            print(f"✗ API 연결 오류: {e}")

        except Exception as e:
            print(f"✗ 전송 오류: {e}")


    # ------------------------------------------------------
    # 메인 루프 (실시간)
    # ------------------------------------------------------
    async def run(self, interval: int = 60):
        print(f"=== 실시간 데이터 수집 시작 (interval={interval}s) ===")
        print(f"CSV: {self.csv_path}")
        print(f"API: {self.api_url}")

        while True:
            try:
                new_data = await self.read_new_data()

                if new_data:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 새 데이터: {len(new_data)}개")
                    await self.send_to_api(new_data)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 새 데이터 없음")

            except Exception as e:
                print(f"실시간 루프 오류: {e}")

            await asyncio.sleep(interval)


# ====================================================================
# 시뮬레이션 모드 (collection_time 기준 시간 흐름 재현)
# ====================================================================
async def simulate_realtime_data(csv_path="data/bus_arrivals.csv",
                                 api_url="http://localhost:8000",
                                 speed_multiplier=60):
    if not os.path.exists(csv_path):
        print(f"CSV 없음: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df['collection_time'] = pd.to_datetime(df['collection_time'])
    df = df.sort_values('collection_time')

    print(f"=== 시뮬레이션 시작 ({speed_multiplier}x) ===")
    print(f"총 {len(df)} rows")

    start_time = df['collection_time'].iloc[0]
    idx = 0

    async with aiohttp.ClientSession() as session:
        while idx < len(df):

            cur_time = df['collection_time'].iloc[idx]
            batch = []

            # 같은 timestamp 묶음 처리
            while idx < len(df) and df['collection_time'].iloc[idx] == cur_time:
                row = df.iloc[idx].copy()

                # Timestamp 직렬화
                row['collection_time'] = row['collection_time'].isoformat()

                # NaN -> None
                row = row.where(pd.notnull(row), None)

                batch.append(row.to_dict())
                idx += 1

            # API 전송
            try:
                async with session.post(
                    f"{api_url}/api/bus-arrival/batch",
                    json=batch
                ) as resp:
                    if resp.status == 200:
                        print(f"[{cur_time}] {len(batch)}건 전송 완료")
                    else:
                        print(f"[{cur_time}] API 오류: {resp.status}")
            except Exception as e:
                print(f"전송 오류: {e}")

            # 다음 timestamp까지 대기
            if idx < len(df):
                next_time = df['collection_time'].iloc[idx]
                wait = (next_time - cur_time).total_seconds() / speed_multiplier
                await asyncio.sleep(max(0.05, wait))


# ====================================================================
# CLI
# ====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="버스 데이터 수집기")
    parser.add_argument("--csv", default="bus_arrivals.csv")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--speed", type=int, default=60)

    args = parser.parse_args()

    if args.simulate:
        asyncio.run(simulate_realtime_data(
            csv_path=args.csv,
            api_url=args.api,
            speed_multiplier=args.speed
        ))
    else:
        collector = BusDataCollector(args.csv, args.api)
        asyncio.run(collector.run(args.interval))
