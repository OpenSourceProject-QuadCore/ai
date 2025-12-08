import requests
import csv
import time
import os
import threading
from datetime import datetime, timedelta, timezone

SERVER_URL = "http://localhost:8000/api/bus-arrival/batch"

# ==============================
# 1) 사용자 설정
# ==============================
SERVICE_KEY = "YOUR_CODE"
KMA_API_KEY = "YOUR_CODE"

CITY_CODE = "37050"
LOOP_INTERVAL_SECONDS = 60

# 구미 기준 기상청 격자 좌표
NX, NY = 86, 96

NODE_ID_LIST = ['GMB130', 'GMB131', 'GMB132']

CSV_ARRIVALS_FILE = "bus_arrivals.csv"
CSV_ARRIVALS_HEADER = [
    'collection_time', 'weekday', 'time_slot',
    'weather', 'temp', 'humidity', 'rain_mm', 'snow_mm',
    'nodeid', 'nodenm', 'routeid',
    'routeno', 'routetp', 'arrprevstationcnt', 'arrtime', 'vehicletp'
]

def now_kst():
    return datetime.now(timezone.utc) + timedelta(hours=9)

# =========================
# 2) CSV 저장
# =========================
def append_to_csv(data_list, file_name, header, extra_info):
    if not data_list:
        return
    
    file_exists = os.path.isfile(file_name)
    try:
        with open(file_name, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
            if not file_exists:
                w.writeheader()

            now = now_kst().isoformat()
            for row in data_list:
                row['collection_time'] = now
                row.update(extra_info)
                w.writerow(row)

        print(f"  -> {file_name}에 {len(data_list)}개 항목 저장 완료.")
    except Exception as e:
        print(f"[CSV ERROR] {e}")

def send_to_server(data_list, extra_info):
    """서버로 데이터 전송 (API 버그 자동 보정 포함)"""
    if not data_list:
        return
    
    try:
        server_data = []
        now = now_kst().isoformat()
        skipped = 0
        invalid_arrtime_count = 0
        auto_corrected_count = 0  # 자동 보정 카운터
        
        for row in data_list:
            # ============================================================
            # ★★★ STEP 1: arrtime 안전하게 파싱 ★★★
            # ============================================================
            raw_arrtime = row.get('arrtime')
            
            # 1-1. arrtime 없으면 스킵
            if raw_arrtime is None or raw_arrtime == '':
                skipped += 1
                continue
            
            # 1-2. 안전한 정수 변환
            try:
                # 문자열이면 숫자만 추출
                if isinstance(raw_arrtime, str):
                    # "1234초", "1,234" → "1234"
                    cleaned = ''.join(c for c in raw_arrtime if c.isdigit() or c == '-')
                    if not cleaned:
                        print(f"  ⚠️  arrtime 파싱 실패: '{raw_arrtime}' "
                              f"(노선: {row.get('routeno')}, 정류장: {row.get('nodenm')})")
                        invalid_arrtime_count += 1
                        continue
                    arrtime = int(cleaned)
                else:
                    # 숫자면 그냥 int 변환
                    arrtime = int(raw_arrtime)
                
            except (ValueError, TypeError) as e:
                print(f"  ⚠️  arrtime 변환 실패: '{raw_arrtime}' → {e} "
                      f"(노선: {row.get('routeno')}, 정류장: {row.get('nodenm')})")
                invalid_arrtime_count += 1
                continue
            
            # ============================================================
            # ★★★ STEP 2: arrprevstationcnt 파싱 ★★★
            # ============================================================
            try:
                arrprevstationcnt = int(row.get('arrprevstationcnt', 0))
            except (ValueError, TypeError):
                arrprevstationcnt = 0
            
            # ============================================================
            # ★★★ STEP 3: API 버그 자동 감지 및 보정 ★★★
            # ============================================================
            original_arrtime = arrtime  # 보정 전 값 저장
            
            # 정류장 수가 5개 이상이고, arrtime이 있을 때만 검증
            if arrprevstationcnt >= 5 and arrtime > 0:
                sec_per_station = arrtime / arrprevstationcnt
                
                # ============================================================
                # ★ 휴리스틱: 정류장당 10초 미만이면 마지막 자리 누락 의심!
                # ============================================================
                if sec_per_station < 10:
                    # 10배 보정 시도
                    corrected_arrtime = arrtime * 10
                    corrected_sec_per_station = corrected_arrtime / arrprevstationcnt
                    
                    # 보정 후 10~600초 범위면 적용
                    if 10 <= corrected_sec_per_station <= 600:
                        print(f"  🔧 [자동 보정] 노선 {row.get('routeno')}, "
                              f"정류장 {arrprevstationcnt}개: "
                              f"{arrtime}초 → {corrected_arrtime}초 "
                              f"(정류장당 {sec_per_station:.1f}초 → {corrected_sec_per_station:.1f}초)")
                        arrtime = corrected_arrtime
                        auto_corrected_count += 1
                    else:
                        # 10배 해도 이상하면 경고만
                        print(f"  ⚠️  의심스러운 arrtime: 노선 {row.get('routeno')}, "
                              f"정류장 {arrprevstationcnt}개, "
                              f"arrtime {arrtime}초 (정류장당 {sec_per_station:.1f}초)")
            
            # ============================================================
            # ★★★ STEP 4: arrtime 검증 ★★★
            # ============================================================
            # 4-1. 범위 체크 (0-7200초 = 0-2시간)
            if arrtime < 0:
                print(f"  ⚠️  음수 arrtime 감지: {arrtime}초 → 스킵 "
                      f"(노선: {row.get('routeno')}, 정류장: {row.get('nodenm')})")
                invalid_arrtime_count += 1
                continue
            
            if arrtime > 7200:
                print(f"  ⚠️  비현실적 arrtime: {arrtime}초 ({arrtime/60:.1f}분) → 스킵 "
                      f"(노선: {row.get('routeno')}, 정류장: {row.get('nodenm')})")
                invalid_arrtime_count += 1
                continue
            
            # ============================================================
            # ★★★ STEP 5: 데이터 준비 ★★★
            # ============================================================
            row_copy = row.copy()
            row_copy['collection_time'] = now
            row_copy.update(extra_info)
            
            # 날씨 문자열
            row_copy['weather'] = str(row_copy.get('weather') or 'Unknown')
            
            # 숫자 필드 (None 안전 처리)
            row_copy['temp'] = float(row_copy.get('temp') if row_copy.get('temp') is not None else 15.0)
            row_copy['humidity'] = float(row_copy.get('humidity') if row_copy.get('humidity') is not None else 50.0)
            row_copy['rain_mm'] = float(row_copy.get('rain_mm') if row_copy.get('rain_mm') is not None else 0.0)
            row_copy['snow_mm'] = float(row_copy.get('snow_mm') if row_copy.get('snow_mm') is not None else 0.0)
            
            # 문자열 필드
            for field in ['routeid', 'routeno', 'nodeid', 'nodenm', 
                         'vehicletp', 'routetp', 'weekday', 'time_slot']:
                row_copy[field] = str(row_copy.get(field) or '')
            
            # 정수 필드
            row_copy['arrprevstationcnt'] = arrprevstationcnt
            row_copy['arrtime'] = arrtime  # ★ 보정된 값 사용!
            
            server_data.append(row_copy)
        
        # ============================================================
        # ★★★ STEP 6: 전송 및 통계 ★★★
        # ============================================================
        if not server_data:
            msg = f"  ⚠️  전송할 데이터 없음"
            if skipped > 0:
                msg += f" (arrtime 누락: {skipped}개)"
            if invalid_arrtime_count > 0:
                msg += f" (arrtime 오류: {invalid_arrtime_count}개)"
            print(msg)
            return
        
        # 전송
        response = requests.post(
            SERVER_URL,
            json=server_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            msg = f"  ✓ 서버 전송: {len(server_data)}개"
            
            # 통계 추가
            stats_parts = []
            if auto_corrected_count > 0:
                stats_parts.append(f"자동 보정 {auto_corrected_count}개")
            if skipped > 0:
                stats_parts.append(f"누락 {skipped}개")
            if invalid_arrtime_count > 0:
                stats_parts.append(f"오류 {invalid_arrtime_count}개")
            
            if stats_parts:
                msg += f" ({', '.join(stats_parts)})"
            
            msg += f" → {result.get('message', 'OK')}"
            print(msg)
        elif response.status_code == 422:
            print(f"  ✗ 서버: 422 Validation Error")
            print(f"    응답: {response.text[:500]}")
            # 디버그: 첫 번째 아이템 출력
            if server_data:
                import json
                print(f"    첫 번째 데이터:")
                print(json.dumps(server_data[0], indent=2, ensure_ascii=False))
        else:
            print(f"  ✗ 서버: {response.status_code}")
            print(f"    응답: {response.text[:200]}")
            
    except requests.exceptions.ConnectionError:
        print(f"  ✗ 서버 연결 실패")
    except Exception as e:
        print(f"  ✗ 전송 오류: {e}")
        import traceback
        traceback.print_exc()

# =========================
# 3) 시간/요일 태그
# =========================
def get_weekday():
    return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][now_kst().weekday()]

def get_time_slot():
    now = now_kst()
    h = now.hour
    weekday = now.weekday()  # Monday = 0 ... Sunday = 6

    is_weekend = weekday >= 5  # 토(5), 일(6)

    # 새벽
    if 5 <= h < 7:
        return "dawn"

    # 출근 시간대
    if 7 <= h < 9:
        return "weekend_morning" if is_weekend else "rushhour"
    
    if 9 <= h <12:
        return "morning"

    # 점심~오후
    if 12 <= h < 17:
        return "afternoon"

    # 퇴근 시간대
    if 17 <= h < 19:
        return "weekend_evening" if is_weekend else "rushhour"

    # 저녁
    if 19 <= h < 21:
        return "evening"

    # 그 외 시간
    return "night"


# =========================
# 4) 기상청 API
# =========================

# ---- 초단기실황: 현재 실제 값 ----
def get_kma_current():
    try:
        now = now_kst()
        base = now.replace(minute=0, second=0, microsecond=0)
        if now.minute < 40: base -= timedelta(hours=1)

        params = {
            "serviceKey": KMA_API_KEY,
            "pageNo": "1",
            "numOfRows": "100",
            "dataType": "JSON",
            "base_date": base.strftime("%Y%m%d"),
            "base_time": base.strftime("%H%M"),
            "nx": NX, "ny": NY
        }

        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
        res = requests.get(url, params=params, timeout=15)
        print("📡 [DEBUG] CURRENT URL:", res.url)
        print("📄 [DEBUG] CURRENT RAW:", res.text[:200])
        js = res.json()

        items = js['response']['body']['items']['item']

        temp = hum = rain = snow = None
        weather = None

        for i in items:
            cat = i['category']
            val = float(i.get('obsrValue', 0))

            if cat == "T1H": temp = val
            elif cat == "REH": hum = val
            elif cat == "RN1": rain = val
            elif cat == "SNO": snow = val
            elif cat == "PTY":
                m = {0:"Clear",1:"Rain",2:"Rain/Snow",3:"Snow",5:"Drizzle"}
                weather = m.get(int(val), "Unknown")

        return weather, temp, hum, rain, snow

    except Exception as e:
        print("[KMA_NOW ERROR]", e)
        return None, None, None, None, None


# ---- 단기예보: 예측 값 ----
def get_kma_forecast():
    try:
        now = now_kst()
        base = now.replace(minute=0, second=0, microsecond=0)
        if now.minute < 40: base -= timedelta(hours=1)

        params = {
            "serviceKey": KMA_API_KEY,
            "pageNo": "1",
            "numOfRows": "100",
            "dataType": "JSON",
            "base_date": base.strftime("%Y%m%d"),
            "base_time": base.strftime("%H00"),
            "nx": NX, "ny": NY
        }

        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        res = requests.get(url, params=params, timeout=15)
        print("📡 [DEBUG] FORECAST URL:", res.url)
        print("📄 [DEBUG] FORECAST RAW:", res.text[:200])
        js = res.json()

        items = js['response']['body']['items']['item']

        temp = hum = rain = snow = None
        weather = None

        for item in items:
            cat = item['category']
            val = item.get('fcstValue')

            if cat == "TMP": temp = float(val)
            elif cat == "REH": hum = float(val)
            elif cat == "PTY":
                m = {0:"Clear",1:"Rain",2:"Rain/Snow",3:"Snow",5:"Drizzle"}
                weather = m.get(int(val), "Unknown")
            elif cat == "PCP":
                # ============================================================
                # ★★★ 다양한 형식 지원! ★★★
                # ============================================================
                if val == "강수없음":
                    rain = 0.0
                elif "mm" in val:
                    # "1.0mm", "1.0mm 미만", "30.0 mm" 모두 지원
                    num_str = val.split("mm")[0].strip()
                    try:
                        rain = float(num_str)
                    except ValueError:
                        # "미만" 등의 경우 → None 처리 (현재값 사용)
                        rain = None
                else:
                    rain = None
            elif cat == "SNO":
                snow = 0 if val == "적설없음" else float(val.replace("cm", "")) * 10.0

        return weather, temp, hum, rain, snow

    except Exception as e:
        print("[KMA_FC ERROR]", e)
        return None, None, None, None, None


# ---- 결합 + 캐싱 ----
class WeatherCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_update = 0
        self._cached_data = (None, None, None, None, None)
    
    def get(self):
        now = time.time()
        
        # 캐시 유효성 체크 (락 없이)
        if now - self._last_update < 120:
            return self._cached_data
        
        # 캐시 갱신 (락 사용)
        with self._lock:
            # Double-checked locking
            if now - self._last_update < 120:
                return self._cached_data
            
            # 날씨 데이터 가져오기
            nw = get_kma_current()
            fc = get_kma_forecast()
            
            weather = nw[0] or fc[0]
            temp = nw[1] if nw[1] is not None else fc[1]
            hum  = nw[2] if nw[2] is not None else fc[2]
            rain = nw[3] if nw[3] is not None else fc[3]
            snow = nw[4] if nw[4] is not None else fc[4]
            
            self._cached_data = (weather, temp, hum, rain, snow)
            self._last_update = now
            
            return self._cached_data

# 전역 인스턴스
weather_cache = WeatherCache()

def get_kma_weather_combined():
    """날씨 정보 조회 (캐싱 적용)"""
    return weather_cache.get()


# =========================
# 5) 버스 API
# =========================
def get_json_response(url, params):
    try:
        r = requests.get(url, params=params, timeout=10).json()
        hdr = r['response']['header']
        if hdr['resultCode'] != '00':
            return []
        items = r['response']['body']['items']
        item = items.get('item')
        if isinstance(item, dict): return [item]
        return item or []
    except:
        return []

def get_arrival_data(node_id):
    url = "http://apis.data.go.kr/1613000/ArvlInfoInqireService/getSttnAcctoArvlPrearngeInfoList"
    params = {
        "serviceKey": SERVICE_KEY,
        "cityCode": CITY_CODE,
        "nodeId": node_id,
        "_type": "json", "numOfRows": "100", "pageNo": "1"
    }
    return get_json_response(url, params)

# =========================
# 6) 메인 루프
# =========================
def main():
    print("=" * 60)
    print("=== 구미 버스 + 기상청 통합 수집 시작 ===")
    print("=== arrtime 자동 보정 기능 활성화 ===")
    print("=" * 60)

    while True:
        start = time.time()

        wk  = get_weekday()
        ts  = get_time_slot()
        weather, temp, hum, rain, snow = get_kma_weather_combined()

        extra = {
            'weekday': wk,
            'time_slot': ts,
            'weather': weather,
            'temp': temp,
            'humidity': hum,
            'rain_mm': rain,
            'snow_mm': snow
        }

        print(f"\n[{now_kst()}] 날씨 {extra}")

        all_data = []
        for node in NODE_ID_LIST:
            data = get_arrival_data(node)
            if data:
                all_data.extend(data)
            time.sleep(0.2)

        append_to_csv(all_data, CSV_ARRIVALS_FILE, CSV_ARRIVALS_HEADER, extra)

        send_to_server(all_data, extra)

        elapsed = time.time() - start
        wait = max(0, LOOP_INTERVAL_SECONDS - elapsed)
        print(f"수집 완료 (소요 {elapsed:.2f}s) / {wait:.1f}s 대기\n")
        time.sleep(wait)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ 수집 종료")