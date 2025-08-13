# ========================= [BLOCK 1] 기본 설정 & 라이브러리 =========================
import math
import requests
import folium
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime, timedelta

st.set_page_config(page_title="배달 예측(메인)", layout="wide")
st.title("🚚 배달 예측 대시보드 — 메인 (Mapbox + 로컬 CSV)")

# ========================= [BLOCK 2] 전역 상수(토큰/파일 경로/컬럼 매핑/색상) =========================
MAPBOX_TOKEN = "mapboxToken"  
LOCAL_CSV_PATH   = "final_merged_df_sample.csv"

# 샘플 CSV 기준 컬럼 매핑
COL = {
    "id": "ID",
    "date": "Order_Date",                   # 주문 날짜
    "order_time": "Time_Orderd",            # 주문 수락 시각(HH:MM 등)
    "pickup_time": "Time_Order_picked",     # 픽업 시각(HH:MM 등)
    "total_min": "Time_taken_min",          # 총 소요(분)
    "prep_min": "prep_time",                # 준비 소요(분)
    "store_lat": "Restaurant_lat_real",
    "store_lng": "Restaurant_lon_real",
    "cust_lat": "Delivery_lat_real",
    "cust_lng": "Delivery_lon_real",
}

CONGESTION_COLOR = {
    "low": "#1DB954",        # 원활(초록)
    "moderate": "#FFA500",   # 서행(주황)
    "heavy": "#FF4D4D",      # 정체(빨강)
    "severe": "#B30000",     # 심한 정체(진빨강)
    "unknown": "#999999"     # 정보없음(회색)
}

# ========================= [BLOCK 3] 데이터 로드/전처리 =========================
@st.cache_data
def load_orders(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 문자열 시간 파싱을 위해 공백/결측 정리
    for c in [COL["date"], COL["order_time"], COL["pickup_time"]]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

orders = load_orders(LOCAL_CSV_PATH)

# ========================= [BLOCK 4] 세션 상태 초기화 =========================
if "route_data" not in st.session_state:
    st.session_state["route_data"] = None
if "start_coords" not in st.session_state:
    st.session_state["start_coords"] = (37.5665, 126.9780)
if "end_coords" not in st.session_state:
    st.session_state["end_coords"] = (37.5700, 126.9920)
if "eta_minutes" not in st.session_state:
    st.session_state["eta_minutes"] = None

# ========================= [BLOCK 5] 보조 함수(시간 파싱/형식/혼잡분류/Mapbox) =========================
def parse_datetime(date_str: str | None, time_str: str | None) -> datetime | None:
    """CSV의 날짜+시간 텍스트를 datetime으로 합쳐 파싱."""
    if not date_str or date_str.lower() == "nan":
        return None
    try:
        d = pd.to_datetime(date_str, errors="coerce", dayfirst=False)  # 'YYYY-MM-DD' or 'DD-MM-YYYY' 모두 허용
    except Exception:
        d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return None
    if not time_str or time_str.lower() == "nan":
        return d.to_pydatetime()
    # 시간 문자열이 'HH:MM' 또는 'H:M:S' 등 다양할 수 있음
    try:
        t = pd.to_datetime(time_str, errors="coerce").time()
        return datetime.combine(d.date(), t)
    except Exception:
        try:
            # 수동 파싱(예: '16:15'만 온 경우)
            h, m = time_str.split(":")[:2]
            return datetime(d.year, d.month, d.day, int(h), int(m))
        except Exception:
            return d.to_pydatetime()

def fmt_kor(dt: datetime | None) -> str:
    if not dt:
        return "-"
    h = dt.hour
    m = dt.minute
    ampm = "오전" if h < 12 else "오후"
    h12 = h if 1 <= h <= 12 else (12 if h % 12 == 0 else h % 12)
    return f"{ampm} {h12}시 {m}분"

def classify_congestion_dynamic(speeds_kmh: pd.Series | None, v_kmh: float | None):
    if v_kmh is None:
        return None
    if speeds_kmh is None or len(speeds_kmh) < 5:
        if v_kmh >= 35: return "low"
        if v_kmh >= 20: return "moderate"
        if v_kmh >= 10: return "heavy"
        return "severe"
    q20, q40, q60 = speeds_kmh.quantile([0.2, 0.4, 0.6])
    if v_kmh < q20: return "severe"
    if v_kmh < q40: return "heavy"
    if v_kmh < q60: return "moderate"
    return "low"

def fetch_route_mapbox_traffic(start_lat, start_lng, end_lat, end_lng, mapbox_token, speed_scale=1.0):
    """현재 시점 교통 기준의 경로 + 세그먼트 혼잡도/속도를 가져와 색상용 세그먼트로 변환."""
    url = (
        f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
        f"{start_lng},{start_lat};{end_lng},{end_lat}"
    )
    params = {
        "alternatives": "false",
        "geometries": "geojson",
        "overview": "full",
        "annotations": "congestion,speed",
        "steps": "false",
        "access_token": mapbox_token,
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()

    route = data["routes"][0]
    distance_m = route["distance"]
    duration_s = route["duration"]

    coords = route["geometry"]["coordinates"]         # [[lon,lat], ...]
    leg = route["legs"][0]
    ann = leg.get("annotation", {}) or {}
    congestion = ann.get("congestion", [])
    speed_ms = ann.get("speed", [])

    n = max(len(coords) - 1, 0)
    speeds_kmh_all = []
    for i in range(n):
        if i < len(speed_ms) and speed_ms[i] is not None:
            speeds_kmh_all.append(float(speed_ms[i]) * 3.6 * speed_scale)
    speeds_kmh_series = pd.Series(speeds_kmh_all) if speeds_kmh_all else None

    segments = []
    for i in range(n):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        cong = congestion[i] if (i < len(congestion) and congestion[i]) else "unknown"
        v = None
        if i < len(speed_ms) and speed_ms[i] is not None:
            v = float(speed_ms[i]) * 3.6 * speed_scale
        if cong == "unknown":
            cong = classify_congestion_dynamic(speeds_kmh_series, v) or "unknown"
        segments.append({"coords": [(lat1, lon1), (lat2, lon2)], "congestion": cong})

    return {"distance_m": distance_m, "duration_s": duration_s, "segments": segments}

def draw_route_segments(m, segments, weight=8, opacity=0.95):
    for seg in segments:
        color = CONGESTION_COLOR.get(seg["congestion"], CONGESTION_COLOR["unknown"])
        folium.PolyLine(seg["coords"], color=color, weight=weight, opacity=opacity).add_to(m)

def add_congestion_legend(m):
    import branca
    html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15); font-size: 13px;">
        <b>교통 혼잡도</b><br>
        <div style="margin-top:6px"><span style="display:inline-block;width:14px;height:10px;background:#1DB954;margin-right:6px;border:1px solid #999"></span> 원활</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#FFA500;margin-right:6px;border:1px solid #999"></span> 서행</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#FF4D4D;margin-right:6px;border:1px solid #999"></span> 정체</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#B30000;margin-right:6px;border:1px solid #999"></span> 심한 정체</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#999999;margin-right:6px;border:1px solid #999"></span> 정보 없음</div>
    </div>"""
    macro = branca.element.MacroElement()
    macro._template = branca.element.Template(html)
    m.get_root().add_child(macro)

# ========================= [BLOCK 6] 상단 컨트롤(주문 선택 + 좌표/경로 갱신) =========================
st.markdown("### 🔧 주문 선택 & 경로")
c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 2.6])

with c1:
    order_ids = orders[COL["id"]].tolist()
    selected_id = st.selectbox("주문 ID", order_ids, index=len(order_ids)-1 if order_ids else 0)

# 선택된 주문의 좌표를 자동 적용
sel = orders[orders[COL["id"]] == selected_id].iloc[0] if len(order_ids) else None
start_lat = float(sel[COL["store_lat"]]) if sel is not None else st.session_state["start_coords"][0]
start_lng = float(sel[COL["store_lng"]]) if sel is not None else st.session_state["start_coords"][1]
end_lat   = float(sel[COL["cust_lat"]])  if sel is not None else st.session_state["end_coords"][0]
end_lng   = float(sel[COL["cust_lng"]])  if sel is not None else st.session_state["end_coords"][1]

with c2:
    start_lat = st.number_input("출발 위도", value=start_lat, format="%.6f")
with c3:
    start_lng = st.number_input("출발 경도", value=start_lng, format="%.6f")
with c4:
    end_lat = st.number_input("도착 위도", value=end_lat, format="%.6f")
with c5:
    end_lng = st.number_input("도착 경도", value=end_lng, format="%.6f")

col_go1, col_go2 = st.columns([1, 6])
with col_go1:
    speed_scale = st.slider("속도 보정", 0.5, 1.2, 1.0, 0.05, help="값을 낮추면 더 혼잡하게 색상 분류됨")
with col_go2:
    if st.button("경로 찾기 / 갱신", use_container_width=True):
        try:
            data = fetch_route_mapbox_traffic(start_lat, start_lng, end_lat, end_lng, MAPBOX_TOKEN, speed_scale=speed_scale)
            st.session_state["route_data"] = data
            st.session_state["start_coords"] = (start_lat, start_lng)
            st.session_state["end_coords"]   = (end_lat, end_lng)
            st.session_state["eta_minutes"]  = math.ceil(data["duration_s"] / 60)
            st.success("경로 갱신 완료 (현재 교통 반영)")
        except Exception as e:
            st.session_state["route_data"] = None
            st.session_state["eta_minutes"] = None
            st.error(f"경로 조회 실패: {e}")

# ========================= [BLOCK 7] 3분할 레이아웃 (지도 / ETA 요약 / 변수중요도) =========================
left_col, mid_col, right_col = st.columns([1.3, 1.0, 1.0])

# ---- 좌: 지도
with left_col:
    st.subheader("지도 / 실시간 교통 혼잡도")
    if st.session_state["route_data"]:
        # 지도 렌더
        start = st.session_state["start_coords"]; end = st.session_state["end_coords"]
        center = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
        folium.Marker(start, tooltip="출발", icon=folium.Icon(color="green", icon="motorcycle", prefix="fa")).add_to(m)
        folium.Marker(end, tooltip="도착", icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(m)
        draw_route_segments(m, st.session_state["route_data"]["segments"], weight=8, opacity=0.95)
        add_congestion_legend(m)
        st_folium(m, width=None, height=520)
        km = st.session_state["route_data"]["distance_m"] / 1000
        dur = st.session_state["route_data"]["duration_s"]
        st.caption(f"거리: {km:.2f} km  |  현재 교통 ETA: {math.ceil(dur/60)}분")
    else:
        st.info("상단에서 경로를 조회하면 혼잡도 색상으로 표시됩니다.")

# ---- 중: ETA 통계 (CSV 기반)
with mid_col:
    st.subheader("배달 시간 통계 (CSV 기반 ETA)")
    if COL["total_min"] in orders.columns:
        eta_series = pd.to_numeric(orders[COL["total_min"]], errors="coerce").dropna()
        st.metric("평균 ETA(분)", f"{eta_series.mean():.1f}")
        cA, cB, cC = st.columns(3)
        cA.metric("중앙값", f"{eta_series.median():.1f}")
        cB.metric("표준편차", f"{eta_series.std():.1f}")
        cC.metric("최대", f"{eta_series.max():.1f}")

        import altair as alt
        hist = (
            alt.Chart(pd.DataFrame({"eta_min": eta_series}))
            .mark_bar()
            .encode(x=alt.X("eta_min:Q", bin=alt.Bin(maxbins=30), title="ETA(분)"), y=alt.Y("count()", title="건수"))
            .properties(height=220)
        )
        st.altair_chart(hist, use_container_width=True)

        qs = eta_series.quantile([0.1,0.25,0.5,0.75,0.9])
        st.dataframe(pd.DataFrame({"quantile": qs.index, "ETA(분)": qs.values}), use_container_width=True)
    else:
        st.info("CSV에 Time_taken_min 컬럼이 필요합니다.")

# ---- 우: 변수 중요도(데모)
with right_col:
    st.subheader("변수 중요도 (데모)")
    rng = np.random.default_rng(0)
    feats = [f"feat_{i}" for i in range(1, 13)]
    vals = rng.random(12); vals = vals/vals.sum()
    fi_top = pd.DataFrame({"feature": feats, "importance": vals}).sort_values("importance", ascending=False).head(10)

    import altair as alt
    bar = (
        alt.Chart(fi_top)
        .mark_bar()
        .encode(x=alt.X("importance:Q", title="중요도"), y=alt.Y("feature:N", sort="-x", title=None), tooltip=["feature","importance"])
        .properties(height=240)
    )
    st.altair_chart(bar, use_container_width=True)
    st.caption("※ 실제 모델 중요도 파일 연결 시 이 영역 교체 예정")

# ========================= [BLOCK 8] 하단 파이프라인 (주문→준비→배달→완료) =========================
st.markdown("---")
st.subheader("주문 파이프라인")

# 현재 선택 주문의 타임라인을 구성
order_dt   = parse_datetime(sel[COL["date"]], sel[COL["order_time"]]) if sel is not None else None
pickup_dt  = parse_datetime(sel[COL["date"]], sel[COL["pickup_time"]]) if sel is not None else None
prep_min   = float(sel[COL["prep_min"]]) if sel is not None and pd.notna(sel[COL["prep_min"]]) else np.nan
total_min  = float(sel[COL["total_min"]]) if sel is not None and pd.notna(sel[COL["total_min"]]) else np.nan

# 누락된 값 보간
if (pickup_dt is None) and (order_dt is not None) and (not np.isnan(prep_min)):
    pickup_dt = order_dt + timedelta(minutes=prep_min)
delivered_dt = None
if (order_dt is not None) and (not np.isnan(total_min)):
    delivered_dt = order_dt + timedelta(minutes=total_min)

# 표기용 문자열
now_str = datetime.now().strftime("%H:%M")
ot_str  = fmt_kor(order_dt)
prep_str= f"약 {int(round(prep_min))}분 소요" if not np.isnan(prep_min) else "-"
pk_str  = fmt_kor(pickup_dt)
dl_str  = f"약 {int(round(total_min - (prep_min if not np.isnan(prep_min) else 0)))}분 소요" \
            if (not np.isnan(total_min)) else "-"
dv_str  = fmt_kor(delivered_dt)

# 스타일 타임라인 (네가 준 디자인 느낌)
timeline_html = f"""
<style>
.step-wrap{{display:flex;align-items:center;gap:48px;margin-top:10px;margin-bottom:10px}}
.step{{text-align:center}}
.badge{{width:82px;height:82px;border-radius:50%;background:#E07A18;color:white;
        display:flex;align-items:center;justify-content:center;font-weight:800;font-size:28px;
        box-shadow:inset -6px -6px 0 rgba(0,0,0,0.15)}}
.badge-empty{{width:82px;height:82px;border-radius:50%;border:10px solid #5A615D;background:#fff}}
.line{{height:10px;background:#5A615D;flex:1}}
.step-title{{font-size:24px;font-weight:700;margin-bottom:6px}}
.step-sub{{color:#8B8F90;font-size:20px}}
.big-clock{{font-size:64px;color:#5A754D;font-weight:900;margin:0}}
</style>

<div style="display:flex;justify-content:space-between;align-items:center;">
  <div class="step">
    <div class="step-title">현재 시각</div>
    <div class="big-clock">{now_str}</div>
  </div>
  <div class="step-wrap" style="flex:1;margin-left:24px;margin-right:24px;">
    <div class="step">
      <div class="step-title">주문 수락됨</div>
      <div class="badge">✓</div>
      <div class="step-sub">{ot_str}</div>
    </div>
    <div class="line"></div>
    <div class="step">
      <div class="step-title">메뉴 준비중</div>
      <div class="badge-empty"></div>
      <div class="step-sub">{prep_str}</div>
    </div>
    <div class="line"></div>
    <div class="step">
      <div class="step-title">배달중</div>
      <div class="badge-empty"></div>
      <div class="step-sub">{pk_str}</div>
    </div>
    <div class="line"></div>
    <div class="step">
      <div class="step-title">배달 완료</div>
      <div class="badge-empty"></div>
      <div class="step-sub">{dv_str}</div>
    </div>
  </div>
</div>
"""
st.markdown(timeline_html, unsafe_allow_html=True)

# ========================= [BLOCK 9] 참고: 완료 데이터로 과거 경로 재현 가능? =========================
st.caption("ℹ️ 이 지도 경로/혼잡도는 '현재' 교통 기준입니다. 과거 주문의 당시 경로/혼잡은 지도 API만으로는 복원되지 않습니다.")
