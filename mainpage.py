# app_main_local_density_route.py
# 단일 주문 선택 → CSV Road_traffic_density 색상으로 "실제 도로 경로" 표시 (Mapbox driving, no traffic)
# 업로드 UI 없음, 로컬 CSV 사용, 하단 파이프라인 유지

# ========================= [BLOCK 1] 기본 설정 & 라이브러리 =========================
import folium
import requests
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import base64
from pathlib import Path

st.set_page_config(page_title="배달 예측(실제경로 + CSV 색)", layout="wide")
st.title("🚚 배달 예측 대시보드")

# ========================= [BLOCK 2] 전역 상수(토큰/파일/컬럼/색상) =========================
MAPBOX_TOKEN   = ""
LOCAL_CSV_PATH = "final_merged_df_sample.csv"         # ← CSV 경로

COL = {
    "id": "ID",
    "date": "Order_Date",
    "order_time": "Time_Orderd",
    "pickup_time": "Time_Order_picked",
    "total_min": "Time_taken_min",
    "prep_min": "prep_time",
    "store_lat": "Restaurant_lat_real",
    "store_lng": "Restaurant_lon_real",
    "cust_lat": "Delivery_lat_real",
    "cust_lng": "Delivery_lon_real",
    "traffic": "Road_traffic_density",  # Low / Medium / High / Jam
}

ROAD_TRAFFIC_COLOR = {
    "low": "#1DB954",      # 초록
    "medium": "#FFA500",   # 주황
    "high": "#FF4D4D",     # 빨강
    "jam": "#B30000",      # 진빨강
    "unknown": "#999999",  # 회색
}

# ========================= [BLOCK 3] 데이터 로드 =========================
@st.cache_data
def load_orders(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in [COL["date"], COL["order_time"], COL["pickup_time"], COL["traffic"]]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in [COL["store_lat"], COL["store_lng"], COL["cust_lat"], COL["cust_lng"], COL["total_min"], COL["prep_min"]]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

orders = load_orders(LOCAL_CSV_PATH)

# ========================= [BLOCK 4] 보조 함수(시간/색/경로 API) =========================
def parse_datetime(date_str: str | None, time_str: str | None) -> datetime | None:
    if not date_str or date_str.lower() == "nan":
        return None
    d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return None
    if not time_str or time_str.lower() == "nan":
        return d.to_pydatetime()
    try:
        t = pd.to_datetime(time_str, errors="coerce").time()
        return datetime.combine(d.date(), t)
    except Exception:
        try:
            h, m = time_str.split(":")[:2]
            return datetime(d.year, d.month, d.day, int(h), int(m))
        except Exception:
            return d.to_pydatetime()

def fmt_kor(dt: datetime | None) -> str:
    if not dt:
        return "-"
    h = dt.hour; m = dt.minute
    ampm = "오전" if h < 12 else "오후"
    h12 = h if 1 <= h <= 12 else (12 if h % 12 == 0 else h % 12)
    return f"{ampm} {h12}시 {m}분"

def traffic_to_color(val: str | None) -> str:
    key = "unknown"
    if isinstance(val, str) and val.strip():
        key = val.strip().lower()
    return ROAD_TRAFFIC_COLOR.get(key, ROAD_TRAFFIC_COLOR["unknown"])

@st.cache_data(show_spinner=False)
def fetch_route_mapbox_geometry(start_lat, start_lng, end_lat, end_lng, token: str):
    """
    실시간 교통 미사용: 'driving' 프로필로 도로를 따르는 라인 좌표만 받음.
    반환: {"distance_m": float, "duration_s": float, "coords_latlon": [(lat,lon), ...]}
    """
    url = (
        f"https://api.mapbox.com/directions/v5/mapbox/driving/"
        f"{start_lng},{start_lat};{end_lng},{end_lat}"
    )
    params = {
        "alternatives": "false",
        "geometries": "geojson",
        "overview": "full",
        "steps": "false",
        "access_token": token,
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()

    route = data["routes"][0]
    distance_m = route.get("distance", None)
    duration_s = route.get("duration", None)
    coords = route["geometry"]["coordinates"]  # [[lon,lat],...]
    coords_latlon = [(lat, lon) for lon, lat in coords]
    return {"distance_m": distance_m, "duration_s": duration_s, "coords_latlon": coords_latlon}

# ========================= [BLOCK 5] 주문 선택 =========================
st.markdown("### 🔎 주문 선택")
order_ids = orders[COL["id"]].tolist() if COL["id"] in orders.columns else []
default_idx = len(order_ids) - 1 if order_ids else 0
selected_id = st.selectbox("주문 ID", order_ids, index=default_idx)

sel = orders[orders[COL["id"]] == selected_id].iloc[0] if order_ids else None

# ========================= [BLOCK 6] 3분할 레이아웃 =========================
left_col, mid_col, right_col = st.columns([1.3, 1.0, 1.0])

# ---- 좌: 지도 (실제 경로 + CSV 색상)

def local_image_to_data_url(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

# 예: 로컬 PNG 경로
start_icon_path = Path("restaurant.png")
end_icon_path   = Path("home.png")

start_icon_url = local_image_to_data_url(start_icon_path)
end_icon_url   = local_image_to_data_url(end_icon_path)
with left_col:
    st.subheader("지도 / 실제 도로 경로 (CSV 혼잡도 색)")
    if sel is None:
        st.info("주문을 선택하세요.")
    else:
        s_lat = float(sel[COL["store_lat"]]); s_lng = float(sel[COL["store_lng"]])
        e_lat = float(sel[COL["cust_lat"]]);  e_lng = float(sel[COL["cust_lng"]])
        if np.isnan(s_lat) or np.isnan(s_lng) or np.isnan(e_lat) or np.isnan(e_lng):
            st.warning("이 주문에는 좌표가 없습니다. 다른 주문을 선택하세요.")
        else:
            traffic_val = sel.get(COL["traffic"], "Unknown")
            color = traffic_to_color(traffic_val)

            # 실제 경로 좌표 가져오기 (교통 미반영, driving)
            try:
                route = fetch_route_mapbox_geometry(s_lat, s_lng, e_lat, e_lng, MAPBOX_TOKEN)
                coords = route["coords_latlon"]
            except Exception as e:
                coords = [(s_lat, s_lng), (e_lat, e_lng)]  # 실패 시 직선 대체
                st.error(f"경로 조회 실패(직선으로 표시): {e}")

            center = ((s_lat + e_lat) / 2, (s_lng + e_lng) / 2)
            m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
            # folium.Marker((s_lat, s_lng), tooltip="출발(매장)", icon=folium.Icon(color="green", icon="motorcycle", prefix="fa")).add_to(m)
            # folium.Marker((e_lat, e_lng), tooltip="도착(고객)", icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(m)
            folium.Marker((s_lat, s_lng),tooltip="출발",icon=folium.CustomIcon(start_icon_url, icon_size=(50, 50))).add_to(m)
            folium.Marker((e_lat, e_lng),tooltip="도착",icon=folium.CustomIcon(end_icon_url, icon_size=(50, 50))).add_to(m)

            # ▼ 전체 경로에 CSV 혼잡도 색상 적용 (단일 색)
            folium.PolyLine(coords, color=color, weight=8, opacity=0.95).add_to(m)

            # 간단 범례
            import branca
            legend = """
            <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                        background: white; padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.15); font-size: 13px;">
              <b>CSV Road_traffic_density</b><br>
              <div style="margin-top:6px"><span style="display:inline-block;width:14px;height:10px;background:#1DB954;margin-right:6px;border:1px solid #999"></span> Low</div>
              <div><span style="display:inline-block;width:14px;height:10px;background:#FFA500;margin-right:6px;border:1px solid #999"></span> Medium</div>
              <div><span style="display:inline-block;width:14px;height:10px;background:#FF4D4D;margin-right:6px;border:1px solid #999"></span> High</div>
              <div><span style="display:inline-block;width:14px;height:10px;background:#B30000;margin-right:6px;border:1px solid #999"></span> Jam</div>
              <div><span style="display:inline-block;width:14px;height:10px;background:#999999;margin-right:6px;border:1px solid #999"></span> Unknown</div>
            </div>"""
            macro = branca.element.MacroElement()
            macro._template = branca.element.Template(legend)
            m.get_root().add_child(macro)

            st_folium(m, width=None, height=520)
            if "distance_m" in route and route["distance_m"] is not None:
                km = route["distance_m"] / 1000
                st.caption(f"경로 길이: {km:.2f} km  |  색상 근거: Road_traffic_density = {traffic_val}")
            else:
                st.caption(f"색상 근거: Road_traffic_density = {traffic_val}")

# ---- 중: ETA 통계 (CSV 전체)
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
            .encode(x=alt.X("eta_min:Q", bin=alt.Bin(maxbins=30), title="ETA(분)"),
                    y=alt.Y("count()", title="건수"))
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
        .encode(x=alt.X("importance:Q", title="중요도"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature","importance"])
        .properties(height=240)
    )
    st.altair_chart(bar, use_container_width=True)
    st.caption("※ 실제 모델 중요도 연결 시 이 영역 교체 예정")

# ========================= [BLOCK 7] 하단 파이프라인 (선택 주문) =========================
st.markdown("---")
st.subheader("주문 파이프라인")

if sel is None:
    st.info("주문을 선택하세요.")
else:
    order_dt   = parse_datetime(sel[COL["date"]], sel[COL["order_time"]])
    pickup_dt  = parse_datetime(sel[COL["date"]], sel[COL["pickup_time"]])
    prep_min   = float(sel[COL["prep_min"]]) if pd.notna(sel[COL["prep_min"]]) else np.nan
    total_min  = float(sel[COL["total_min"]]) if pd.notna(sel[COL["total_min"]]) else np.nan

    if (pickup_dt is None) and (order_dt is not None) and (not np.isnan(prep_min)):
        pickup_dt = order_dt + timedelta(minutes=prep_min)
    delivered_dt = None
    if (order_dt is not None) and (not np.isnan(total_min)):
        delivered_dt = order_dt + timedelta(minutes=total_min)

    now_str = datetime.now().strftime("%H:%M")
    ot_str  = fmt_kor(order_dt)
    prep_str= f"약 {int(round(prep_min))}분 소요" if not np.isnan(prep_min) else "-"
    pk_str  = fmt_kor(pickup_dt)
    deliver_only = None
    if not np.isnan(total_min) and not np.isnan(prep_min):
        deliver_only = max(0, total_min - prep_min)
    dl_str  = f"약 {int(round(deliver_only))}분 소요" if deliver_only is not None else "-"
    dv_str  = fmt_kor(delivered_dt)

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

# ========================= [BLOCK 8] 주의사항 =========================
st.caption("ℹ️ 경로는 Mapbox Directions(driving)로 계산된 '현재' 기준 도로 경로이며, 선 색상은 CSV의 Road_traffic_density 값을 그대로 반영합니다(실시간 교통 미사용).")
