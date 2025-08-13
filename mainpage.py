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
import time

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

# ========================= [BLOCK 7] 하단 파이프라인 (시뮬 시간, 동적 체크, 오토바이 애니메이션) =========================
import time
import streamlit.components.v1 as components

st.markdown("---")
st.subheader("주문 파이프라인")

if sel is None:
    st.info("주문을 선택하세요.")
else:
    # --- 원본 시간/수치 파싱 ---
    order_dt   = parse_datetime(sel[COL["date"]], sel[COL["order_time"]])
    pickup_dt  = parse_datetime(sel[COL["date"]], sel[COL["pickup_time"]])
    prep_min   = float(sel[COL["prep_min"]])  if pd.notna(sel[COL["prep_min"]])  else np.nan
    total_min  = float(sel[COL["total_min"]]) if pd.notna(sel[COL["total_min"]]) else np.nan

    # ✅ 실제 배달 완료까지 걸린 시간(분) = Time_real (없으면 fallback)
    if "Time_real" in sel and pd.notna(sel["Time_real"]):
        deliver_only_min = float(sel["Time_real"])
    else:
        deliver_only_min = (max(0, total_min - prep_min)
                            if (not np.isnan(total_min) and not np.isnan(prep_min))
                            else None)

    # 보간
    if (pickup_dt is None) and (order_dt is not None) and (not np.isnan(prep_min)):
        pickup_dt = order_dt + timedelta(minutes=prep_min)
    delivered_dt = (pickup_dt + timedelta(minutes=deliver_only_min)) if (pickup_dt and deliver_only_min is not None) \
                   else (order_dt + timedelta(minutes=total_min) if (order_dt and not np.isnan(total_min)) else None)

    # --- ⏱ 시뮬레이터: 현실 3초 = 시뮬 1분 ---
    # 상태 저장 키: 주문 변경/픽업시각 변경 시 앵커 리셋
    need_reset = False
    if st.session_state.get("sim_id") != sel[COL["id"]]:
        need_reset = True
    if st.session_state.get("sim_pickup") != (pickup_dt.isoformat() if pickup_dt else None):
        need_reset = True

    if need_reset:
        st.session_state["sim_id"] = sel[COL["id"]]
        st.session_state["sim_pickup"] = (pickup_dt.isoformat() if pickup_dt else None)
        st.session_state["sim_anchor_real"] = datetime.now()
        st.session_state["sim_anchor_sim"]  = pickup_dt or order_dt or datetime.now()

    anchor_real = st.session_state["sim_anchor_real"]
    anchor_sim  = st.session_state["sim_anchor_sim"]

    # 현실 경과초 → 시뮬 경과분 (3초 = 1분 → 배속 20x)
    real_elapsed_sec = (datetime.now() - anchor_real).total_seconds()
    sim_elapsed_min  = real_elapsed_sec / 3.0
    sim_now = anchor_sim + timedelta(minutes=sim_elapsed_min)

    # 픽업~완료 구간으로 클램프(픽업 전이면 픽업에 고정, 완료 지나면 완료에 고정)
    if pickup_dt:
        if delivered_dt:
            sim_now = min(max(sim_now, pickup_dt), delivered_dt)
        else:
            sim_now = max(sim_now, pickup_dt)

    # --- 진행률/ETA(시뮬 시간 기준) ---
    progress_pct = 0.0
    eta_remain_min = None
    if pickup_dt and delivered_dt and delivered_dt > pickup_dt:
        total_sec   = (delivered_dt - pickup_dt).total_seconds()
        elapsed_sec = (sim_now - pickup_dt).total_seconds()
        progress_pct = max(0.0, min(elapsed_sec / total_sec, 1.0))
        if 0 <= progress_pct < 1:
            eta_remain_min = max(0, int(round((1 - progress_pct) * total_sec / 60)))

    # --- 상태 체크(시뮬 시간 기준) ---
    accepted_done  = (order_dt is not None)    and (sim_now >= order_dt)
    prepared_done  = (pickup_dt is not None)   and (sim_now >= pickup_dt)
    delivered_done = (delivered_dt is not None) and (sim_now >= delivered_dt)

    # --- 표기 문자열 ---
    now_str  = sim_now.strftime("%H:%M")                     # "현재 시각(픽업 기준)" = 시뮬 시간
    ot_str   = fmt_kor(order_dt)
    pk_str   = fmt_kor(pickup_dt)
    dv_str   = fmt_kor(delivered_dt)
    prep_str = f"약 {int(round(prep_min))}분 소요" if not np.isnan(prep_min) else "-"

    # 뱃지 HTML
    def badge_html(checked: bool) -> str:
        return '<div class="badge">✓</div>' if checked else '<div class="badge-empty"></div>'

    accepted_badge  = badge_html(accepted_done)
    prepared_badge  = badge_html(prepared_done)
    delivered_badge = badge_html(delivered_done)

    # 진행 게이지/오토바이 위치
    prog_width = f"{progress_pct * 100:.1f}%"
    remain_text = (
        f"남은 약 {eta_remain_min}분" if eta_remain_min is not None
        else ("완료" if delivered_done else ("곧 시작" if not prepared_done else "-"))
    )
    bike_left = prog_width

    # --- HTML/CSS + 렌더 ---
    pipeline_html = f"""
    <style>
    .step-wrap{{display:flex;align-items:center;gap:48px;margin-top:10px;margin-bottom:10px}}
    .step{{text-align:center}}
    .badge{{width:82px;height:82px;border-radius:50%;background:#5A754D;color:white;
           display:flex;align-items:center;justify-content:center;font-weight:800;font-size:28px;
           box-shadow:inset -6px -6px 0 rgba(0,0,0,0.15)}}
    .badge-empty{{width:82px;height:82px;border-radius:50%;border:10px solid #5A615D;background:#fff}}
    .line{{height:10px;background:#5A615D;flex:1}}
    .step-title{{font-size:24px;font-weight:700;margin-bottom:6px}}
    .step-sub{{color:#8B8F90;font-size:20px}}
    .big-clock{{font-size:64px;color:#5A754D;font-weight:900;margin:0}}

    /* 진행 게이지 + 오토바이 */
    .progress-wrap{{min-width:320px}}
    .progress-track{{position:relative;height:16px;background:#E9ECEB;border-radius:10px;overflow:hidden}}
    .progress-fill{{position:absolute;left:0;top:0;height:100%;width:{prog_width};background:#E07A18;
                    transition:width 0.8s ease;}}
    .progress-bike{{position:absolute;top:50%;left:{bike_left};transform:translate(-50%,-50%);
                    font-size:22px; line-height:1; transition:left 0.8s ease;}}
    .progress-label{{margin-top:6px;color:#6B7072;font-size:14px}}
    </style>

    <div style="display:flex;justify-content:space-between;align-items:center;">
      <!-- 현재 시각(픽업 기준: 시뮬 시간) -->
      <div class="step">
        <div class="step-title">현재 시각 (픽업 기준)</div>
        <div class="big-clock">{now_str}</div>
      </div>

      <div class="step-wrap" style="flex:1;margin-left:24px;margin-right:24px;">
        <!-- 주문 수락 -->
        <div class="step">
          <div class="step-title">주문 수락됨</div>
          {accepted_badge}
          <div class="step-sub">{ot_str}</div>
        </div>

        <div class="line"></div>

        <!-- 메뉴 준비 -->
        <div class="step">
          <div class="step-title">메뉴 준비중</div>
          {prepared_badge}
          <div class="step-sub">{prep_str}</div>
        </div>

        <div class="line"></div>

        <!-- 배달중: 게이지 + 오토바이 -->
        <div class="step progress-wrap">
          <div class="step-title">배달중</div>
          <div class="progress-track">
            <div class="progress-fill"></div>
            <div class="progress-bike">🛵</div>
          </div>
          <div class="progress-label">{pk_str} · {remain_text}</div>
        </div>

        <div class="line"></div>

        <!-- 배달 완료 -->
        <div class="step">
          <div class="step-title">배달 완료</div>
          {delivered_badge}
          <div class="step-sub">{dv_str}</div>
        </div>
      </div>
    </div>
    """

    components.html(pipeline_html, height=320, scrolling=False)

    # ─ 자동 리프레시: 배달중 구간에서 3초마다 재실행(현실 3초 → 시뮬 1분)
    if (pickup_dt and delivered_dt) and (sim_now < delivered_dt):
        time.sleep(3)
        st.rerun()

# ========================= [BLOCK 8] 주의사항 =========================
st.caption("ℹ️ 경로는 Mapbox Directions(driving)로 계산된 '현재' 기준 도로 경로이며, 선 색상은 CSV의 Road_traffic_density 값을 그대로 반영합니다(실시간 교통 미사용).")
