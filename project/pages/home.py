# ========================= [BLOCK 1] 기본 설정 & 라이브러리 =========================
import folium
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from streamlit.components.v1 import html as components_html
import streamlit.components.v1 as components
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]     # -> project/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))              # utils 등 import 할 때도 편함

DATA_DIR   = ROOT / "data"                 
ASSETS_DIR = ROOT / "assets"                   
from utils.paths import LOCAL_CSV_PATH

st.set_page_config(page_title="🐘 Deliphant : 설명가능한 AI 배달예측", layout="wide")
# 🔝 타이틀 위 전용 슬롯
FX_SLOT = st.container()

st.markdown("""
<style>
.custom-title {
    font-size: 35px; /* 원하는 크기 */
    font-weight: 700;
    color: #000000;
}
</style>
<div class="custom-title">🐘 Deliphant : 설명가능한 AI 배달예측</div>
""", unsafe_allow_html=True)
st.markdown("---")


######### 페이지 변환 네비게이션 ########3
qp = st.query_params
to = qp.get("to")
if to == "prob":
    qid = qp.get("id")
    if qid:
        st.session_state["selected_id"] = qid  # ★ 쿼리 → 세션 복사
    # (선택) URL 깨끗하게: 이동 전에 파라미터 지우기
    st.query_params.clear()
    st.switch_page("pages/prob_distribution.py")

elif to == "fi":
    qid = qp.get("id")
    if qid:
        st.session_state["selected_id"] = qid  # ★ 쿼리 → 세션 복사
    st.query_params.clear()
    st.switch_page("pages/feature_importance.py")


# ========================= [BLOCK 2] 전역 상수(토큰/파일/컬럼/색상) =========================
MAPBOX_TOKEN   = ""
target_ids = ['0x8fdd', '0x23d4', '0x6461', '0x9d32', '0x7ab9', '0x8b39', '0x6f80', '0xa512', '0xd740', '0xd200']

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
    "festival": "Festival",        # yes / no
    "peak_flag": "Peak_flag",      # 1 / 0
    "region": "region",
    "restaurant_name": "Restaurant_name_real",
    "courier_id": "Delivery_person_ID",
    "courier_age": "Delivery_person_Age",
    "courier_rating": "Delivery_person_Ratings",
    "city":"City",
    "long":"long_distance",
    "weather":"Weatherconditions"
}


ROAD_TRAFFIC_COLOR = {
    "low": "#1DB954",      # 초록
    "medium": "#FFA500",   # 주황
    "high": "#FF4D4D",     # 빨강
    "jam": "#B30000",      # 진빨강
    "unknown": "#999999",  # 회색
}

# ========================= [BLOCK 3] 데이터 로드/전처리 ===============================
@st.cache_data
def load_orders(path: str | Path, filter_ids: list, col_map: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[col_map["id"]] = df[col_map["id"]].astype(str).str.strip()
    df = df[df[col_map["id"]].isin(filter_ids)].reset_index(drop=True)
    for c in [col_map["date"], col_map["order_time"], col_map["pickup_time"]]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

orders = load_orders(LOCAL_CSV_PATH, target_ids, COL)

##### 이미지 로드 #####
def local_image_to_data_url(img_path: str | Path) -> str:
    p = Path(img_path)
    with open(p, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

# 아이콘 경로 
start_icon_path = ASSETS_DIR / "icons" / "map" / "resturant.png"
end_icon_path   = ASSETS_DIR / "icons" / "map" / "home.png"
ele_path        = ASSETS_DIR / "icons" / "home" / "elephant.png"
bike_image_path = ASSETS_DIR / "icons" / "home" / "elephant_person.png"

start_icon_url = local_image_to_data_url(start_icon_path)
end_icon_url   = local_image_to_data_url(end_icon_path)
ele_src        = local_image_to_data_url(ele_path)
bike_img_url   = local_image_to_data_url(bike_image_path)

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

# ================================ [BLOCK 5] 주문 선택 ==================================
def clean_id(x):
    s = str(x)
    return s.replace("\u00A0", " ").replace("\ufeff", "").replace("\u200b", "").strip()

def to_int_or_none(v):
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except:
        return None

def to_float_or_none(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except:
        return None

# --- 레이아웃: 주문ID / 매장정보 / 배달원정보 / 특이사항 ---
col_id, col_store, col_courier, col_special = st.columns([0.8, 1.0, 1.2, 1.5])

orders[COL["id"]] = orders[COL["id"]].apply(clean_id)
order_ids = orders[COL["id"]].tolist()
default_idx = len(order_ids) - 1 if order_ids else 0
selected_id_clean = None
sel = None

# ===== 주문 ID =====
with col_id:
    selected_id = st.selectbox(
        "주문 ID",
        order_ids,
        index=default_idx,
        format_func=clean_id
    )
    selected_id_clean = clean_id(selected_id)
    sel = orders[orders[COL["id"]] == selected_id_clean].iloc[0] if order_ids else None
    st.session_state["selected_id"] = selected_id_clean

# ===== 매장정보 =====
with col_store:
    region = sel.get(COL["region"], "—") if sel is not None else "—"
    rname  = sel.get(COL["restaurant_name"], "—") if sel is not None else "—"

    st.markdown(
        f"""
        <div style="background:#f2f2f2; padding:10px 12px; border-radius:8px;
                    display:flex; flex-direction:column; justify-content:flex-start;">
          <div style="font-weight:700; font-size:16px; margin-bottom:6px;">🍽️ 매장 정보</div>
          <div style="display:grid; gap:6px;">
            <div style="background:#fff; border:1px solid #e3e3e3; border-radius:6px; padding:6px 8px;">
              <div style="color:#70757a; font-size:12px; font-weight:600;">지역</div>
              <div style="font-size:14px; font-weight:600;">{region}</div>
            </div>
            <div style="background:#fff; border:1px solid #e3e3e3; border-radius:6px; padding:6px 8px;">
              <div style="color:#70757a; font-size:12px; font-weight:600;">매장명</div>
              <div style="font-size:14px; font-weight:600;">{rname}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== 배달원 정보 (중간) =====
with col_courier:
    if sel is not None:
        courier_id_raw     = sel.get(COL["courier_id"], "—")
        courier_age_raw    = sel.get(COL["courier_age"], None)
        courier_rating_raw = sel.get(COL["courier_rating"], None)

        age_int = to_int_or_none(courier_age_raw)
        courier_age = f"{(age_int // 10) * 10}대" if age_int is not None else "—"

        rating_f = to_float_or_none(courier_rating_raw)
        courier_rating = round(rating_f, 2) if rating_f is not None else "—"

        courier_id = courier_id_raw if str(courier_id_raw).strip() else "—"
    else:
        courier_id = courier_age = courier_rating = "—"

    st.markdown(
        f"""
        <style>
          /* 2칼럼 그리드 (폭 좁아지면 자동 1열) */
          @media (min-width: 720px) {{
            .two-col-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:6px; }}
          }}
          @media (max-width: 719px) {{
            .two-col-grid {{ display:grid; grid-template-columns: 1fr; gap:6px; }}
          }}
          .card {{ background:#fff; border:1px solid #e3e3e3; border-radius:6px; padding:6px 8px; }}
          .label {{ color:#70757a; font-size:12px; font-weight:600; margin-bottom:2px; }}
          .value {{ font-size:14px; font-weight:600; }}
        </style>

        <div style="background:#f2f2f2; padding:10px 12px; border-radius:8px;
                    display:flex; flex-direction:column; justify-content:flex-start;">
          <div style="font-weight:700; font-size:16px; margin-bottom:6px;">🛵 배달원 정보</div>

          <!-- 1행: ID (풀폭) -->
          <div class="card" style="margin-bottom:6px;">
            <div class="label">배달원 ID</div>
            <div class="value">{courier_id}</div>
          </div>

          <!-- 2행: 나이 | 평점 (2칼럼) -->
          <div class="two-col-grid">
            <div class="card">
              <div class="label">나이</div>
              <div class="value">{courier_age}</div>
            </div>
            <div class="card">
              <div class="label">평점</div>
              <div class="value">{courier_rating}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== 특이사항 =====
with col_special:
    if sel is not None:
        city    = str(sel.get(COL["city"], "")).strip()
        peak    = sel.get(COL["peak_flag"], None)
        road    = str(sel.get(COL["traffic"], "")).strip()
        fest    = str(sel.get(COL["festival"], "")).strip()
        longd   = sel.get(COL["long"], None)
        weather = str(sel.get(COL["weather"], "")).strip()

        notes = []
        if city.lower() == "semi-urban":
            notes.append("🏙️ 평균적으로 배달이 오래 걸리는 지역이에요.")
        if to_int_or_none(peak) == 1:
            notes.append("⏰ 피크타임이어서 배달이 늦어지고 있어요.")
        if road.lower() == "jam":
            notes.append("🚗🚗 도로 정체로 배달이 늦어지고 있어요.")
        elif road.lower() == "high":
            notes.append("🚙 도로 혼잡으로 배달이 늦어지고 있어요.")
        if fest.lower() == "yes":
            notes.append("🎉 축제기간이라 배달이 늦어요.")
        if to_int_or_none(longd) == 1:
            notes.append("📍 10km 이상 장거리 배달이에요.")

        weather_mapping = {
            "cloudy": "☁️ 현재 비가 오고 있어 배달이 늦어질 수 있어요.",
            "fog": "🌫️ 현재 안개가 껴 있어 배달이 늦어질 수 있어요.",
            "windy": "💨 현재 강풍이 불고 있어요.",
            "stormy": "⛈️ 현재 폭우가 내리고 있어요.",
            "sandstorms": "🌪️ 현재 모래폭풍이 불고 있어요.",
            "sunny": "☀️ 현재 날씨는 맑음이에요."
        }

        if weather:
            weather_key = weather.lower()
            if weather_key in weather_mapping:
                notes.append(weather_mapping[weather_key])


    else:
        notes = []

    if notes:
        li_html = "".join([f"<li style='margin:2px 0; font-size:13px; font-weight:600;'>{n}</li>" for n in notes])
    else:
        li_html = "<li style='margin:2px 0; color:#777; font-size:13px;'>표시할 특이사항이 없어요.</li>"

    st.markdown(
        f"""
        <div style="background:#f2f2f2; padding:10px 12px; border-radius:8px;
                    display:flex; flex-direction:column; justify-content:flex-start;">
          <div style="font-weight:700; font-size:16px; margin-bottom:6px;">📝 특이사항</div>
          <ul style="padding-left:18px; margin:0;">{li_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ========================= [BLOCK 6] 축제 및 피크 시간대 경고 =========================
def trigger_fireworks(duration_sec: float = 2.5, height: int = 120):
    """전체 화면에 폭죽 애니메이션 (canvas-confetti)."""
    components_html(f"""
    <div id="fw-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js"></script>
    <script>
    (function () {{
        const duration = {duration_sec} * 1000;
        const end = Date.now() + duration;
        (function frame() {{
            confetti({{ particleCount: 4, startVelocity: 35, spread: 360, ticks: 70,
                       origin: {{ x: Math.random(), y: Math.random()*0.6 }} }});
            if (Date.now() < end) requestAnimationFrame(frame);
        }})();
    }})();
    </script>
    """, height=height, scrolling=False)

# ======= 상태 플래그 =======
is_festival = False
is_peak = False

if sel is not None:
    # Festival: 'yes'면 True
    if COL["festival"] in sel.index:
        val = str(sel[COL["festival"]]).strip().lower()
        is_festival = (val == "yes")

    # Peak_flag: 1이면 True
    if COL["peak_flag"] in sel.index:
        try:
            is_peak = int(float(sel[COL["peak_flag"]])) == 1
        except Exception:
            is_peak = False

# Festival 효과 + 알림  (🔝 타이틀 위에 표시)
if is_festival:
    with FX_SLOT:
        trigger_fireworks(duration_sec=5.0, height=130)
    st.toast("축제 기간이라 배달이 늦어지고 있어요 🥹", icon="🎆")
else:
    # 축제가 아니면 슬롯 비우기(이전 렌더 지우기)
    FX_SLOT.empty()

# 피크 효과 + 알림 
if is_peak:
    if not st.session_state.get("_peak_toast_shown"):
        st.toast("피크 시간대라 배달이 늦어지고 있어요 🥹")
        st.session_state["_peak_toast_shown"] = True
else:
    # 피크 해제 시 다음 번에 다시 토스트 보낼 수 있도록 플래그 리셋
    st.session_state["_peak_toast_shown"] = False

peak_text_html = ""
if is_peak:
    peak_text_html = """
        <style>
        @keyframes flash {
            0%, 100% { color: #e11d48; text-shadow: 0 0 6px rgba(255,0,0,0.8); }
            50% { color: #ff4d6d; text-shadow: 0 0 16px rgba(255,0,0,1); }
        }
        .flash-text {
            font-weight: 700;
            font-size: 16px;
            margin-top: 4px;
            animation: flash 1s infinite;
        }
        </style>
        <div class="flash-text">🚨 피크 시간대 입니다 🚨</div>
        """

# ========================= [BLOCK 7] 3분할 레이아웃 =========================
# -------------------------------------------------
# (A) 먼저 상태/시간/진행률을 모두 계산
# -------------------------------------------------
order_dt   = parse_datetime(sel[COL["date"]], sel[COL["order_time"]]) if sel is not None else None
pickup_dt  = parse_datetime(sel[COL["date"]], sel[COL["pickup_time"]]) if sel is not None else None
prep_min   = float(sel[COL["prep_min"]])  if (sel is not None and pd.notna(sel[COL["prep_min"]]))  else np.nan
total_min  = float(sel[COL["total_min"]]) if (sel is not None and pd.notna(sel[COL["total_min"]])) else np.nan

if sel is not None and "Time_real" in sel and pd.notna(sel["Time_real"]):
    deliver_only_min = float(sel["Time_real"])
else:
    deliver_only_min = (max(0, total_min - prep_min)
                        if (not np.isnan(total_min) and not np.isnan(prep_min)) else None)

if (pickup_dt is None) and (order_dt is not None) and (not np.isnan(prep_min)):
    pickup_dt = order_dt + timedelta(minutes=prep_min)

delivered_dt = (pickup_dt + timedelta(minutes=deliver_only_min)) if (pickup_dt and deliver_only_min is not None) \
               else (order_dt + timedelta(minutes=total_min) if (order_dt and not np.isnan(total_min)) else None)

# 앵커/시뮬레이터 상태 갱신
need_reset = False
if st.session_state.get("sim_id") != (sel[COL["id"]] if sel is not None else None):
    need_reset = True
if st.session_state.get("sim_pickup") != (pickup_dt.isoformat() if pickup_dt else None):
    need_reset = True
if need_reset:
    st.session_state["sim_id"]         = sel[COL["id"]] if sel is not None else None
    st.session_state["sim_pickup"]     = (pickup_dt.isoformat() if pickup_dt else None)
    st.session_state["sim_anchor_real"]= datetime.now()
    st.session_state["sim_anchor_sim"] = pickup_dt or order_dt or datetime.now()
    st.session_state.pop("done_banner_for", None)
    st.session_state.pop("sim_now", None)
    st.session_state["pipe_sim_id"] = None

anchor_real = st.session_state["sim_anchor_real"]
anchor_sim  = st.session_state["sim_anchor_sim"]

real_elapsed_sec = (datetime.now() - anchor_real).total_seconds()
sim_elapsed_min  = real_elapsed_sec / 3.0   # 3초 = 1분
sim_now = anchor_sim + timedelta(minutes=sim_elapsed_min)

if pickup_dt:
    if delivered_dt:
        sim_now = min(max(sim_now, pickup_dt), delivered_dt)
    else:
        sim_now = max(sim_now, pickup_dt)

progress_pct = 0.0; eta_remain_min = None
if pickup_dt and delivered_dt and delivered_dt > pickup_dt:
    total_sec   = (delivered_dt - pickup_dt).total_seconds()
    elapsed_sec = (sim_now - pickup_dt).total_seconds()
    progress_pct = max(0.0, min(elapsed_sec / total_sec, 1.0))
    if 0 <= progress_pct < 1:
        eta_remain_min = max(0, int(round((1-progress_pct)*total_sec/60)))

accepted_done  = (order_dt is not None)     and (sim_now >= order_dt)
prepared_done  = (pickup_dt is not None)    and (sim_now >= pickup_dt)
delivered_done = (delivered_dt is not None) and (sim_now >= delivered_dt)

# 오버레이 텍스트에 쓸 n분
n_min = None
if sel is not None and "Time_real" in sel and pd.notna(sel["Time_real"]):
    n_min = int(round(float(sel["Time_real"])))
elif not np.isnan(total_min):
    n_min = int(round(total_min))

####### 오버레이 함수 (subtitle 지원하도록 확장)
def show_top_overlay_full(minutes_text: int | str, ele_data_url: str = "", auto_close_ms: int | None = None,
                          subtitle_text: str = ""):
    """화면 전체를 덮는 오버레이 + 중앙 카드 (클릭/ESC로 닫힘)."""
    # 줄바꿈 처리
    subtitle_html = ""
    if subtitle_text:
        subtitle_html = f"<div style='font-size:16px; opacity:.95; font-weight:700;'>{str(subtitle_text).replace('\\n','<br>')}</div>"

    card_html = f"""
    <div style="
      background: rgba(0,0,0,0.65);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 28px;
      padding: 28px 36px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
      display: flex; flex-direction: column; align-items: center; gap: 12px;
      min-width: 520px; max-width: 86%;
      color: #fff; text-align: center; font-weight: 800;">
      {f"<img src='{ele_data_url}' alt='elephant' style='width:180px;height:auto;' />" if ele_data_url else ""}
      <div style="font-size:20px; font-weight:700;">{minutes_text}분 만에 배달이 완료되었어요! <span style="font-size:22px">☺️</span></div>
      {subtitle_html}
      <div style="font-size:13px; opacity:.85; font-weight:600;">(화면을 클릭하면 닫혀요)</div>
    </div>
    """

    components.html(f"""
    <div></div>
    <script>
    (function(){{
      const doc = window.parent?.document || document;

      // 기존 것이 있으면 재사용
      let ov = doc.getElementById("dlp-top-overlay");
      if(!ov){{
        ov = doc.createElement("div");
        ov.id = "dlp-top-overlay";
        doc.body.appendChild(ov);
        Object.assign(ov.style, {{
          position: "fixed",
          left: "0px",
          top: "0px",
          width: "100vw",
          height: "100vh",
          background: "rgba(0,0,0,0.60)",
          zIndex: "999999",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "opacity .18s ease",
          opacity: "0"
        }});
        requestAnimationFrame(() => ov.style.opacity = "1");
      }}

      ov.innerHTML = `{card_html.replace("`","\\`")}`;

      const remove = () => {{
        ov.style.opacity = "0";
        setTimeout(() => ov && ov.remove && ov.remove(), 200);
      }};
      ov.onclick = remove;
      doc.addEventListener("keydown", (ev) => {{ if (ev.key === "Escape") remove(); }}, {{ once: true }});

      { f"setTimeout(remove, {auto_close_ms});" if auto_close_ms else "" }
    }})();
    </script>
    """, height=0)

# -------------------------------------------------
# (B) 상단 3분할을 먼저 렌더 + 오버레이는 top_scope 안에
# -------------------------------------------------
top_scope = st.container()
with top_scope:
    # 고유 앵커 id (주문이 바뀌어도 겹치지 않도록)
    scope_key = f"topscope-{selected_id_clean}"
    start_id  = f"{scope_key}-start"
    end_id    = f"{scope_key}-end"
    st.markdown(f"<div id='{start_id}'></div>", unsafe_allow_html=True)
    
    left_col, mid_col, right_col = st.columns([1.3, 1.0, 1.0])
    # ---- 좌: 지도 (실제 경로 + CSV 색상)
    with left_col:
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
                # ▼ 경로 전체가 화면에 들어오도록 자동 맞춤
                lats = [lat for (lat, lon) in coords]
                lngs = [lon for (lat, lon) in coords]
                bounds = [[min(lats), min(lngs)], [max(lats), max(lngs)]]
                m.fit_bounds(bounds, padding=(30, 30))  # 여백(px) 적당히 조절
                            
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
                    st.caption(f"경로 길이: {km:.2f} km  |  교통 상황: {traffic_val}")
                else:
                    st.caption(f"교통 상황: {traffic_val}")

    # ---- 중: ETA 통계 (CSV 기반)
    with mid_col:
        # 전체 orders 데이터프레임이 비어있지 않은 경우에만 실행
        if not orders.empty:
            target_row = orders[orders[COL["id"]] == selected_id].iloc[0]
            
            # 클래스를 시간(분) 범위로 매핑하는 딕셔너리
            time_map = {
                1.0: "10~14분", 1.5: "15~19분", 2.0: "20~24분",
                2.5: "25~29분", 3.0: "30~34분", 3.5: "35~39분",
                4.0: "40~44분", 4.5: "45~49분", 5.0: "50~54분"
            }

            pred_class = None
            if 'max_after_class_key' in target_row and pd.notna(target_row['max_after_class_key']):
                try:
                    pred_class = float(target_row['max_after_class_key'])  # 예: 2.0, 2.5 ...
                except (ValueError, TypeError):
                    pass

            # ▶ 끝 값(최댓값) 맵 생성: "10~14분" → 14
            upper_bound_map = {
                k: int(v.split('~')[1].replace('분', '')) for k, v in time_map.items()
            }

            # 1-1. 출력: "24분 이내" 형태
            if pred_class is not None and pred_class in upper_bound_map:
                upper_bound_min = upper_bound_map[pred_class] + 1        # 예: 25
                time_range_str = f"{upper_bound_min}분 이내"          # "25분 이내"
            else:
                time_range_str = "정보 없음"  # 또는 "계산 불가"

            # 1-2. 예상 도착 시각 (예: "오후 10시 33분 도착 예정") 또는 에러 메시지
            arrival_text = ""
            error_text = ""
            if pred_class is not None and COL["pickup_time"] in orders.columns:
                pickup_time_dt = parse_datetime(target_row.get(COL["date"]), target_row.get(COL["pickup_time"]))

                if pickup_time_dt:
                    minutes_to_add = upper_bound_map[pred_class] + 1
                    estimated_arrival_time = pickup_time_dt + timedelta(minutes=minutes_to_add)
                    arrival_text = f"{fmt_kor(estimated_arrival_time)} 전 도착 예정"
                else:
                    error_text = "픽업 시간이 없어 도착 예정 시간을 계산할 수 없습니다."
            else:
                error_text = "예측에 필요한 컬럼이 없거나 데이터가 유효하지 않습니다."


            # 1-3. 준비된 변수들을 사용하여 하나의 HTML 블록으로 모든 정보를 한 번에 출력합니다.
            # 도착 시각이 정상적으로 계산되었는지, 아니면 에러가 발생했는지에 따라 세 번째 줄의 내용이 바뀝니다.
            if error_text:
                third_line_html = f"<h4 style='text-align: left; color: #FF4B4B; margin-top: 5px;'>{error_text}</h4>"
            else:
                third_line_html = f"<h5 style='text-align: left; margin-top: -5px;'>{arrival_text}</h5>"

            eta_inner_html = f"""
                <div style="line-height: 1.0; padding: 8px 8px 8px 14px;">
                    <h3 style='text-align: left; font-weight: bold; margin-bottom: -20px;'>배달 예상 소요 시간</h3>
                    <h1 style='text-align: left; color: #1E90FF; margin-top: -20px;'>{time_range_str}</h1>
                    {"<h4 style='text-align:left; color:#FF4B4B; margin-top:5px;'>" + error_text + "</h4>" if error_text
                    else f"<h5 style='text-align:left; margin-top:-5px;'>{arrival_text}</h5>"}
                    <p>주문하신 곳으로 가고 있어요. 🛵</p>
                </div>
            """

            # 카드 전체 클릭(hover 확대 + 클릭 시 이동)
            selected_id_clean = str(selected_id)  # 이미 있으시면 그 변수 사용
            eta_card = f"""
                <div class="click-card" style="background:#ffffff; padding: 8px; border-radius:16px;">
                <a class="cover-link" href="?to=prob&id={selected_id_clean}" aria-label="확률분포 상세보기"></a>
                {eta_inner_html}
                </div>
            """
            st.markdown(eta_card, unsafe_allow_html=True)
            st.write("")

            # 2. 가로 막대 그래프 생성 (시간대 텍스트만, 값 라벨 표시, x축 숨김)
            chart_data = []

            pairs = [
                ('max_before_class_key', 'max_before_class_value'),
                ('max_class_key',        'max_class_value'),
                ('max_after_class_key',  'max_after_class_value'),
            ]

            def _to_float_or_none(x):
                try:
                    if pd.isna(x):
                        return None
                    return float(x)
                except Exception:
                    return None

            def _time_label_from_key(key_val):
                """time_map 우선 사용, 없으면 key로 5분 구간 자동 생성."""
                # 1) time_map에 문자열 키로
                if key_val in time_map:
                    return time_map[key_val]
                # 2) float 변환해서 time_map에
                kf = _to_float_or_none(key_val)
                if kf in time_map:
                    return time_map[kf]
                # 3) time_map이 없거나 키가 없으면 규칙으로 생성 (예: 3.0 -> 30~34분)
                if kf is not None:
                    start = int(round(kf * 10))
                    end = start + 4
                    return f"{start}~{end}분"
                # 4) 최후 fallback
                return f"{key_val} 구간"

            for key_col, val_col in pairs:
                if key_col not in target_row.index or val_col not in target_row.index:
                    continue
                key_val = target_row[key_col]
                val = target_row[val_col]
                if pd.isna(val) or pd.isna(key_val):
                    continue

                time_label = _time_label_from_key(key_val)
                chart_data.append({
                    "time_range": time_label,
                    "value": round(float(val) * 100, 1)  # %로 변환
                })

            # 3. chart_data에 유효한 데이터가 하나라도 있으면 차트를 출력합니다.
            if chart_data:
                import altair as alt
                chart_df = pd.DataFrame(chart_data).dropna()

                # 값/형식
                chart_df["value"] = chart_df["value"].astype(float)        # 0~100 (%)
                chart_df["percent_str"] = chart_df["value"].round(0).astype(int).astype(str) + "%"

                # 하이라이트(최대값)
                vmax = chart_df["value"].max()
                chart_df["is_max"] = chart_df["value"] == vmax

                # 색상 정의
                COLOR_TRACK   = "#E9EEF2"
                COLOR_INACTIVE= "#8C8F93"
                COLOR_ACTIVE  = "#D97706"

                # 트랙(100%) 값
                chart_df["track"] = 100

                # 공통 y 인코딩
                y_enc = alt.Y("time_range:N", title=None, sort=None, axis=None)

                # 왼쪽: 시간대 텍스트 (왼쪽 위치)
                left_labels = (
                    alt.Chart(chart_df)
                    .mark_text(align="left", baseline="middle", fontSize=18, dx=-20)  # dx로 왼쪽 이동
                    .encode(
                        y=y_enc,
                        text="time_range:N",
                        color=alt.condition("datum.is_max", alt.value(COLOR_ACTIVE), alt.value(COLOR_INACTIVE))
                    )
                    .properties(width=140, height=120)  # 폭 살짝 넓힘
                )

                # 가운데: 트랙 + 채워진 막대 (짧게 & 얇게)
                base = alt.Chart(chart_df).encode(y=y_enc)

                track = (
                    base.mark_bar(size=5, color=COLOR_TRACK)
                    .encode(x=alt.X("track:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 100])))
                    .properties(width=140, height=120)  # 막대 길이 더 짧게
                )

                filled = (
                    base.mark_bar(size=5)
                    .encode(
                        x=alt.X("value:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 100])),
                        color=alt.condition("datum.is_max", alt.value(COLOR_ACTIVE), alt.value(COLOR_INACTIVE))
                    )
                    .properties(width=140, height=120)
                )

                middle = track + filled

                # 오른쪽: % 숫자 (더 크게, 굵게)
                right_values = (
                    alt.Chart(chart_df)
                    .mark_text(align="right", baseline="middle", fontSize=18, fontWeight="bold", dx=-10)
                    .encode(
                        y=y_enc,
                        text="percent_str:N",
                        color=alt.condition("datum.is_max", alt.value(COLOR_ACTIVE), alt.value(COLOR_INACTIVE))
                    )
                    .properties(width=60, height=120)
                )

                # 좌우 붙이기 + y 공유
                chart_comp = alt.hconcat(left_labels, middle, right_values).resolve_scale(y='shared')

                st.altair_chart(chart_comp, use_container_width=True)

                # 상세보기 버튼
                st.session_state['selected_id'] = selected_id
                st.markdown("""
                    <style>
                    .click-card {
                        position: relative;
                        border-radius: 16px;
                        background: #ffffff;
                        padding: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.06); /* 기본 그림자 */
                        border: 1px solid rgba(0,0,0,0.05);     /* 기본 경계선 */
                        transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
                        will-change: transform;
                    }
                    .click-card:hover {
                        transform: scale(1.02);
                        box-shadow: 0 10px 28px rgba(0,0,0,.12);
                        border-color: rgba(0,0,0,0.15); /* hover 시 테두리 강조 */
                    }
                    .click-card .cover-link {
                        position: absolute; inset: 0;
                        z-index: 3;
                        text-indent: -9999px;
                    }
                    .click-card, .click-card * { cursor: pointer; }
                    .click-card * { pointer-events: none; }
                    .click-card .cover-link { pointer-events: auto; }
                    </style>
                    """, unsafe_allow_html=True)
            else:
                st.warning("차트를 표시할 예측 확률 데이터가 없습니다.")

    # ---- 우: 변수 중요도 (박스 제거 + 세로 간격 줄임 + 칼럼명 볼드 제거 + 상세보기 버튼 중앙)
    with right_col:

        @st.cache_data
        def load_fi_csv(path: str | Path) -> pd.DataFrame:
            df_fi = pd.read_csv(path)
            df_fi.columns = [str(c).strip() for c in df_fi.columns]
            return df_fi

        try:
            feat_path = DATA_DIR / "feature_importance.csv"   # ✅ ROOT 기준
            df_fi = load_fi_csv(feat_path)
        except Exception as e:
            st.warning(f"feature_importance.csv 로드 실패: {e}  → 이 섹션만 숨기고 아래 콘텐츠는 계속 렌더합니다.")
            df_fi = None

        if df_fi is not None:
            if "ID" not in df_fi.columns:
                st.warning("feature_importance.csv에 'ID' 컬럼이 없어 변수 중요도 섹션만 건너뜁니다.")
            else:
                row = df_fi[df_fi["ID"] == selected_id]
                if row.empty:
                    st.warning("선택한 ID에 대한 변수 중요도 데이터가 없습니다.")
                else:
                    row = row.iloc[0]

                use_cols = [
                    ("distance_km",          "거리 🧭"),
                    ("Weatherconditions",    "날씨 🌈"),
                    ("region_city",          "지역 🏙️"),
                    ("multiple_deliveries",  "배달 수 🏍️"),
                    ("Road_traffic_density", "교통 🚗"),
                ]

                chart_data = []
                for col, label in use_cols:
                    if col not in df_fi.columns:
                        continue
                    val = row[col]
                    if pd.isna(val):
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    v = v*100 if 0.0 <= v <= 1.0 else v
                    v = max(0, min(v, 100))
                    chart_data.append({"feature": label, "value": v})

                if not chart_data:
                    st.warning("표시할 변수 중요도 값이 없습니다.")
                else:
                    import altair as alt
                    chart_df = pd.DataFrame(chart_data)
                    chart_df = chart_df.sort_values("value", ascending=False).reset_index(drop=True)

                    COLOR_TRACK = "#E9E7F3"
                    COLOR_FILL  = "#6C7F45"
                    COLOR_LABEL = "#111111"
                    COLOR_PCT   = "#D97706"

                    chart_df["track"] = 100
                    chart_df["pct_str"] = chart_df["value"].round(0).astype(int).astype(str) + "%"

                    y_order = chart_df["feature"].tolist()
                    y_enc = alt.Y("feature:N", title=None, sort=y_order, axis=None)

                    # 세로 간격 더 좁게
                    row_h = 40
                    total_h = max(120, len(chart_df) * row_h)

                    # 왼쪽 라벨 (볼드 제거)
                    left_labels = (
                        alt.Chart(chart_df)
                        .mark_text(
                            align="right",
                            baseline="middle",
                            fontSize=17,
                            fontWeight="normal",
                            dx=5, 
                            color=COLOR_LABEL
                        )
                        .encode(y=y_enc, text="feature:N")
                        .properties(width=20, height=total_h)
                    )

                    base = alt.Chart(chart_df).encode(y=y_enc)
                    track = base.mark_bar(size=18, color=COLOR_TRACK, cornerRadius=999).encode(
                        x=alt.X("track:Q", scale=alt.Scale(domain=[0,100]), axis=None, title=None)
                    ).properties(width=200, height=total_h)
                    fill = base.mark_bar(size=18, color=COLOR_FILL, cornerRadius=999).encode(
                        x=alt.X("value:Q", scale=alt.Scale(domain=[0,100]), axis=None, title=None)
                    ).properties(width=200, height=total_h)

                    middle = track + fill

                    right_values = (
                        alt.Chart(chart_df)
                        .mark_text(align="left", baseline="middle", fontSize=18, fontWeight="bold", dx=6, color=COLOR_PCT)
                        .encode(y=y_enc, text="pct_str:N")
                        .properties(width=10, height=total_h)
                    )

                    chart_comp = alt.hconcat(left_labels, middle, right_values).resolve_scale(y="shared")
                    
                    st.session_state['selected_id'] = selected_id
                    html_code = f"""
                    <div class="click-card" style="background:#ffffff; padding: 8px 8px 8px 20px; border-radius:16px;">
                        <a class="cover-link" href="?to=fi&id={selected_id_clean}" aria-label="변수 중요도 상세보기"></a>
                        <div style="line-height: 1.4;">
                            <h3 style='text-align: left; font-weight: bold; margin-bottom: 6px;'>변수 중요도</h3>
                            <p style='text-align: left; color: #555; font-size:18px; margin-top: 0;'>예상시간에 영향을 끼치고 있는 변수들이에요.</p>
                        </div>
                    </div>
                    """

                    st.markdown(html_code, unsafe_allow_html=True)
                    
                    st.write("")
                    st.altair_chart(chart_comp, use_container_width=True)
                    # 카드 닫기
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div id='{end_id}'></div>", unsafe_allow_html=True)

# ========================= [BLOCK 8] 하단 파이프라인 (선택 주문) =========================
st.markdown("---")
# st.subheader("배달 현황")

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

    # --- 시뮬 시계 (3초=1분) ---
    if st.session_state.get("pipe_sim_id") != selected_id_clean or "sim_now" not in st.session_state:
        st.session_state["pipe_sim_id"] = selected_id_clean
        st.session_state["sim_now"] = pickup_dt or order_dt or datetime.now()

    sim_now = st.session_state["sim_now"]

    # --- 진행률 ---
    progress_ratio = 0.0
    if pickup_dt and delivered_dt and delivered_dt > pickup_dt:
        total_delivery_sec = (delivered_dt - pickup_dt).total_seconds()
        elapsed_sec = (sim_now - pickup_dt).total_seconds()
        progress_ratio = max(0, min(1, elapsed_sec / total_delivery_sec))
    progress_percent = progress_ratio * 100

    # --- 상태 체크 ---
    accepted_done  = (order_dt is not None) and (sim_now >= order_dt)
    prepared_done  = (pickup_dt is not None) and (sim_now >= pickup_dt)
    delivered_done = progress_ratio >= 1.0  # 100%면 완료

    # --- 표기 문자열 ---
    ot_str  = fmt_kor(order_dt)
    pk_str  = fmt_kor(pickup_dt)
    prep_str = f"약 {int(round(prep_min))}분 소요" if not np.isnan(prep_min) else "-"
    dv_str  = fmt_kor(delivered_dt) if delivered_done else ""  # 완료 후에만 시간 표시

    def badge_html(checked: bool) -> str:
        return '<div class="badge">✓</div>' if checked else '<div class="badge-empty"></div>'

    accepted_badge  = badge_html(accepted_done)
    prepared_badge  = badge_html(prepared_done)
    delivered_badge = badge_html(delivered_done)

    # est_delivered_dt 계산
    est_delivered_dt = None
    if pickup_dt and 'upper_bound_min' in locals() and upper_bound_min is not None:
        est_delivered_dt = pickup_dt + timedelta(minutes=upper_bound_min)

    eta_remain_min = None
    if pickup_dt and est_delivered_dt and est_delivered_dt > pickup_dt:
        total_sec   = (est_delivered_dt - pickup_dt).total_seconds()
        elapsed_sec = (sim_now - pickup_dt).total_seconds()
        progress_pct = max(0.0, min(elapsed_sec / total_sec, 1.0))
        if 0 <= progress_pct < 1:
            eta_remain_min = max(0, int(round((1 - progress_pct) * total_sec / 60)))

    remain_text = (
        f"남은 예상 시간 {eta_remain_min}분" if eta_remain_min is not None
        else ("완료" if delivered_done else ("곧 시작" if not prepared_done else "-"))
    )

    ############### 배달 완료 오버레이
    n_min = None
    if "Time_real" in sel and pd.notna(sel["Time_real"]):
        n_min = int(round(float(sel["Time_real"])))
    elif not np.isnan(total_min):
        n_min = int(round(total_min))

    # --- HTML/CSS + 렌더 ---
    bike_node = (
        f'<img class="progress-bike-img" src="{bike_img_url}" alt="bike" />'
        if bike_img_url else
        '<div class="progress-bike-emoji">🛵</div>'
    )

    pipeline_html = f"""
    <style>
    .pipeline-container {{
        display:flex;justify-content:space-between;align-items:center; overflow: visible; padding-top:50px;

    }}

    .step-wrap{{
    display:flex;align-items:center;justify-content:space-between;gap:40px;
    margin-top:10px;margin-bottom:10px;flex:1;overflow: visible;
    }}
    .step{{text-align:center;flex-shrink:0;overflow: visible;}}
    .badge{{width:82px;height:82px;border-radius:50%;background:#5A754D;color:white;
        display:flex;align-items:center;justify-content:center;font-weight:800;font-size:28px;
        box-shadow:inset -6px -6px 0 rgba(0,0,0,0.15)}}
    .badge-empty{{width:82px;height:82px;border-radius:50%;border:10px solid #5A615D;background:#fff}}

    .line{{height:10px;background:#5A615D;flex:1;position:relative}}
    .line-short{{flex:0 0 35px}} 

    .step-title{{font-size:20px;font-weight:700;margin-bottom:6px}}
    .step-sub{{color:#8B8F90;font-size:16px;min-height:22px;margin-top:8px}}
    .big-clock{{font-size:48px;color:#5A754D;font-weight:900;margin:0}}

    /* ▶ 진행바 영역 넓히기 */
    .progress-wrap{{position:relative;flex:3;min-width:420px;overflow: visible;}}  
    .progress-track{{position:relative;height:20px;background:#E9ECEB;border-radius:7px;overflow:hidden}}
    .progress-fill{{height:100%;background:#E07A18;width:{progress_percent}%;transition:width 0.5s linear}}

    /* 게이지 위 오토바이 */
    .progress-bike-img{{position:absolute;left:{progress_percent}%;bottom:100%;
                        transform:translateX(-50%);height:130px;width:auto;transition:left 0.5s linear;}}
    .progress-bike-emoji{{position:absolute;left:{progress_percent}%;bottom:100%;
                        transform:translateX(-50%);font-size:28px;line-height:1;transition:left 0.5s linear;}}
    .progress-label {{text-align:center;font-weight:500;margin-top:10px;}}
    </style>

    <div class="pipeline-container">
    <div class="step">
        <div class="step-title">현재 시각 (데이터 기준)</div>
        <div class="big-clock">{sim_now.strftime("%H:%M")}</div>
        {peak_text_html}
    </div>

    <div class="step-wrap" style="margin-left:24px;margin-right:24px;">
        <!-- 주문 수락 -->
        <div class="step">
        <div class="step-title">주문 수락됨</div>
        {accepted_badge}
        <div class="step-sub">{ot_str}</div>
        </div>

        <div class="line line-short"></div>

        <!-- 메뉴 준비 -->
        <div class="step">
        <div class="step-title">메뉴 준비</div>
        {prepared_badge}
        <div class="step-sub">{pk_str}</div>
        </div>

        <!-- 배달중(진행바 영역을 넓힘: .progress-wrap {{ flex:3 }}) -->
        <div class="step progress-wrap">
        {bike_node}
        <div class="progress-track">
            <div class="progress-fill"></div>
        </div>
        <div class="progress-label"> {pk_str} 배달 시작 / {remain_text}</div>
        </div>

        <!-- 배달 완료 -->
        <div class="step">
        <div class="step-title">배달 완료</div>
        {delivered_badge}
        <div class="step-sub">{dv_str}</div>
        </div>
    </div>
    </div>
    """

    components.html(pipeline_html, height=350, scrolling=False)

    ############### 배달 완료 오버레이 (트리거/문구 생성만 수정)
# ✅ 실제 배달 완료까지 걸린 시간(분) = Time_real (없으면 fallback)
if "Time_real" in sel and pd.notna(sel["Time_real"]):
    deliver_only_min = float(sel["Time_real"])
else:
    deliver_only_min = (max(0, total_min - prep_min)
                        if (not np.isnan(total_min) and not np.isnan(prep_min))
                        else None)

# 숫자 값과 텍스트 동시 준비
minutes_val = int(round(deliver_only_min)) if deliver_only_min is not None else None
minutes_text = str(minutes_val) if minutes_val is not None else "예상"

# ▶ upper_bound_min 과 비교해서 예정 대비 문구 만들기
subtitle = ""
if (minutes_val is not None) and ('upper_bound_min' in locals()) and (upper_bound_min is not None):
    try:
        extra_min = int(upper_bound_min) - int(minutes_val)  # (+) 일찍 / 0 정시 / (-) 늦게
        if extra_min > 0:
            subtitle = f"예상보다 {extra_min}분 일찍 도착했어요."
        elif extra_min == 0:
            subtitle = "예정 시간에 정확히 도착했어요."
        else:
            subtitle = f"예상보다 {abs(extra_min)}분 늦게 도착했어요."
    except Exception:
        # upper_bound_min이 숫자가 아니거나 minutes_val 변환 실패 시 안전하게 스킵
        subtitle = ""

# 완료 시 1회 오버레이
if delivered_done and (st.session_state.get("done_banner_for") != selected_id_clean):
    st.session_state["done_banner_for"] = selected_id_clean
    show_top_overlay_full(
        minutes_text=minutes_text if isinstance(minutes_text, (int, str)) else "예상",
        ele_data_url=ele_src,
        subtitle_text=subtitle
    )

# --- 3초마다 업데이트 (렌더 끝난 뒤 실행되도록 플래그만 세팅) ---
rerun_needed = False
if (pickup_dt and delivered_dt) and (sim_now < delivered_dt):
    st.session_state["sim_now"] = sim_now + timedelta(minutes=1)
    rerun_needed = True

# ========================= [BLOCK 9] 주의사항 =========================
st.caption("ℹ️ 경로는 Mapbox Directions(driving)로 계산된 '현재' 기준 도로 경로이며, 선 색상은 과거 도로 교통상황을 그대로 반영합니다.")


# ========================= [BLOCK 10] 상황별 평균 배달소요시간 (요약 차트) =========================
import plotly.express as px

st.markdown("---")
html_variables = """
            <div style="line-height: 1.2;">
                <h3 style='text-align: left; font-weight: bold; margin-bottom: -8px;'>
                    상황별 평균 배달소요시간
                </h3>
                <p style='text-align: left; color: #555; font-size:20px; margin-top: 0;'>
                    상황별로 과거에 평균적으로 소요된 배달 소요시간이에요.
                </p>
            </div>
            """
st.markdown(html_variables, unsafe_allow_html=True)
st.write("")

# Pastel palettes
PASTEL_ORANGES = ["#FAD7A0", "#F9CB9C", "#FFD1A6", "#FDE2B6"]
PASTEL_GREENS  = ["#CDECCF", "#BDE0C6", "#D4EDDA", "#C3E6CB"]

@st.cache_data(show_spinner=False)
def load_summary_dfs(data_dir: Path):
    dfs = {}
    dfs["city"]      = pd.read_csv(data_dir / "mean_time_by_City.csv",      encoding="utf-8-sig")
    dfs["peak"]      = pd.read_csv(data_dir / "mean_time_by_Peak_flag.csv", encoding="utf-8-sig")
    dfs["region"]    = pd.read_csv(data_dir / "mean_time_by_region.csv",    encoding="utf-8-sig")
    dfs["long"]      = pd.read_csv(data_dir / "mean_time_by_long_distance.csv", encoding="utf-8-sig")
    dfs["weather"]   = pd.read_csv(data_dir / "mean_time_by_Weatherconditions.csv", encoding="utf-8-sig")
    dfs["traffic"]   = pd.read_csv(data_dir / "mean_time_by_Road_traffic_density.csv", encoding="utf-8-sig")
    dfs["multiple"]  = pd.read_csv(data_dir / "mean_time_by_multiple_deliveries.csv", encoding="utf-8-sig")
    dfs["festival"]  = pd.read_csv(data_dir / "mean_time_by_Festival.csv",  encoding="utf-8-sig")
    return dfs

def _boolify_if_binary(s: pd.Series) -> pd.Series:
    vals = set(pd.Series(s).dropna().unique().tolist())
    if vals.issubset({0, 1}) or vals.issubset({0.0, 1.0}) or vals.issubset({"0", "1"}) or vals.issubset({0, 1, 0.0, 1.0}):
        return s.replace({0: False, 1: True, 0.0: False, 1.0: True, "0": False, "1": True})
    return s

def small_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "Time_real_mean",
    title: str = "",
    x_title: str = "",
    y_title: str = "평균 소요시간(분)",
    color: str | None = None,
    horizontal: bool = False,
    height: int = 300
):
    d = df.copy()
    # x 이진값이면 불리언으로 표시
    d[x_col] = _boolify_if_binary(d[x_col])
    # 정렬 및 카테고리 순서 고정
    d = d.sort_values(y_col, ascending=False)
    d[x_col] = pd.Categorical(d[x_col], categories=d[x_col], ordered=True)
    d["label_min"] = d[y_col].round().astype(int).astype(str) + "분"

    if horizontal:
        fig = px.bar(
            d, x=y_col, y=x_col, orientation="h",
            hover_data={y_col: ":.2f"},
            title=title
        )
    else:
        fig = px.bar(
            d, x=x_col, y=y_col, text="label_min",
            hover_data={y_col: ":.2f"},
            title=title
        )

    if color:
        fig.update_traces(marker_color=color, marker_line_color="rgba(0,0,0,0.08)", marker_line_width=1)

    # 라벨/폰트/여백
    fig.update_traces(texttemplate="%{text}", textposition="inside", textfont_size=14, cliponaxis=False)
    if horizontal:
        xmax = float(d[y_col].max())
        fig.update_xaxes(range=[0, xmax * 1.25], tickfont=dict(size=11), title_font=dict(size=13), title=y_title)
        fig.update_yaxes(tickfont=dict(size=11), title_font=dict(size=13), title=x_title)
    else:
        ymax = float(d[y_col].max())
        fig.update_xaxes(tickfont=dict(size=11), title_font=dict(size=13), title=x_title)
        fig.update_yaxes(range=[0, ymax * 1.25], tickfont=dict(size=11), title_font=dict(size=13), title=y_title)

    fig.update_layout(
        title=dict(x=0.5, xanchor="center"),
        title_font=dict(size=16),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height
    )
    return fig

_sum = load_summary_dfs(DATA_DIR)

# 1행(주황 계열)
r1 = st.columns(4)
with r1[0]:
    st.plotly_chart(
        small_bar(_sum["city"], x_col="City", title="도시유형별 평균 배달소요시간",
                  x_title="도시 유형", color=PASTEL_ORANGES[0]),
        use_container_width=True
    )
with r1[1]:
    st.plotly_chart(
        small_bar(_sum["peak"], x_col="Peak_flag", title="피크타임 평균 배달소요시간",
                  x_title="피크타임 여부", color=PASTEL_ORANGES[1]),
        use_container_width=True
    )
with r1[2]:
    st.plotly_chart(
        small_bar(_sum["region"], x_col="region", title="지역별 평균 배달소요시간",
                  x_title="지역", horizontal=True, color=PASTEL_ORANGES[2]),
        use_container_width=True
    )
with r1[3]:
    st.plotly_chart(
        small_bar(_sum["long"], x_col="long_distance", title="장거리(10km이상) 평균 배달소요시간",
                  x_title="장거리 여부", color=PASTEL_ORANGES[3]),
        use_container_width=True
    )

# 2행(초록 계열)
r2 = st.columns(4)
with r2[0]:
    st.plotly_chart(
        small_bar(_sum["weather"], x_col="Weatherconditions", title="날씨별 평균 배달소요시간",
                  x_title="날씨", color=PASTEL_GREENS[0]),
        use_container_width=True
    )
with r2[1]:
    st.plotly_chart(
        small_bar(_sum["traffic"], x_col="Road_traffic_density", title="교통상황별 평균 배달소요시간",
                  x_title="교통상황", color=PASTEL_GREENS[1]),
        use_container_width=True
    )
with r2[2]:
    st.plotly_chart(
        small_bar(_sum["multiple"], x_col="multiple_deliveries", title="동시배달 수 평균 배달소요시간",
                  x_title="동시배달 수", color=PASTEL_GREENS[2]),
        use_container_width=True
    )
with r2[3]:
    st.plotly_chart(
        small_bar(_sum["festival"], x_col="Festival", title="축제기간 평균 배달소요시간",
                  x_title="축제 여부", color=PASTEL_GREENS[3]),
        use_container_width=True
    )


if 'rerun_needed' in locals() and rerun_needed:
    time.sleep(1)
    st.rerun()
