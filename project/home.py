# app_main_local_density_route.py
# 단일 주문 선택 → CSV Road_traffic_density 색상으로 "실제 도로 경로" 표시 (Mapbox driving, no traffic)
# 업로드 UI 없음, 로컬 CSV 사용, 하단 파이프라인 유지

# ========================= [BLOCK 1] 기본 설정 & 라이브러리 =========================
import folium
import time
import textwrap
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
ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.paths import LOCAL_CSV_PATH

st.set_page_config(page_title="배달 예측(메인)", layout="wide")
# 🔝 타이틀 위 전용 슬롯
FX_SLOT = st.container()

st.title("🚚 배달 예측 대시보드")

# ========================= [BLOCK 2] 전역 상수(토큰/파일/컬럼/색상) =========================
MAPBOX_TOKEN   = ""
target_ids = ['0x9d32', '0x23d4', '0x8b39', '0xce01', '0x8fdd', '0x7ab9', '0x6f80', '0xa512', '0xd740', '0xb478', '0xd200', '0x2a85', '0x1ef8', '0x972b']

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
def load_orders(path: str, filter_ids: list, col_map: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    # isin()을 사용하여 ID 필터링
    df[col_map["id"]] = df[col_map["id"]].astype(str).str.strip()
    df = df[df[col_map["id"]].isin(filter_ids)].reset_index(drop=True)

    # 문자열 시간 파싱을 위해 공백/결측 정리
    for c in [col_map["date"], col_map["order_time"], col_map["pickup_time"]]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

# 함수 호출 시 인수를 명확하게 전달
orders = load_orders(LOCAL_CSV_PATH, target_ids, COL)

##### 이미지 로드 #####
def local_image_to_data_url(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

start_icon_path = "assets/icons/map/resturant.png" 
end_icon_path   = "assets/icons/map/home.png"
ele_path = "assets/icons/home/elephant.png"
bike_image_path = "assets/icons/home/elephant_person.png" 

start_icon_url = local_image_to_data_url(start_icon_path)
end_icon_url   = local_image_to_data_url(end_icon_path)
ele_src  = local_image_to_data_url(ele_path)
bike_img_url = local_image_to_data_url(bike_image_path)

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
st.markdown("### 🔎 주문 선택 & 정보")

# ID 컬럼 먼저 정규화(앞뒤 공백/눈에 안 보이는 공백 제거)
def clean_id(x: object) -> str:
    s = str(x)
    # 흔한 보이지 않는 공백 제거(\u00A0=non-breaking space, \ufeff=BOM, \u200b=zero-width space)
    s = s.replace("\u00A0", " ").replace("\ufeff", "").replace("\u200b", "")
    return s.strip()

# 좌우로 분할: 왼쪽(좁게) = 주문 선택, 오른쪽(넓게) = 매장/배달원 정보
sel_left, sel_right = st.columns([1.0, 2.2])

with sel_left:
    orders[COL["id"]] = orders[COL["id"]].apply(clean_id)
    order_ids = orders[COL["id"]].tolist()
    default_idx = len(order_ids) - 1 if order_ids else 0
    selected_id = st.selectbox("주문 ID", order_ids, index=len(order_ids) - 1 if order_ids else 0, format_func=lambda x: clean_id(x))

# 선택된 행
selected_id_clean = clean_id(selected_id)
sel = orders[orders[COL["id"]] == selected_id_clean].iloc[0] if order_ids else None
st.session_state["selected_id"] = selected_id_clean

with sel_right:
    # 값 준비
    if sel is None:
        region = rname = courier_id = courier_age = courier_rating = "—"
    else:
        region = sel.get(COL["region"], "—") if COL["region"] in sel else "—"
        rname  = sel.get(COL["restaurant_name"], "—") if COL["restaurant_name"] in sel else "—"

        courier_id_raw     = sel.get(COL["courier_id"], "—") if COL["courier_id"] in sel else "—"
        courier_age_raw    = sel.get(COL["courier_age"], "—") if COL["courier_age"] in sel else "—"
        courier_rating_raw = sel.get(COL["courier_rating"], "—") if COL["courier_rating"] in sel else "—"

        # 나이 → 연령대 변환
        try:
            age_int = int(float(courier_age_raw)) if pd.notna(courier_age_raw) else None
            if age_int is not None:
                decade = (age_int // 10) * 10
                courier_age = f"{decade}대"
            else:
                courier_age = "—"
        except Exception:
            courier_age = "—"


        try:
            courier_rating = round(float(courier_rating_raw), 2) if pd.notna(courier_rating_raw) else "—"
        except Exception:
            courier_rating = courier_rating_raw if str(courier_rating_raw).strip().lower() != "nan" else "—"

        courier_id = courier_id_raw if str(courier_id_raw).strip() else "—"

    # 회색 박스 하나만 렌더
    panel_html = f"""
    <style>
      .info-panel {{
        background:#f2f2f2; padding:16px 18px; border-radius:10px;
      }}
      .section-title {{ margin:0 0 10px 0; font-weight:700; font-size:20px; }}
      .grid-2 {{ display:grid; grid-template-columns: 1fr 2fr; gap:12px; }}
      .grid-3 {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; }}
      .info-card {{
        background:#ffffff; border:1px solid #e3e3e3; border-radius:8px; padding:10px 12px;
      }}
      .label {{ color:#70757a; font-size:14px; font-weight:600; margin-bottom:4px; }}
      .value {{ font-size:15x; font-weight:700; }}
    </style>

    <div class="info-panel">
      <div class="section-title">🍽️ 매장 정보</div>
      <div class="grid-2" style="margin-bottom:14px;">
        <div class="info-card">
          <div class="label">지역</div>
          <div class="value">{region}</div>
        </div>
        <div class="info-card">
          <div class="label">매장명</div>
          <div class="value">{rname}</div>
        </div>
      </div>

      <div class="section-title">🛵 배달원 정보</div>
      <div class="grid-3">
        <div class="info-card">
          <div class="label">배달원 ID</div>
          <div class="value">{courier_id}</div>
        </div>
        <div class="info-card">
          <div class="label">나이</div>
          <div class="value">{courier_age}</div>
        </div>
        <div class="info-card">
          <div class="label">평점</div>
          <div class="value">{courier_rating}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(panel_html, unsafe_allow_html=True)

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
    st.toast("축제 기간이라 배달이 늦어지고 있어요 🎉", icon="🎆")
else:
    # 축제가 아니면 슬롯 비우기(이전 렌더 지우기)
    FX_SLOT.empty()

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

####### 오버레이 함수
def show_top_overlay_between(start_id: str, end_id: str, minutes_text: str, ele_data_url: str = ""):
    # components.html 안에서 parent 문서를 건드려서(동일 출처) body에 고정 오버레이를 붙입니다.
    card_html = f"""
    <div style="
      background: rgba(0,0,0,0.65);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 28px;
      padding: 28px 36px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
      display: flex; flex-direction: column; align-items: center; gap: 14px;
      min-width: 520px; max-width: 86%;
      color: #fff; text-align: center; font-weight: 800;">
      {f"<img src='{ele_data_url}' alt='elephant' style='width:180px;height:auto;' />" if ele_data_url else ""}
      <div style="font-size:20px; font-weight:700;">{minutes_text}분 만에 배달이 완료되었어요 <span style="font-size:22px">☺️</span></div>
      <div style="font-size:13px; opacity:.85; font-weight:600;">(화면을 클릭하면 닫혀요)</div>
    </div>
    """
    components.html(f"""
    <div></div>
    <script>
    (function(){{
      const startId = "{start_id}";
      const endId   = "{end_id}";

      function mount(){{
        const doc = window.parent?.document || document;
        const s = doc.getElementById(startId);
        const e = doc.getElementById(endId);
        if(!s || !e) {{ setTimeout(mount, 120); return; }}

        const r1 = s.getBoundingClientRect();
        const r2 = e.getBoundingClientRect();
        const top = r1.top + window.parent.scrollY;
        const height = (r2.bottom + window.parent.scrollY) - top;

        let ov = doc.getElementById("dlp-top-overlay");
        if(!ov){{
          ov = doc.createElement("div");
          ov.id = "dlp-top-overlay";
          doc.body.appendChild(ov);
          // 기본 스타일 + 페이드인
          Object.assign(ov.style, {{
            position: "fixed",
            left: "0px",
            top: top + "px",
            width: "100vw",
            height: height + "px",
            background: "rgba(0,0,0,0.60)",
            zIndex: "999999",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "opacity .18s ease",
            opacity: "0"
          }});
          // 다음 프레임에 불투명
          requestAnimationFrame(() => ov.style.opacity = "1");
        }} else {{
          ov.style.top = top + "px";
          ov.style.height = height + "px";
        }}

        ov.innerHTML = `{card_html.replace("`","\\`")}`;

        // ===== 닫기 핸들러 =====
        const remove = () => {{
          ov.style.opacity = "0";
          setTimeout(() => ov && ov.remove && ov.remove(), 200);
        }};
        ov.onclick = remove;  // 화면 어느 곳을 클릭해도 닫힘
        doc.addEventListener("keydown", (ev) => {{
          if (ev.key === "Escape") remove();
        }}, {{ once: true }});

        // (선택) 자동 닫기 원하면 주석 해제
        // setTimeout(remove, 4000);
      }}
      mount();
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
                    st.caption(f"경로 길이: {km:.2f} km  |  색상 근거: Road_traffic_density = {traffic_val}")
                else:
                    st.caption(f"색상 근거: Road_traffic_density = {traffic_val}")

    # ---- 중: ETA 통계 (CSV 기반)
    with mid_col:
        # 전체 orders 데이터프레임이 비어있지 않은 경우에만 실행
        if not orders.empty:
            # **수정된 부분**: `st.selectbox`에서 선택된 ID에 맞는 행을 가져옵니다.
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
                    # str 타입이므로 float으로 변환
                    pred_class = float(target_row['max_after_class_key'])
                except (ValueError, TypeError):
                    # 변환 실패 시 None
                    pass
            
            # 클래스를 실제 더할 시간(분)으로 매핑 (범위의 최소값 사용)
            minute_map = {
                key: int(value.split('~')[0]) for key, value in time_map.items()
            }

            # 1-1. 예상 소요 시간 (예: "10~14분")
            if pred_class is not None:
                time_range_str = time_map.get(pred_class, "계산 불가")
            else:
                time_range_str = "정보 없음"

            # 1-2. 예상 도착 시각 (예: "오후 10시 33분 도착 예정") 또는 에러 메시지
            arrival_text = ""
            error_text = ""
            if pred_class is not None and COL["pickup_time"] in orders.columns:
                pickup_time_dt = parse_datetime(target_row.get(COL["date"]), target_row.get(COL["pickup_time"]))

                if pickup_time_dt:
                    minutes_to_add = minute_map.get(pred_class, 0)
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

            html_code = f"""
            <div style="line-height: 1.0;">
                <h3 style='text-align: left; font-weight: bold; margin-bottom: -20px;'>배달 예상 소요 시간</h3>
                <h1 style='text-align: left; color: #1E90FF; margin-top: -20px;'>{time_range_str}</h1>
                {third_line_html}
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)

            # 2. 상태 메시지 출력
            st.write("주문하신 곳으로 가고 있어요. 🛵")
            st.write("") 

            # 3. 가로 막대 그래프 생성 (시간대 텍스트만, 값 라벨 표시, x축 숨김)
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
                .stButton button {
                    background-color: #f0f2f6;
                    color: #000000;
                    border-radius: 20px;
                    border: 1px solid #dcdcdc;
                    padding: 10px 20px;
                    font-size: 16px;
                    font-weight: bold;
                    width: 100%;
                }
                </style>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("상세 보기", use_container_width=True):
                        st.switch_page("pages/prob_distribution.py")
            else:
                st.warning("차트를 표시할 예측 확률 데이터가 없습니다.")

    # ---- 우: 변수 중요도 (박스 제거 + 세로 간격 줄임 + 칼럼명 볼드 제거 + 상세보기 버튼 중앙)
    with right_col:
        html_code = """
            <div style="line-height: 1.2;">
                <h3 style='text-align: left; font-weight: bold; margin-bottom: -8px;'>
                    변수 중요도
                </h3>
                <p style='text-align: left; color: #555; font-size:20px; margin-top: 0;'>
                    예상시간에 영향을 끼치고 있는 변수들이에요.
                </p>
            </div>
            """
        st.markdown(html_code, unsafe_allow_html=True)


        @st.cache_data
        def load_fi_csv(path: str) -> pd.DataFrame:
            df_fi = pd.read_csv(path)
            df_fi.columns = [str(c).strip() for c in df_fi.columns]
            return df_fi

        try:
            feat_path = "data/feature_importance.csv"
            df_fi = load_fi_csv(feat_path)
        except Exception as e:
            st.error(f"feature_importance.csv 로드 실패: {e}")
            st.stop()

        if "ID" not in df_fi.columns:
            st.error("feature_importance.csv에 'ID' 컬럼이 없습니다.")
        else:
            row = df_fi[df_fi["ID"] == selected_id]
            if row.empty:
                st.warning("선택한 ID에 대한 변수 중요도 데이터가 없습니다.")
            else:
                row = row.iloc[0]

                use_cols = [
                    ("distance_km",          "distance"),
                    ("Weatherconditions",    "Weather"),
                    ("region_city",          "region"),
                    ("multiple_deliveries",  "multiple"),
                    ("Road_traffic_density", "Traffic"),
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
                    row_h = 24
                    total_h = max(80, len(chart_df) * row_h)

                    # 왼쪽 라벨 (볼드 제거)
                    left_labels = (
                        alt.Chart(chart_df)
                        .mark_text(
                            align="right",
                            baseline="middle",
                            fontSize=14,
                            fontWeight="normal",
                            dx=5, 
                            color=COLOR_LABEL
                        )
                        .encode(y=y_enc, text="feature:N")
                        .properties(width=20, height=total_h)
                    )

                    base = alt.Chart(chart_df).encode(y=y_enc)
                    track = base.mark_bar(size=10, color=COLOR_TRACK, cornerRadius=999).encode(
                        x=alt.X("track:Q", scale=alt.Scale(domain=[0,100]), axis=None, title=None)
                    ).properties(width=140, height=total_h)
                    fill = base.mark_bar(size=10, color=COLOR_FILL, cornerRadius=999).encode(
                        x=alt.X("value:Q", scale=alt.Scale(domain=[0,100]), axis=None, title=None)
                    ).properties(width=140, height=total_h)

                    middle = track + fill

                    right_values = (
                        alt.Chart(chart_df)
                        .mark_text(align="left", baseline="middle", fontSize=14, fontWeight="bold", dx=6, color=COLOR_PCT)
                        .encode(y=y_enc, text="pct_str:N")
                        .properties(width=10, height=total_h)
                    )

                    chart_comp = alt.hconcat(left_labels, middle, right_values).resolve_scale(y="shared")
                    st.altair_chart(chart_comp, use_container_width=True)

                    # 상세보기 버튼 (우)
                    st.session_state['selected_id'] = selected_id
                    st.markdown("""
                    <style>
                    .stButton button {
                        white-space: nowrap;           /* 줄바꿈 금지 */
                        word-break: keep-all;          /* 한글도 단어 단위로 */
                        background-color: #f0f2f6;
                        color: #000000;
                        border-radius: 20px;
                        border: 1px solid #dcdcdc;
                        padding: 10px 20px;
                        font-size: 16px;
                        font-weight: bold;
                        width: 100%; max-width: 320px; /* 충분한 폭 고정 */
                        display: block; margin: 6px auto; /* 가운데 정렬 */
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns([1,2,1])
                    with c2:
                        # ✅ key를 고유하게: 선택된 ID를 붙이면 충돌 없음
                        if st.button("상세 보기", key=f"detail_btn_{selected_id_clean}"):
                            # 세션/쿼리파라미터에 ID 저장 (둘 다 써도 OK)
                            st.session_state["selected_id"] = selected_id_clean
                            st.query_params.update({"id": selected_id_clean})
                            st.switch_page("pages/feature_importance.py")

    st.markdown(f"<div id='{end_id}'></div>", unsafe_allow_html=True)

# ========================= [BLOCK 8] 하단 파이프라인 (선택 주문) =========================
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

    eta_remain_min = None
    if pickup_dt and delivered_dt and delivered_dt > pickup_dt:
        total_sec   = (delivered_dt - pickup_dt).total_seconds()
        elapsed_sec = (sim_now - pickup_dt).total_seconds()
        progress_pct = max(0.0, min(elapsed_sec / total_sec, 1.0))
        if 0 <= progress_pct < 1:
            eta_remain_min = max(0, int(round((1 - progress_pct) * total_sec / 60)))
    remain_text = (
        f"남은 시간 약 {eta_remain_min}분" if eta_remain_min is not None
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
        display:flex;justify-content:space-between;align-items:center;
    }}
    .step-wrap{{display:flex;align-items:center;gap:48px;margin-top:10px;margin-bottom:10px;flex:1}}
    .step{{text-align:center;flex-shrink:0}}
    .badge{{width:82px;height:82px;border-radius:50%;background:#5A754D;color:white;
           display:flex;align-items:center;justify-content:center;font-weight:800;font-size:28px;
           box-shadow:inset -6px -6px 0 rgba(0,0,0,0.15)}}
    .badge-empty{{width:82px;height:82px;border-radius:50%;border:10px solid #5A615D;background:#fff}}
    .line{{height:10px;background:#5A615D;flex:1;position:relative}}
    .step-title{{font-size:20px;font-weight:700;margin-bottom:6px}}
    .step-sub{{color:#8B8F90;font-size:16px;min-height:22px;margin-top:8px}}
    .big-clock{{font-size:48px;color:#5A754D;font-weight:900;margin:0}}

    /* 게이지 & 오토바이 */
    .progress-wrap{{position:relative;min-width:360px;}}
    .progress-track{{position:relative;height:20px;background:#E9ECEB;border-radius:7px;overflow:hidden}}
    .progress-fill{{height:100%;background:#E07A18;width:{progress_percent}%;transition:width 0.5s linear}}
    /* 게이지 바로 위에 붙임 */
    .progress-bike-img{{position:absolute;left:{progress_percent}%;bottom:100%;
                        transform:translateX(-50%);height:70px;width:auto;transition:left 0.5s linear;}}
    .progress-bike-emoji{{position:absolute;left:{progress_percent}%;bottom:100%;
                          transform:translateX(-50%);font-size:28px;line-height:1;transition:left 0.5s linear;}}
    .progress-label {{text-align:center;font-weight:500;margin-top:10px;}}

    </style>

    <div class="pipeline-container">
      <div class="step">
        <div class="step-title">현재 시각 (픽업 기준)</div>
        <div class="big-clock">{sim_now.strftime("%H:%M")}</div>
      </div>

      <div class="step-wrap" style="margin-left:24px;margin-right:24px;">
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

        <!-- 배달중 -->
        <div class="step progress-wrap">
          
          {bike_node}
          <div class="progress-track">
            <div class="progress-fill"></div>
          </div>
          <div class="progress-label"> {pk_str} 배달 시작 / {remain_text}</div>
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
    components.html(pipeline_html, height=260, scrolling=False)

    # ===== 오버레이 트리거: 하단 진행바(세션 시계) 기준 =====
    minutes_text = str(n_min if n_min is not None else "예상")
    if delivered_done and (st.session_state.get("done_banner_for") != selected_id_clean):
        st.session_state["done_banner_for"] = selected_id_clean
        show_top_overlay_between(start_id, end_id, minutes_text, ele_src)

    # --- 3초마다 업데이트 ---
    if (pickup_dt and delivered_dt) and (sim_now < delivered_dt):
        st.session_state["sim_now"] = sim_now + timedelta(minutes=1)
        time.sleep(3)
        st.rerun()

# ========================= [BLOCK 9] 주의사항 =========================
st.caption("ℹ️ 경로는 Mapbox Directions(driving)로 계산된 '현재' 기준 도로 경로이며, 선 색상은 CSV의 Road_traffic_density 값을 그대로 반영합니다(실시간 교통 미사용).")

