import streamlit as st
import re 
import time
import base64
import pandas as pd
import numpy as np
import matplotlib as mpl
from pathlib import Path
import matplotlib.pyplot as plt
from functools import lru_cache
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
from matplotlib import patheffects as pe
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from utils.paths import DATA_PATH, FONT_PATH, PREPROCESSED_PATH

# -----------------------------
# 0) 폰트
# -----------------------------
def use_korean_font(font_path):
    """로컬 TTF 등록 후 전역 기본 폰트 설정 + FontProperties 반환"""
    fm.fontManager.addfont(str(font_path))
    prop = FontProperties(fname=str(font_path))
    name = prop.get_name()
    mpl.rcParams.update({
        "font.family": name,
        "font.sans-serif": [name],  
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return prop

myfont = use_korean_font(FONT_PATH)

def inject_webfont(ttf_path: Path, css_family: str, weight: int = 400):
    """로컬 TTF를 base64로 임베드하고 css_family 이름으로 등록"""
    if not ttf_path or not Path(ttf_path).exists():
        return
    b64 = base64.b64encode(Path(ttf_path).read_bytes()).decode("utf-8")
    st.markdown(f"""
    <style>
      @font-face {{
        font-family: '{css_family}';
        src: url(data:font/ttf;base64,{b64}) format('truetype');
        font-weight: {weight};
        font-style: normal;
        font-display: swap;
      }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 1) 데이터 로드
# -----------------------------
meta_df = pd.read_csv(PREPROCESSED_PATH)
df = pd.read_csv(DATA_PATH)
icon_base = ROOT / "assets" / "icons" 

# -----------------------------
# 2) 실제 데이터 값
# -----------------------------
# 한국어 매핑
def map_weather(v):
    s = str(v).strip().lower()
    m = {
        "sunny":"맑음", "cloudy":"흐림", "fog":"안개", "sandstorms":"황사",
        "windy":"바람", "stormy":"폭풍"
    }
    return m.get(s, str(v))

def map_traffic(v):
    s = str(v).strip().lower()
    m = {"jam":"정체", "high":"혼잡", "medium":"서행", "low":"원활"}
    return m.get(s, str(v))

def map_multi(v):
    s = str(v).strip().lower()
    if s in {"1.0", "2.0", "3.0"}:  return "동시 배달"
    if s in {"0.0"}:  return "한 집 배달"
    return str(v)  # 숫자 2,3 등 특수 케이스 그대로

def is_multi_delivery(v) -> bool:
    """'1.0','2.0','3.0' → True, '0.0' → False (그 외는 값 보고 판단)"""
    s = str(v).strip().lower()
    if s in {"1", "1.0", "2", "2.0", "3", "3.0"}:
        return True
    if s in {"0", "0.0"}:
        return False
    # 혹시 숫자로 들어오면 0 초과면 동시배달로 간주
    try:
        return float(s) > 0.0
    except:
        return False

# -----------------------------
# 3) 버블 차트
# -----------------------------
def plot_feature_bubbles_for_id(
    df: pd.DataFrame,
    id_value: str,
    *,
    id_col: str = "ID",
    prob_cols: list[str] | None = None,
    figsize=(12, 10),
    area_gamma: float = 1.0,

    # === 스타일 파라미터 (조금 키움) ===
    ring_scale: float = 1.45,    # 중앙 원 대비 바깥 링 반지름 배율
    radius_scale: float = 5.00,  # 전체 원 크기 배율 
    gap_ratio: float = 0.73,     # 원 사이 상대 간격(겹침 제어)
    gap_abs: float   = 0.65,     # 원 사이 절대 간격(겹침 제어)

    # === 표현 옵션 ===
    cmap=None,
    fontprop=None,
    show_percent: bool = True,
    text_fontsize: int = 11,

    # === 애니메이션 옵션 ===
    animate: bool = True,
    frames: int = 6,
    frame_delay: float = 0.02,   # seconds
    st_placeholder=None,         # Streamlit이면 st.empty() 넘겨주기
):

    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' 컬럼을 찾을 수 없습니다.")

    if prob_cols is None:
        prob_cols = [c for c in df.columns if c != id_col]

    # --- 1) 해당 ID 행 추출 ---
    row_df = df.loc[df[id_col].astype(str).str.strip() == str(id_value).strip()]
    if row_df.empty:
        raise ValueError(f"ID '{id_value}'에 해당하는 행이 없습니다.")
    row = row_df.iloc[0][prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # --- 2) 확률 정규화 + 라벨 ---
    s = row.sum()
    probs = (row / s).to_numpy(dtype=float) if s > 0 else np.ones(len(row), dtype=float) / len(row)
    labels = [str(c) for c in prob_cols]


    def norm_key(s: str) -> str:
        return (
            s.strip().lower()
            .replace("_", " ").replace("-", " ")
        )
    
    kor_map = {
        "distance_km": "거리",
        "Road_traffic_density": "교통 상황",
        "region_city": "지역",
        "Weatherconditions": "날씨",
        "multiple_deliveries": "동시 배달",
    }

    # 내림차순 정렬 → 가장 큰 값이 중앙
    order = np.argsort(-probs)
    probs = probs[order]
    labels = [labels[i] for i in order]
    n = len(probs)

    kor_map_norm = {norm_key(k): v for k, v in kor_map.items()}

    # 정렬/순서 적용 이후 labels에 대해 한국어 라벨 생성
    labels_ko = [kor_map_norm.get(norm_key(s), s) for s in labels]

    # --- 3) 반지름(면적 ∝ 확률) ---
    adj = np.power(probs, area_gamma)
    k = 12.0 / np.sqrt(probs.max() + 1e-12)
    base_radii = (k * np.sqrt(adj)) * radius_scale  # scale_factor=1일 때의 반지름

    # 중앙(최대) 원 강조
    center_boost = 1.00   # ← 1.0~1.25 사이로 취향껏
    base_radii[0] *= center_boost

    # --- 4) 좌표 배치: 중앙(0,0), 나머지 링에 균등 ---
    cx, cy = [0.0], [0.0]
    if n > 1:
        ring_radius = base_radii[0] * ring_scale
        angles = np.linspace(0, 2*np.pi, n-1, endpoint=False)
        for ang in angles:
            cx.append(ring_radius * np.cos(ang))
            cy.append(ring_radius * np.sin(ang))

    # --- 5) 겹침 해소(최종 반지름 기준으로 한 번 계산) ---
    def resolve_overlaps(cx, cy, r, gap_ratio=0.25, gap_abs=0.4, steps=1200, lr=0.006):
        cx = np.array(cx, dtype=float)
        cy = np.array(cy, dtype=float)
        r  = np.array(r,  dtype=float)
        for _ in range(steps):
            moved = False
            # (A) 쌍별 충돌 해결
            for i in range(len(cx)):
                for j in range(i+1, len(cx)):
                    dx, dy = cx[j]-cx[i], cy[j]-cy[i]
                    dist = np.hypot(dx, dy) + 1e-12
                    min_dist = (r[i] + r[j]) * (1.0 + gap_ratio) + gap_abs
                    if dist < min_dist:
                        push = (min_dist - dist) * lr
                        ux, uy = dx/dist, dy/dist
                        if i == 0 and j != 0:   # 중앙은 고정
                            cx[j] += ux * push; cy[j] += uy * push
                        elif j == 0 and i != 0:
                            cx[i] -= ux * push; cy[i] -= uy * push
                        else:
                            cx[i] -= ux * (push * 0.5); cy[i] -= uy * (push * 0.5)
                            cx[j] += ux * (push * 0.5); cy[j] += uy * (push * 0.5)
                        moved = True
            # (B) 바깥으로 미세 드리프트
            for i in range(1, len(cx)):
                vx, vy = cx[i], cy[i]
                norm = np.hypot(vx, vy) + 1e-12
                cx[i] += (vx / norm) * lr * 0.3
                cy[i] += (vy / norm) * lr * 0.3
            if not moved:
                break
        return cx, cy

    cx, cy = resolve_overlaps(cx, cy, base_radii, gap_ratio=gap_ratio, gap_abs=gap_abs)

    # --- 6) 색상맵(큰 값일수록 진하게) ---
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "custom_green", ["#dfecc4", "#c7ceb8", "#9eac85", "#9da982", "#758950"]
        )
    v = (probs - probs.min()) / (probs.max() - probs.min() + 1e-12)
    colors = cmap(0.15 + 0.85 * v)

    # --- 7) 이징 함수 + 프레임 드로어 ---
    def ease_out_cubic(t: float) -> float:
        return 1 - (1 - t) ** 3

    def draw_frame(scale_factor: float):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal"); ax.axis("off")

        # 중앙 제외 먼저
        for i in range(1, n):
            circle = plt.Circle(
                (cx[i], cy[i]),
                base_radii[i] * scale_factor,
                facecolor=colors[i], edgecolor="none",
                linewidth=0, alpha=0.86, zorder=1
            )
            ax.add_patch(circle)

        # 중앙(가장 큰 원) 마지막
        if n >= 1:
            circle_c = plt.Circle(
                (cx[0], cy[0]),
                base_radii[0] * scale_factor,
                facecolor=colors[0], edgecolor="none",
                linewidth=0, alpha=0.92, zorder=5
            )
            ax.add_patch(circle_c)

        def _fp(base_fp: FontProperties | None, size: int, weight: str="bold") -> FontProperties | None:
            if base_fp is None:
                return None
            fp = base_fp.copy()
            fp.set_size(size)
            fp.set_weight(weight)
            return fp


        # 라벨용/중앙라벨용 폰트
        fp_label  = _fp(fontprop, text_fontsize, "bold")
        fp_center = _fp(fontprop, text_fontsize + 2, "bold")

        # --- 버블 라벨들 ---
        for i in range(1, n):
            txt = f"{labels_ko[i]}\n{probs[i]*100:.1f}%" if show_percent else labels_ko[i]
            if fp_label is not None:
                ax.text(cx[i], cy[i], txt, ha="center", va="center",
                        color="#111", fontproperties=fp_label, zorder=6)
            else:
                ax.text(cx[i], cy[i], txt, ha="center", va="center",
                        fontsize=text_fontsize, fontweight="bold", color="#111", zorder=6)

        # 중앙(최대) 라벨
        txt0 = f"{labels_ko[0]}\n{probs[0]*100:.1f}%" if show_percent else labels_ko[0]
        if fp_center is not None:
            ax.text(cx[0], cy[0], txt0, ha="center", va="center",
                    color="#111", fontproperties=fp_center, zorder=7)
        else:
            ax.text(cx[0], cy[0], txt0, ha="center", va="center",
                    fontsize=text_fontsize+2, fontweight="bold", color="#111", zorder=7)

        # 여백
        pad = (base_radii.max() if len(base_radii) else 1.0) 
        ax.set_xlim(min(cx) - pad, max(cx) + pad)
        ax.set_ylim(min(cy) - pad, max(cy) + pad)

        return fig, ax

    # --- 8) 렌더링 ---
    if animate:
        # Streamlit placeholder가 있으면 거기에 애니메이션 렌더
        if st_placeholder is not None:
            last_fig = None
            for f in range(frames):
                t = ease_out_cubic((f + 1) / frames)
                sf = max(0.1, t)
                fig, _ = draw_frame(sf)
                st_placeholder.pyplot(fig)
                last_fig = fig
                plt.close(fig)
                time.sleep(frame_delay)
            return last_fig, None
        else:
            sf = 1.0
            fig, ax = draw_frame(sf)
            return fig, ax
    else:
        fig, ax = draw_frame(1.0)
        return fig, ax


# -----------------------------
# 4) 칩 (실제 데이터 값)
# -----------------------------
def _norm(s): 
    return str(s).strip().lower().replace(" ", "_")

def extract_meta(meta_df: pd.DataFrame, id_value: str, id_col="ID") -> dict:
    row = meta_df.loc[meta_df[id_col].astype(str).str.strip()==str(id_value).strip()]
    if row.empty:
        return {}

    r = row.iloc[0]
    out = {}

    # 원본 값
    out["distance_km"] = float(r.get("distance_km", float("nan")))
    out["region_city"] = str(r.get("region_city", "NA"))
    out["multiple_deliveries"] = r.get("multiple_deliveries", "NA")

    w_raw = r.get("Weatherconditions", "NA")
    t_raw = r.get("Road_traffic_density", "NA")

    # 표시 텍스트(한글)
    out["weather_txt"] = map_weather(w_raw)
    out["traffic_txt"] = map_traffic(t_raw)
    out["multi_txt"]   = map_multi(out["multiple_deliveries"])

    # 아이콘 파일명 키
    out["weather_key"] = _norm(w_raw)     # fog, sunny, ...
    out["traffic_key"] = _norm(t_raw)     # low, medium, high, jam

    return out

# 로컬 PNG -> DATA URI
def img_to_data_uri(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = path.suffix.lower().lstrip(".") or "png"
        mime = "png" if ext == "png" else "jpeg"
        return f"data:image/{mime};base64,{b64}"
    except Exception:
        return None

def render_meta_chips(meta: dict, icon_dir: Path):
    """제목 아래에 아이콘+텍스트 칩을 한 줄로 렌더"""
    if not meta:
        return

    chips = []

    # (라벨, 값, 아이콘 경로)
    multi_val  = meta.get("multiple_deliveries", "NA")
    multi_icon = icon_dir / "delivery" / ("multiple.png" if is_multi_delivery(multi_val) else "only.png")

    chips.append(("교통", meta.get("traffic_txt","NA"),
                  icon_dir / "traffic" / f"{meta.get('traffic_key','')}.png"))
    chips.append(("날씨", meta.get("weather_txt","NA"),
                  icon_dir / "weather" / f"{meta.get('weather_key','')}.png"))
    chips.append(("거리", f"{meta.get('distance_km', float('nan')):.1f} km" if pd.notna(meta.get('distance_km')) else "NA",
                  icon_dir / "distance" / "distance.png"))
    chips.append(("배달 수",
              meta.get("multi_txt", "NA"),   # 이미 '동시 배달' / '한 집 배달' 매핑되어 있으면 이게 더 보기 좋아요
              multi_icon))
    chips.append(("지역", meta.get("region_city","NA"),
                  icon_dir / "misc" / "city.png"))

    parts = []
    for label, value, p in chips:
        img_html = ""
        if p and p.exists():
            uri = img_to_data_uri(p)
            if uri:
                img_html = f'<img src="{uri}" class="chip-icon" />'
        parts.append(
            f'<span class="chip">{img_html}<span class="chip-text">{label}: {value}</span></span>'
        )

    html = f"""
    <style>
      .page-title {{
        font-family: 'TitleFont', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 700; font-size: 42px; line-height: 1.25;
        text-align: center; margin: 6px 0 8px 0;
      }}
      .chip-row {{
        display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
        margin: 6px 0 14px;
        justify-content: center;
      }}
      .chip {{
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 12px; border-radius: 14px;
        background: #ffffff; border: 1px solid #c9d2b3;
      }}
      .chip-icon {{ width: 18px; height: 18px; }}
      .chip-text {{
        font-family: 'BodyFont', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 15px; color: #263018; font-weight: 500;
      }}
    </style>
    <div class="chip-row">{''.join(parts)}</div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -----------------------------
# 5) 특정 ID로 그리기
# -----------------------------
def clean_id(x: object) -> str:
    s = str(x)
    s = s.replace("\u00A0", " ").replace("\ufeff", "").replace("\u200b", "")
    return s.strip()

selected_id = st.session_state.get("selected_id")
if not selected_id:
    st.warning("선택된 주문 ID가 없습니다. 먼저 홈 화면에서 주문을 선택하세요.")
    st.stop()

id_value = clean_id(selected_id)

st.markdown(f"<div class='page-title'>⭐️ ID ({id_value})의 변수 중요도 ⭐️</div>", unsafe_allow_html=True)
meta = extract_meta(meta_df, id_value)
render_meta_chips(meta, icon_base)

# 그 다음 버블 플롯 (아이콘은 플롯 안에서는 끄기 추천)
fig, _ = plot_feature_bubbles_for_id(
    df, id_value,
    fontprop=myfont,
    animate=True, frames=5, frame_delay=0.012,
    ring_scale=1.08, radius_scale=6.0,
    gap_ratio=-0.05, gap_abs=0.0,
    area_gamma=1.6, text_fontsize=18,
    st_placeholder=st.empty()
)
