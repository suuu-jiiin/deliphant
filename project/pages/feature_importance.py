import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from utils.paths import DATA_PATH, FONT_PATH

# 한국어 폰트 
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

# -----------------------------
# 1) 플랏
# -----------------------------
def plot_feature_bubbles_for_id(
    df: pd.DataFrame,
    id_value: str,
    *,
    id_col: str = "ID",
    prob_cols: list[str] | None = None,
    title: str | None = None,
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

        # 제목
        def _apply_title(fig, text, fontprop=None, size=24, weight="bold"):
            fig.subplots_adjust(top=0.86)
            if fontprop is not None:
                fp = fontprop.copy(); fp.set_size(size); fp.set_weight(weight)
                fig.suptitle(text, x=0.5, y=0.98, fontproperties=fp)
            else:
                fig.suptitle(text, x=0.5, y=0.98, fontsize=size, fontweight=weight)

        _title = title if title is not None else f"ID ({id_value})의 변수 중요도"
        _apply_title(fig, _title, fontprop=fontprop, size=24, weight="bold")

        # 여백
        pad = (base_radii.max() if len(base_radii) else 1.0) * 2.2
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
            # 일반 파이썬/노트북에서도 마지막 프레임 반환
            sf = 1.0
            fig, ax = draw_frame(sf)
            return fig, ax
    else:
        fig, ax = draw_frame(1.0)
        return fig, ax

# -----------------------------
# 2) 데이터 로드
# -----------------------------
df = pd.read_csv(DATA_PATH)  

# -----------------------------
# 3) 특정 ID로 그리기
# -----------------------------
id_value = "0x8b39"
ph = st.empty()
fig, _ = plot_feature_bubbles_for_id(
    df, id_value,
    fontprop=myfont,
    figsize=(14, 12),        # 캔버스 넓게
    ring_scale=1.1,         # 주변 원을 중앙에 바짝
    radius_scale=7.0,        # 원 자체를 크게
    gap_ratio=-0.20,         # 일부러 겹치도록(음수면 살짝 겹침 허용)
    gap_abs=-0.10,
    area_gamma=1.6,
    text_fontsize=18,        # 라벨도 큼직하게
    animate=True, frames=5, frame_delay=0.012,  # 더 빠른 애니메이션
    st_placeholder=ph
)