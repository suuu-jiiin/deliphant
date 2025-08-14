# ========================= [BLOCK 7] 하단 파이프라인 =========================
import streamlit.components.v1 as components
import time
from pathlib import Path

st.markdown("---")
st.subheader("주문 파이프라인")

if sel is None:
    st.info("주문을 선택하세요.")
else:
    # --- 시간 파싱 ---
    order_dt   = parse_datetime(sel[COL["date"]], sel[COL["order_time"]])
    pickup_dt  = parse_datetime(sel[COL["date"]], sel[COL["pickup_time"]])
    prep_min   = float(sel[COL["prep_min"]])  if pd.notna(sel[COL["prep_min"]])  else np.nan
    total_min  = float(sel[COL["total_min"]]) if pd.notna(sel[COL["total_min"]]) else np.nan

    if "Time_real" in sel and pd.notna(sel["Time_real"]):
        deliver_only_min = float(sel["Time_real"])
    else:
        deliver_only_min = (max(0, total_min - prep_min)
                            if (not np.isnan(total_min) and not np.isnan(prep_min))
                            else None)

    if (pickup_dt is None) and (order_dt is not None) and (not np.isnan(prep_min)):
        pickup_dt = order_dt + timedelta(minutes=prep_min)

    delivered_dt = (pickup_dt + timedelta(minutes=deliver_only_min)) if (pickup_dt and deliver_only_min is not None) \
                   else (order_dt + timedelta(minutes=total_min) if (order_dt and not np.isnan(total_min)) else None)

    # --- 시뮬 시계 (3초=1분) ---
    if "sim_now" not in st.session_state or st.session_state.get("sim_id") != sel[COL["id"]]:
        st.session_state["sim_id"] = sel[COL["id"]]
        st.session_state["sim_now"] = pickup_dt if pickup_dt else datetime.now()

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

    # --- 오토바이 이미지 ---
    bike_image_path = Path("assets/icons/home/elephant_perseon.png" )
    try:
        if bike_image_path.exists():
            bike_img_url = local_image_to_data_url(bike_image_path)
        else:
            bike_img_url = None
    except Exception:
        bike_img_url = None

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
    # --- HTML ---
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

    # --- 3초마다 업데이트 ---
    if (pickup_dt and delivered_dt) and (sim_now < delivered_dt):
        st.session_state["sim_now"] = sim_now + timedelta(minutes=1)
        time.sleep(3)
        st.rerun()
