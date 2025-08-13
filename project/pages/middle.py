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

        # Time_pred_class 컬럼이 있는지, 값이 유효한지 확인
        if 'Time_pred_class' in orders.columns and pd.notna(target_row['Time_pred_class']):
            pred_class = target_row['Time_pred_class']
            # time_map에서 예측 클래스에 해당하는 시간 범위 문자열을 가져옴
            time_range_str = time_map.get(pred_class, "계산 불가")
        else:
            time_range_str = "정보 없음"
        
        # 클래스를 실제 더할 시간(분)으로 매핑 (범위의 최소값 사용)
        minute_map = {
            key: int(value.split('~')[0]) for key, value in time_map.items()
        }

        # 1-1. 예상 소요 시간 (예: "10~14분")
        if 'Time_pred_class' in orders.columns and pd.notna(target_row['Time_pred_class']):
            pred_class = target_row['Time_pred_class']
            time_range_str = time_map.get(pred_class, "계산 불가")
        else:
            time_range_str = "정보 없음"

        # 1-2. 예상 도착 시각 (예: "오후 10시 33분 도착 예정") 또는 에러 메시지
        arrival_text = ""
        error_text = ""
        if 'Time_pred_class' in orders.columns and COL["pickup_time"] in orders.columns:
            pickup_time_dt = parse_datetime(target_row.get(COL["date"]), target_row.get(COL["pickup_time"]))

            if pickup_time_dt:
                pred_class = target_row['Time_pred_class']
                minutes_to_add = minute_map.get(pred_class, 0)
                estimated_arrival_time = pickup_time_dt + timedelta(minutes=minutes_to_add)
                arrival_text = f"{fmt_kor(estimated_arrival_time)} 도착 예정"
            else:
                error_text = "픽업 시간이 없어 도착 예정 시간을 계산할 수 없습니다."
        else:
            error_text = "예측에 필요한 컬럼이 없습니다."


        # 1-3. 준비된 변수들을 사용하여 하나의 HTML 블록으로 모든 정보를 한 번에 출력합니다.
        # 도착 시각이 정상적으로 계산되었는지, 아니면 에러가 발생했는지에 따라 세 번째 줄의 내용이 바뀝니다.
        if error_text:
            third_line_html = f"<h4 style='text-align: left; color: #FF4B4B; margin-top: 5px;'>{error_text}</h4>"
        else:
            third_line_html = f"<h5 style='text-align: left; margin-top: -5px;'>{arrival_text}</h5>"

        html_code = f"""
        <div style="line-height: 1.0;">
            <h3 style='text-align: left; font-weight: bold; margin-bottom: -20px;'>배달 예상 시간</h3>
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
        else:
            st.warning("차트를 표시할 예측 확률 데이터가 없습니다.")