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
        df_fi = load_fi_csv("feature_importance.csv")
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

                # 상세보기 버튼 (센터)
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

                col1, col2, col3 = st.columns([0.5, 1, 1.5])
                with col2:
                    # ✅ key를 고유하게: 선택된 ID를 붙이면 충돌 없음
                    if st.button("상세 보기", use_container_width=True, key=f"detail_btn_{selected_id}"):
                        st.switch_page("pages/feature_importance.py")
