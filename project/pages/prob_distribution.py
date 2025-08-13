import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast

st.set_page_config(page_title="확률분포 대시보드")
st.title("예상 배달소요시간별 확률 상세보기")

# --- 데이터 로드 및 준비 ---
@st.cache_data
def load_data():
    """데이터를 로드하고 캐싱합니다."""
    try:
        df = pd.read_csv('pages/prob_distribution.csv')
    except FileNotFoundError:
        st.warning("⚠️ prob_distribution.csv 파일을 찾을 수 없어 예제 데이터로 실행합니다.", icon="⚠️")
        data = {
            'ID': [f'ID_{i}' for i in range(1, 11)],
            '1.0': [0.1, 0.2, 0.05, 0.15, 0.05, 0.1, 0.2, 0.05, 0.15, 0.05],
            '1.5': [0.5, 0.3, 0.15, 0.25, 0.1, 0.5, 0.3, 0.15, 0.25, 0.1],
            '2.0': [0.3, 0.4, 0.10, 0.40, 0.2, 0.3, 0.4, 0.10, 0.40, 0.2],
            '2.5': [0.05, 0.05, 0.20, 0.15, 0.5, 0.05, 0.05, 0.20, 0.15, 0.5],
            '3.0': [0.05, 0.05, 0.50, 0.05, 0.15, 0.05, 0.05, 0.50, 0.05, 0.15],
            '3.5': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        df = pd.DataFrame(data)
    df['ID'] = df['ID'].astype(str).str.strip()
    return df

df = load_data()

prob_cols_potential = [f"{i:.1f}" for i in np.arange(1.0, 6.0, 0.5)]
prob_cols = [col for col in prob_cols_potential if col in df.columns]
x_labels_map = {i/10: f"{i}~{i+4}분" for i in range(10, 60, 5)}

if 'selected_id' in st.session_state:
    selected_id = st.session_state['selected_id']
    st.markdown(f"**배달 ID:** `{selected_id}`")
    id_data = df[df['ID'] == selected_id]
else:
    st.info("메인 페이지로 돌아가서 배달 ID를 선택해주세요.")
    st.page_link(page="0813_1630_main.py", label="메인 페이지로 돌아가기")
    id_data = pd.DataFrame()

if not id_data.empty:
    prob_values = id_data[prob_cols].iloc[0].astype(float)
    
    # --- Plotly 데이터 준비 ---
    bin_width = 5
    prob_density = prob_values / bin_width
    x_numeric_base = [i for i in range(10, 10 + bin_width * len(prob_cols), bin_width)]
    x_tick_labels = [f"{i}분" for i in x_numeric_base]
    x_interp = np.linspace(min(x_numeric_base), max(x_numeric_base) + bin_width, 400)
    y_interp = np.interp(x_interp, [x + bin_width/2 for x in x_numeric_base], prob_density)

    fig = go.Figure()

    # --- 1. 배경 영역 그리기 ---
    max_prob_idx = prob_values.argmax()
    for i in range(len(prob_values)):
        is_highlight_zone = (i == max_prob_idx)
        start_x, end_x = x_numeric_base[i], x_numeric_base[i] + bin_width
        mask = (x_interp >= start_x) & (x_interp <= end_x)
        x_segment = x_interp[mask]
        x_fill = np.concatenate([[start_x], x_segment, [end_x]])
        y_fill = np.concatenate([[0], y_interp[mask], [0]])

        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill, mode='lines', line=dict(width=0),
            fill='tozeroy',
            fillcolor='rgba(255, 223, 0, 0.6)' if is_highlight_zone else 'rgba(180, 180, 180, 0.2)',
            hoverinfo='none'
        ))

    # --- 2. 전체 분포 곡선 ---
    fig.add_trace(go.Scatter(
        x=x_interp, y=y_interp, mode='lines',
        line=dict(width=2.5, color='rgba(135, 206, 250, 1)', shape='spline'),
        name='확률 분포', hoverinfo='none'
    ))

    # --- 3. Hover 인터랙션을 위한 보이지 않는 Bar (고정 팝업용) ---
    bar_labels = [x_labels_map.get(float(col)) for col in prob_cols]
    fig.add_trace(go.Bar(
        x=[x + bin_width/2 for x in x_numeric_base],
        y=[max(prob_density) * 1.2] * len(prob_values),
        width=bin_width,
        customdata=np.stack([prob_values, bar_labels], axis=-1),
        hovertemplate="<b>%{customdata[1]}</b><br>도착확률: %{customdata[0]:.1%}<extra></extra>",
        marker=dict(color='rgba(0,0,0,0)'), # 완전히 투명하게
        name=''
    ))

    # --- 4. 동적 기준선 ---
    if max_prob_idx < len(x_numeric_base) - 1:
        next_interval_idx = max_prob_idx + 1
        red_line_x = x_numeric_base[next_interval_idx] + bin_width
        
        fig.add_shape(
            type="line", x0=red_line_x, y0=0, x1=red_line_x, y1=max(prob_density) * 1.2,
            line=dict(color="rgba(255, 82, 82, 1)", width=2, dash="dash")
        )
        fig.add_annotation(
            x=red_line_x, y=max(prob_density) * 1.15,
            text=f"<b>{red_line_x}분 이내 도착 예상</b>",
            showarrow=False, font=dict(color="rgba(255, 82, 82, 1)"),
            xanchor='left', xshift=5
        )

    # --- 5. 차트 레이아웃 최종 수정 ---
    fig.update_layout(
        xaxis_title="소요시간 (분)",
        yaxis_title="확률 밀도",
        yaxis_tickformat='.2%',
        showlegend=False,
        plot_bgcolor='#2c2c2c',
        paper_bgcolor='#2c2c2c',
        font=dict(family="Inter, sans-serif", size=12, color='#BDBDBD'),
        margin=dict(l=60, r=40, t=80, b=60),
        xaxis=dict(
            tickmode='array',
            tickvals=x_numeric_base,
            ticktext=x_tick_labels,
            range=[min(x_numeric_base), max(x_numeric_base) + bin_width],
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False
        ),
        yaxis=dict(
            range=[0, max(prob_density) * 1.2],
            showgrid=False,
            zeroline=False
        ),
        hoverlabel=dict(
            bgcolor="rgba(44, 44, 44, 0.9)",
            bordercolor="rgba(255, 255, 255, 0.3)",
            font_size=14,
            font_family="Inter, sans-serif",
            font_color="white",
            align="auto",
            namelength=-1,
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("각 구간의 **면적**이 해당 5분 구간에 도착할 확률을 나타내며, 곡선 아래의 총면적은 100%입니다.", icon="ℹ️")

else:
    st.error("선택된 ID에 대한 데이터를 찾을 수 없습니다.")