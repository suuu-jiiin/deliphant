'''
<작업내역>
final_merged_df.csv에서 선택한 ID만 남김.
ID값 뒤부분 spacebar 공백제거
'''
# ========================= [BLOCK 1] 기본 설정 & 라이브러리 =========================
import math
import requests
import folium
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import ast

st.set_page_config(page_title="배달 예측(메인)", layout="wide")
st.title("🚚 배달 예측 대시보드 — 메인 (Mapbox + 로컬 CSV)")

# ========================= [BLOCK 2] 전역 상수(토큰/파일 경로/컬럼 매핑/색상) =========================
MAPBOX_TOKEN = "mapboxToken"  
LOCAL_CSV_PATH   = "final_merged_df.csv"
target_ids = ['0x9d32', '0x23d4', '0x8b39', '0xce01', '0x8fdd', '0x7ab9', '0x6f80', '0xa512', '0xd740', '0xb478', '0xd200', '0x2a85', '0x1ef8', '0x972b']

# 샘플 CSV 기준 컬럼 매핑
COL = {
    "id": "ID",
    "date": "Order_Date",                   # 주문 날짜
    "order_time": "Time_Orderd",            # 주문 수락 시각(HH:MM 등)
    "pickup_time": "Time_Order_picked",     # 픽업 시각(HH:MM 등)
    "total_min": "Time_taken_min",          # 총 소요(분)
    "prep_min": "prep_time",                # 준비 소요(분)
    "store_lat": "Restaurant_lat_real",
    "store_lng": "Restaurant_lon_real",
    "cust_lat": "Delivery_lat_real",
    "cust_lng": "Delivery_lon_real",
}

CONGESTION_COLOR = {
    "low": "#1DB954",        # 원활(초록)
    "moderate": "#FFA500",   # 서행(주황)
    "heavy": "#FF4D4D",      # 정체(빨강)
    "severe": "#B30000",     # 심한 정체(진빨강)
    "unknown": "#999999"     # 정보없음(회색)
}

# ========================= [BLOCK 3] 데이터 로드/전처리 =========================
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
