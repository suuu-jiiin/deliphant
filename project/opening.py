import streamlit as st

st.set_page_config(page_title="Deliphant", page_icon="🐘", layout="wide")

# --- 네비 훅: 쿼리파라미터로 홈으로 이동 (신규 API만 사용) ---
qp = st.query_params
if qp.get("go") == "home":
    st.query_params.clear()              # URL 깔끔히
    st.switch_page("pages/home.py")            # pages/home.py 라면 경로 맞춰주세요

# --- 스타일 & 본문 ---
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { height: 100%; }

/* 인도 느낌 배경 */
.bg {
  position: fixed; inset: 0;
  background: linear-gradient(180deg, #FF9933 0%, #FFF6E9 35%, #FFFFFF 50%, #ECF7EC 65%, #138808 100%);
  background-size: 100% 120%;
  animation: breathe 6s ease-in-out infinite;
}
@keyframes breathe { 0%,100%{background-position-y:0%} 50%{background-position-y:10%} }

.center-wrap { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; padding: 24px; }
.card {
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(6px);
  border-radius: 28px;
  padding: 40px 48px;
  box-shadow: 0 18px 60px rgba(0,0,0,.20);
  text-align: center; max-width: 680px;
}

/* ✅ 타이틀 그라데이션 텍스트 */
.title {
  font-size: 64px; line-height: 1.05; font-weight: 900; letter-spacing: 1px; margin: 0 0 8px 0;
  background: linear-gradient(90deg, #FF7A1A 0%, #1AA34A 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  /* 살짝 더 선명하게 보이도록 그림자 */
  text-shadow: 0 0 0 rgba(0,0,0,0); /* 사파리 투명 텍스트 그림자 버그 방지 */
}

.subtitle { font-size: 16px; color: #475569; margin: 0 0 24px 0; }

/* ✅ 버튼은 단색 (그라데이션 제거) */
a.cta {
  display: inline-block; padding: 14px 26px; border-radius: 999px; font-weight: 800; text-decoration: none;
  color: #fff; background: #333;   /* 여기만 단색으로 */
  box-shadow: 0 8px 24px rgba(0,0,0,.18);
  transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}
a.cta:hover { transform: translateY(-1px); box-shadow: 0 12px 32px rgba(0,0,0,.22); filter: brightness(1.03); }
a.cta:active { transform: translateY(0); }
</style>

<div class="bg"></div>
<div class="center-wrap">
  <div class="card">
    <div class="title">Deliphant 🐘</div>
    <div class="subtitle">설명 가능한 AI 배달 예측 서비스</div>
    <a class="cta" href="?go=home">배달현황 보러가기</a>
  </div>
</div>
""", unsafe_allow_html=True)
