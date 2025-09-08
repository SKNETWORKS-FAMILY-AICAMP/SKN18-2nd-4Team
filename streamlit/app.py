import streamlit as st
import os

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="24/25 Premier League | Main",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Style (디자인에 맞춘 CSS)
# -----------------------------
CSS = """
<style>
:root{
    --brand:#7B19BD;      /* 24/25 보라 포인트 */
    --accent:#7B19BD;     /* 강조 텍스트 */
    --radius:14px;
    --green:#6b5cff;      /* Transfer Predictor 색상 */
    --blue:#6b5cff;       /* Player Search 색상 */
}

html, body, [class*="css"]  { 
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; 
    background: white;
}

.stMarkdown a, a, a:visited { color:#333; text-decoration:none; }
.stCaption, small { color:#666 !important; }

/* Top Navigation */
.page-title { font-size: 22px; font-weight: 700; color: #333; margin: 0; padding-top: 8px;}
/* 네비게이션 버튼(상단 우측) - 텍스트 링크처럼 보이도록 */
div[data-testid="stHorizontalBlock"] .stButton>button {
    background: transparent !important;
    color: #333 !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    font-weight: 600;
    font-size: 14px;
}

/* Hero Section */
.hero-wrap { 
    padding: 40px 20px 30px; 
    text-align: left;
    margin-bottom: 30px;
}
.hero-eyebrow { color: var(--brand); font-weight: 800; font-size: 22px; margin-bottom: 10px; }
.hero-title { font-weight: 900; font-size: 45px; line-height: 1.15; margin: 0 0 12px 0; letter-spacing: -0.01em; color: #333; }
.hero-title .accent { color: var(--accent); }
.hero-sub { color: rgba(0,0,0,.75); font-size: 18px; margin: 0 0 18px 0; font-weight: 500; }

/* Section Header */
.section-title { font-size: 26px; font-weight: 800; margin: 0; color: #333; }

/* --- Push 버튼 전용 (primary만 타겟) --- */
button[kind="primary"] {
    background: #000 !important;
    color: #fff !important;
    border: 1px solid #000 !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    box-shadow: none !important;
}

/* Footer */
.footer {
    text-align: right;
    padding: 20px 20px;
    color: #666;
    border-top: 1px solid #eee;
    margin-top: 40px;
}
.footer-text { font-size: 14px; font-weight: 500; }

/* Responsive */
@media (max-width: 768px) {
    .hero-title { font-size: 28px; }
    .hero-sub { font-size: 14px; }
}

/* Streamlit 기본 스타일 보정 */
.main > div { padding-top: 1rem; padding-bottom: 1rem; }
div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    padding-left: 20px;
    padding-right: 20px;
}
.stApp > header { display:none; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# 간단 라우팅 (session_state 사용)
# -----------------------------
if "route" not in st.session_state:
    st.session_state.route = "home"

def go(route: str):
    st.session_state.route = route
    st.rerun()

# -----------------------------
# Main Page Content
# -----------------------------
def page_home():
    # Top Navigation
    nav_cols = st.columns([0.6, 0.2, 0.2])
    with nav_cols[0]:
        st.markdown('<p class="page-title">Home</p>', unsafe_allow_html=True)
    with nav_cols[1]:
        if st.button("Player Search", key="nav_players", use_container_width=True):
            go("players")
    with nav_cols[2]:
        if st.button("Transfer Predictor", key="nav_predict", use_container_width=True):
            go("predict")

    # Hero Section
    st.markdown("""
        <div class="hero-wrap">
            <div class="hero-eyebrow">24/25 Premier League</div>
            <div class="hero-title">From Player Search to Transfer Predictions, <span class="accent">All in One Place!</span></div>
            <p class="hero-sub">Check real-time player information and AI-driven transfer probabilities here.</p>
        </div>
    """, unsafe_allow_html=True)

    # Player Search Section
    with st.container():
        header_cols = st.columns([0.8, 0.2])
        with header_cols[0]:
            st.markdown('<h3 class="section-title">Player Search</h3>', unsafe_allow_html=True)
        with header_cols[1]:
            if st.button("Push", key="push_players", use_container_width=True, type="primary"):
                go("players")
        st.image("pages/imgs/main-img1.jpg", use_container_width=True)

    st.divider()

    # Transfer Predictor Section
    with st.container():
        header_cols = st.columns([0.8, 0.2])
        with header_cols[0]:
            st.markdown('<h3 class="section-title">Transfer Predictor</h3>', unsafe_allow_html=True)
        with header_cols[1]:
            if st.button("Push", key="push_predict", use_container_width=True, type="primary"):
                go("predict")
        st.image("pages/imgs/main-img2.jpg", use_container_width=True)

    # Footer
    st.markdown("""
        <div class="footer">
            <p class="footer-text">©2024 Project_XferStats by SKN18 2nd Project 4th team</p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Router
# -----------------------------
if st.session_state.route == "home":
    page_home()
elif st.session_state.route == "players":
    st.switch_page("pages/player_search.py")
elif st.session_state.route == "predict":
    st.switch_page("pages/transfer_predictor.py")