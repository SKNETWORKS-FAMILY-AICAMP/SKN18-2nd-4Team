# 📄 pages/player_search.py — 4-column (centered) + club/player tiles (5 cols) + player header/detail @ 969px
import os, re, base64, html
from io import StringIO
import pandas as pd
import streamlit as st
import plotly.graph_objects as pgo 
import pathlib

def cwd():
    """현재 작업 디렉토리를 반환"""
    return pathlib.Path.cwd()

# =============================
# Config
# =============================
st.set_page_config(
    page_title="Player Search | 24/25 Premier League",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CSV_PATH = cwd() / "data" / "streamlit" / "data" / "DB1.csv"
TARGET_SEASON = "24/25"

# =============================
# Style
# =============================
CSS = """
<style>
:root{
  --brand:#7B19BD; --ink:#111827; --muted:#6B7280;
  --bg:#fff; --card:#fff; --line:#E5E7EB; --radius:14px;
  --content-width:982px;       /* 전체 페이지 컨테이너 폭 */
  --players-5col:969px;        /* 5열(181×5 + 16×4) 총폭 */
}

/* 기본 */
html, body, [class*="css"]{
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  background:rgba(0, 0, 0, 0.7); color:#e5e7eb !important;
}

/* 배경 설정 - 반투명 다크그레이 배경 */
.stApp {
  background-color: rgba(0, 0, 0, 0.7) !important;
}

/* 메인 컨테이너 배경 설정 */
.main {
  background-color: rgba(0, 0, 0, 0.7) !important;
}

/* 모든 텍스트를 밝은 회색으로 */
h1, h2, h3, h4, h5, h6, p, div, span, label, .stText, .stMarkdown {
  color: #e5e7eb !important;
}

/* 선수정보 카드 내부 텍스트 색상 설정 */
.card, .card * {
  color: #111827 !important;  /* 기본 검정색 */
}

/* 키(라벨)는 회색 */
.card .k, .card .metric-label {
  color: #6B7280 !important;  /* 회색 */
}

/* 값은 검정색 */
.card .v, .card .metric-val {
  color: #111827 !important;  /* 검정색 */
}

/* 선수 이름은 검정색 */
.card h1, .card h2, .card h3, .card h4, .card h5, .card h6 {
  color: #111827 !important;  /* 검정색 */
}

/* 최강 우선순위로 선수 이름 강제 검정색 설정 */
html body div[data-testid="stApp"] .card h1,
html body div[data-testid="stApp"] .card h2,
html body div[data-testid="stApp"] .card h3,
html body div[data-testid="stApp"] .card h4,
html body div[data-testid="stApp"] .card h5,
html body div[data-testid="stApp"] .card h6 {
  color: #111827 !important;  /* 검정색 */
}

/* 최종 강제 설정 - 모든 선수 이름 관련 요소 */
html body div[data-testid="stApp"] h1,
html body div[data-testid="stApp"] h2,
html body div[data-testid="stApp"] h3,
html body div[data-testid="stApp"] h4,
html body div[data-testid="stApp"] h5,
html body div[data-testid="stApp"] h6 {
  color: #111827 !important;  /* 검정색 */
}

/* 팀멤버 그리드 전체 컨테이너 배경 제거 */
.player-grid-wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.player-grid {
  background: transparent !important;
}
.main > div { padding-top:0 !important; }
.stApp > header { display:none; }

/* 전체 컨테이너 중앙 정렬 강화 */
.stApp > div {
  display: flex !important;
  justify-content: center !important;
  align-items: flex-start !important;
}

.stApp .main {
  width: 100% !important;
  max-width: var(--content-width) !important;
  margin: 0 auto !important;
}

/* Top Navigation */
.page-title { font-size: 22px; font-weight: 700; color: #333; margin: 0; padding-top: 8px;}
/* 네비게이션 버튼(상단 우측) - 텍스트 링크처럼 보이도록 */
div[data-testid="stHorizontalBlock"] .stButton>button {
    background: transparent !important;
    color: #333 !important;
    border: none !important;
    padding: 0 !important;
    box_shadow: none !important;
    font-weight: 600;
    font-size: 14px;
}


/* ▶ 4열 폭에 맞춰 중앙 정렬 */
.main .block-container{
  max-width:var(--content-width) !important;
  width:var(--content-width) !important;
  padding-left:0px !important; 
  padding-right:24px !important;
  margin:0 auto !important;
}

.page-wrap { max-width:var(--content-width); width:var(--content-width); margin:24px auto 60px; }
.h1{font-weight:800; font-size:45px; line-height:1.2; letter-spacing:-0.02em; margin:8px 0 8px; text-align:left; color:#e5e7eb !important;}
.h1 span{color:#7B19BD !important;}
.subtitle{font-size:18px; color:var(--muted); text-align:left;}
.h2{font-weight:700; font-size:26px; letter-spacing:-0.01em; margin:24px 0 12px; text-align:left;}
.card, .chartbox { text-align:left; }
.page-wrap, .page-wrap * { text-align:left; }


/* 공용 카드/차트 */
.hr{height:1px; background:var(--line); margin:32px 0;}
.card{background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:18px; box-shadow:0 1px 2px rgba(0,0,0,.03);}
.chartbox{background:#F5F5F7; border:1px solid #ECECEF; border-radius:12px; padding:16px;}

/* ===== Club tiles (4열, 227×105) ===== */
.club-grid{
  display:grid;
  grid-template-columns:repeat(4, 227.43px);
  gap:24px;
  margin:6px 0 8px;
  justify-content:flex-start;
}
.club-tile{
  position:relative;
  display:grid; grid-template-columns:96px 1fr; align-items:center;
  width:227.43px; height:105px; padding:0;
  background:#8A898B; border-radius:12px; overflow:hidden;
  box-shadow:0 1px 2px rgba(0,0,0,.06);
  transition:transform .15s, box-shadow .15s, filter .15s;
}
.club-tile:hover{ transform:translateY(-2px); box-shadow:0 6px 16px rgba(0,0,0,.15); filter:saturate(1.05); }
.club-logo{ width:80px; height:80px; object-fit:contain; justify-self:start; margin-left:10px; }
.club-name{ color:#fff; font-weight:700; font-size:18px; line-height:26px; display:flex; align-items:center; padding-left:1px;
  white-space:normal; word-break:keep-all; overflow-wrap:break-word; text-shadow:0 2px 6px rgba(0,0,0,.25); text-align:left; }
/* 전체 클릭 영역 */
.club-tile form{ position:absolute; inset:0; margin:0; }
.club-tile .overlay-btn{ position:absolute; inset:0; background:transparent; border:0; cursor:pointer; }

/* ===== Player tiles (고정: 5열 × 181px, gap 16px) ===== */
.player-grid-wrap{
  background:#F3F1F1; border-radius:12px;
  width:var(--players-5col); margin:8px 0 2px; padding:8px 0;
}
.player-grid{
  display:grid; grid-template-columns:repeat(5, 181px);
  gap:16px; justify-content:flex-start;
}
.player-tile{
  position:relative; display:flex; align-items:center; justify-content:center;
  height:60px; padding:0 10px; background:#8A898B; border-radius:12px; overflow:hidden;
  box-shadow:0 1px 2px rgba(0,0,0,.06);
  transition:transform .15s, box-shadow .15s, filter .15s;
}
.player-tile:hover{ transform:translateY(-2px); box-shadow:0 6px 16px rgba(0,0,0,.15); filter:saturate(1.05); }
.player-name{ width:100%; text-align:center; color:#fff; font-weight:700; font-size:18px; line-height:22px;
  white-space:normal; word-break:keep-all; overflow-wrap:anywhere; text-shadow:0 2px 6px rgba(0,0,0,.25); }
.player-tile form{ position:absolute; inset:0; margin:0; }
.player-tile .overlay-btn{ position:absolute; inset:0; background:transparent; border:0; cursor:pointer; }

/* ====== 섹션 폭 강제 (마커 :has 사용) ====== */
[data-testid="stVerticalBlock"]:has(#pi-wrap),
[data-testid="stVerticalBlock"]:has(#detail-wrap){
  width: var(--players-5col) !important;
  max-width: var(--players-5col) !important;
  margin: 8px auto 2px !important;
  padding: 0 !important;
}
/* 내부 max-width 해제 */
[data-testid="stVerticalBlock"]:has(#pi-wrap) .element-container,
[data-testid="stVerticalBlock"]:has(#pi-wrap) [data-testid="stMarkdown"],
[data-testid="stVerticalBlock"]:has(#pi-wrap) [data-testid="stPlotlyChart"],
[data-testid="stVerticalBlock"]:has(#detail-wrap) .element-container,
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-testid="stMarkdown"],
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-testid="stPlotlyChart"]{
  max-width:none !important; width:100% !important;
}
/* detail 컬럼 간격 고정 */
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-testid="stHorizontalBlock"]{ gap:16px !important; }
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-testid="column"]{ min-width:0 !important; }

/* ===== Player header ===== */
.player-header{
  display:grid; grid-template-columns:320px 1fr; gap:24px; align-items:start;
}
@media (max-width:740px){ .player-header{ grid-template-columns:1fr; } }
.player-photo-img{
  width:100%; height:auto; object-fit:contain;
  border-radius:10px; border:1px solid #ECECEF; background:#FAFAFB;
}

/* ===== Meta (two-line ::label + value) ===== */
.kv-grid{
  display:grid; grid-template-columns:repeat(3, 1fr); gap:18px 24px; margin:6px 0 0;
}
@media (max-width:740px){ .kv-grid{ grid-template-columns:repeat(2, 1fr);} }
.kv{ padding:6px 0 10px; border-bottom:1px dashed #eee; }
.kv .k{ color:#8A8F98; font-size:12px; letter-spacing:.02em; }
.kv .k::before{ content:":: "; }
.kv .v{ margin-top:4px; font-weight:800; font-size:18px; color:#111; }

/* ===== Season metrics ===== */
.metrics-wide{ display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; margin-top:16px; }
@media (max-width:740px){ .metrics-wide{ grid-template-columns:repeat(3, 1fr);} }
@media (max-width:480px){ .metrics-wide{ grid-template-columns:repeat(2, 1fr);} }
.metric-box{ background:#F5F5F7; border:1px solid #ECECEF; border-radius:10px; padding:10px 12px; }
.metric-box .metric-label{ color:#666; font-size:12px; }
.metric-box .metric-val{ font-weight:800; font-size:20px; line-height:1; }

/* ===== Responsive ===== */
@media (max-width:980px){
  .club-grid{ grid-template-columns:repeat(3, 227.43px); }
  .player-grid{ grid-template-columns:repeat(4, 181px); }
  .player-grid-wrap{ width:min(772px, 100%); }
}
@media (max-width:740px){
  .club-grid{ grid-template-columns:repeat(2, 227.43px); }
  .player-grid{ grid-template-columns:repeat(3, 181px); }
  .player-grid-wrap{ width:min(575px, 100%); }
}
@media (max-width:600px){
  .player-grid{ grid-template-columns:repeat(2, 181px); }
  .player-grid-wrap{ width:min(378px, 100%); }
}
@media (max-width:420px){
  .club-grid{ grid-template-columns:repeat(1, 227.43px); }
  .player-grid{ grid-template-columns:repeat(1, 181px); }
  .player-grid-wrap{ width:min(181px, 100%); }
}

/* ✅ detail의 '얇은 캡션 바' 및 입력 컨트롤 제거 */
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-testid="column"] > div:first-child,
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-testid="column"] > div:first-child > div:first-child,
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-testid="column"] > div:first-child > div:first-child > div:first-child{
  display:none !important; visibility:hidden !important; height:0 !important; margin:0 !important; padding:0 !important; border:0 !important;
}
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-testid="stTextInput"],
[data-testid="stVerticalBlock"]:has(#detail-wrap) .stTextInput,
[data-testid="stVerticalBlock"]:has(#detail-wrap) [data-baseweb="input"],
[data-testid="stVerticalBlock"]:has(#detail-wrap) [role="textbox"],
[data-testid="stVerticalBlock"]:has(#detail-wrap) input[type="text"],
[data-testid="stVerticalBlock"]:has(#detail-wrap) textarea{
  display:none !important; visibility:hidden !important; height:0 !important; margin:0 !important; padding:0 !important; border:0 !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
wrap_start, wrap_end = "<div class='page-wrap'>", "</div>"


# =============================
# Helpers
# =============================

# [수정] 1. 페이지 이동을 위한 go() 함수 정의
def go(page_name: str):
    # pages 폴더에 있는 파일명을 기반으로 페이지를 전환합니다.
    # 예: go("players") -> st.switch_page("pages/player_search.py")
    # 파일명은 실제 프로젝트 구조에 맞게 수정하세요.
    page_map = {
        "home": "./app.py",
        "players": "pages/player_search.py",
        "predict": "pages/transfer_predictor.py", # 예시 파일명
    }
    target_page = page_map.get(page_name)
    if target_page:
        st.switch_page(target_page)

def page_home():
    # Top Navigation
    nav_cols = st.columns([0.6, 0.2, 0.2])
    with nav_cols[0]:
        st.markdown('<p class="page-title">Player Search</p>', unsafe_allow_html=True)
    with nav_cols[1]:
        # 현재 페이지이므로 비활성화하거나 홈으로 가게 할 수 있습니다. 여기서는 그대로 둡니다.
        if st.button("Home", key="nav_home", use_container_width=True):
            go("home") 
    with nav_cols[2]:
        if st.button("Transfer Predictor", key="nav_predict", use_container_width=True):
            go("predict")
            
def normalize_season(s):
  if pd.isna(s): return s
  s = str(s).strip()
  if "-" in s:
    a, b = s.split("-", 1)
    if a.isdigit() and b.isdigit(): return f"{a[2:]}/{b}"
  if "/" in s:
    a, b = s.split("/", 1)
    if a.isdigit() and b.isdigit():
      return f"{a[2:]}/{b}" if len(a) == 4 else f"{a}/{b}"
  return s

def read_csv_robust(path):
  encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
  for enc in encodings:
    try: return pd.read_csv(path, encoding=enc), enc
    except UnicodeDecodeError: pass
  with open(path, "rb") as f: raw = f.read()
  text = raw.decode("utf-8", errors="replace")
  return pd.read_csv(StringIO(text)), "utf-8(replace)"

def canonical_club_name(name: str) -> str:
  if name is None or (isinstance(name, float) and pd.isna(name)): return ""
  s = str(name).strip()
  s = re.sub(r"^Association\s+Football\s+Club\s+", "AFC ", s, flags=re.I)
  s = re.sub(r"\s+Football\s+Club\b", "", s, flags=re.I)
  s = re.sub(r"\s+", " ", s).strip()
  return s

def _norm_token(x: str) -> str:
  x = x.lower().replace("&", "and")
  return re.sub(r"[^a-z0-9]", "", x)

def img_to_data_uri(path: str):
  if (not path) or (not os.path.exists(path)): return None
  mime = "image/png" if path.lower().endswith(".png") else \
         "image/webp" if path.lower().endswith(".webp") else \
         "image/gif"  if path.lower().endswith(".gif")  else \
         "image/svg+xml" if path.lower().endswith(".svg") else "image/jpeg"
  with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
  return f"data:{mime};base64,{b64}"

def logo_src_for_club(club_display_name: str):
  search_dirs = ["data/streamlit/data/imgs/clubs", "pages/imgs", "assets/clubs", "assets", "static/clubs", "static"]
  target_norm = _norm_token(club_display_name)
  exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")
  for d in search_dirs:
    for candidate in (
      os.path.join(d, f"{club_display_name}.png"),
      os.path.join(d, f"{club_display_name}.jpg"),
      os.path.join(d, f"{club_display_name}.jpeg"),
      os.path.join(d, f"{club_display_name}.webp"),
      os.path.join(d, f"{club_display_name}.svg"),
    ):
      if os.path.exists(candidate):
        return img_to_data_uri(candidate)
  for d in search_dirs:
    if not os.path.isdir(d): continue
    try:
      for fn in os.listdir(d):
        if not fn.lower().endswith(exts): continue
        stem = os.path.splitext(fn)[0]
        if _norm_token(stem) == target_norm:
          return img_to_data_uri(os.path.join(d, fn))
    except Exception:
      continue
  return None

def h2(t): st.markdown(f"<div class='h2'>▾ {t}</div>", unsafe_allow_html=True)

def as_int(x, default="-"):
  try:
    if pd.isna(x): return default
  except Exception: pass
  try: return int(float(x))
  except Exception: return default

def as_float(x, default=None):
  try:
    if pd.isna(x): return default
  except Exception: pass
  try: return float(x)
  except Exception: return default

def fmt_millions(x):
  v = as_float(x, None)
  if v is None: return "-"
  return f"{v/1_000_000:.1f}"

# =============================
# Load df & map fields
# =============================
if not os.path.exists(CSV_PATH):
  st.error(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
  st.stop()

df, used_enc = read_csv_robust(CSV_PATH)

# 필수 기본 컬럼
for c in ["season","player_name","club_name"]:
  if c not in df.columns: df[c] = pd.NA

df["season"] = df["season"].map(normalize_season)

# 새 스키마 → 내부 표준 컬럼 매핑
if "Appreances" in df.columns: df["games"] = pd.to_numeric(df["Appreances"], errors="coerce")
if "goals" in df.columns: df["goals"] = pd.to_numeric(df["goals"], errors="coerce")
if "assists" in df.columns: df["assists"] = pd.to_numeric(df["assists"], errors="coerce")
if "season_avg_minutes" in df.columns: df["minutes"] = pd.to_numeric(df["season_avg_minutes"], errors="coerce")
if "market_value" in df.columns: df["market_value_eur"] = pd.to_numeric(df["market_value"], errors="coerce")
if "player_highest_market_value_in_eur" in df.columns: df["highest_mv_eur"] = pd.to_numeric(df["player_highest_market_value_in_eur"], errors="coerce")
if "season_win_count" in df.columns: df["wins"] = pd.to_numeric(df["season_win_count"], errors="coerce")
if "height_in_cm" in df.columns: df["height_cm"] = pd.to_numeric(df["height_in_cm"], errors="coerce")
if "Nationality" in df.columns: df["nationality"] = df["Nationality"].astype(str).str.strip()
if "imag_url" in df.columns: df["photo_url"] = df["imag_url"].astype(str).str.strip()

for keep in ["date_of_birth","foot","position","sub_position","agent_name","net_transfer_record","transfer"]:
  if keep in df.columns:
    df[keep] = df[keep].astype(str).str.strip()
  else:
    df[keep] = ""

df["club_name"] = df["club_name"].astype(str).str.strip()
df["club_name_canon"] = df["club_name"].apply(canonical_club_name)

# =============================
# State
# =============================
if "selected_team" not in st.session_state: st.session_state.selected_team = None
if "selected_player" not in st.session_state: st.session_state.selected_player = None

# Streamlit 1.28.0 이상에서는 st.query_params 사용 권장
try:
  qp = st.query_params
except Exception:
  qp = st.experimental_get_query_params()

def _qp_get(name):
  try:
    v = qp.get(name, None)
    if isinstance(v, list): v = v[-1]
    return v
  except Exception:
    return None

sel_from_qp = _qp_get("club")
if sel_from_qp:
  st.session_state.selected_team = str(sel_from_qp)

sel_player_from_qp = _qp_get("player")
if sel_player_from_qp:
  st.session_state.selected_player = str(sel_player_from_qp)


# =============================
# Page
# =============================

# [수정] 2. 페이지 상단에 내비게이션 바를 표시하기 위해 함수 호출
page_home()

st.markdown(wrap_start, unsafe_allow_html=True)

st.markdown("""
<div>
  <div class="h1">Explore <span style="color:#7B19BD !important; font-weight:800 !important;">24/25 Premier League</span> <br/>Football Player in One Place</div>
  <div class="subtitle">Easily search across teams, key positions.</div>
</div>
""", unsafe_allow_html=True)

# --- Team(Club) Search ---
h2("Team Search")
mask_2425 = df["season"] == TARGET_SEASON
team_list = sorted(df.loc[mask_2425, "club_name_canon"].dropna().astype(str).unique().tolist())

if not team_list:
  st.warning("24/25 시즌 데이터가 없거나 club_name 값이 비어있습니다. CSV의 season(24/25) / club_name을 확인하세요.")
else:
  tiles_html = '<div class="club-grid">'
  for club_display in team_list:
    logo_uri = logo_src_for_club(club_display)
    logo_tag = f'<img class="club-logo" src="{logo_uri}"/>' if logo_uri else ''
    safe = html.escape(club_display)
    tiles_html += (
      '<div class="club-tile">'
      f'{logo_tag}'
      f'<div class="club-name">{safe}</div>'
      '<form method="get">'
      f'<input type="hidden" name="club" value="{safe}">'
      f'<button class="overlay-btn" type="submit" aria-label="{safe}"></button>'
      '</form>'
      '</div>'
    )
  tiles_html += '</div>'
  st.markdown(tiles_html, unsafe_allow_html=True)

# --- Team Member Tiles (24/25) ---
if st.session_state.selected_team:
  h2(f"{st.session_state.selected_team} Team Member")
  mask_team = mask_2425 & (df["club_name_canon"] == st.session_state.selected_team)
  tdf = (
    df.loc[mask_team, ["player_name","club_name_canon"]]
      .dropna(subset=["player_name"]).drop_duplicates(subset=["player_name"])
      .sort_values("player_name").reset_index(drop=True)
  )
  if tdf.empty:
    st.info("해당 클럽의 24/25 시즌 선수 데이터가 없습니다.")
  else:
    ptiles = '<div class="player-grid-wrap"><div class="player-grid">'
    safe_club = html.escape(st.session_state.selected_team)
    for name in tdf["player_name"].astype(str):
      safe_name = html.escape(name)
      ptiles += (
        '<div class="player-tile">'
        f'  <div class="player-name">{safe_name}</div>'
        '  <form method="get">'
        f'    <input type="hidden" name="club" value="{safe_club}">'
        f'    <input type="hidden" name="player" value="{safe_name}">'
        '    <button class="overlay-btn" type="submit" aria-label="Select player"></button>'
        '  </form>'
        '</div>'
      )
    ptiles += '</div></div>'
    st.markdown(ptiles, unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

def get_local_img(player_name):
    if player_name is None:
        return None  # 선수 이름이 없으면 바로 None 반환

<<<<<<< Updated upstream
<<<<<<< Updated upstream
    img_folder = cwd() / "data" / "streamlit" / "data" / "imgs" / "player"
=======
    img_folder = "data/streamlit/data/imgs/player"
>>>>>>> Stashed changes
=======
    img_folder = "data/streamlit/data/imgs/player"
>>>>>>> Stashed changes
    name_variants = [
        f"{player_name}.png", f"{player_name}.jpg",
        f"{player_name.replace(' ', '_')}.png",
        f"{player_name.replace(' ', '_')}.jpg"
    ]
    for fname in name_variants:
        full_path = os.path.join(img_folder, fname)
        if os.path.exists(full_path):
            return full_path

    # fallback default 이미지
    default_img = os.path.join(img_folder, "default.png")
    return default_img if os.path.exists(default_img) else None

pname = st.session_state.selected_player
# 이미지 불러오기 및 base64 변환
img_path = get_local_img(pname)
if img_path:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(img_path)[-1][1:]  # 확장자 추출 (e.g. png, jpg)
    img_tag = f"<img class='player-photo-img' src='data:image/{ext};base64,{b64}'/>"
else:
    # fallback도 없을 경우 최소한의 placeholder
    img_tag = "<div class='player-photo-img' style='background:#F1F1F4;'></div>"

# ─────────────────────────────
# 📊 숫자 처리 유틸
# ─────────────────────────────
def as_int(x, fallback="-"):
    try:
        return int(float(x))
    except Exception:
        return fallback

def fmt_millions(x):
    try:
        return f"{round(float(x) / 1e6, 1)}"
    except Exception:
        return "-"

# ── Player information (team grid width: 969px) ──
with st.container():
    st.markdown("<i id='pi-wrap'></i>", unsafe_allow_html=True)
    st.markdown("## Player information")

    if st.session_state.selected_team and not st.session_state.selected_player:
        mask_team = mask_2425 & (df["club_name_canon"] == st.session_state.selected_team)
        auto_list = df.loc[mask_team, "player_name"].dropna().astype(str).drop_duplicates().sort_values().tolist()
        if auto_list:
            st.session_state.selected_player = auto_list[0]

    pname = st.session_state.selected_player
    if pname:
        cols = ["player_name","season","games","goals","assists","minutes","market_value_eur","wins",
                "club_name","nationality","photo_url","date_of_birth","foot","position","sub_position",
                "height_cm","agent_name","net_transfer_record","transfer"]
        for c in cols:
            if c not in df.columns: df[c] = pd.NA
        hist = df.loc[df["player_name"] == pname, cols].dropna(subset=["season"]).sort_values("season").reset_index(drop=True)
        cap = (hist.loc[hist["season"] == TARGET_SEASON].iloc[0]
              if (not hist.empty and TARGET_SEASON in hist["season"].values) else None)

        def pick(col):
            if cap is not None and col in cap.index and pd.notna(cap[col]):
                return cap[col]
            if col in hist.columns and not hist[col].dropna().empty:
                return hist[col].dropna().iloc[0]
            return None

        def fmt_date(x):
            if x is None or (isinstance(x, float) and pd.isna(x)): return "-"
            ts = pd.to_datetime(x, errors="coerce")
            if pd.notna(ts): return ts.strftime("%Y-%m-%d")
            s = str(x).split()[0].strip()
            ts2 = pd.to_datetime(s, dayfirst=True, errors="coerce")
            return ts2.strftime("%Y-%m-%d") if pd.notna(ts2) else s

        def fmt_height(x):
            try: return f"{int(float(x))} cm"
            except Exception: return "-" if (x is None or (isinstance(x, float) and pd.isna(x))) else f"{x} cm"

        # 🖼️ 이미지 base64 삽입
        img_path = get_local_img(pname)
        if img_path:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(img_path)[-1][1:]
            img_tag = f"<img class='player-photo-img' src='data:image/{ext};base64,{b64}'/>"
        else:
            img_tag = "<div class='player-photo-img' style='background:#F1F1F4;'></div>"

        kv_items = [
            ("Team",            pick("club_name") or "-"),
            ("Nationality",     pick("nationality") or "-"),
            ("Date of Birth",   fmt_date(pick("date_of_birth"))),
            ("Preference Foot", pick("foot") or "-"),
            ("Position",        pick("position") or "-"),
            ("Sub Position",    pick("sub_position") or "-"),
            ("Appearances",     str(as_int(pick("games"), "-"))),
            ("Goals",           str(as_int(pick("goals"), "-"))),
            ("Assists",         str(as_int(pick("assists"), "-"))),
            ("Height",          fmt_height(pick("height_cm"))),
            ("Transfer Count",  str(as_int(pick("transfer"), "-"))),
        ]
        kv_html = ["<div class='kv-grid'>"]
        for k, v in kv_items:
            kv_html.append(
                f"<div class='kv'><div class='k'>{html.escape(k)}</div>"
                f"<div class='v'>{html.escape(str(v))}</div></div>"
            )
        kv_html.append("</div>")

        header_html = f"""
        <div class="card">
          <div class="player-header">
            <div>{img_tag}</div>
            <div>
              <h3 style="margin:0 0 12px 0; font-weight:800; font-size:26px;">{html.escape(pname)}</h3>
              {''.join(kv_html)}
            </div>
          </div>
          {"".join([
            "<div class='metrics-wide'>",
            f"<div class='metric-box'><div class='metric-label'>Appearances</div><div class='metric-val'>{as_int(cap.get('games'), '-') if cap is not None else '-'}</div></div>",
            f"<div class='metric-box'><div class='metric-label'>Goals</div><div class='metric-val'>{as_int(cap.get('goals'), '-') if cap is not None else '-'}</div></div>",
            f"<div class='metric-box'><div class='metric-label'>Assists</div><div class='metric-val'>{as_int(cap.get('assists'), '-') if cap is not None else '-'}</div></div>",
            f"<div class='metric-box'><div class='metric-label'>Minutes</div><div class='metric-val'>{as_int(cap.get('minutes'), '-') if cap is not None else '-'}</div></div>",
            f"<div class='metric-box'><div class='metric-label'>Mkt Value (€M)</div><div class='metric-val'>{fmt_millions(cap.get('market_value_eur')) if cap is not None else '-'}</div></div>",
            "</div>"
          ]) if cap is not None else "<div></div>"}
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

# ── detail (team grid width: 969px)
with st.container():
    st.markdown("<i id='detail-wrap'></i>", unsafe_allow_html=True)
    st.markdown("## detail")

    left, right = st.columns(2, gap="large")

    # 📊 왼쪽: Seasonal Wins
    with left:
        if 'pname' in locals() and pname and not (df.loc[df["player_name"] == pname, "wins"].isna().all()):
            wins_df = df.loc[df["player_name"] == pname, ["season", "wins"]].dropna()
            wins_df = wins_df.groupby("season", as_index=False)["wins"].max()

            fig_w = pgo.Figure()
            fig_w.add_bar(x=wins_df["season"], y=wins_df["wins"], marker_color="#7B19BD", name="Wins")
            fig_w.update_layout(
                height=320, showlegend=False,
                margin=dict(t=40, l=40, r=40, b=40),
                plot_bgcolor="#F5F5F7", paper_bgcolor="white",
                title_text="Seasonal Wins", title_x=0.0, title_y=0.98,
                title_font=dict(size=14)
            )
            st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar": False})

    # 📈 오른쪽: Market Value
    with right:
        if 'pname' in locals() and pname and not (df.loc[df["player_name"] == pname, "market_value_eur"].isna().all()):
            mv_df = df.loc[df["player_name"] == pname, ["season", "market_value_eur"]].dropna()
            mv_df = mv_df.groupby("season", as_index=False)["market_value_eur"].mean()

            fig_mv = pgo.Figure()
            fig_mv.add_scatter(x=mv_df["season"], y=mv_df["market_value_eur"], mode="lines+markers",
                              line=dict(color="#7B19BD"), name="Market Value")
            fig_mv.update_layout(
                height=320, showlegend=False,
                margin=dict(t=40, l=40, r=40, b=40),
                plot_bgcolor="#F5F5F7", paper_bgcolor="white",
                title_text="Seasonal Market Value (€)", title_x=0.0, title_y=0.98,
                title_font=dict(size=14)
            )
            st.plotly_chart(fig_mv, use_container_width=True, config={"displayModeBar": False})

# Footer
    # Footer
st.markdown("""
        <div class="footer" style="text-align: right;">
            <p>©2025 Project by SKN18 2nd Project 4th team</p>
        </div>
    """, unsafe_allow_html=True)