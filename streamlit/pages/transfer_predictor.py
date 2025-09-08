import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pathlib

def cwd():
    """현재 작업 디렉토리를 반환"""
    return pathlib.Path.cwd()

# ============================================================================
# ⚙️ Page Setup
# ============================================================================
st.set_page_config(
    page_title="Transfer Predictor",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# 🎨 Style (CSS)
# ============================================================================
st.markdown("""
<style>
:root {
  --primary-color: #7B19BD;
  --bg-color: #F8F9FA;
  --card-bg-color: #FFFFFF;
  --text-color: #212529;
  --subtext-color: #6c757d;
  --border-radius: 12px;
  --content-width: 969px;
}
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--bg-color);
}
.stApp > header { display: none; }
.custom-header {
  display: flex; justify-content: space-between; align-items: center;
  width: 100%; padding: 16px 40px; background-color: #FFFFFF;
  border-bottom: 1px solid #DEE2E6; position: fixed; top: 0; left: 0; right: 0;
  z-index: 99;
}
.custom-header .logo { font-size: 22px; font-weight: 700; color: var(--text-color); }
.block-container {
  max-width: var(--content-width) !important;
  margin: 0 auto !important;
  padding: 100px 16px 48px 16px !important;
}
.page-title-container { text-align: left; margin-bottom: 2.5rem; }
.page-title {
  font-size: 45px !important; font-weight: 700 !important;
  line-height: 1.2 !important; color: var(--text-color);
  letter-spacing: -0.02em; margin: 0;
}
.page-title .highlight { color: var(--primary-color); }
.section-header {
  font-weight: 700; font-size: 22px; color: var(--text-color);
  margin-top: 2.5rem; margin-bottom: 1rem;
  border-bottom: 2px solid #F1F3F5; padding-bottom: 0.5rem;
}
.footer {
  text-align: center; padding: 2rem 0; color: var(--subtext-color);
  font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 💾 Data Loader
# ============================================================================
@st.cache_data
def load_and_merge_data(path_db2, path_db1, name_col_db1, birth_date_col_db1):
    def read_csv_robust(path):
        encs = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
        for enc in encs:
            try:
                return pd.read_csv(path, encoding=enc), enc
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                raise
        # 마지막 시도: 기본 디코드 실패 시 바이트 읽어 replace 처리
        with open(path, "rb") as f:
            data = f.read().decode("utf-8", errors="replace")
        from io import StringIO
        return pd.read_csv(StringIO(data)), "utf-8(replace)"

    try:
        df2, enc2 = read_csv_robust(path_db2)
        df1, enc1 = read_csv_robust(path_db1)
    except FileNotFoundError as e:
        st.error(f"[ERROR] 파일을 찾을 수 없습니다: {e.filename}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"[ERROR] 파일 불러오기 실패: {e}")
        return pd.DataFrame()

    season_col_list = [col for col in df1.columns if "season" in col.lower()]
    if not season_col_list:
        st.error("DB1에 'season' 관련 컬럼이 없습니다.")
        return pd.DataFrame()
    season_col = season_col_list[0]
    
    df1 = df1[df1[season_col] == "24/25"].copy()

    required_cols = [name_col_db1, birth_date_col_db1, 'position', 'market_value', 'foot']
    if not all(col in df1.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df1.columns]
        st.error(f"DB1에서 다음 컬럼이 누락됨: {missing_cols}")
        return pd.DataFrame()
    df1 = df1[required_cols].drop_duplicates(subset=[name_col_db1])

    df = pd.merge(df2, df1, left_on='player_name', right_on=name_col_db1, how='left')

    if 'position_x' in df.columns and 'position_y' in df.columns:
        df['position'] = df['position_y'].fillna(df['position_x'])
        df.drop(columns=['position_x', 'position_y'], inplace=True)

    if 'foot_x' in df.columns and 'foot_y' in df.columns:
        df['foot'] = df['foot_y'].fillna(df['foot_x'])
        df.drop(columns=['foot_x', 'foot_y'], inplace=True)

    if 'market_value' in df.columns:
        df['market_value_numeric'] = (
            df['market_value'].astype(str)
            .str.replace('€', '', regex=False)
            .str.replace('m', 'e6', regex=False)
            .str.replace('k', 'e3', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
            .apply(lambda x: pd.to_numeric(x, errors='coerce'))
        )
    else:
        df['market_value_numeric'] = np.nan

    df['date_of_birth'] = pd.to_datetime(df[birth_date_col_db1], errors='coerce')
    df.dropna(subset=['date_of_birth'], inplace=True)
    df['birth_year'] = df['date_of_birth'].dt.year
    df['age'] = datetime.now().year - df['birth_year']

    return df

# ============================================================================
# 📁 File Paths
# ============================================================================
DB2_FILE_PATH = str(cwd() / "data" / "streamlit" / "data" / "DB2.csv")
DB1_FILE_PATH = str(cwd() / "data" / "streamlit" / "data" / "DB1.csv")
DB1_PLAYER_NAME_COLUMN = 'player_name'
DB1_BIRTH_DATE_COLUMN = 'date_of_birth'

df = load_and_merge_data(DB2_FILE_PATH, DB1_FILE_PATH, DB1_PLAYER_NAME_COLUMN, DB1_BIRTH_DATE_COLUMN)
if not df.empty and 'transfer_probability' in df.columns:
    df['transfer_probability_percent'] = (df['transfer_probability'] * 100).round(1).astype(str) + "%"

# ============================================================================
# 🔍 Search Callback
# ============================================================================
def run_name_search():
    query = st.session_state.player_name_input
    if not df.empty:
        if query:
            st.session_state.filtered_results = df[df['player_name'].str.contains(query, case=False, na=False)]
        else:
            st.session_state.filtered_results = df.copy()

# ============================================================================
# 🖥️ UI Layout
# ============================================================================
st.markdown("""<div class="custom-header"><div class="logo">Transfer Predictor</div></div>""", unsafe_allow_html=True)
st.markdown("""<div class="page-title-container"><h1 class="page-title">Predict Transfer Probabilities Across <span class="highlight">25/26</span> Premier League Players</h1></div>""", unsafe_allow_html=True)

st.markdown('<p class="section-header">▼ Player Filters</p>', unsafe_allow_html=True)

filter_button = False

with st.container():
    st.text_input("Search by Player Name", key="player_name_input", placeholder="e.g., Haaland", on_change=run_name_search)
    st.markdown("<hr style='margin:1.5rem 0;'>", unsafe_allow_html=True)

    if not df.empty:
        c1, c2, c3, c4, c5 = st.columns([2, 1.8, 2.2, 1.5, 1.5])
        with c1:
            pos_options = ['All'] + sorted(df['position'].dropna().unique().tolist())
            position = st.selectbox("Position", options=pos_options)
        with c2:
            min_birth_year = st.number_input("Born After", min_value=1980, max_value=datetime.now().year, value=1980)
        with c3:
            market_value_million = st.number_input("Min Market Value (€M)", min_value=0, value=0)
        with c4:
            foot_options = ['All'] + sorted(df['foot'].dropna().unique().tolist())
            foot = st.selectbox("Foot", options=foot_options)
        with c5:
            filter_button = st.button("🔍 Search Players")
    else:
        st.warning("데이터가 로드되지 않아 필터를 표시할 수 없습니다. 파일 경로와 내용을 확인해 주세요.")

# ============================================================================
# 📄 Filter + Display
# ============================================================================
if 'filtered_results' not in st.session_state:
    st.session_state.filtered_results = df.copy() if not df.empty else pd.DataFrame()

if filter_button:
    result_df = df.copy()
    query = st.session_state.player_name_input
    if query:
        result_df = result_df[result_df['player_name'].str.contains(query, case=False, na=False)]
    if position != 'All':
        result_df = result_df[result_df['position'] == position]
    if min_birth_year:
        result_df = result_df[result_df['birth_year'] >= min_birth_year]
    if market_value_million > 0:
        result_df = result_df[result_df['market_value_numeric'] >= market_value_million * 1_000_000]
    if foot != 'All':
        result_df = result_df[result_df['foot'] == foot]
    st.session_state.filtered_results = result_df

result_df = st.session_state.filtered_results

if not result_df.empty:
    st.markdown('<p class="section-header">📄 Search Results</p>', unsafe_allow_html=True)

    sort_by_options = {
        'Market Value': 'market_value_numeric',
        'Age': 'age',
        'Player Name': 'player_name'
    }
    sort_by_label = st.selectbox("Sort by", options=list(sort_by_options.keys()), index=0)
    sort_by_col = sort_by_options[sort_by_label]
    
    sort_order = st.selectbox("Order", options=["Descending", "Ascending"])
    sorted_df = result_df.sort_values(by=sort_by_col, ascending=(sort_order == "Ascending")).reset_index(drop=True)
    
    display_cols = ['player_name', 'age', 'position', 'foot', 'market_value', 'transfer_probability_percent']
    display_cols_exist = [col for col in display_cols if col in sorted_df.columns]
    df_display = sorted_df[display_cols_exist]

    st.dataframe(df_display, use_container_width=True)

    player_names = sorted_df['player_name'].tolist()
    if player_names:
        selected_player_name = st.selectbox("👤 Select a player to view details", player_names)
        player_row = sorted_df[sorted_df['player_name'] == selected_player_name].iloc[0]

        st.markdown('<p class="section-header">⭐ Selected Player Info</p>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2.5, 1, 1.5, 1, 2])
            with col1:
                st.subheader(player_row.get("player_name", "N/A"))
                st.write(f"{player_row.get('club_name', 'N/A')} | {player_row.get('position', 'N/A')}")
            with col2:
                st.metric("Age", f"{player_row.get('age', 0):.0f}")
            with col3:
                st.metric("Market Value", player_row.get('market_value', '€0M'))
            with col4:
                st.metric("Foot", player_row.get("foot", "N/A"))
            with col5:
                st.metric("Transfer Probability", player_row.get("transfer_probability_percent", "N/A"))
        
        try:
            # [수정] 차트용 데이터 로드 시에도 encoding='cp949' 지정
            db1 = pd.read_csv(DB1_FILE_PATH, encoding='cp949')

            # 이하 시각화 로직은 동일
            # ...

        except Exception as e:
            st.error(f"시즌별 데이터 시각화 중 오류 발생: {e}")

else:
    st.info("조건에 맞는 선수가 없습니다. 필터를 조정해 주세요.")


st.markdown("""
        <div class="footer">
            <p>©2025 Project by SKN18 2nd Project 4th team</p>
        </div>
    """, unsafe_allow_html=True)