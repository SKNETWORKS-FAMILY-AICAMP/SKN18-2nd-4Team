import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
<<<<<<< Updated upstream
<<<<<<< Updated upstream
import pathlib

def cwd():
    """현재 작업 디렉토리를 반환"""
    return pathlib.Path.cwd()
=======
=======
>>>>>>> Stashed changes
import os
import base64
import plotly.graph_objects as pgo

# CSS 스타일 추가
CSS = """
<style>
:root{
  --brand:#7B19BD; --ink:#111827; --muted:#6B7280;
  --bg:#fff; --card:#fff; --line:#E5E7EB; --radius:14px;
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
* {
  color: #e5e7eb !important;
}

/* 제목과 서브텍스트 강제 설정 */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
.stMarkdown p, .stMarkdown div, .stMarkdown span,
.stText, .stText *,
.stMarkdown, .stMarkdown * {
  color: #e5e7eb !important;
}

/* 선수정보 카드 내부 텍스트 색상 설정 - 예외 처리 */
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

/* 필터 입력 요소들의 텍스트는 검정색으로 */
.stTextInput input,
.stSelectbox select,
.stNumberInput input,
.stTextArea textarea,
.stSlider input,
.stSelectbox > div > div,
.stTextInput > div > div,
.stNumberInput > div > div,
.stSelectbox .stSelectbox > div,
.stTextInput .stTextInput > div,
.stNumberInput .stNumberInput > div,
input[type="text"],
input[type="number"],
select,
textarea {
  color: #111827 !important;  /* 검정색 */
}

/* 드롭다운 메뉴 내부 옵션들도 검정색으로 */
.stSelectbox [role="listbox"] *,
.stSelectbox [role="option"],
.stSelectbox .stSelectbox [role="listbox"] *,
.stSelectbox .stSelectbox [role="option"],
div[data-baseweb="select"] [role="listbox"] *,
div[data-baseweb="select"] [role="option"],
div[data-baseweb="select"] ul li,
div[data-baseweb="select"] ul li *,
div[data-baseweb="select"] div[role="option"],
div[data-baseweb="select"] div[role="option"] *,
.stSelectbox div[role="listbox"] ul li,
.stSelectbox div[role="listbox"] ul li *,
.stSelectbox div[role="listbox"] div[role="option"],
.stSelectbox div[role="listbox"] div[role="option"] * {
  color: #111827 !important;  /* 검정색 */
}

/* 필터 설명 부분(라벨)은 밝은 회색으로 */
.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stSlider label {
  color: #e5e7eb !important;  /* 밝은 회색 */
}

/* Streamlit의 모든 텍스트 요소 강제 설정 */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp div, .stApp span, .stApp label,
.stApp .stMarkdown, .stApp .stText,
.stApp .stMarkdown *, .stApp .stText *,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4,
div[data-testid="stMarkdownContainer"] h5,
div[data-testid="stMarkdownContainer"] h6,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] div,
div[data-testid="stMarkdownContainer"] span {
  color: #e5e7eb !important;
}

/* 드롭다운 메뉴 내부 모든 텍스트 강제 검정색 */
div[data-baseweb="select"] *,
.stSelectbox *,
.stSelectbox div *,
.stSelectbox span *,
div[data-baseweb="select"] div *,
div[data-baseweb="select"] span *,
div[data-baseweb="select"] ul *,
div[data-baseweb="select"] li * {
  color: #111827 !important;  /* 검정색 */
}

/* 필터 입력 요소들만 검정색으로 - 더 구체적인 선택자 */
.stTextInput input,
.stNumberInput input,
.stSelectbox select,
.stSelectbox input,
input[type="text"],
input[type="number"],
select,
textarea {
  color: #111827 !important;  /* 검정색 */
}

/* 드롭다운 옵션들 검정색 */
.stSelectbox [role="listbox"] *,
.stSelectbox [role="option"],
div[data-baseweb="select"] [role="listbox"] *,
div[data-baseweb="select"] [role="option"],
div[data-baseweb="select"] ul li * {
  color: #111827 !important;  /* 검정색 */
}

/* 필터 라벨들만 밝은 회색으로 - 더 강력한 선택자 */
.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stSlider label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stTextInput"] > label,
div[data-testid="stNumberInput"] > label,
.stSelectbox > label,
.stTextInput > label,
.stNumberInput > label {
  color: #e5e7eb !important;  /* 밝은 회색 */
}

/* 모든 라벨을 밝은 회색으로 강제 설정 */
label {
  color: #e5e7eb !important;  /* 밝은 회색 */
}

/* 최강 우선순위로 라벨 강제 설정 */
.stApp label,
.stApp .stSelectbox label,
.stApp .stTextInput label,
.stApp .stNumberInput label,
.stApp .stSlider label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
.stSelectbox > div > label,
.stTextInput > div > label,
.stNumberInput > div > label {
  color: #e5e7eb !important;  /* 밝은 회색 */
}

/* 최강 우선순위로 입력 요소 강제 검정색 */
.stApp input,
.stApp select,
.stApp textarea,
.stApp .stTextInput input,
.stApp .stNumberInput input,
.stApp .stSelectbox select,
.stApp .stSelectbox input,
div[data-testid="stSelectbox"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  color: #111827 !important;  /* 검정색 */
}

/* 선수정보 관련 모든 요소들 */
.player-card, .player-card *,
.player-info, .player-info *,
.metric-container, .metric-container *,
.stMetric, .stMetric * {
  color: #111827 !important;  /* 기본 검정색 */
}

/* 선수정보 카드에서 키(라벨)는 회색 */
.player-card .k, .player-card .metric-label,
.player-info .k, .player-info .metric-label,
.metric-container .k, .metric-container .metric-label,
.stMetric .k, .stMetric .metric-label,
.card .k, .card .metric-label {
  color: #6B7280 !important;  /* 회색 */
}

/* 선수정보 카드에서 값은 검정색 */
.player-card .v, .player-card .metric-val,
.player-info .v, .player-info .metric-val,
.metric-container .v, .metric-container .metric-val,
.stMetric .v, .stMetric .metric-val,
.card .v, .card .metric-val {
  color: #111827 !important;  /* 검정색 */
}

/* 선수 이름은 검정색 - 더 강력한 선택자 */
.player-card h1, .player-card h2, .player-card h3, .player-card h4, .player-card h5, .player-card h6,
.player-info h1, .player-info h2, .player-info h3, .player-info h4, .player-info h5, .player-info h6,
.metric-container h1, .metric-container h2, .metric-container h3, .metric-container h4, .metric-container h5, .metric-container h6,
.stMetric h1, .stMetric h2, .stMetric h3, .stMetric h4, .stMetric h5, .stMetric h6,
.card h1, .card h2, .card h3, .card h4, .card h5, .card h6,
.player-name, .player-title, .player-header,
div[data-testid="metric-container"] h1,
div[data-testid="metric-container"] h2,
div[data-testid="metric-container"] h3,
div[data-testid="metric-container"] h4,
div[data-testid="metric-container"] h5,
div[data-testid="metric-container"] h6 {
  color: #111827 !important;  /* 검정색 */
}

/* 최강 우선순위로 선수 이름 강제 검정색 설정 */
html body div[data-testid="stApp"] .card h1,
html body div[data-testid="stApp"] .card h2,
html body div[data-testid="stApp"] .card h3,
html body div[data-testid="stApp"] .card h4,
html body div[data-testid="stApp"] .card h5,
html body div[data-testid="stApp"] .card h6,
html body div[data-testid="stApp"] .player-card h1,
html body div[data-testid="stApp"] .player-card h2,
html body div[data-testid="stApp"] .player-card h3,
html body div[data-testid="stApp"] .player-card h4,
html body div[data-testid="stApp"] .player-card h5,
html body div[data-testid="stApp"] .player-card h6,
html body div[data-testid="stApp"] .player-info h1,
html body div[data-testid="stApp"] .player-info h2,
html body div[data-testid="stApp"] .player-info h3,
html body div[data-testid="stApp"] .player-info h4,
html body div[data-testid="stApp"] .player-info h5,
html body div[data-testid="stApp"] .player-info h6,
html body div[data-testid="stApp"] .metric-container h1,
html body div[data-testid="stApp"] .metric-container h2,
html body div[data-testid="stApp"] .metric-container h3,
html body div[data-testid="stApp"] .metric-container h4,
html body div[data-testid="stApp"] .metric-container h5,
html body div[data-testid="stApp"] .metric-container h6,
html body div[data-testid="stApp"] .stMetric h1,
html body div[data-testid="stApp"] .stMetric h2,
html body div[data-testid="stApp"] .stMetric h3,
html body div[data-testid="stApp"] .stMetric h4,
html body div[data-testid="stApp"] .stMetric h5,
html body div[data-testid="stApp"] .stMetric h6 {
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

/* highlight 클래스 - 25/26 Premier League 색상 */
.highlight {
  color: #9337D1 !important;  /* 보라색 */
  font-weight: 800 !important;
}

/* 더 강력한 선택자로 highlight 색상 강제 설정 */
.page-title .highlight,
.page-title-container .highlight,
h1 .highlight,
h1 span,
.page-title span,
.stApp .highlight,
html body .highlight,
div[data-testid="stMarkdownContainer"] .highlight,
div[data-testid="stMarkdownContainer"] h1 span {
  color: #9337D1 !important;  /* 보라색 */
  font-weight: 800 !important;
  background: none !important;
  text-shadow: none !important;
}

/* 최강 우선순위로 span 요소 강제 설정 */
html body div[data-testid="stApp"] h1 span,
html body div[data-testid="stApp"] .page-title span,
html body div[data-testid="stApp"] .page-title-container span,
html body div[data-testid="stApp"] div[data-testid="stMarkdownContainer"] h1 span {
  color: #9337D1 !important;  /* 보라색 */
  font-weight: 800 !important;
  background: none !important;
  text-shadow: none !important;
}

/* 최종 강제 설정 - 모든 라벨을 밝은 회색으로 */
html body .stApp label,
html body .stApp .stSelectbox label,
html body .stApp .stTextInput label,
html body .stApp .stNumberInput label,
html body .stApp .stSlider label {
  color: #e5e7eb !important;  /* 밝은 회색 */
}

/* 최종 강제 설정 - 모든 입력 요소를 검정색으로 */
html body .stApp input,
html body .stApp select,
html body .stApp textarea,
html body .stApp .stTextInput input,
html body .stApp .stNumberInput input,
html body .stApp .stSelectbox select {
  color: #111827 !important;  /* 검정색 */
}

/* 최강 우선순위 CSS - 모든 요소 강제 설정 */
html body div[data-testid="stApp"] label,
html body div[data-testid="stApp"] .stSelectbox label,
html body div[data-testid="stApp"] .stTextInput label,
html body div[data-testid="stApp"] .stNumberInput label,
html body div[data-testid="stApp"] .stSlider label,
html body div[data-testid="stApp"] div[data-testid="stSelectbox"] label,
html body div[data-testid="stApp"] div[data-testid="stTextInput"] label,
html body div[data-testid="stApp"] div[data-testid="stNumberInput"] label {
  color: #e5e7eb !important;  /* 밝은 회색 */
}

html body div[data-testid="stApp"] input,
html body div[data-testid="stApp"] select,
html body div[data-testid="stApp"] textarea,
html body div[data-testid="stApp"] .stTextInput input,
html body div[data-testid="stApp"] .stNumberInput input,
html body div[data-testid="stApp"] .stSelectbox select,
html body div[data-testid="stApp"] div[data-testid="stSelectbox"] input,
html body div[data-testid="stApp"] div[data-testid="stTextInput"] input,
html body div[data-testid="stApp"] div[data-testid="stNumberInput"] input {
  color: #111827 !important;  /* 검정색 */
}

/* 드롭다운 옵션들 최강 설정 */
html body div[data-testid="stApp"] .stSelectbox [role="listbox"] *,
html body div[data-testid="stApp"] .stSelectbox [role="option"],
html body div[data-testid="stApp"] div[data-baseweb="select"] *,
html body div[data-testid="stApp"] div[data-baseweb="select"] ul li * {
  color: #111827 !important;  /* 검정색 */
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
  max-width: 982px !important;
  margin: 0 auto !important;
}

/* block-container 설정 - 콘텐츠 영역은 흰색 유지 */
.main .block-container {
  max-width: 982px !important;
  margin: 0 auto !important;
  padding: 1rem 24px !important;
  background: white !important;
  border-radius: 12px !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

/* Top Navigation */
p.page-title { font-size: 14px !important; font-weight: 700; color: #333; margin: 0; padding-top: 8px;}
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

/* 공용 카드/차트 */
.card{background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:18px; box-shadow:0 1px 2px rgba(0,0,0,.03);}

/* 플레이어 헤더 */
.player-header{
  display:grid; grid-template-columns:320px 1fr; gap:24px; align-items:center;
}
@media (max-width:740px){ .player-header{ grid-template-columns:1fr; } }
.player-photo-img{
  width:100%; height:auto; object-fit:contain;
  border-radius:10px; border:1px solid #ECECEF; background:#FAFAFB;
}

/* 메타 정보 (키-값) */
.kv-grid{
  display:grid; grid-template-columns:repeat(3, 1fr); gap:18px 24px; margin:6px 0 0;
}
@media (max-width:740px){ .kv-grid{ grid-template-columns:repeat(2, 1fr);} }
.kv{ padding:6px 0 10px; border-bottom:1px dashed #eee; }
.kv .k{ color:#8A8F98; font-size:12px; letter-spacing:.02em; }
.kv .k::before{ content:":: "; }
.kv .v{ margin-top:4px; font-weight:800; font-size:18px; color:#111; }

/* 시즌 메트릭 */
.metrics-wide{ display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; margin-top:16px; }
@media (max-width:740px){ .metrics-wide{ grid-template-columns:repeat(3, 1fr);} }
@media (max-width:480px){ .metrics-wide{ grid-template-columns:repeat(2, 1fr);} }
.metric-box{ background:#F5F5F7; border:1px solid #ECECEF; border-radius:10px; padding:10px 12px; }
.metric-box .metric-label{ color:#666; font-size:12px; }
.metric-box .metric-val{ font-weight:800; font-size:20px; line-height:1; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# JavaScript로 CSS 강제 적용
st.markdown("""
<script>
// 페이지 로드 후 CSS 강제 적용
document.addEventListener('DOMContentLoaded', function() {
    // 모든 라벨을 밝은 회색으로
    const labels = document.querySelectorAll('label');
    labels.forEach(label => {
        label.style.color = '#e5e7eb !important';
    });
    
    // 모든 입력 요소를 검정색으로
    const inputs = document.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.style.color = '#111827 !important';
    });
    
    // 드롭다운 옵션들을 검정색으로
    const options = document.querySelectorAll('[role="option"], [role="listbox"] *');
    options.forEach(option => {
        option.style.color = '#111827 !important';
    });
    
    // highlight 클래스 요소들을 보라색으로
    const highlights = document.querySelectorAll('.highlight');
    highlights.forEach(highlight => {
        highlight.style.color = '#9337D1 !important';
        highlight.style.fontWeight = '800 !important';
        highlight.style.background = 'none !important';
        highlight.style.textShadow = 'none !important';
    });
    
    // h1 내의 모든 span 요소들을 보라색으로
    const h1Spans = document.querySelectorAll('h1 span');
    h1Spans.forEach(span => {
        span.style.color = '#9337D1 !important';
        span.style.fontWeight = '800 !important';
        span.style.background = 'none !important';
        span.style.textShadow = 'none !important';
    });
    
    // "25/26 Premier League" 텍스트가 포함된 span 찾기
    const allSpans = document.querySelectorAll('span');
    allSpans.forEach(span => {
        if (span.textContent.includes('25/26 Premier League')) {
            span.style.color = '#9337D1 !important';
            span.style.fontWeight = '800 !important';
            span.style.background = 'none !important';
            span.style.textShadow = 'none !important';
        }
    });
    
    // 선수 이름을 검정색으로 설정 - 더 강력한 방법
    const playerNames = document.querySelectorAll('.player-card h1, .player-card h2, .player-card h3, .player-card h4, .player-card h5, .player-card h6, .player-info h1, .player-info h2, .player-info h3, .player-info h4, .player-info h5, .player-info h6, .metric-container h1, .metric-container h2, .metric-container h3, .metric-container h4, .metric-container h5, .metric-container h6, .stMetric h1, .stMetric h2, .stMetric h3, .stMetric h4, .stMetric h5, .stMetric h6, .card h1, .card h2, .card h3, .card h4, .card h5, .card h6, .player-name, .player-title, .player-header');
    playerNames.forEach(name => {
        name.style.setProperty('color', '#111827', 'important');
    });
    
    // 모든 h1, h2, h3 요소를 검정색으로 (선수 이름 포함)
    const allHeadings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    allHeadings.forEach(heading => {
        // 선수 정보 카드 내부의 제목들만 검정색으로
        if (heading.closest('.player-card') || heading.closest('.player-info') || 
            heading.closest('.metric-container') || heading.closest('.stMetric') || 
            heading.closest('.card') || heading.closest('[data-testid="metric-container"]')) {
            heading.style.setProperty('color', '#111827', 'important');
        }
    });
    
    // 모든 div 요소에서 선수 이름 찾기
    const allDivs = document.querySelectorAll('div');
    allDivs.forEach(div => {
        if (div.textContent && div.textContent.length > 0 && div.textContent.length < 50) {
            // 선수 이름으로 보이는 텍스트를 검정색으로
            if (div.closest('.player-card') || div.closest('.player-info') || 
                div.closest('.metric-container') || div.closest('.stMetric') || 
                div.closest('.card')) {
                div.style.setProperty('color', '#111827', 'important');
            }
        }
    });
});

// Streamlit이 동적으로 요소를 추가할 때마다 실행
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.addedNodes.length) {
            // 새로 추가된 라벨들
            const newLabels = document.querySelectorAll('label');
            newLabels.forEach(label => {
                label.style.color = '#e5e7eb !important';
            });
            
            // 새로 추가된 입력 요소들
            const newInputs = document.querySelectorAll('input, select, textarea');
            newInputs.forEach(input => {
                input.style.color = '#111827 !important';
            });
            
            // 새로 추가된 highlight 요소들
            const newHighlights = document.querySelectorAll('.highlight');
            newHighlights.forEach(highlight => {
                highlight.style.color = '#9337D1 !important';
                highlight.style.fontWeight = '800 !important';
            });
            
            // 새로 추가된 선수 이름들
            const newPlayerNames = document.querySelectorAll('.player-card h1, .player-card h2, .player-card h3, .player-card h4, .player-card h5, .player-card h6, .player-info h1, .player-info h2, .player-info h3, .player-info h4, .player-info h5, .player-info h6, .metric-container h1, .metric-container h2, .metric-container h3, .metric-container h4, .metric-container h5, .metric-container h6, .stMetric h1, .stMetric h2, .stMetric h3, .stMetric h4, .stMetric h5, .stMetric h6, .card h1, .card h2, .card h3, .card h4, .card h5, .card h6');
            newPlayerNames.forEach(name => {
                name.style.setProperty('color', '#111827', 'important');
            });
            
            // 새로 추가된 모든 제목 요소들
            const newHeadings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            newHeadings.forEach(heading => {
                if (heading.closest('.player-card') || heading.closest('.player-info') || 
                    heading.closest('.metric-container') || heading.closest('.stMetric') || 
                    heading.closest('.card') || heading.closest('[data-testid="metric-container"]')) {
                    heading.style.setProperty('color', '#111827', 'important');
                }
            });
        }
    });
});

// 전체 문서를 관찰
observer.observe(document.body, {
    childList: true,
    subtree: true
});
</script>
""", unsafe_allow_html=True)

# ============================================================================
# 🖼️ 이미지 로드 함수
# ============================================================================
def get_local_img(player_name):
    if player_name is None or player_name == "" or not isinstance(player_name, str):
        return None  # 선수 이름이 없거나 유효하지 않으면 바로 None 반환

    img_folder = "data/streamlit/data/imgs/player"
    name_variants = [
        player_name,
        player_name.replace(" ", "_"),
        player_name.replace(" ", "-"),
        player_name.replace(" ", ""),
    ]
    
    exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")
    for variant in name_variants:
        if variant:  # 빈 문자열이 아닌 경우만 처리
            for ext in exts:
                img_path = os.path.join(img_folder, f"{variant}{ext}")
                if os.path.exists(img_path):
                    return img_path
    
    # fallback default 이미지
    default_img = os.path.join(img_folder, "default.png")
    return default_img if os.path.exists(default_img) else None

# ============================================================================
# 🧭 Navigation Helper
# ============================================================================
def go(page_name: str):
    # pages 폴더에 있는 파일명을 기반으로 페이지를 전환합니다.
    page_map = {
        "home": "./app.py",
        "players": "pages/player_search.py",
        "predict": "pages/transfer_predictor.py",
    }
    target_page = page_map.get(page_name)
    if target_page:
        st.switch_page(target_page)

def page_home():
    # Top Navigation
    nav_cols = st.columns([0.6, 0.2, 0.2])
    with nav_cols[0]:
        st.markdown('<p class="page-title">Transfer Predictor</p>', unsafe_allow_html=True)
    with nav_cols[1]:
        if st.button("Home", key="nav_home", use_container_width=True):
            go("home") 
    with nav_cols[2]:
        if st.button("Player Search", key="nav_players", use_container_width=True):
            go("players")
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

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
  background: rgba(0, 0, 0, 0.5);
  color: #e5e7eb !important;
}
.stApp > header { display: none; }
.custom-header {
  display: flex; justify-content: space-between; align-items: center;
  width: 100%; padding: 16px 40px; background-color: rgba(0, 0, 0, 0.5);
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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
DB2_FILE_PATH = str(cwd() / "data" / "streamlit" / "data" / "DB2.csv")
DB1_FILE_PATH = str(cwd() / "data" / "streamlit" / "data" / "DB1.csv")
=======
DB2_FILE_PATH = "./data/streamlit/data/DB2.csv"
DB1_FILE_PATH = "./data/streamlit/data/DB1.csv"
>>>>>>> Stashed changes
=======
DB2_FILE_PATH = "./data/streamlit/data/DB2.csv"
DB1_FILE_PATH = "./data/streamlit/data/DB1.csv"
>>>>>>> Stashed changes
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
# [수정] 2. 페이지 상단에 내비게이션 바를 표시하기 위해 함수 호출
page_home()

st.markdown("""<div class="page-title-container"><h1 class="page-title">Predict Transfer Probabilities<br>Across <span style="color:#9337D1 !important; font-weight:800 !important; background:none !important; text-shadow:none !important;">25/26 Premier League</span> Players</h1></div>""", unsafe_allow_html=True)

# 필터 섹션만 컨테이너로 감싸기
st.markdown('<p class="section-header">▼ Player Filters</p>', unsafe_allow_html=True)

with st.container():
    # 배경 이미지 로드
    try:
        with open("streamlit/pages/imgs/transfer-1.jpg", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        bg_image = f"data:image/jpeg;base64,{b64}"
    except:
        bg_image = None
    
    st.markdown(f"""
    <style>
    /* 필터 컨테이너 스타일링 */
    .stContainer {{
        background: {'url(' + bg_image + ')' if bg_image else '#f8fafc'} !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin: 16px 0 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        position: relative !important;
    }}
    
    /* 필터 컨테이너 내부 요소들에 반투명 배경 추가 */
    .stContainer .stTextInput,
    .stContainer .stSelectbox,
    .stContainer .stNumberInput,
    .stContainer .stButton,
    .stContainer .stAlert {{
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(5px) !important;
    }}
    
     /* 필터 컨테이너 내부 입력 필드들 */
    .stContainer .stTextInput > div > div > input,
    .stContainer .stSelectbox > div > div > select,
    .stContainer .stNumberInput > div > div > input {{
        background: white !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        color: #111827 !important;
    }}
    
    /* 필터 라벨들 - 밝은 회색으로 설정 */
    .stContainer label,
    .stContainer .stSelectbox label,
    .stContainer .stTextInput label,
    .stContainer .stNumberInput label,
    .stContainer .stSlider label,
    div[data-testid="stVerticalBlock"] label,
    div[data-testid="stVerticalBlock"] .stSelectbox label,
    div[data-testid="stVerticalBlock"] .stTextInput label,
    div[data-testid="stVerticalBlock"] .stNumberInput label,
    .stSelectbox > label,
    .stTextInput > label,
    .stNumberInput > label,
    .stSlider > label {{
        color: #e5e7eb !important;  /* 밝은 회색 */
        font-weight: 600 !important;
    }}
    
    /* 드롭다운만 검정색으로 설정 */
    .stContainer .stSelectbox select,
    .stContainer .stSelectbox [role="listbox"] *,
    .stContainer .stSelectbox [role="option"],
    .stContainer div[data-baseweb="select"] *,
    .stContainer div[data-baseweb="select"] ul li *,
    div[data-testid="stVerticalBlock"] .stSelectbox select,
    div[data-testid="stVerticalBlock"] .stSelectbox [role="listbox"] *,
    div[data-testid="stVerticalBlock"] .stSelectbox [role="option"],
    div[data-testid="stVerticalBlock"] div[data-baseweb="select"] *,
    div[data-testid="stVerticalBlock"] div[data-baseweb="select"] ul li *,
    .stSelectbox select,
    .stSelectbox [role="listbox"] *,
    .stSelectbox [role="option"],
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] ul li * {{
        color: #111827 !important;  /* 검정색 */
    }}
    
    /* 입력 필드들 - 검정색으로 설정 */
    .stContainer input,
    .stContainer textarea,
    .stContainer .stTextInput input,
    .stContainer .stNumberInput input,
    div[data-testid="stVerticalBlock"] input,
    div[data-testid="stVerticalBlock"] textarea,
    div[data-testid="stVerticalBlock"] .stTextInput input,
    div[data-testid="stVerticalBlock"] .stNumberInput input {{
        color: #111827 !important;  /* 검정색 */
    }}
    
    /* 필터 컨테이너 내 기타 텍스트 - 밝은 회색으로 설정 */
    .stContainer .stText,
    .stContainer .stMarkdown,
    .stContainer p,
    .stContainer div,
    .stContainer span,
    div[data-testid="stVerticalBlock"] .stText,
    div[data-testid="stVerticalBlock"] .stMarkdown,
    div[data-testid="stVerticalBlock"] p,
    div[data-testid="stVerticalBlock"] div,
    div[data-testid="stVerticalBlock"] span {{
        color: #e5e7eb !important;  /* 밝은 회색 */
    }}
    
    /* 필터 컨테이너 내부 버튼 */
    .stContainer .stButton > button {{
        background: rgba(59, 130, 246, 0.9) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
    }}
    
    .stContainer .stButton > button:hover {{
        background: rgba(37, 99, 235, 0.9) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    filter_button = False

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
    df_display = sorted_df[display_cols_exist].copy()
    
    # Market Value를 €M 단위로 변환
    if 'market_value' in df_display.columns:
        df_display['market_value'] = df_display['market_value'].apply(
            lambda x: f"€{x/1_000_000:.1f}M" if isinstance(x, (int, float)) and x > 0 else "N/A"
        )

    st.dataframe(df_display, use_container_width=True)

    player_names = sorted_df['player_name'].tolist()
    if player_names:
        selected_player_name = st.selectbox("👤 Select a player to view details", player_names)
        player_row = sorted_df[sorted_df['player_name'] == selected_player_name].iloc[0]

       
        
        # 선수 정보 카드 (player_search.py 스타일 적용)
        player_name = player_row.get("player_name", "N/A")
        club_name = player_row.get('club_name', 'N/A')
        position = player_row.get('position', 'N/A')
        age = player_row.get('age', 0)
        foot = player_row.get("foot", "N/A")
        market_value = player_row.get('market_value', 0)
        transfer_prob = player_row.get("transfer_probability_percent", "N/A")
        
        # Market Value 포맷팅
        if isinstance(market_value, (int, float)) and market_value > 0:
            market_value_display = f"{market_value/1_000_000:.1f}"
        else:
            market_value_display = "-"
        
        # 선수 이미지 로드 (player_search.py와 동일한 방식)
        img_path = get_local_img(player_name)
        if img_path:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(img_path)[-1][1:]  # 확장자 추출 (e.g. png, jpg)
            img_tag = f"<img class='player-photo-img' src='data:image/{ext};base64,{b64}'/>"
        else:
            # fallback도 없을 경우 최소한의 placeholder
            img_tag = "<div class='player-photo-img' style='background:#F1F1F4;'></div>"
        
        # 키-값 아이템들 (요청된 정보만)
        kv_items = [
            ("Team", club_name or "-"),
            ("Position", position or "-"),
            ("Age", f"{age:.0f}" if age > 0 else "-"),
            ("Foot", foot or "-"),
            ("Market Value", f"€{market_value_display}M" if market_value_display != "-" else "-"),
            ("Transfer Prob", transfer_prob if transfer_prob != "N/A" else "-"),
        ]
        
        kv_html = ["<div class='kv-grid'>"]
        for k, v in kv_items:
            kv_html.append(
                f"<div class='kv'><div class='k'>{k}</div>"
                f"<div class='v'>{v}</div></div>"
            )
        kv_html.append("</div>")
        
        # 하단 메트릭 박스 제거
        metrics_html = ""
        
        header_html = f"""
        <div class="card">
            <div class="player-header">
            <div>{img_tag}</div>
            <div>
                <h3 style="margin:0 0 12px 0; font-weight:800; font-size:26px;">{player_name}</h3>
                {''.join(kv_html)}
            </div>
            </div>
            {metrics_html}
        </div>
        """
        
        st.markdown(header_html, unsafe_allow_html=True)
        
        try:
            # [수정] 차트용 데이터 로드 시에도 encoding='cp949' 지정
            db1 = pd.read_csv(DB1_FILE_PATH, encoding='cp949')

            
            # 선수 이름으로 DB1에서 해당 선수 데이터 찾기
            player_name = player_row.get("player_name", "")
            player_data = db1[db1['player_name'] == player_name].copy()
            
            if not player_data.empty:
                # 시즌별 데이터 정리 - 데이터베이스에서 실제 시즌 가져오기
                seasons = sorted(player_data['season'].unique().tolist())
                market_values = []
                wins = []
                avg_market_values = []
                avg_wins = []
                
                for season in seasons:
                    season_data = player_data[player_data['season'] == season]
                    if not season_data.empty:
                        market_value = season_data['market_value'].iloc[0] if 'market_value' in season_data.columns else 0
                        win_count = season_data['season_win_count'].iloc[0] if 'season_win_count' in season_data.columns else 0
                    else:
                        market_value = 0
                        win_count = 0
                    
                    market_values.append(market_value)
                    wins.append(win_count)
                    
                    # 같은 포지션의 평균값 계산
                    position = player_row.get('position', '')
                    if position and position != 'N/A':
                        position_data = db1[(db1['position'] == position) & (db1['season'] == season)]
                        avg_market_value = position_data['market_value'].mean() if 'market_value' in position_data.columns and not position_data.empty else 0
                        avg_win = position_data['season_win_count'].mean() if 'season_win_count' in position_data.columns and not position_data.empty else 0
                    else:
                        avg_market_value = 0
                        avg_win = 0
                    
                    avg_market_values.append(avg_market_value)
                    avg_wins.append(avg_win)
                
                # 그래프 생성
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**<h5>Market Value by Season(€M)</h5>**", unsafe_allow_html=True)
                    
                    # Plotly를 사용한 선형 그래프
                    
                    fig_mv = pgo.Figure()
                    fig_mv.add_scatter(x=seasons, y=[mv/1_000_000 for mv in market_values], 
                                     mode="lines+markers", line=dict(color="#7B19BD"), 
                                     name="Player", marker=dict(size=8))
                    fig_mv.add_scatter(x=seasons, y=[mv/1_000_000 for mv in avg_market_values], 
                                     mode="lines+markers", line=dict(color="#6B7280", dash="dash"), 
                                     name="Position Average", marker=dict(size=6))
                    
                    fig_mv.update_layout(
                        height=320, showlegend=True,
                        margin=dict(t=40, l=40, r=40, b=60),
                        plot_bgcolor="#F5F5F7", paper_bgcolor="white",
                        legend=dict(x=0.5, y=-0.15, xanchor="center", yanchor="top", 
                                  orientation="h", bgcolor="rgba(255,255,255,0.8)", 
                                  bordercolor="rgba(0,0,0,0.2)", borderwidth=1)
                    )
                    st.plotly_chart(fig_mv, use_container_width=True, config={"displayModeBar": False})
                    st.caption("단위: €M (백만 유로)")
                
                with col2:
                    st.markdown("**<h5>Wins by Season</h5>**", unsafe_allow_html=True)
                    
                    # Plotly를 사용한 선형 그래프
                    fig_wins = pgo.Figure()
                    fig_wins.add_scatter(x=seasons, y=wins, 
                                       mode="lines+markers", line=dict(color="#7B19BD"), 
                                       name="Player", marker=dict(size=8))
                    fig_wins.add_scatter(x=seasons, y=avg_wins, 
                                       mode="lines+markers", line=dict(color="#6B7280", dash="dash"), 
                                       name="Position Average", marker=dict(size=6))
                    
                    fig_wins.update_layout(
                        height=320, showlegend=True,
                        margin=dict(t=40, l=40, r=40, b=60),
                        plot_bgcolor="#F5F5F7", paper_bgcolor="white",
                        legend=dict(x=0.5, y=-0.15, xanchor="center", yanchor="top", 
                                  orientation="h", bgcolor="rgba(255,255,255,0.8)", 
                                  bordercolor="rgba(0,0,0,0.2)", borderwidth=1)
                    )
                    st.plotly_chart(fig_wins, use_container_width=True, config={"displayModeBar": False})
                    st.caption("시즌별 승리 수")
            else:
                st.warning("해당 선수의 시즌별 데이터를 찾을 수 없습니다.")

        except Exception as e:
            st.error(f"시즌별 데이터 시각화 중 오류 발생: {e}")

else:
    st.info("조건에 맞는 선수가 없습니다. 필터를 조정해 주세요.")


st.markdown("""
        <div class="footer" style="text-align: right;">
            <p>©2025 Project by SKN18 2nd Project 4th team</p>
        </div>
    """, unsafe_allow_html=True)