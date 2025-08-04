import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from database.database import connect_db
from utilities.money_utility import get_announcement_data, get_subsidy_data, get_top5_models
import numpy as np
import json

# 페이지 설정
st.set_page_config(
    page_title="보조금 정보",
    page_icon="💰",
    layout="wide"
)

st.title("💰 친환경 자동차 보조금 정보")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["공고 현황 분석", "보조금 정보", "지역별 정책 활용 현황"])

with tab1:
    st.header("공고 현황 분석")

    # 데이터 가져오기
    announcement_data = get_announcement_data()

    if announcement_data is not None and not announcement_data.empty:
        # 스택형 막대그래프 생성
        fig = go.Figure()

        # 출고대수 (실제 출고된 수량)
        fig.add_trace(go.Bar(
            x=announcement_data['year'],
            y=announcement_data['released_count'],
            name='출고대수',
            marker_color='green',
            hovertemplate='출고대수: %{y:,}대<br>비율: %{customdata:.1f}%<extra></extra>',
            customdata=announcement_data['released_ratio']
        ))

        # 출고잔여대수 (출고되지 않은 잔여 수량)
        fig.add_trace(go.Bar(
            x=announcement_data['year'],
            y=announcement_data['remaining_count'],
            name='출고잔여대수',
            marker_color='red',
            hovertemplate='잔여대수: %{y:,}대<br>비율: %{customdata:.1f}%<extra></extra>',
            customdata=announcement_data['remaining_ratio']
        ))

        fig.update_layout(
            title="연도별 민간공고 현황",
            xaxis_title="연도",
            yaxis_title="대수",
            barmode='stack',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # 2024년 기준 지역별 공고대수 현황 표
        st.subheader("2024년 기준 지역별 공고대수 현황")

        # 데이터베이스에서 지역별 데이터 가져오기
        try:
            conn = connect_db()
            cursor = conn.cursor()
            region_query = """
            SELECT 
                지역 as region,
                차종 as vehicle_type,
                민간공고대수 as announced_count,
                출고잔여대수 as remaining_count
            FROM electronic_car 
            WHERE 년도 = 2024
            ORDER BY 지역, 차종
            """
            cursor.execute(region_query)
            data = cursor.fetchall()
            columns = ['region', 'vehicle_type', 'announced_count', 'remaining_count']
            region_data = pd.DataFrame(data, columns=columns)
            cursor.close()
            conn.close()

            if not region_data.empty:
                # 지역별로 집계
                region_summary = region_data.groupby('region').agg({
                    'announced_count': 'sum',
                    'remaining_count': 'sum'
                }).reset_index()

                region_summary = region_summary.rename(columns={
                    'region': '지역',
                    'announced_count': '민간공고대수',
                    'remaining_count': '출고잔여대수'
                })

                st.dataframe(region_summary, use_container_width=True)

                # 통계 정보
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_announced = region_summary['민간공고대수'].sum()
                    st.metric("총 민간공고대수", f"{total_announced:,}대")
                with col2:
                    total_remaining = region_summary['출고잔여대수'].sum()
                    st.metric("총 출고잔여대수", f"{total_remaining:,}대")
                with col3:
                    remaining_ratio = (total_remaining / total_announced) * 100
                    st.metric("잔여 비율", f"{remaining_ratio:.1f}%")
            else:
                st.warning("지역별 공고 현황 데이터를 가져올 수 없습니다.")
        except Exception as e:
            st.warning("지역별 공고 현황 데이터를 가져올 수 없습니다.")
    else:
        st.warning("데이터베이스에서 공고 현황 데이터를 가져올 수 없습니다.")

with tab2:
    st.header("보조금 정보")

    # 보조금 정보 탭
    subsidy_tab1, subsidy_tab2 = st.tabs(["2024년 1대당 지원금", "자동차모델 TOP5"])

    with subsidy_tab1:
        st.subheader("2024년 1대당 지원금")

        # 차종 선택
        vehicle_type = st.selectbox(
            "차종을 선택하세요:",
            ["전기차", "수소차"]
        )

        if vehicle_type == "전기차":
            # 전기차 보조금 정보
            electric_subsidy = get_subsidy_data("electric")

            if electric_subsidy is not None and not electric_subsidy.empty:
                st.dataframe(electric_subsidy, use_container_width=True)
            else:
                st.warning("전기차 보조금 데이터를 가져올 수 없습니다.")

        elif vehicle_type == "수소차":
            # 수소차 보조금 정보
            hydrogen_subsidy = get_subsidy_data("hydrogen")

            if hydrogen_subsidy is not None and not hydrogen_subsidy.empty:
                st.dataframe(hydrogen_subsidy, use_container_width=True)
            else:
                st.warning("수소차 보조금 데이터를 가져올 수 없습니다.")

    with subsidy_tab2:
        st.subheader("자동차모델 TOP5")

        # 데이터베이스에서 실제 지역 목록 가져오기
        try:
            conn = connect_db()
            cursor = conn.cursor()
            region_list_query = """
            SELECT DISTINCT 지역 as region
            FROM electronic_car
            WHERE 년도 = 2024
            ORDER BY 지역
            """
            cursor.execute(region_list_query)
            data = cursor.fetchall()
            columns = ['region']
            region_list_data = pd.DataFrame(data, columns=columns)
            cursor.close()
            conn.close()

            if not region_list_data.empty:
                # 지역 목록 생성 (전체 + 실제 지역들)
                available_regions = ["전체"] + region_list_data['region'].tolist()

                # 지역 선택
                region = st.selectbox(
                    "지역을 선택하세요:",
                    available_regions
                )

                # TOP5 모델 데이터 가져오기
                top5_data = get_top5_models(region)

                if top5_data is not None and not top5_data.empty:
                    st.dataframe(top5_data, use_container_width=True)
                else:
                    st.warning("TOP5 모델 데이터를 가져올 수 없습니다.")
            else:
                st.warning("지역 데이터를 가져올 수 없습니다.")
        except Exception as e:
            st.warning("지역 데이터를 가져올 수 없습니다.") 


# -------------------------지역별 정책 활용 현황---------------------------------------------------
with tab3:
    st.header("지역별 정책 활용 현황")

    # --- 연도별 데이터 로드 ---
    try:
        conn = connect_db()
        years_df = pd.read_sql("SELECT DISTINCT 년도 AS year FROM electronic_car ORDER BY 년도", conn)
        years = years_df["year"].tolist()
        sel_year = st.selectbox("연도 선택:", years, index=(len(years) - 1 if years else 0))

        sql = """
        SELECT 
            지역 AS region,
            민간공고대수 AS announced_count,
            출고잔여대수 AS remaining_count
        FROM electronic_car
        WHERE 년도 = %s
        """
        df = pd.read_sql(sql, conn, params=[sel_year])
        conn.close()
    except Exception as e:
        df = pd.DataFrame()
        st.warning("에러", e)

    if not df.empty:
        # --- 지역 합산하여 계산 ---
        region_summary = (df.groupby("region", as_index=False)
                            .agg(announced_count=("announced_count","sum"),
                                remaining_count=("remaining_count","sum")))
        region_summary["released_count"] = (region_summary["announced_count"] - region_summary["remaining_count"]).clip(lower=0)

        safe_den = region_summary["announced_count"].replace(0, np.nan)
        region_summary["정책활용도(%)"] = (region_summary["released_count"] / safe_den * 100).round(1).fillna(0)

        # --- GeoJSON 로드 ---
        geojson_path = "./skorea-provinces-geo.json"
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                korea_geo = json.load(f)
        except FileNotFoundError:
            korea_geo = None
            st.warning("GeoJSON 파일을 찾을 수 없어. 경로를 확인해줘: ./skorea-provinces-geo.json")

        # --- JSON 지역명 속성키를 찾아서 자동감지 ---
        def detect_featureid_key(geo):
            if not geo or "features" not in geo or not geo["features"]:
                return None
            props = geo["features"][0].get("properties", {})
            for k in ["CTP_KOR_NM", "CTP_ENG_NM", "NAME_1", "name"]:
                if k in props:
                    return f"properties.{k}"
            return f"properties.{list(props.keys())[0]}" if props else None

        featureidkey = detect_featureid_key(korea_geo) if korea_geo else None

        # --- 지역명 매핑 ---
        kor_to_eng = {
            "서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon",
            "광주": "Gwangju", "대전": "Daejeon", "울산": "Ulsan", "세종": "Sejong",
            "경기": "Gyeonggi-do", "강원": "Gangwon-do",
            "충북": "Chungcheongbuk-do", "충남": "Chungcheongnam-do",
            "전북": "Jeollabuk-do", "전남": "Jeollanam-do",
            "경북": "Gyeongsangbuk-do", "경남": "Gyeongsangnam-do",
            "제주": "Jeju-do",
        }

        def normalize_for_geo(name, featureidkey_str):
            key = featureidkey_str.split(".")[-1] if featureidkey_str else ""
            if key in ["CTP_KOR_NM", "name"]:
                return name
            return kor_to_eng.get(name, name)

        if korea_geo and featureidkey:
            region_summary["지도매칭명"] = region_summary["region"].apply(
                lambda x: normalize_for_geo(x, featureidkey)
            )

            # --- 레이블 표시 ---
            st.markdown(f"{sel_year}년 표시 지표: 정책활용도(%)")
            fig_map = px.choropleth(
                region_summary,
                geojson=korea_geo,
                locations="지도매칭명",
                featureidkey=featureidkey,
                color="정책활용도(%)", 
                hover_data={
                    "region": True,
                    "announced_count": ":,",
                    "remaining_count": ":,",
                    "정책활용도(%)": ":.1f",
                    "지도매칭명": False
                },
                labels={
                    "region": "지역",
                    "announced_count": "민간공고대수",
                    "remaining_count": "출고잔여대수",
                    "정책활용도(%)": "정책활용도(%)"
                },
                title=None
            )
            # 0~100% 범위로 고정
            fig_map.update_coloraxes(cmin=0, cmax=100)
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                height=600,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar=dict(title="정책활용도(%)")
            )
            st.plotly_chart(fig_map, use_container_width=True)

        else:
            st.info("GeoJSON을 불러오지 못함.")
    else:
        st.warning("선택한 연도에 대한 데이터를 찾지 못함")


