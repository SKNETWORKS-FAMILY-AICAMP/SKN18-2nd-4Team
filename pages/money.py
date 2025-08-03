import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from database.database import connect_db
from utilities.money_utility import get_announcement_data, get_subsidy_data, get_top5_models

# 페이지 설정
st.set_page_config(
    page_title="보조금 정보",
    page_icon="💰",
    layout="wide"
)

st.title("💰 친환경 자동차 보조금 정보")

# 탭 생성
tab1, tab2 = st.tabs(["공고 현황 분석", "보조금 정보"])

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