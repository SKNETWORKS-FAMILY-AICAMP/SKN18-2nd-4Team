import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from database.database import connect_db
from utilities.app_utility import get_vehicle_registration_data, get_environmental_impact_data

# 페이지 설정
st.set_page_config(
    page_title="친환경 자동차 대시보드",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 친환경 자동차 대시보드")

# 탭 생성
tab1, tab2 = st.tabs(["자동차 등록 현황 분석", "환경 영향 분석"])

with tab1:
    st.header("자동차 등록 현황 분석d")
    
    # 데이터 가져오기
    vehicle_data = get_vehicle_registration_data()
    
    if vehicle_data is not None and not vehicle_data.empty:
        # 차종별 하이라이트 기능
        st.subheader("차종별 하이라이트")
        highlight_option = st.selectbox(
            "하이라이트할 차종을 선택하세요:",
            ["전체", "전기차", "수소차", "하이브리드"]
        )
        
        # 이중 축 그래프 생성
        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )
        
        # 첫 번째 그래프: 전체 자동차 등록대수 (선그래프)
        fig.add_trace(
            go.Scatter(
                x=vehicle_data['year'],
                y=vehicle_data['total_vehicles'],
                name="전체 자동차 등록대수",
                line=dict(color='blue', width=3),
                mode='lines+markers'
            ),
            secondary_y=False
        )
        
        # 두 번째 그래프: 친환경 자동차 등록대수 (스택형 막대그래프)
        # 하이라이트 옵션에 따라 색상과 투명도 조정
        electric_color = 'red' if highlight_option == "전기차" else 'green'
        hydrogen_color = 'red' if highlight_option == "수소차" else 'orange'
        hybrid_color = 'red' if highlight_option == "하이브리드" else 'purple'
        
        # 투명도 설정 (하이라이트된 항목은 불투명, 나머지는 반투명)
        electric_opacity = 1.0 if highlight_option == "전기차" else 0.6
        hydrogen_opacity = 1.0 if highlight_option == "수소차" else 0.6
        hybrid_opacity = 1.0 if highlight_option == "하이브리드" else 0.6
        
        # 전체 선택 시 모든 항목을 불투명하게
        if highlight_option == "전체":
            electric_opacity = hydrogen_opacity = hybrid_opacity = 1.0
            electric_color = 'green'
            hydrogen_color = 'orange'
            hybrid_color = 'purple'
        
        fig.add_trace(
            go.Bar(
                x=vehicle_data['year'],
                y=vehicle_data['electric_vehicles'],
                name="전기차",
                marker_color=electric_color,
                marker_opacity=electric_opacity,
                hovertemplate='전기차: %{y:,.0f}대<br>비율: %{customdata:.1f}%<extra></extra>',
                customdata=vehicle_data['electric_ratio']
            ),
            secondary_y=True
        )
        
        fig.add_trace(
            go.Bar(
                x=vehicle_data['year'],
                y=vehicle_data['hydrogen_vehicles'],
                name="수소차",
                marker_color=hydrogen_color,
                marker_opacity=hydrogen_opacity,
                hovertemplate='수소차: %{y:,.0f}대<br>비율: %{customdata:.1f}%<extra></extra>',
                customdata=vehicle_data['hydrogen_ratio']
            ),
            secondary_y=True
        )
        
        fig.add_trace(
            go.Bar(
                x=vehicle_data['year'],
                y=vehicle_data['hybrid_vehicles'],
                name="하이브리드",
                marker_color=hybrid_color,
                marker_opacity=hybrid_opacity,
                hovertemplate='하이브리드: %{y:,.0f}대<br>비율: %{customdata:.1f}%<extra></extra>',
                customdata=vehicle_data['hybrid_ratio']
            ),
            secondary_y=True
        )
        
        # 그래프 업데이트
        fig.update_layout(
            title=f"연도별 자동차 등록 현황 - {highlight_option} 하이라이트",
            xaxis_title="연도",
            barmode='stack',
            height=600
        )
        
        fig.update_yaxes(title_text="전체 자동차 등록대수", secondary_y=False)
        fig.update_yaxes(title_text="친환경 자동차 등록대수", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 통계 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("2024년 전체 등록대수", f"{vehicle_data.iloc[-1]['total_vehicles']:,}대")
        with col2:
            st.metric("2024년 친환경차 등록대수", f"{vehicle_data.iloc[-1]['total_eco_vehicles']:,.0f}대")
        with col3:
            eco_ratio = (vehicle_data.iloc[-1]['total_eco_vehicles'] / vehicle_data.iloc[-1]['total_vehicles']) * 100
            st.metric("친환경차 비율", f"{eco_ratio:.1f}%")
        
        # 선택된 차종의 상세 정보 표시
        if highlight_option != "전체":
            st.subheader(f"📊 {highlight_option} 상세 정보")
            
            if highlight_option == "전기차":
                selected_data = vehicle_data['electric_vehicles']
                selected_ratio = vehicle_data['electric_ratio']
            elif highlight_option == "수소차":
                selected_data = vehicle_data['hydrogen_vehicles']
                selected_ratio = vehicle_data['hydrogen_ratio']
            elif highlight_option == "하이브리드":
                selected_data = vehicle_data['hybrid_vehicles']
                selected_ratio = vehicle_data['hybrid_ratio']
            
            # 선택된 차종의 연도별 변화 그래프
            fig_detail = go.Figure()
            fig_detail.add_trace(go.Bar(
                x=vehicle_data['year'],
                y=selected_data,
                name=highlight_option,
                marker_color='red',
                hovertemplate=f'{highlight_option}: %{{y:,.0f}}대<br>비율: %{{customdata:.1f}}%<extra></extra>',
                customdata=selected_ratio
            ))
            
            fig_detail.update_layout(
                title=f"{highlight_option} 연도별 등록 현황",
                xaxis_title="연도",
                yaxis_title="등록대수",
                height=400
            )
            
            st.plotly_chart(fig_detail, use_container_width=True)
            
            # 선택된 차종의 통계 정보
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"2024년 {highlight_option} 등록대수", f"{selected_data.iloc[-1]:,.0f}대")
            with col2:
                growth_rate = ((selected_data.iloc[-1] - selected_data.iloc[0]) / selected_data.iloc[0]) * 100
                st.metric("2020년 대비 증가율", f"{growth_rate:.1f}%")
            with col3:
                st.metric(f"{highlight_option} 비율", f"{selected_ratio.iloc[-1]:.1f}%")
    else:
        st.warning("데이터베이스에서 자동차 등록 현황 데이터를 가져올 수 없습니다.")

with tab2:
    st.header("환경 영향 분석")
    
    # 데이터 가져오기
    env_data = get_environmental_impact_data()
    
    if env_data is not None and not env_data.empty:
        # 이중 축 그래프 생성
        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )
        
        # 첫 번째 그래프: 온실가스 배출량 (선그래프)
        fig.add_trace(
            go.Scatter(
                x=env_data['year'],
                y=env_data['greenhouse_gas'],
                name="온실가스 배출량",
                line=dict(color='red', width=3),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # 두 번째 그래프: 친환경 자동차 비율 (막대그래프)
        fig.add_trace(
            go.Bar(
                x=env_data['year'],
                y=env_data['eco_vehicle_ratio'],
                name="친환경 자동차 비율",
                marker_color='lightgreen',
                hovertemplate='친환경차 비율: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=False
        )
        
        fig.update_layout(
            title="연도별 환경 영향 분석",
            xaxis_title="연도",
            height=500
        )
        
        fig.update_yaxes(title_text="친환경 자동차 비율 (%)", secondary_y=False, range=[0, 20])
        fig.update_yaxes(title_text="온실가스 배출량", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 추가 분석: 지역별 온실가스 배출량 분석
        st.subheader("🌍 지역별 온실가스 배출량 분석")
        
        try:
            conn = connect_db()
            cursor = conn.cursor()
            # 실제 사용 가능한 최신 연도(2022년)로 수정
            region_gas_query = """
            SELECT 
                년도 as year,
                지역 as region,
                승용 as passenger,
                승합 as bus,
                화물 as cargo,
                특수 as special
            FROM greenhouse_gases 
            WHERE 년도 = 2022
            ORDER BY 지역
            """
            cursor.execute(region_gas_query)
            data = cursor.fetchall()
            columns = ['year', 'region', 'passenger', 'bus', 'cargo', 'special']
            region_gas_data = pd.DataFrame(data, columns=columns)
            cursor.close()
            conn.close()
            
            if not region_gas_data.empty:
                # 지역별 총 온실가스 배출량 계산
                region_gas_data['total_gas'] = region_gas_data['passenger'] + region_gas_data['bus'] + region_gas_data['cargo'] + region_gas_data['special']
                
                # 지역별 온실가스 배출량 차트
                fig_region = go.Figure()
                fig_region.add_trace(go.Bar(
                    x=region_gas_data['region'],
                    y=region_gas_data['total_gas'],
                    name='총 온실가스 배출량',
                    marker_color='orange',
                    hovertemplate='지역: %{x}<br>배출량: %{y:,}<extra></extra>'
                ))
                
                fig_region.update_layout(
                    title="2022년 지역별 온실가스 배출량",
                    xaxis_title="지역",
                    yaxis_title="온실가스 배출량",
                    height=400
                )
                
                st.plotly_chart(fig_region, use_container_width=True)
                
                # 지역별 상세 분석
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 지역별 배출량 순위")
                    
                    # 단위 표시를 위한 컨테이너
                    unit_container = st.container()
                    with unit_container:
                        # CSS를 사용해서 단위를 오른쪽 상단에 배치
                        st.markdown(
                            """
                            <style>
                            .unit-text {
                                text-align: right;
                                font-size: 14px;
                                color: #666;
                                margin-bottom: 5px;
                            }
                            </style>
                            <div class="unit-text">단위: 톤CO₂</div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    region_ranking = region_gas_data.sort_values('total_gas', ascending=False)
                    # 순위 추가 (1부터 시작)
                    region_ranking['rank'] = range(1, len(region_ranking) + 1)
                    region_ranking = region_ranking[['rank', 'region', 'total_gas']].rename(columns={
                        'rank': '순위',
                        'region': '지역',
                        'total_gas': '총 배출량'
                    })
                    # 총 배출량에 1,000 단위 구분 쉼표 추가
                    region_ranking['총 배출량'] = region_ranking['총 배출량'].apply(lambda x: f"{x:,}")
                    st.dataframe(region_ranking, use_container_width=True, hide_index=True)

                with col2:
                    st.subheader("📈 차종별 배출량 분석")
                    
                    # 단위 표시를 위한 컨테이너
                    unit_container2 = st.container()
                    with unit_container2:
                        # CSS를 사용해서 단위를 오른쪽 상단에 배치
                        st.markdown(
                            """
                            <style>
                            .unit-text2 {
                                text-align: right;
                                font-size: 14px;
                                color: #666;
                                margin-bottom: 5px;
                            }
                            </style>
                            <div class="unit-text2">단위: 톤CO₂</div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    vehicle_types = ['passenger', 'bus', 'cargo', 'special']
                    vehicle_names = ['승용', '승합', '화물', '특수']
                    
                    for i, (vehicle_type, vehicle_name) in enumerate(zip(vehicle_types, vehicle_names)):
                        total_emission = region_gas_data[vehicle_type].sum()
                        st.metric(f"{vehicle_name} 총 배출량", f"{total_emission:,}")
            else:
                st.warning("지역별 온실가스 배출량 데이터를 가져올 수 없습니다.")
        except Exception as e:
            st.warning("지역별 온실가스 배출량 데이터를 가져올 수 없습니다.")
    else:
        st.warning("데이터베이스에서 환경 영향 분석 데이터를 가져올 수 없습니다.") 