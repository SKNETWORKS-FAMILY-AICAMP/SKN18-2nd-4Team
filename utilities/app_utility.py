import pandas as pd
import numpy as np
from database.database import connect_db

def get_vehicle_registration_data():
    """
    자동차 등록 현황 데이터를 가져오는 함수
    """
    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        # environmental_vehicles 테이블에서 데이터 가져오기
        query = """
        SELECT 
            연도 as year,
            구분 as category,
            합계 as total
        FROM environmental_vehicles 
        WHERE 연도 BETWEEN 2020 AND 2024
        ORDER BY 연도, 구분
        """
        
        cursor.execute(query)
        data = cursor.fetchall()
        columns = ['year', 'category', 'total']
        df = pd.DataFrame(data, columns=columns)
        
        cursor.close()
        conn.close()
        
        if df.empty:
            return None
        
        # 데이터 재구성
        # 전체 차량 등록과 친환경 전체 데이터 분리
        total_vehicles = df[df['category'].str.contains('전체 차량 등록')]
        eco_vehicles = df[df['category'].str.contains('친환경 전체')]
        
        # 연도별로 데이터 정리
        result_data = []
        for year in range(2020, 2025):
            year_total = total_vehicles[total_vehicles['year'] == year]
            year_eco = eco_vehicles[eco_vehicles['year'] == year]
            
            if not year_total.empty and not year_eco.empty:
                # 실제 데이터베이스에서 친환경 차종별 데이터 가져오기
                try:
                    conn = connect_db()
                    cursor = conn.cursor()
                    eco_detail_query = """
                    SELECT 
                        연도 as year,
                        구분 as category,
                        합계 as total
                    FROM environmental_vehicles 
                    WHERE 연도 = %s AND 구분 IN ('전기차', '수소차', '하이브리드')
                    ORDER BY 구분
                    """
                    cursor.execute(eco_detail_query, (year,))
                    eco_detail_data = cursor.fetchall()
                    eco_detail_df = pd.DataFrame(eco_detail_data, columns=['year', 'category', 'total'])
                    cursor.close()
                    conn.close()
                    
                    # 친환경 차종별 데이터 추출
                    electric_data = eco_detail_df[eco_detail_df['category'] == '전기차']
                    hydrogen_data = eco_detail_df[eco_detail_df['category'] == '수소차']
                    hybrid_data = eco_detail_df[eco_detail_df['category'] == '하이브리드']
                    
                    electric_count = electric_data.iloc[0]['total'] if not electric_data.empty else 0
                    hydrogen_count = hydrogen_data.iloc[0]['total'] if not hydrogen_data.empty else 0
                    hybrid_count = hybrid_data.iloc[0]['total'] if not hybrid_data.empty else 0
                    
                except Exception as e:
                    print(f"친환경 차종별 데이터 조회 실패: {e}")
                    # 데이터베이스에서 차종별 데이터를 가져올 수 없는 경우 기본값 사용
                    electric_count = year_eco.iloc[0]['total'] * 0.4
                    hydrogen_count = year_eco.iloc[0]['total'] * 0.1
                    hybrid_count = year_eco.iloc[0]['total'] * 0.5
                
                result_data.append({
                    'year': year,
                    'total_vehicles': year_total.iloc[0]['total'],
                    'total_eco_vehicles': year_eco.iloc[0]['total'],
                    'electric_vehicles': electric_count,
                    'hydrogen_vehicles': hydrogen_count,
                    'hybrid_vehicles': hybrid_count,
                })
        
        if not result_data:
            return None
            
        result_df = pd.DataFrame(result_data)
        
        # 친환경 자동차 총합 계산
        result_df['total_eco_vehicles'] = result_df['electric_vehicles'] + result_df['hydrogen_vehicles'] + result_df['hybrid_vehicles']
        
        # 비율 계산
        result_df['electric_ratio'] = (result_df['electric_vehicles'] / result_df['total_eco_vehicles']) * 100
        result_df['hydrogen_ratio'] = (result_df['hydrogen_vehicles'] / result_df['total_eco_vehicles']) * 100
        result_df['hybrid_ratio'] = (result_df['hybrid_vehicles'] / result_df['total_eco_vehicles']) * 100
        
        return result_df
        
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        return None

def get_environmental_impact_data():
    """
    환경 영향 분석 데이터를 가져오는 함수
    """
    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        # greenhouse_gases 테이블에서 실제 사용 가능한 연도 범위로 데이터 가져오기
        query = """
        SELECT 
            년도 as year,
            지역 as region,
            승용 as passenger,
            승합 as bus,
            화물 as cargo,
            특수 as special
        FROM greenhouse_gases 
        WHERE 년도 BETWEEN 2019 AND 2022
        ORDER BY 년도, 지역
        """
        
        cursor.execute(query)
        data = cursor.fetchall()
        columns = ['year', 'region', 'passenger', 'bus', 'cargo', 'special']
        df = pd.DataFrame(data, columns=columns)
        
        cursor.close()
        conn.close()
        
        if df.empty:
            return None
        
        # 지역별 온실가스 배출량 합계 계산
        yearly_data = df.groupby('year').agg({
            'passenger': 'sum',
            'bus': 'sum', 
            'cargo': 'sum',
            'special': 'sum'
        }).reset_index()
        
        # 총 온실가스 배출량 계산
        yearly_data['greenhouse_gas'] = yearly_data['passenger'] + yearly_data['bus'] + yearly_data['cargo'] + yearly_data['special']
        
        # 실제 데이터베이스에서 친환경 자동차 비율 데이터 가져오기
        try:
            conn = connect_db()
            cursor = conn.cursor()
            eco_ratio_query = """
            SELECT 
                연도 as year,
                구분 as category,
                합계 as total
            FROM environmental_vehicles 
            WHERE 연도 BETWEEN 2019 AND 2024 AND 구분 IN ('전체 차량 등록', '친환경 전체')
            ORDER BY 연도, 구분
            """
            cursor.execute(eco_ratio_query)
            eco_ratio_data = cursor.fetchall()
            eco_ratio_df = pd.DataFrame(eco_ratio_data, columns=['year', 'category', 'total'])
            cursor.close()
            conn.close()
            
            if not eco_ratio_df.empty:
                # 연도별 친환경 자동차 비율 계산
                eco_ratios = []
                for year in yearly_data['year']:  # yearly_data의 실제 연도 사용
                    year_total = eco_ratio_df[(eco_ratio_df['year'] == year) & (eco_ratio_df['category'] == '전체 차량 등록')]
                    year_eco = eco_ratio_df[(eco_ratio_df['year'] == year) & (eco_ratio_df['category'] == '친환경 전체')]
                    
                    if not year_total.empty and not year_eco.empty:
                        ratio = (year_eco.iloc[0]['total'] / year_total.iloc[0]['total']) * 100
                        eco_ratios.append(ratio)
                    else:
                        # 해당 연도 데이터가 없으면 0으로 설정
                        eco_ratios.append(0)
                
                # 배열 길이 확인 및 조정
                if len(eco_ratios) == len(yearly_data):
                    yearly_data['eco_vehicle_ratio'] = eco_ratios
                else:
                    # 길이가 다르면 0으로 채움
                    eco_ratios.extend([0] * (len(yearly_data) - len(eco_ratios)))
                    yearly_data['eco_vehicle_ratio'] = eco_ratios[:len(yearly_data)]
            else:
                # 데이터가 없는 경우 기본값 사용
                yearly_data['eco_vehicle_ratio'] = [0] * len(yearly_data)
                
        except Exception as e:
            print(f"친환경 자동차 비율 데이터 조회 실패: {e}")
            # 데이터베이스에서 비율 데이터를 가져올 수 없는 경우 기본값 사용
            yearly_data['eco_vehicle_ratio'] = [0] * len(yearly_data)
        
        return yearly_data
        
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        return None 