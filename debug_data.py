import pandas as pd
from database.database import connect_db

def debug_environmental_vehicles():
    """environmental_vehicles 테이블의 데이터를 직접 확인"""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        # 전체 데이터 확인
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
        
        print("=== environmental_vehicles 테이블 데이터 ===")
        print(df)
        print(f"\n총 {len(df)} 개의 레코드")
        
        # 카테고리별 확인
        print("\n=== 카테고리별 데이터 ===")
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            print(f"\n{category}:")
            print(category_data)
        
        cursor.close()
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def debug_greenhouse_gases():
    """greenhouse_gases 테이블의 데이터를 직접 확인"""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        # 전체 데이터 확인
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
        
        print("=== greenhouse_gases 테이블 데이터 ===")
        print(df.head(10))  # 처음 10개만 출력
        print(f"\n총 {len(df)} 개의 레코드")
        
        # 연도별 확인
        print("\n=== 연도별 데이터 ===")
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            print(f"\n{year}년: {len(year_data)}개 지역")
        
        cursor.close()
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    print("데이터 디버깅 시작...\n")
    
    # environmental_vehicles 데이터 확인
    env_data = debug_environmental_vehicles()
    
    print("\n" + "="*50 + "\n")
    
    # greenhouse_gases 데이터 확인
    gas_data = debug_greenhouse_gases() 