import pymysql
import sys

def test_database_connection():
    """데이터베이스 연결을 테스트하는 함수"""
    try:
        print("데이터베이스 연결 테스트 시작...")
        
        # 연결 시도
        conn = pymysql.connect(
            host='localhost', 
            user='root', 
            password='root1234',
            db='car', 
            charset='utf8'
        )
        
        print("✅ 데이터베이스 연결 성공!")
        
        # 커서 생성
        cursor = conn.cursor()
        
        # 테이블 존재 여부 확인
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        print(f"📋 데이터베이스에 있는 테이블들:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # environmental_vehicles 테이블 확인
        cursor.execute("SELECT COUNT(*) FROM environmental_vehicles")
        count = cursor.fetchone()[0]
        print(f"📊 environmental_vehicles 테이블 레코드 수: {count}")
        
        # greenhouse_gases 테이블 확인
        cursor.execute("SELECT COUNT(*) FROM greenhouse_gases")
        count = cursor.fetchone()[0]
        print(f"📊 greenhouse_gases 테이블 레코드 수: {count}")
        
        # 샘플 데이터 확인
        print("\n📈 environmental_vehicles 샘플 데이터:")
        cursor.execute("SELECT * FROM environmental_vehicles LIMIT 5")
        sample_data = cursor.fetchall()
        for row in sample_data:
            print(f"  {row}")
        
        cursor.close()
        conn.close()
        
        print("\n✅ 모든 테스트 통과!")
        return True
        
    except pymysql.Error as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        print(f"   에러 코드: {e.args[0]}")
        print(f"   에러 메시지: {e.args[1]}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False

if __name__ == "__main__":
    test_database_connection() 