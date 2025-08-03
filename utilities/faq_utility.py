from database.database import connect_db
import pandas as pd

def get_con():
    con = connect_db()
    cursor = con.cursor()
    cursor.execute("SELECT * FROM faq")
    categories = cursor.fetchall()
    return categories

def get_faq_data():
    """FAQ 데이터를 가져와서 DataFrame으로 반환"""
    con = connect_db()
    cursor = con.cursor()
    cursor.execute("SELECT * FROM faq")
    data = cursor.fetchall()
    
    # 컬럼명 가져오기
    columns = [desc[0] for desc in cursor.description]
    
    # DataFrame 생성
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    con.close()
    
    return df

def get_categories():
    """데이터베이스에서 실제 카테고리 목록을 가져와서 반환"""
    try:
        con = connect_db()
        cursor = con.cursor()
        
        # 실제 카테고리 가져오기
        cursor.execute("SELECT DISTINCT category FROM faq WHERE category IS NOT NULL AND category != ''")
        categories = cursor.fetchall()
        category_list = [cat[0] for cat in categories if cat[0]]
        
        cursor.close()
        con.close()
        
        # "전체" 카테고리를 맨 앞에 추가
        if "전체" not in category_list:
            category_list.insert(0, "전체")
        
        return category_list
        
    except Exception as e:
        print(f"카테고리 조회 실패: {e}")
        # 오류 시 기본 카테고리 반환
        return ["전체", "차량 구매", "차량 정비", "기아멤버스", "홈페이지", "PBV", "기타"]

def filter_faq_by_category(df, category):
    """카테고리에 따라 FAQ 데이터 필터링"""
    if category == "전체":
        return df
    else:
        # 실제 카테고리 컬럼 사용
        if 'category' in df.columns:
            return df[df['category'] == category]
        else:
            # 카테고리 컬럼이 없으면 전체 반환
            return df

def search_faq(df, search_term):
    """검색어로 FAQ 필터링"""
    if not search_term or search_term == "궁금한 점을 검색해 보세요.":
        return df
    
    # question과 answer 컬럼에서 검색
    search_columns = []
    for col in df.columns:
        if col in ['question', 'answer']:
            search_columns.append(col)
    
    if search_columns:
        mask = df[search_columns].astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        return df[mask]
    else:
        # 모든 컬럼에서 검색
        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        return df[mask]



