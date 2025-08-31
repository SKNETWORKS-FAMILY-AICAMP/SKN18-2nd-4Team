import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data(file_path):
    """
    Excel 파일을 로드하고 기본 정보를 출력합니다.
    """
    print("=" * 60)
    print("📊 E-COMMERCE 데이터셋 EDA 분석")
    print("=" * 60)
    
    # Excel 파일의 시트 정보 확인
    try:
        excel_file = pd.ExcelFile(file_path)
        print(f"📁 파일: {file_path}")
        print(f"📋 시트 목록: {excel_file.sheet_names}")
        print()
        
        # 각 시트의 기본 정보 확인
        for sheet_name in excel_file.sheet_names:
            print(f"🔍 시트 '{sheet_name}' 정보:")
            df_temp = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
            print(f"  - 크기: {df_temp.shape}")
            print(f"  - 컬럼: {list(df_temp.columns)}")
            print(f"  - 처음 3행:")
            print(df_temp.head(3))
            print()
        
        # 'E Comm' 시트를 로드 (실제 데이터)
        df = pd.read_excel(file_path, sheet_name='E Comm')
        print(f"데이터 로드 성공!")
        print(f"데이터 크기: {df.shape[0]} 행 × {df.shape[1]} 열")
        print()
        
        return df
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None

def analyze_features(df):
    """
    데이터 피쳐들의 기본 정보를 분석합니다.
    """
    print("🔍 데이터 피쳐 분석")
    print("-" * 40)
    
    # 데이터 타입 정보
    print("📋 데이터 타입:")
    print(df.dtypes)
    print()
    
    # 결측값 정보
    print("결측값 정보:")
    missing_info = df.isnull().sum()
    missing_percent = (missing_info / len(df)) * 100
    missing_df = pd.DataFrame({
        '결측값 개수': missing_info,
        '결측값 비율(%)': missing_percent
    })
    print(missing_df[missing_df['결측값 개수'] > 0])
    if missing_df['결측값 개수'].sum() == 0:
        print("✅ 결측값이 없습니다!")
    print()
    
    # 수치형 컬럼의 기술통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("📈 수치형 컬럼 기술통계:")
        print(df[numeric_cols].describe())
        print()
    
    # 범주형 컬럼의 unique 값 개수
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("🏷️ 범주형 컬럼 unique 값 개수:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count}개")
        print()

def analyze_unique_values(df, max_unique=20):
    """
    각 컬럼의 unique 값들을 분석합니다.
    """
    print("🎯 Unique 값 분석")
    print("-" * 40)
    
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"\n📌 {col} (총 {unique_count}개 unique 값)")
        
        if unique_count <= max_unique:
            # unique 값이 적은 경우 모든 값 출력
            unique_values = df[col].value_counts()
            print("  값별 개수:")
            for value, count in unique_values.items():
                percentage = (count / len(df)) * 100
                print(f"    {value}: {count}개 ({percentage:.1f}%)")
        else:
            # unique 값이 많은 경우 상위 10개만 출력
            top_values = df[col].value_counts().head(10)
            print("  상위 10개 값:")
            for value, count in top_values.items():
                percentage = (count / len(df)) * 100
                print(f"    {value}: {count}개 ({percentage:.1f}%)")
            print(f"    ... 기타 {unique_count - 10}개 값")

def create_visualizations(df):
    """
    데이터 시각화를 생성합니다.
    """
    print("\n📊 데이터 시각화 생성 중...")
    
    # 그래프 스타일 설정
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('E-Commerce 데이터셋 EDA 시각화', fontsize=16, fontweight='bold')
    
    # 1. 수치형 컬럼 분포 (히스토그램)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for i, col in enumerate(numeric_cols[:4]):  # 최대 4개 컬럼만
            row, col_idx = i // 2, i % 2
            axes[row, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[row, col_idx].set_title(f'{col} 분포')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('빈도')
    
    # 2. 범주형 컬럼 분포 (막대그래프)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        col = categorical_cols[0]  # 첫 번째 범주형 컬럼
        value_counts = df[col].value_counts().head(10)
        axes[0, 1].bar(range(len(value_counts)), value_counts.values)
        axes[0, 1].set_title(f'{col} 상위 10개 값')
        axes[0, 1].set_xlabel('값')
        axes[0, 1].set_ylabel('개수')
        axes[0, 1].set_xticks(range(len(value_counts)))
        axes[0, 1].set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    # 3. 상관관계 히트맵 (수치형 컬럼이 2개 이상인 경우)
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('수치형 컬럼 상관관계')
        axes[1, 0].set_xticks(range(len(numeric_cols)))
        axes[1, 0].set_yticks(range(len(numeric_cols)))
        axes[1, 0].set_xticklabels(numeric_cols, rotation=45, ha='right')
        axes[1, 0].set_yticklabels(numeric_cols)
        
        # 상관계수 값 표시
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 결측값 시각화
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_data = missing_data[missing_data > 0]
        axes[1, 1].bar(range(len(missing_data)), missing_data.values)
        axes[1, 1].set_title('결측값 개수')
        axes[1, 1].set_xlabel('컬럼')
        axes[1, 1].set_ylabel('결측값 개수')
        axes[1, 1].set_xticks(range(len(missing_data)))
        axes[1, 1].set_xticklabels(missing_data.index, rotation=45, ha='right')
    else:
        axes[1, 1].text(0.5, 0.5, '결측값 없음', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('결측값 현황')
    
    plt.tight_layout()
    plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ 시각화가 'eda_visualization.png'로 저장되었습니다!")
    # plt.show() 제거하여 자동으로 창이 열리지 않도록 함

def generate_summary_report(df):
    """
    데이터 요약 리포트를 생성합니다.
    """
    print("\n📋 데이터 요약 리포트")
    print("=" * 50)
    
    # 기본 정보
    print(f"📊 전체 데이터 크기: {df.shape[0]:,} 행 × {df.shape[1]} 열")
    print(f"💾 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 컬럼별 정보
    print(f"\n🔍 컬럼별 정보:")
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        print(f"  {col}:")
        print(f"    - 타입: {dtype}")
        print(f"    - Unique 값: {unique_count}개")
        print(f"    - 결측값: {missing_count}개 ({missing_percent:.1f}%)")
        
        # 범주형 컬럼의 경우 최빈값 정보
        if dtype == 'object' or dtype.name == 'category':
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            print(f"    - 최빈값: {mode_value}")
        
        print()

def main():
    """
    메인 실행 함수
    """
    # Excel 파일 경로
    file_path = "data/raw/E Commerce Dataset.xlsx"
    
    # 데이터 로드
    df = load_and_explore_data(file_path)
    if df is None:
        return
    
    # 데이터 피쳐 분석
    analyze_features(df)
    
    # Unique 값 분석
    analyze_unique_values(df)
    
    # 시각화 생성
    create_visualizations(df)
    
    # 요약 리포트 생성
    generate_summary_report(df)
    
    print("\n🎉 EDA 분석이 완료되었습니다!")
    print("📁 생성된 파일:")
    print("  - eda_visualization.png: 데이터 시각화")

if __name__ == "__main__":
    main()
