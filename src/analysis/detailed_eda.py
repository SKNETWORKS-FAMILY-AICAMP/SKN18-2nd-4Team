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

def load_data():
    """데이터를 로드합니다."""
    df = pd.read_excel("data/raw/E Commerce Dataset.xlsx", sheet_name='E Comm')
    return df

def analyze_churn_patterns(df):
    """이탈(Churn) 패턴을 분석합니다."""
    print("🔍 이탈(Churn) 패턴 분석")
    print("=" * 50)
    
    # 전체 이탈률
    churn_rate = df['Churn'].mean() * 100
    print(f"📊 전체 이탈률: {churn_rate:.1f}%")
    print(f"📈 이탈 고객 수: {df['Churn'].sum():,}명")
    print(f"📉 유지 고객 수: {(df['Churn'] == 0).sum():,}명")
    print()
    
    # 성별별 이탈률
    print("👥 성별별 이탈률:")
    gender_churn = df.groupby('Gender')['Churn'].agg(['count', 'sum', 'mean'])
    for gender in gender_churn.index:
        count = gender_churn.loc[gender, 'count']
        churn_count = gender_churn.loc[gender, 'sum']
        churn_rate = gender_churn.loc[gender, 'mean'] * 100
        print(f"  {gender}: {churn_rate:.1f}% ({churn_count}/{count}명)")
    print()
    
    # 결혼상태별 이탈률
    print("💍 결혼상태별 이탈률:")
    marital_churn = df.groupby('MaritalStatus')['Churn'].agg(['count', 'sum', 'mean'])
    for status in marital_churn.index:
        count = marital_churn.loc[status, 'count']
        churn_count = marital_churn.loc[status, 'sum']
        churn_rate = marital_churn.loc[status, 'mean'] * 100
        print(f"  {status}: {churn_rate:.1f}% ({churn_count}/{count}명)")
    print()
    
    # 도시 등급별 이탈률
    print("🏙️ 도시 등급별 이탈률:")
    city_churn = df.groupby('CityTier')['Churn'].agg(['count', 'sum', 'mean'])
    for tier in city_churn.index:
        count = city_churn.loc[tier, 'count']
        churn_count = city_churn.loc[tier, 'sum']
        churn_rate = city_churn.loc[tier, 'mean'] * 100
        print(f"  Tier {tier}: {churn_rate:.1f}% ({churn_count}/{count}명)")
    print()

def analyze_customer_behavior(df):
    """고객 행동 패턴을 분석합니다."""
    print("🎯 고객 행동 패턴 분석")
    print("=" * 50)
    
    # 선호 결제 방식
    print("💳 선호 결제 방식:")
    payment_counts = df['PreferredPaymentMode'].value_counts()
    for payment, count in payment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {payment}: {count}명 ({percentage:.1f}%)")
    print()
    
    # 선호 주문 카테고리
    print("🛍️ 선호 주문 카테고리:")
    category_counts = df['PreferedOrderCat'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count}명 ({percentage:.1f}%)")
    print()
    
    # 만족도 점수 분포
    print("⭐ 만족도 점수 분포:")
    satisfaction_counts = df['SatisfactionScore'].value_counts().sort_index()
    for score, count in satisfaction_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {score}점: {count}명 ({percentage:.1f}%)")
    print()
    
    # 앱 사용 시간
    print("⏰ 앱 사용 시간 분포:")
    hour_counts = df['HourSpendOnApp'].value_counts().sort_index()
    for hour, count in hour_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {hour}시간: {count}명 ({percentage:.1f}%)")
    print()

def analyze_numerical_features(df):
    """수치형 피쳐들의 상관관계를 분석합니다."""
    print("📊 수치형 피쳐 상관관계 분석")
    print("=" * 50)
    
    # 수치형 컬럼 선택
    numeric_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
                   'NumberOfDeviceRegistered', 'SatisfactionScore', 
                   'NumberOfAddress', 'OrderAmountHikeFromlastYear', 
                   'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 
                   'CashbackAmount', 'Churn']
    
    # 결측값이 있는 컬럼 제외
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # 상관관계 계산
    corr_matrix = df[numeric_cols].corr()
    
    # Churn과의 상관관계
    print("🎯 Churn과의 상관관계:")
    churn_corr = corr_matrix['Churn'].sort_values(ascending=False)
    for feature, corr in churn_corr.items():
        if feature != 'Churn':
            print(f"  {feature}: {corr:.3f}")
    print()
    
    # 상관관계 히트맵 생성
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('수치형 피쳐 상관관계 히트맵', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ 상관관계 히트맵이 'correlation_heatmap.png'로 저장되었습니다!")

def create_feature_distributions(df):
    """주요 피쳐들의 분포를 시각화합니다."""
    print("📈 주요 피쳐 분포 시각화")
    print("=" * 50)
    
    # 서브플롯 설정
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('주요 피쳐 분포 분석', fontsize=16, fontweight='bold')
    
    # 1. Tenure (고객 유지 기간)
    axes[0, 0].hist(df['Tenure'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('고객 유지 기간 (Tenure)')
    axes[0, 0].set_xlabel('개월')
    axes[0, 0].set_ylabel('고객 수')
    
    # 2. Satisfaction Score
    satisfaction_counts = df['SatisfactionScore'].value_counts().sort_index()
    axes[0, 1].bar(satisfaction_counts.index, satisfaction_counts.values, color='lightgreen')
    axes[0, 1].set_title('만족도 점수 분포')
    axes[0, 1].set_xlabel('만족도 점수')
    axes[0, 1].set_ylabel('고객 수')
    
    # 3. Hour Spend On App
    hour_counts = df['HourSpendOnApp'].value_counts().sort_index()
    axes[0, 2].bar(hour_counts.index, hour_counts.values, color='lightcoral')
    axes[0, 2].set_title('앱 사용 시간')
    axes[0, 2].set_xlabel('시간')
    axes[0, 2].set_ylabel('고객 수')
    
    # 4. Order Count
    order_counts = df['OrderCount'].value_counts().sort_index()
    axes[1, 0].bar(order_counts.index, order_counts.values, color='gold')
    axes[1, 0].set_title('주문 횟수')
    axes[1, 0].set_xlabel('주문 횟수')
    axes[1, 0].set_ylabel('고객 수')
    
    # 5. Cashback Amount
    axes[1, 1].hist(df['CashbackAmount'], bins=30, alpha=0.7, color='plum', edgecolor='black')
    axes[1, 1].set_title('캐시백 금액')
    axes[1, 1].set_xlabel('캐시백 금액')
    axes[1, 1].set_ylabel('고객 수')
    
    # 6. Number of Address
    address_counts = df['NumberOfAddress'].value_counts().sort_index()
    axes[1, 2].bar(address_counts.index, address_counts.values, color='lightblue')
    axes[1, 2].set_title('등록된 주소 수')
    axes[1, 2].set_xlabel('주소 수')
    axes[1, 2].set_ylabel('고객 수')
    
    # 7. Gender vs Churn
    gender_churn = pd.crosstab(df['Gender'], df['Churn'])
    gender_churn.plot(kind='bar', ax=axes[2, 0], color=['lightblue', 'lightcoral'])
    axes[2, 0].set_title('성별별 이탈률')
    axes[2, 0].set_xlabel('성별')
    axes[2, 0].set_ylabel('고객 수')
    axes[2, 0].legend(['유지', '이탈'])
    
    # 8. Marital Status vs Churn
    marital_churn = pd.crosstab(df['MaritalStatus'], df['Churn'])
    marital_churn.plot(kind='bar', ax=axes[2, 1], color=['lightgreen', 'lightcoral'])
    axes[2, 1].set_title('결혼상태별 이탈률')
    axes[2, 1].set_xlabel('결혼상태')
    axes[2, 1].set_ylabel('고객 수')
    axes[2, 1].legend(['유지', '이탈'])
    
    # 9. City Tier vs Churn
    city_churn = pd.crosstab(df['CityTier'], df['Churn'])
    city_churn.plot(kind='bar', ax=axes[2, 2], color=['lightyellow', 'lightcoral'])
    axes[2, 2].set_title('도시 등급별 이탈률')
    axes[2, 2].set_xlabel('도시 등급')
    axes[2, 2].set_ylabel('고객 수')
    axes[2, 2].legend(['유지', '이탈'])
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("✅ 피쳐 분포 시각화가 'feature_distributions.png'로 저장되었습니다!")

def generate_insights_report(df):
    """데이터 인사이트 리포트를 생성합니다."""
    print("💡 데이터 인사이트 리포트")
    print("=" * 50)
    
    # 주요 통계
    print("📊 주요 통계:")
    print(f"  - 총 고객 수: {len(df):,}명")
    print(f"  - 이탈률: {df['Churn'].mean()*100:.1f}%")
    print(f"  - 평균 만족도: {df['SatisfactionScore'].mean():.1f}점")
    print(f"  - 평균 주문 횟수: {df['OrderCount'].mean():.1f}회")
    print(f"  - 평균 캐시백: {df['CashbackAmount'].mean():.1f}원")
    print()
    
    # 고위험 고객 프로필
    print("⚠️ 고위험 고객 프로필 (이탈 가능성이 높은 고객):")
    high_risk = df[df['Churn'] == 1]
    print(f"  - 평균 만족도: {high_risk['SatisfactionScore'].mean():.1f}점")
    print(f"  - 평균 주문 횟수: {high_risk['OrderCount'].mean():.1f}회")
    print(f"  - 평균 앱 사용 시간: {high_risk['HourSpendOnApp'].mean():.1f}시간")
    print(f"  - 불만 제기 비율: {high_risk['Complain'].mean()*100:.1f}%")
    print()
    
    # 충성 고객 프로필
    print("💎 충성 고객 프로필 (이탈하지 않는 고객):")
    loyal = df[df['Churn'] == 0]
    print(f"  - 평균 만족도: {loyal['SatisfactionScore'].mean():.1f}점")
    print(f"  - 평균 주문 횟수: {loyal['OrderCount'].mean():.1f}회")
    print(f"  - 평균 앱 사용 시간: {loyal['HourSpendOnApp'].mean():.1f}시간")
    print(f"  - 불만 제기 비율: {loyal['Complain'].mean()*100:.1f}%")
    print()
    
    # 비즈니스 인사이트
    print("🚀 비즈니스 인사이트:")
    print("  1. 이탈률이 16.8%로 상당히 높은 편입니다.")
    print("  2. 만족도가 낮은 고객의 이탈 가능성이 높습니다.")
    print("  3. 앱 사용 시간이 적은 고객이 이탈할 가능성이 높습니다.")
    print("  4. 불만을 제기한 고객의 이탈률이 높습니다.")
    print("  5. 주문 횟수가 적은 고객이 이탈할 가능성이 높습니다.")

def main():
    """메인 실행 함수"""
    print("🔍 상세 E-Commerce 데이터 분석 시작")
    print("=" * 60)
    
    # 데이터 로드
    df = load_data()
    print(f"✅ 데이터 로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
    print()
    
    # 각종 분석 실행
    analyze_churn_patterns(df)
    analyze_customer_behavior(df)
    analyze_numerical_features(df)
    create_feature_distributions(df)
    generate_insights_report(df)
    
    print("\n🎉 상세 EDA 분석이 완료되었습니다!")
    print("📁 생성된 파일:")
    print("  - correlation_heatmap.png: 상관관계 히트맵")
    print("  - feature_distributions.png: 피쳐 분포 시각화")

if __name__ == "__main__":
    main()
