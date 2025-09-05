"""
Football Transfer Prediction - Post-Transfer Care Strategy Analysis (Simplified)
이적 후 케어 전략 분석 (간단 버전)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🛡️ Football Transfer Care Strategy Analysis (Simplified)")
print("=" * 60)

# ============================================================================
# 1. 데이터 로딩 및 기본 분석
# ============================================================================

print("\n📁 데이터 로딩...")
DATA_DIR = Path.cwd() / "data" / "curated"
df = pd.read_csv(DATA_DIR / "player_final.csv", low_memory=True)

# 미래 데이터 제외
df = df[~df['season'].isin(['23/24', '24/25'])].copy()
print(f"✅ 데이터 로딩 완료: {df.shape}")

# ============================================================================
# 2. 기본 피쳐 생성
# ============================================================================

print("\n🔧 기본 피쳐 생성...")

# 시즌 시작 연도
df['season_start_year'] = df['season'].apply(lambda x: 2000 + int(x.split('/')[0]))

# 나이 계산
if 'date_of_birth' in df.columns:
    by = df['date_of_birth'].astype(str).str.extract(r"^(\d{4})")[0]
    birth_year = pd.to_numeric(by, errors='coerce')
    df['age_at_season'] = (df['season_start_year'] - birth_year).astype('float')

# 로그 시장가치
if 'player_market_value_in_eur' in df.columns:
    df['log_market_value'] = np.log1p(pd.to_numeric(df['player_market_value_in_eur'], errors='coerce'))

# 공격 기여도
if 'goals' in df.columns and 'assists' in df.columns:
    df['attack_contribution'] = df['goals'] + df['assists']

# 외국인 선수 여부
if 'country_of_birth' in df.columns:
    df['is_foreigner'] = (df['country_of_birth'] != 'England').astype(int)

print("✅ 기본 피쳐 생성 완료")

# ============================================================================
# 3. 고위험 선수 식별 (규칙 기반)
# ============================================================================

print("\n🎯 고위험 선수 식별 (규칙 기반)...")

# 고위험 선수 식별 규칙
high_risk_conditions = []

# 1. 나이 조건 (22세 이하 또는 30세 이상)
if 'age_at_season' in df.columns:
    age_risk = (df['age_at_season'] <= 22) | (df['age_at_season'] >= 30)
    high_risk_conditions.append(age_risk)

# 2. 시장가치 조건 (높은 시장가치)
if 'player_market_value_in_eur' in df.columns:
    high_value_threshold = df['player_market_value_in_eur'].quantile(0.7)
    value_risk = df['player_market_value_in_eur'] >= high_value_threshold
    high_risk_conditions.append(value_risk)

# 3. 출전시간 조건 (낮은 출전시간)
if 'season_avg_minutes' in df.columns:
    low_minutes_threshold = df['season_avg_minutes'].quantile(0.3)
    minutes_risk = df['season_avg_minutes'] <= low_minutes_threshold
    high_risk_conditions.append(minutes_risk)

# 4. 포지션 조건 (공격수)
if 'position' in df.columns:
    position_risk = df['position'] == 'Attack'
    high_risk_conditions.append(position_risk)

# 5. 외국인 선수
if 'is_foreigner' in df.columns:
    foreigner_risk = df['is_foreigner'] == 1
    high_risk_conditions.append(foreigner_risk)

# 고위험 선수 식별 (2개 이상 조건 만족)
if high_risk_conditions:
    risk_score = sum(high_risk_conditions)
    high_risk_players = df[risk_score >= 2].copy()
    high_risk_players['risk_score'] = risk_score[risk_score >= 2]
else:
    high_risk_players = df.copy()
    high_risk_players['risk_score'] = 0

print(f"✅ 고위험 선수 식별 완료: {len(high_risk_players):,}명")
print(f"   - 전체 대비 비율: {len(high_risk_players)/len(df)*100:.1f}%")

# ============================================================================
# 4. 포지션별 리스크 분석
# ============================================================================

print("\n📊 포지션별 리스크 분석...")

if 'position' in high_risk_players.columns:
    position_analysis = high_risk_players.groupby('position').agg({
        'risk_score': ['count', 'mean'],
        'age_at_season': 'mean',
        'player_market_value_in_eur': 'mean',
        'season_avg_minutes': 'mean'
    }).round(3)
    
    position_analysis.columns = ['선수수', '평균리스크점수', '평균나이', '평균시장가치', '평균출전시간']
    
    print("  📈 포지션별 고위험 선수 분석:")
    for pos in position_analysis.index:
        count = position_analysis.loc[pos, '선수수']
        avg_risk = position_analysis.loc[pos, '평균리스크점수']
        avg_age = position_analysis.loc[pos, '평균나이']
        avg_value = position_analysis.loc[pos, '평균시장가치']
        print(f"    {pos}: {count}명 (리스크: {avg_risk:.1f}, 나이: {avg_age:.1f}세, 가치: €{avg_value:,.0f})")
    
    # 포지션별 리스크 시각화
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    position_analysis['선수수'].plot(kind='bar', color='skyblue')
    plt.title('포지션별 고위험 선수 수')
    plt.ylabel('선수 수')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 2)
    position_analysis['평균리스크점수'].plot(kind='bar', color='lightcoral')
    plt.title('포지션별 평균 리스크 점수')
    plt.ylabel('리스크 점수')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 3)
    position_analysis['평균나이'].plot(kind='bar', color='lightgreen')
    plt.title('포지션별 평균 나이')
    plt.ylabel('나이')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 4)
    position_analysis['평균시장가치'].plot(kind='bar', color='gold')
    plt.title('포지션별 평균 시장가치')
    plt.ylabel('시장가치 (€)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 5)
    position_analysis['평균출전시간'].plot(kind='bar', color='lightpink')
    plt.title('포지션별 평균 출전시간')
    plt.ylabel('출전시간 (분)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. 연령대별 리스크 분석
# ============================================================================

print("\n👥 연령대별 리스크 분석...")

if 'age_at_season' in high_risk_players.columns:
    # 연령대 분류
    high_risk_players['age_group'] = pd.cut(
        high_risk_players['age_at_season'], 
        bins=[0, 22, 26, 30, 100], 
        labels=['22세 이하', '23-26세', '27-30세', '30세 이상']
    )
    
    age_analysis = high_risk_players.groupby('age_group').agg({
        'risk_score': ['count', 'mean'],
        'player_market_value_in_eur': 'mean',
        'season_avg_minutes': 'mean'
    }).round(3)
    
    age_analysis.columns = ['선수수', '평균리스크점수', '평균시장가치', '평균출전시간']
    
    print("  📈 연령대별 고위험 선수 분석:")
    for age in age_analysis.index:
        count = age_analysis.loc[age, '선수수']
        avg_risk = age_analysis.loc[age, '평균리스크점수']
        avg_value = age_analysis.loc[age, '평균시장가치']
        avg_minutes = age_analysis.loc[age, '평균출전시간']
        print(f"    {age}: {count}명 (리스크: {avg_risk:.1f}, 가치: €{avg_value:,.0f}, 출전: {avg_minutes:.0f}분)")
    
    # 연령대별 리스크 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    age_analysis['선수수'].plot(kind='bar', color='skyblue')
    plt.title('연령대별 고위험 선수 수')
    plt.ylabel('선수 수')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    age_analysis['평균리스크점수'].plot(kind='bar', color='lightcoral')
    plt.title('연령대별 평균 리스크 점수')
    plt.ylabel('리스크 점수')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    age_analysis['평균시장가치'].plot(kind='bar', color='lightgreen')
    plt.title('연령대별 평균 시장가치')
    plt.ylabel('시장가치 (€)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 6. 클럽별 리스크 분석
# ============================================================================

print("\n🏟️ 클럽별 리스크 분석...")

if 'club_name' in high_risk_players.columns:
    club_analysis = high_risk_players.groupby('club_name').agg({
        'risk_score': ['count', 'mean'],
        'age_at_season': 'mean',
        'player_market_value_in_eur': 'mean'
    }).round(3)
    
    club_analysis.columns = ['고위험선수수', '평균리스크점수', '평균나이', '평균시장가치']
    
    # 상위 10개 클럽
    top_clubs = club_analysis.nlargest(10, '고위험선수수')
    
    print("  📈 고위험 선수가 많은 클럽 TOP 10:")
    for club in top_clubs.index:
        count = top_clubs.loc[club, '고위험선수수']
        avg_risk = top_clubs.loc[club, '평균리스크점수']
        avg_age = top_clubs.loc[club, '평균나이']
        print(f"    {club}: {count}명 (리스크: {avg_risk:.1f}, 나이: {avg_age:.1f}세)")
    
    # 클럽별 리스크 시각화
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    top_clubs['고위험선수수'].plot(kind='bar', color='skyblue')
    plt.title('클럽별 고위험 선수 수 (TOP 10)')
    plt.ylabel('선수 수')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    top_clubs['평균리스크점수'].plot(kind='bar', color='lightcoral')
    plt.title('클럽별 평균 리스크 점수 (TOP 10)')
    plt.ylabel('리스크 점수')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. 이적 후 케어 전략 수립
# ============================================================================

print("\n🛡️ 이적 후 케어 전략 수립...")

# 전략 1: 우선순위 기반 재계약 전략
print("\n  📋 전략 1: 우선순위 기반 재계약 전략")
priority_players = high_risk_players[
    (high_risk_players['risk_score'] >= 3) & 
    (high_risk_players['player_market_value_in_eur'] >= high_risk_players['player_market_value_in_eur'].quantile(0.5))
].copy()

print(f"    - 최우선 재계약 대상: {len(priority_players)}명")
if len(priority_players) > 0:
    print(f"    - 평균 시장가치: €{priority_players['player_market_value_in_eur'].mean():,.0f}")
    print(f"    - 평균 리스크 점수: {priority_players['risk_score'].mean():.1f}")

# 전략 2: 임대 전략
print("\n  📋 전략 2: 임대 전략")
loan_candidates = high_risk_players[
    (high_risk_players['age_at_season'] <= 22) & 
    (high_risk_players['risk_score'] >= 2)
].copy()

print(f"    - 임대 추천 대상: {len(loan_candidates)}명 (22세 이하)")
if len(loan_candidates) > 0:
    print(f"    - 평균 나이: {loan_candidates['age_at_season'].mean():.1f}세")
    print(f"    - 평균 리스크 점수: {loan_candidates['risk_score'].mean():.1f}")

# 전략 3: 인센티브 설계
print("\n  📋 전략 3: 인센티브 설계")
incentive_players = high_risk_players[
    (high_risk_players['risk_score'] >= 2) & 
    (high_risk_players['season_avg_minutes'] >= 30)
].copy()

print(f"    - 인센티브 대상: {len(incentive_players)}명 (출전시간 30분 이상)")
if len(incentive_players) > 0:
    print(f"    - 평균 출전시간: {incentive_players['season_avg_minutes'].mean():.0f}분")
    print(f"    - 평균 리스크 점수: {incentive_players['risk_score'].mean():.1f}")

# 전략 4: 스쿼드 리스크 관리
print("\n  📋 전략 4: 스쿼드 리스크 관리")
if 'position' in high_risk_players.columns:
    position_risk = high_risk_players.groupby('position')['risk_score'].agg(['count', 'mean']).round(3)
    high_risk_positions = position_risk[position_risk['mean'] >= 2.0]
    
    print(f"    - 고위험 포지션: {len(high_risk_positions)}개")
    for pos in high_risk_positions.index:
        count = high_risk_positions.loc[pos, 'count']
        avg_risk = high_risk_positions.loc[pos, 'mean']
        print(f"      {pos}: {count}명 (평균 리스크: {avg_risk:.1f})")

# ============================================================================
# 8. 모니터링 및 알림 시스템 설계
# ============================================================================

print("\n📊 모니터링 및 알림 시스템 설계...")

# 위험도 등급 분류
def classify_risk_level(score):
    if score >= 4:
        return "매우 높음"
    elif score >= 3:
        return "높음"
    elif score >= 2:
        return "보통"
    else:
        return "낮음"

high_risk_players['risk_level'] = high_risk_players['risk_score'].apply(classify_risk_level)

risk_distribution = high_risk_players['risk_level'].value_counts()
print("  📈 위험도 등급 분포:")
for level, count in risk_distribution.items():
    print(f"    {level}: {count}명 ({count/len(high_risk_players)*100:.1f}%)")

# 알림 시스템 설계
print("\n  🔔 알림 시스템 설계:")
print("    - 매우 높음 (≥4): 즉시 알림, 긴급 재계약 검토")
print("    - 높음 (≥3): 주간 모니터링, 재계약 준비")
print("    - 보통 (≥2): 월간 모니터링, 상황 관찰")
print("    - 낮음 (<2): 분기별 모니터링")

# ============================================================================
# 9. 비용-효과 분석
# ============================================================================

print("\n💰 비용-효과 분석...")

if 'player_market_value_in_eur' in high_risk_players.columns:
    avg_market_value = high_risk_players['player_market_value_in_eur'].mean()
    avg_risk_score = high_risk_players['risk_score'].mean()
    
    # 가정: 재계약 비용 = 시장가치의 20%, 이적 손실 = 시장가치의 100%
    renewal_cost = avg_market_value * 0.2
    transfer_loss = avg_market_value * 1.0
    
    # 리스크 점수에 따른 이적 확률 추정 (0.1 * risk_score)
    estimated_transfer_prob = min(0.1 * avg_risk_score, 0.9)
    
    expected_renewal_cost = renewal_cost * estimated_transfer_prob
    expected_transfer_loss = transfer_loss * estimated_transfer_prob
    
    print(f"  📊 비용 분석 (평균 기준):")
    print(f"    - 평균 시장가치: €{avg_market_value:,.0f}")
    print(f"    - 평균 리스크 점수: {avg_risk_score:.1f}")
    print(f"    - 추정 이적 확률: {estimated_transfer_prob:.3f}")
    print(f"    - 재계약 비용: €{renewal_cost:,.0f}")
    print(f"    - 이적 손실: €{transfer_loss:,.0f}")
    print(f"    - 예상 재계약 비용: €{expected_renewal_cost:,.0f}")
    print(f"    - 예상 이적 손실: €{expected_transfer_loss:,.0f}")
    
    if expected_renewal_cost < expected_transfer_loss:
        print(f"    ✅ 재계약이 경제적으로 유리 (절약: €{expected_transfer_loss - expected_renewal_cost:,.0f})")
    else:
        print(f"    ❌ 이적이 경제적으로 유리 (절약: €{expected_renewal_cost - expected_transfer_loss:,.0f})")

# ============================================================================
# 10. 실행 계획 수립
# ============================================================================

print("\n📅 실행 계획 수립...")

print("  🎯 단기 계획 (1-3개월):")
print("    1. 고위험 선수 1:1 면담 및 재계약 협상")
print("    2. 임대 후보자 선정 및 임대 시장 조사")
print("    3. 인센티브 체계 설계 및 적용")
print("    4. 모니터링 시스템 구축")

print("\n  🎯 중기 계획 (3-6개월):")
print("    1. 스쿼드 리스크 분산을 위한 영입 계획")
print("    2. 선수별 맞춤형 케어 프로그램 운영")
print("    3. 성과 지표 모니터링 및 개선")
print("    4. A/B 테스트를 통한 전략 검증")

print("\n  🎯 장기 계획 (6-12개월):")
print("    1. 예방적 선수 관리 시스템 고도화")
print("    2. 데이터 기반 의사결정 문화 정착")
print("    3. 선수 만족도 및 유지율 개선")
print("    4. 경쟁사 대비 우위 확보")

print("\n✅ 이적 후 케어 전략 분석 완료!")
print("=" * 60)
