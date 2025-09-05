"""
Football Transfer Prediction - Post-Transfer Care Strategy Analysis
이적 후 케어 전략 분석 및 비즈니스 인사이트 도출
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🛡️ Football Transfer Care Strategy Analysis")
print("=" * 60)

# ============================================================================
# 1. 모델 및 데이터 로딩
# ============================================================================

print("\n📁 모델 및 데이터 로딩...")

# 모델 로딩
try:
    model = joblib.load('logistic_regression_model.pkl')
    preprocessor = joblib.load('feature_preprocessor.pkl')
    print("✅ 모델 로딩 완료")
except FileNotFoundError:
    print("❌ 모델 파일을 찾을 수 없습니다. 먼저 04_고급_모델링_및_분석.py를 실행하세요.")
    exit(1)

# 데이터 로딩
DATA_DIR = Path.cwd() / "data" / "curated"
df = pd.read_csv(DATA_DIR / "player_final.csv", low_memory=True)

# 미래 데이터 제외
df = df[~df['season'].isin(['23/24', '24/25'])].copy()
print(f"✅ 데이터 로딩 완료: {df.shape}")

# ============================================================================
# 2. 고위험 선수 식별 및 분석
# ============================================================================

print("\n🎯 고위험 선수 식별...")

# 피쳐 엔지니어링 (전체 버전)
def create_all_features(df):
    """전체 피쳐 생성 (04번 파일과 동일)"""
    df = df.copy()
    
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
    
    # 고급 피쳐들
    # 1. 시즌 평균 출전시간 / 클럽 시즌 평균 러닝타임
    if 'season_avg_minutes' in df.columns and 'club_average_age' in df.columns:
        club_running_time = df.groupby(['club_name', 'season'])['season_avg_minutes'].mean().reset_index()
        club_running_time.columns = ['club_name', 'season', 'club_season_avg_minutes']
        df = df.merge(club_running_time, on=['club_name', 'season'], how='left')
        df['minutes_vs_club_avg'] = df['season_avg_minutes'] / (df['club_season_avg_minutes'] + 1e-6)
    
    # 2. 나이 차이
    if 'age_at_season' in df.columns and 'club_average_age' in df.columns:
        df['age_difference'] = df['age_at_season'] - df['club_average_age']
        df['age_relative_position'] = df['age_difference'] / (df['club_average_age'] + 1e-6)
    
    # 3. 공격 기여도 vs 팀 성과
    if 'goals' in df.columns and 'assists' in df.columns and 'season_win_count' in df.columns:
        df['attack_contribution'] = df['goals'] + df['assists']
        df['attack_vs_team_success'] = df['attack_contribution'] * df['season_win_count']
        df['attack_efficiency'] = df['attack_contribution'] / (df['season_win_count'] + 1e-6)
    
    # 4. 외국인 선수 여부
    if 'country_of_birth' in df.columns and 'club_foreigners_percentage' in df.columns:
        df['is_foreigner'] = (df['country_of_birth'] != 'England').astype(int)
        df['foreigner_vs_club_ratio'] = df['is_foreigner'] * df['club_foreigners_percentage']
    
    # 5. 포지션별 키 적합성
    if 'position' in df.columns and 'height_in_cm' in df.columns:
        position_height = df.groupby('position')['height_in_cm'].mean().reset_index()
        position_height.columns = ['position', 'position_avg_height']
        df = df.merge(position_height, on='position', how='left')
        df['height_vs_position'] = df['height_in_cm'] - df['position_avg_height']
        df['height_advantage'] = df['height_vs_position'] / (df['position_avg_height'] + 1e-6)
    
    # 6. 경고장과 출전시간
    if 'yellow_cards' in df.columns and 'season_avg_minutes' in df.columns:
        df['cards_per_minute'] = df['yellow_cards'] / (df['season_avg_minutes'] + 1e-6)
        df['discipline_score'] = 1 / (df['cards_per_minute'] + 1e-6)
    
    # 7. 클럽 재적 기간
    if 'season' in df.columns and 'club_name' in df.columns:
        player_club_first_season = df.groupby(['player_name', 'club_name'])['season'].min().reset_index()
        player_club_first_season.columns = ['player_name', 'club_name', 'first_season']
        df = df.merge(player_club_first_season, on=['player_name', 'club_name'], how='left')
        
        season_order = ['12/13', '13/14', '14/15', '15/16', '16/17', '17/18', '18/19', '19/20', '20/21', '21/22', '22/23']
        season_to_num = {s: i for i, s in enumerate(season_order)}
        df['season_num'] = df['season'].map(season_to_num)
        df['first_season_num'] = df['first_season'].map(season_to_num)
        df['club_tenure_seasons'] = df['season_num'] - df['first_season_num'] + 1
        df['club_tenure_seasons'] = df['club_tenure_seasons'].fillna(1)
    
    # 8. 포지션별 경쟁
    if 'position' in df.columns and 'club_name' in df.columns:
        position_club_count = df.groupby(['position', 'club_name']).size().reset_index(name='position_club_count')
        df = df.merge(position_club_count, on=['position', 'club_name'], how='left')
        df['position_competition'] = df['position_club_count'] - 1
    
    # 9. 시장가치 관련
    if 'player_highest_market_value_in_eur' in df.columns and 'player_market_value_in_eur' in df.columns:
        mv = pd.to_numeric(df['player_market_value_in_eur'], errors='coerce')
        mv_hi = pd.to_numeric(df['player_highest_market_value_in_eur'], errors='coerce')
        df['value_growth'] = (mv_hi - mv)
        df['negotiation_proxy'] = 0.6 * mv + 0.4 * mv_hi
    
    return df

df_processed = create_all_features(df)

# 기본 피쳐만 사용 (안전한 버전)
basic_features = [
    'goals', 'assists', 'yellow_cards', 'red_cards', 'season_avg_minutes',
    'player_market_value_in_eur', 'club_squad_size', 'club_average_age',
    'club_foreigners_percentage', 'club_national_team_players',
    'player_highest_market_value_in_eur', 'height_in_cm', 'season_win_count',
    'season_start_year', 'age_at_season', 'log_market_value',
    'season', 'position', 'sub_position', 'club_name', 'country_of_birth', 'foot'
]

# 존재하는 피쳐만 선택
available_features = [col for col in basic_features if col in df_processed.columns]
X_features = df_processed[available_features]

print(f"  📊 사용된 피쳐: {len(available_features)}개")
print(f"  📊 누락된 피쳐: {len(basic_features) - len(available_features)}개")

# 전처리
X_processed = preprocessor.transform(X_features)

# 이적 확률 예측
if hasattr(model, 'predict_proba'):
    transfer_proba = model.predict_proba(X_processed)[:, 1]
    df_processed['transfer_probability'] = transfer_proba
    
    # 고위험 선수 식별 (상위 20%)
    high_risk_threshold = np.percentile(transfer_proba, 80)
    high_risk_players = df_processed[df_processed['transfer_probability'] >= high_risk_threshold].copy()
    
    print(f"✅ 고위험 선수 식별 완료: {len(high_risk_players):,}명 (상위 20%)")
    print(f"   - 평균 이적 확률: {high_risk_players['transfer_probability'].mean():.3f}")
    print(f"   - 최고 이적 확률: {high_risk_players['transfer_probability'].max():.3f}")
else:
    print("❌ 확률 예측 불가능한 모델")
    exit(1)

# ============================================================================
# 3. 포지션별 리스크 분석
# ============================================================================

print("\n📊 포지션별 리스크 분석...")

if 'position' in high_risk_players.columns:
    position_analysis = high_risk_players.groupby('position').agg({
        'transfer_probability': ['count', 'mean', 'std'],
        'age_at_season': 'mean',
        'player_market_value_in_eur': 'mean'
    }).round(3)
    
    position_analysis.columns = ['선수수', '평균확률', '확률표준편차', '평균나이', '평균시장가치']
    
    print("  📈 포지션별 고위험 선수 분석:")
    for pos in position_analysis.index:
        count = position_analysis.loc[pos, '선수수']
        avg_prob = position_analysis.loc[pos, '평균확률']
        avg_age = position_analysis.loc[pos, '평균나이']
        avg_value = position_analysis.loc[pos, '평균시장가치']
        print(f"    {pos}: {count}명 (확률: {avg_prob:.3f}, 나이: {avg_age:.1f}세, 가치: €{avg_value:,.0f})")
    
    # 포지션별 리스크 시각화
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    position_analysis['선수수'].plot(kind='bar', color='skyblue')
    plt.title('포지션별 고위험 선수 수')
    plt.ylabel('선수 수')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    position_analysis['평균확률'].plot(kind='bar', color='lightcoral')
    plt.title('포지션별 평균 이적 확률')
    plt.ylabel('이적 확률')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    position_analysis['평균나이'].plot(kind='bar', color='lightgreen')
    plt.title('포지션별 평균 나이')
    plt.ylabel('나이')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    position_analysis['평균시장가치'].plot(kind='bar', color='gold')
    plt.title('포지션별 평균 시장가치')
    plt.ylabel('시장가치 (€)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 4. 연령대별 리스크 분석
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
        'transfer_probability': ['count', 'mean', 'std'],
        'player_market_value_in_eur': 'mean',
        'season_avg_minutes': 'mean'
    }).round(3)
    
    age_analysis.columns = ['선수수', '평균확률', '확률표준편차', '평균시장가치', '평균출전시간']
    
    print("  📈 연령대별 고위험 선수 분석:")
    for age in age_analysis.index:
        count = age_analysis.loc[age, '선수수']
        avg_prob = age_analysis.loc[age, '평균확률']
        avg_value = age_analysis.loc[age, '평균시장가치']
        avg_minutes = age_analysis.loc[age, '평균출전시간']
        print(f"    {age}: {count}명 (확률: {avg_prob:.3f}, 가치: €{avg_value:,.0f}, 출전: {avg_minutes:.0f}분)")
    
    # 연령대별 리스크 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    age_analysis['선수수'].plot(kind='bar', color='skyblue')
    plt.title('연령대별 고위험 선수 수')
    plt.ylabel('선수 수')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    age_analysis['평균확률'].plot(kind='bar', color='lightcoral')
    plt.title('연령대별 평균 이적 확률')
    plt.ylabel('이적 확률')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    age_analysis['평균시장가치'].plot(kind='bar', color='lightgreen')
    plt.title('연령대별 평균 시장가치')
    plt.ylabel('시장가치 (€)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. 클럽별 리스크 분석
# ============================================================================

print("\n🏟️ 클럽별 리스크 분석...")

if 'club_name' in high_risk_players.columns:
    club_analysis = high_risk_players.groupby('club_name').agg({
        'transfer_probability': ['count', 'mean'],
        'age_at_season': 'mean',
        'player_market_value_in_eur': 'mean'
    }).round(3)
    
    club_analysis.columns = ['고위험선수수', '평균확률', '평균나이', '평균시장가치']
    
    # 상위 10개 클럽
    top_clubs = club_analysis.nlargest(10, '고위험선수수')
    
    print("  📈 고위험 선수가 많은 클럽 TOP 10:")
    for club in top_clubs.index:
        count = top_clubs.loc[club, '고위험선수수']
        avg_prob = top_clubs.loc[club, '평균확률']
        avg_age = top_clubs.loc[club, '평균나이']
        print(f"    {club}: {count}명 (확률: {avg_prob:.3f}, 나이: {avg_age:.1f}세)")
    
    # 클럽별 리스크 시각화
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    top_clubs['고위험선수수'].plot(kind='bar', color='skyblue')
    plt.title('클럽별 고위험 선수 수 (TOP 10)')
    plt.ylabel('선수 수')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    top_clubs['평균확률'].plot(kind='bar', color='lightcoral')
    plt.title('클럽별 평균 이적 확률 (TOP 10)')
    plt.ylabel('이적 확률')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 6. 이적 후 케어 전략 수립
# ============================================================================

print("\n🛡️ 이적 후 케어 전략 수립...")

# 전략 1: 우선순위 기반 재계약 전략
print("\n  📋 전략 1: 우선순위 기반 재계약 전략")
priority_players = high_risk_players[
    (high_risk_players['transfer_probability'] >= 0.8) & 
    (high_risk_players['player_market_value_in_eur'] >= high_risk_players['player_market_value_in_eur'].quantile(0.5))
].copy()

print(f"    - 최우선 재계약 대상: {len(priority_players)}명")
print(f"    - 평균 시장가치: €{priority_players['player_market_value_in_eur'].mean():,.0f}")
print(f"    - 평균 이적 확률: {priority_players['transfer_probability'].mean():.3f}")

# 전략 2: 임대 전략
print("\n  📋 전략 2: 임대 전략")
loan_candidates = high_risk_players[
    (high_risk_players['age_at_season'] <= 22) & 
    (high_risk_players['transfer_probability'] >= 0.7)
].copy()

print(f"    - 임대 추천 대상: {len(loan_candidates)}명 (22세 이하)")
print(f"    - 평균 나이: {loan_candidates['age_at_season'].mean():.1f}세")
print(f"    - 평균 이적 확률: {loan_candidates['transfer_probability'].mean():.3f}")

# 전략 3: 인센티브 설계
print("\n  📋 전략 3: 인센티브 설계")
incentive_players = high_risk_players[
    (high_risk_players['transfer_probability'] >= 0.6) & 
    (high_risk_players['season_avg_minutes'] >= 30)
].copy()

print(f"    - 인센티브 대상: {len(incentive_players)}명 (출전시간 30분 이상)")
print(f"    - 평균 출전시간: {incentive_players['season_avg_minutes'].mean():.0f}분")
print(f"    - 평균 이적 확률: {incentive_players['transfer_probability'].mean():.3f}")

# 전략 4: 스쿼드 리스크 관리
print("\n  📋 전략 4: 스쿼드 리스크 관리")
if 'position' in high_risk_players.columns:
    position_risk = high_risk_players.groupby('position')['transfer_probability'].agg(['count', 'mean']).round(3)
    high_risk_positions = position_risk[position_risk['mean'] >= 0.7]
    
    print(f"    - 고위험 포지션: {len(high_risk_positions)}개")
    for pos in high_risk_positions.index:
        count = high_risk_positions.loc[pos, 'count']
        avg_prob = high_risk_positions.loc[pos, 'mean']
        print(f"      {pos}: {count}명 (평균 확률: {avg_prob:.3f})")

# ============================================================================
# 7. 모니터링 및 알림 시스템 설계
# ============================================================================

print("\n📊 모니터링 및 알림 시스템 설계...")

# 위험도 등급 분류
def classify_risk_level(prob):
    if prob >= 0.8:
        return "매우 높음"
    elif prob >= 0.6:
        return "높음"
    elif prob >= 0.4:
        return "보통"
    else:
        return "낮음"

high_risk_players['risk_level'] = high_risk_players['transfer_probability'].apply(classify_risk_level)

risk_distribution = high_risk_players['risk_level'].value_counts()
print("  📈 위험도 등급 분포:")
for level, count in risk_distribution.items():
    print(f"    {level}: {count}명 ({count/len(high_risk_players)*100:.1f}%)")

# 알림 시스템 설계
print("\n  🔔 알림 시스템 설계:")
print("    - 매우 높음 (≥0.8): 즉시 알림, 긴급 재계약 검토")
print("    - 높음 (≥0.6): 주간 모니터링, 재계약 준비")
print("    - 보통 (≥0.4): 월간 모니터링, 상황 관찰")
print("    - 낮음 (<0.4): 분기별 모니터링")

# ============================================================================
# 8. 비용-효과 분석
# ============================================================================

print("\n💰 비용-효과 분석...")

# 이적 방지 비용 vs 이적 손실 비교
avg_market_value = high_risk_players['player_market_value_in_eur'].mean()
avg_transfer_prob = high_risk_players['transfer_probability'].mean()

# 가정: 재계약 비용 = 시장가치의 20%, 이적 손실 = 시장가치의 100%
renewal_cost = avg_market_value * 0.2
transfer_loss = avg_market_value * 1.0

expected_renewal_cost = renewal_cost * avg_transfer_prob
expected_transfer_loss = transfer_loss * avg_transfer_prob

print(f"  📊 비용 분석 (평균 기준):")
print(f"    - 평균 시장가치: €{avg_market_value:,.0f}")
print(f"    - 평균 이적 확률: {avg_transfer_prob:.3f}")
print(f"    - 재계약 비용: €{renewal_cost:,.0f}")
print(f"    - 이적 손실: €{transfer_loss:,.0f}")
print(f"    - 예상 재계약 비용: €{expected_renewal_cost:,.0f}")
print(f"    - 예상 이적 손실: €{expected_transfer_loss:,.0f}")

if expected_renewal_cost < expected_transfer_loss:
    print(f"    ✅ 재계약이 경제적으로 유리 (절약: €{expected_transfer_loss - expected_renewal_cost:,.0f})")
else:
    print(f"    ❌ 이적이 경제적으로 유리 (절약: €{expected_renewal_cost - expected_transfer_loss:,.0f})")

# ============================================================================
# 9. 실행 계획 수립
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
