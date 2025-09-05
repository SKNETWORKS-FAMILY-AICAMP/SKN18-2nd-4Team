"""
Football Transfer Prediction - Advanced Modeling & Analysis
고급 모델링 및 분석 (모듈화, 피쳐 엔지니어링, 오버피팅 체크)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 모듈 임포트
from src.features.feature_engineering import FootballFeatureEngineer, DataLeakageChecker, OverfittingChecker
from src.models.football_modeling import FootballModelingPipeline

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Football Transfer Prediction - Advanced Modeling")
print("=" * 60)

# ============================================================================
# 1. 데이터 로딩
# ============================================================================

print("\n📁 데이터 로딩...")
DATA_DIR = Path.cwd() / "data" / "curated"
df = pd.read_csv(DATA_DIR / "player_final.csv", low_memory=True)
print(f"✅ 데이터 로드 완료: {df.shape}")

# ============================================================================
# 2. 데이터 누수 검사
# ============================================================================

print("\n🔍 데이터 누수 검사...")
leakage_checker = DataLeakageChecker()

# 시간적 누수 검사
temporal_check = leakage_checker.check_temporal_leakage(df, 'season', 'transfer')
print(f"  - 미래 데이터 포함: {temporal_check.get('has_future_data', False)}")
print(f"  - 시간적 일관성: {temporal_check.get('temporal_consistency', False)}")

# 피쳐 누수 검사
feature_check = leakage_checker.check_feature_leakage(df, 'transfer')
if feature_check['suspicious_features']:
    print(f"  - 의심스러운 피쳐: {feature_check['suspicious_features']}")
else:
    print("  - 피쳐 누수 없음")

# 데이터 품질 검사
quality_check = leakage_checker.check_data_quality(df)
print(f"  - 중복 행: {quality_check['duplicate_rows']}개")
print(f"  - 상수 피쳐: {quality_check['constant_features']}")

# ============================================================================
# 3. 피쳐 엔지니어링
# ============================================================================

print("\n🔧 피쳐 엔지니어링...")
feature_engineer = FootballFeatureEngineer()

# 기본 피쳐 생성
df_processed = feature_engineer.create_basic_features(df)
print("  ✅ 기본 피쳐 생성 완료")

# 고급 피쳐 생성
df_processed = feature_engineer.create_advanced_features(df_processed)
print("  ✅ 고급 피쳐 생성 완료")

# 피쳐 타입 분류
feature_types = feature_engineer.get_feature_types(df_processed)
print(f"  📊 피쳐 분류:")
print(f"    - 수치형: {len(feature_types['numeric'])}개")
print(f"    - 순서형: {len(feature_types['ordinal'])}개")
print(f"    - 명목형: {len(feature_types['nominal'])}개")

# ============================================================================
# 4. 데이터 분할 (시간 순서 고려)
# ============================================================================

print("\n📅 데이터 분할 (시간 순서 고려)...")

# 미래 데이터 제외
before = len(df_processed)
df_processed = df_processed[~df_processed['season'].isin(['23/24', '24/25'])].copy()
after = len(df_processed)
print(f"  🧹 미래 시즌 제외: {before-after:,}건 제거")

# 타겟 설정
target_col = 'transfer'
df_processed[target_col] = pd.to_numeric(df_processed[target_col], errors='coerce').fillna(0).astype(int)

# 22/23 시즌을 테스트로 사용
test_mask = df_processed['season'] == '22/23'
X_train = df_processed[~test_mask]
X_test = df_processed[test_mask]
y_train = df_processed.loc[~test_mask, target_col]
y_test = df_processed.loc[test_mask, target_col]

print(f"  📊 분할 결과:")
print(f"    - 훈련 데이터: {len(X_train):,}개 (12/13~21/22)")
print(f"    - 테스트 데이터: {len(X_test):,}개 (22/23)")
print(f"    - 훈련 이탈률: {y_train.mean()*100:.1f}%")
print(f"    - 테스트 이탈률: {y_test.mean()*100:.1f}%")

# ============================================================================
# 5. 전처리 파이프라인
# ============================================================================

print("\n⚙️ 전처리 파이프라인...")

# 전처리기 생성
preprocessor = feature_engineer.create_preprocessor(feature_types)

# 피쳐 선택
modeling_features = (feature_types['numeric'] + 
                    feature_types['ordinal'] + 
                    feature_types['nominal'])

X_train_features = X_train[modeling_features]
X_test_features = X_test[modeling_features]

# 전처리 실행
X_train_processed = preprocessor.fit_transform(X_train_features)
X_test_processed = preprocessor.transform(X_test_features)

print(f"  ✅ 전처리 완료: {X_train_processed.shape[1]}개 피쳐")

# ============================================================================
# 6. 오버피팅 검사
# ============================================================================

print("\n🔍 오버피팅 검사...")
overfitting_checker = OverfittingChecker()

# 교차검증 일관성 검사
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
cv_results = overfitting_checker.check_cv_consistency(rf_model, X_train_processed, y_train)

print(f"  📊 교차검증 결과:")
print(f"    - CV 평균: {cv_results['cv_mean']:.4f}")
print(f"    - CV 표준편차: {cv_results['cv_std']:.4f}")
print(f"    - 안정성: {'✅ 안정' if cv_results['is_stable'] else '❌ 불안정'}")

# ============================================================================
# 7. 모델링 파이프라인 실행
# ============================================================================

print("\n🤖 모델링 파이프라인 실행...")
modeling_pipeline = FootballModelingPipeline(random_state=42)

# 모델링 실행
best_model, results = modeling_pipeline.fit(
    X_train_processed, y_train, 
    X_test_processed, y_test,
    preprocessor, feature_types
)

# ============================================================================
# 8. 비즈니스 인사이트 분석
# ============================================================================

print("\n💡 비즈니스 인사이트 분석...")

# 이적 후 케어 전략 분석
def analyze_transfer_care_strategy(df: pd.DataFrame, model, preprocessor, feature_types):
    """이적 후 케어 전략 분석"""
    
    # 고위험 선수 식별 (이적 확률 상위 20%)
    modeling_features = (feature_types['numeric'] + 
                        feature_types['ordinal'] + 
                        feature_types['nominal'])
    
    X_features = df[modeling_features]
    X_processed = preprocessor.transform(X_features)
    
    if hasattr(model, 'predict_proba'):
        transfer_proba = model.predict_proba(X_processed)[:, 1]
        df['transfer_probability'] = transfer_proba
        
        # 고위험 선수
        high_risk_threshold = np.percentile(transfer_proba, 80)
        high_risk_players = df[df['transfer_probability'] >= high_risk_threshold].copy()
        
        print(f"  🎯 고위험 선수 분석 (상위 20%):")
        print(f"    - 고위험 선수 수: {len(high_risk_players):,}명")
        print(f"    - 평균 이적 확률: {high_risk_players['transfer_probability'].mean():.3f}")
        
        # 포지션별 고위험 선수
        if 'position' in high_risk_players.columns:
            position_risk = high_risk_players.groupby('position').agg({
                'transfer_probability': ['count', 'mean']
            }).round(3)
            print(f"    - 포지션별 고위험 선수:")
            for pos in position_risk.index:
                count = position_risk.loc[pos, ('transfer_probability', 'count')]
                avg_prob = position_risk.loc[pos, ('transfer_probability', 'mean')]
                print(f"      {pos}: {count}명 (평균 확률: {avg_prob:.3f})")
        
        # 연령대별 고위험 선수
        if 'age_at_season' in high_risk_players.columns:
            high_risk_players['age_group'] = pd.cut(
                high_risk_players['age_at_season'], 
                bins=[0, 22, 26, 30, 100], 
                labels=['22세 이하', '23-26세', '27-30세', '30세 이상']
            )
            age_risk = high_risk_players.groupby('age_group')['transfer_probability'].agg(['count', 'mean']).round(3)
            print(f"    - 연령대별 고위험 선수:")
            for age in age_risk.index:
                count = age_risk.loc[age, 'count']
                avg_prob = age_risk.loc[age, 'mean']
                print(f"      {age}: {count}명 (평균 확률: {avg_prob:.3f})")
        
        # 케어 전략 제안
        print(f"\n  🛡️ 이적 후 케어 전략 제안:")
        print(f"    1. 재계약 우선순위: 고위험 선수 중 핵심 포지션 선수")
        print(f"    2. 임대 전략: 고위험 선수 중 젊은 선수 (22세 이하)")
        print(f"    3. 인센티브 설계: 시장가치 연계 성과 보너스")
        print(f"    4. 스쿼드 관리: 포지션별 리스크 분산")
        
        return high_risk_players
    else:
        print("  ❌ 확률 예측 불가능한 모델")
        return None

# 비즈니스 인사이트 분석 실행
high_risk_players = analyze_transfer_care_strategy(
    df_processed, best_model, preprocessor, feature_types
)

# ============================================================================
# 9. 모델 성능 요약
# ============================================================================

print("\n📋 최종 모델 성능 요약:")
print("=" * 60)

best_result = results[modeling_pipeline.best_model_name]
print(f"🏆 최고 성능 모델: {modeling_pipeline.best_model_name}")
print(f"  - Accuracy: {best_result['accuracy']:.4f}")
print(f"  - Precision: {best_result['precision']:.4f}")
print(f"  - Recall: {best_result['recall']:.4f}")
print(f"  - F1-Score: {best_result['f1']:.4f}")
print(f"  - AUC: {best_result['auc']:.4f}" if best_result['auc'] else "  - AUC: N/A")

print(f"\n🎯 주요 성과:")
print(f"  1. 모듈화된 피쳐 엔지니어링 파이프라인 구축")
print(f"  2. 8개 고급 피쳐 생성 및 적용")
print(f"  3. 데이터 누수 검사 및 시간 순서 고려")
print(f"  4. 오버피팅 검사 및 모델 안정성 확보")
print(f"  5. 비즈니스 인사이트 기반 케어 전략 수립")

print(f"\n🚀 향후 개선 방향:")
print(f"  1. 실시간 모니터링 시스템 구축")
print(f"  2. A/B 테스트를 통한 전략 검증")
print(f"  3. 딥러닝 모델 적용 검토")
print(f"  4. 앙상블 모델 구축")

print("\n✅ 고급 모델링 및 분석 완료!")
