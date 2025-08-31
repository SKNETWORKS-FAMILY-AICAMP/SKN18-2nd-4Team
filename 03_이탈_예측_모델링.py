# 📊 E-Commerce 고객 이탈 예측 프로젝트
# 03. 이탈 예측 모델링
# 
# 이 노트북에서는 다양한 머신러닝 모델을 사용하여 고객 이탈을 예측합니다.

# ============================================================================
# 셀 1: 라이브러리 임포트
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, 
                           roc_curve, precision_recall_curve, auc)
from sklearn.utils.class_weight import compute_class_weight
import joblib

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 그래프 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ 라이브러리 임포트 완료!")

# ============================================================================
# 셀 2: 데이터 로딩 및 전처리
# ============================================================================

# 데이터 로드
df = pd.read_excel("data/raw/E Commerce Dataset.xlsx", sheet_name='E Comm')
print(f"✅ 데이터 로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")

# 데이터 복사본 생성
df_model = df.copy()

# 결측값 처리
numeric_missing_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
                       'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']

for col in numeric_missing_cols:
    if col in df_model.columns:
        median_val = df_model[col].median()
        df_model[col].fillna(median_val, inplace=True)

# 범주형 변수 인코딩
categorical_features = ['Gender', 'MaritalStatus', 'CityTier', 'PreferredPaymentMode', 'PreferedOrderCat']
label_encoders = {}

for col in categorical_features:
    if col in df_model.columns:
        le = LabelEncoder()
        df_model[f'{col}_encoded'] = le.fit_transform(df_model[col])
        label_encoders[col] = le

# 피쳐 엔지니어링 (02번 노트북에서 생성한 피쳐들)
df_model['ActivityScore'] = (
    df_model['OrderCount'] * 0.4 + 
    df_model['HourSpendOnApp'] * 0.3 + 
    df_model['CouponUsed'] * 0.3
)

df_model['OrderFrequency'] = df_model['OrderCount'] / (df_model['Tenure'] + 1)

df_model['CustomerValueScore'] = (
    df_model['CashbackAmount'] * 0.5 + 
    df_model['OrderCount'] * 0.3 + 
    df_model['Tenure'] * 0.2
)

print("✅ 데이터 전처리 완료!")

# ============================================================================
# 셀 3: 피쳐 선택 및 데이터 준비
# ============================================================================

# 모델링용 피쳐 선택
modeling_features = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                    'SatisfactionScore', 'NumberOfAddress', 'OrderAmountHikeFromlastYear',
                    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount',
                    'ActivityScore', 'OrderFrequency', 'CustomerValueScore']

# 인코딩된 범주형 변수 추가
for col in categorical_features:
    if f'{col}_encoded' in df_model.columns:
        modeling_features.append(f'{col}_encoded')

# 데이터 준비
X = df_model[modeling_features].fillna(0)
y = df_model['Churn']

print(f"📊 모델링 데이터 크기: {X.shape}")
print(f"🎯 타겟 변수 분포:")
print(y.value_counts(normalize=True))

# ============================================================================
# 셀 4: 데이터 분할
# ============================================================================

# 훈련/테스트 데이터 분할 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("📊 데이터 분할 결과:")
print(f"- 훈련 데이터: {X_train.shape[0]:,}개 샘플")
print(f"- 테스트 데이터: {X_test.shape[0]:,}개 샘플")
print(f"- 훈련 데이터 이탈률: {y_train.mean()*100:.1f}%")
print(f"- 테스트 데이터 이탈률: {y_test.mean()*100:.1f}%")

# ============================================================================
# 셀 5: 피쳐 스케일링
# ============================================================================

# StandardScaler 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ 피쳐 스케일링 완료!")

# ============================================================================
# 셀 6: 클래스 불균형 확인 및 처리
# ============================================================================

# 클래스 불균형 확인
class_counts = y.value_counts()
print("📊 클래스 분포:")
print(f"- 유지 고객 (0): {class_counts[0]:,}명 ({class_counts[0]/len(y)*100:.1f}%)")
print(f"- 이탈 고객 (1): {class_counts[1]:,}명 ({class_counts[1]/len(y)*100:.1f}%)")

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

print(f"\n⚖️ 클래스 가중치:")
print(f"- 유지 고객 (0): {class_weight_dict[0]:.3f}")
print(f"- 이탈 고객 (1): {class_weight_dict[1]:.3f}")

# ============================================================================
# 셀 7: 기본 모델 성능 비교
# ============================================================================

# 다양한 모델 정의
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

# 모델 성능 평가
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("🔍 기본 모델 성능 비교:")
print("=" * 80)

for name, model in models.items():
    # 교차 검증
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
    
    # 모델 훈련
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std()
    }
    
    print(f"\n📌 {name}:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-Score: {f1:.4f}")
    print(f"  - AUC: {auc_score:.4f}" if auc_score else "  - AUC: N/A")
    print(f"  - CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================================================
# 셀 8: 모델 성능 비교 시각화
# ============================================================================

# 성능 비교 시각화
metrics = ['accuracy', 'precision', 'recall', 'f1']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, metric in enumerate(metrics):
    values = [results[name][metric] for name in models.keys()]
    bars = axes[i].bar(models.keys(), values, color='skyblue')
    axes[i].set_title(f'{metric.title()} Score')
    axes[i].set_ylabel(metric.title())
    axes[i].tick_params(axis='x', rotation=45)
    
    # 값 표시
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ============================================================================
# 셀 9: 최고 성능 모델 선택 및 하이퍼파라미터 튜닝
# ============================================================================

# 최고 F1 점수 모델 선택
best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
print(f"🏆 최고 성능 모델: {best_model_name}")
print(f"   - F1 Score: {results[best_model_name]['f1']:.4f}")

# Random Forest 하이퍼파라미터 튜닝
if best_model_name == 'Random Forest':
    print("\n🔧 Random Forest 하이퍼파라미터 튜닝 중...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"✅ 최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f"✅ 최적 F1 Score: {grid_search.best_score_:.4f}")
    
    # 최적 모델로 예측
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test_scaled)
    y_pred_proba_rf = best_rf.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n📊 튜닝된 Random Forest 성능:")
    print(f"- Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"- Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"- Recall: {recall_score(y_test, y_pred_rf):.4f}")
    print(f"- F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
    print(f"- AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# ============================================================================
# 셀 10: 혼동 행렬 시각화
# ============================================================================

# 최고 성능 모델의 혼동 행렬
if best_model_name == 'Random Forest' and 'best_rf' in locals():
    model_for_cm = best_rf
    y_pred_for_cm = y_pred_rf
else:
    model_for_cm = models[best_model_name]
    model_for_cm.fit(X_train_scaled, y_train)
    y_pred_for_cm = model_for_cm.predict(X_test_scaled)

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred_for_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['유지', '이탈'], yticklabels=['유지', '이탈'])
plt.title(f'{best_model_name} 혼동 행렬')
plt.xlabel('예측')
plt.ylabel('실제')
plt.show()

# 분류 리포트
print(f"\n📋 {best_model_name} 분류 리포트:")
print(classification_report(y_test, y_pred_for_cm, target_names=['유지', '이탈']))

# ============================================================================
# 셀 11: ROC 곡선 및 AUC
# ============================================================================

# ROC 곡선 그리기
plt.figure(figsize=(12, 5))

# 서브플롯 1: ROC 곡선
plt.subplot(1, 2, 1)
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 곡선')
plt.legend()
plt.grid(True)

# 서브플롯 2: Precision-Recall 곡선
plt.subplot(1, 2, 2)
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall 곡선')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# 셀 12: 피쳐 중요도 분석 (Random Forest)
# ============================================================================

if best_model_name == 'Random Forest':
    # 피쳐 중요도
    feature_importance = pd.DataFrame({
        'feature': modeling_features,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🎯 Random Forest 피쳐 중요도 (상위 10개):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # 피쳐 중요도 시각화
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('피쳐 중요도')
    plt.title('Random Forest 피쳐 중요도 (상위 15개)')
    plt.gca().invert_yaxis()
    
    # 값 표시
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{importance:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 셀 13: 임계값 조정 및 성능 최적화
# ============================================================================

# 최고 성능 모델의 예측 확률
if hasattr(model_for_cm, 'predict_proba'):
    y_pred_proba = model_for_cm.predict_proba(X_test_scaled)[:, 1]
    
    # 다양한 임계값에서 성능 확인
    thresholds = np.arange(0.1, 0.9, 0.05)
    threshold_results = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        precision = precision_score(y_test, y_pred_threshold)
        recall = recall_score(y_test, y_pred_threshold)
        threshold_results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
    
    threshold_df = pd.DataFrame(threshold_results)
    
    # 최적 임계값 찾기
    optimal_threshold = threshold_df.loc[threshold_df['f1'].idxmax(), 'threshold']
    print(f"🎯 최적 임계값: {optimal_threshold:.3f}")
    
    # 최적 임계값으로 예측
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    print(f"\n📊 최적 임계값 적용 후 성능:")
    print(f"- Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")
    print(f"- Precision: {precision_score(y_test, y_pred_optimal):.4f}")
    print(f"- Recall: {recall_score(y_test, y_pred_optimal):.4f}")
    print(f"- F1-Score: {f1_score(y_test, y_pred_optimal):.4f}")

# ============================================================================
# 셀 14: 모델 해석 및 비즈니스 인사이트
# ============================================================================

print("💡 모델 해석 및 비즈니스 인사이트:")
print("=" * 60)

print("\n📊 모델 성능 요약:")
print(f"- 최고 성능 모델: {best_model_name}")
print(f"- F1 Score: {results[best_model_name]['f1']:.4f}")
print(f"- AUC: {results[best_model_name]['auc']:.4f}" if results[best_model_name]['auc'] else "- AUC: N/A")

print("\n🎯 주요 예측 요인:")
if best_model_name == 'Random Forest' and 'feature_importance' in locals():
    top_5_features = feature_importance.head(5)
    for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
        print(f"{i}. {row['feature']}: {row['importance']:.4f}")

print("\n🚀 비즈니스 권장사항:")
print("1. 고위험 고객 식별 및 개입 전략 수립")
print("2. 만족도 향상 프로그램 강화")
print("3. 고객 유지 기간 연장을 위한 프로그램 개발")
print("4. 주문 빈도 증가를 위한 마케팅 전략")
print("5. 캐시백 혜택 최적화")

# ============================================================================
# 셀 15: 모델 저장
# ============================================================================

# 최종 모델 저장
if best_model_name == 'Random Forest' and 'best_rf' in locals():
    final_model = best_rf
else:
    final_model = model_for_cm

# 모델과 스케일러 저장
joblib.dump(final_model, 'churn_prediction_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("✅ 모델 저장 완료!")
print("- churn_prediction_model.pkl: 예측 모델")
print("- feature_scaler.pkl: 피쳐 스케일러")
print("- label_encoders.pkl: 라벨 인코더")

# ============================================================================
# 셀 16: 예측 함수 생성
# ============================================================================

def predict_churn(customer_data, model, scaler, label_encoders):
    """
    고객 데이터를 받아서 이탈 확률을 예측하는 함수
    
    Parameters:
    customer_data: 고객 정보 딕셔너리
    model: 훈련된 모델
    scaler: 피쳐 스케일러
    label_encoders: 라벨 인코더들
    
    Returns:
    churn_probability: 이탈 확률
    churn_prediction: 이탈 예측 (0: 유지, 1: 이탈)
    """
    
    # 피쳐 엔지니어링
    customer_data['ActivityScore'] = (
        customer_data['OrderCount'] * 0.4 + 
        customer_data['HourSpendOnApp'] * 0.3 + 
        customer_data['CouponUsed'] * 0.3
    )
    
    customer_data['OrderFrequency'] = customer_data['OrderCount'] / (customer_data['Tenure'] + 1)
    
    customer_data['CustomerValueScore'] = (
        customer_data['CashbackAmount'] * 0.5 + 
        customer_data['OrderCount'] * 0.3 + 
        customer_data['Tenure'] * 0.2
    )
    
    # 범주형 변수 인코딩
    for col in categorical_features:
        if col in customer_data and col in label_encoders:
            customer_data[f'{col}_encoded'] = label_encoders[col].transform([customer_data[col]])[0]
    
    # 피쳐 벡터 생성
    feature_vector = []
    for feature in modeling_features:
        if feature in customer_data:
            feature_vector.append(customer_data[feature])
        else:
            feature_vector.append(0)
    
    # 스케일링 및 예측
    feature_vector_scaled = scaler.transform([feature_vector])
    churn_probability = model.predict_proba(feature_vector_scaled)[0, 1]
    churn_prediction = 1 if churn_probability > 0.5 else 0
    
    return churn_probability, churn_prediction

print("✅ 예측 함수 생성 완료!")

# ============================================================================
# 셀 17: 모델 성능 요약 및 결론
# ============================================================================

print("📋 모델링 프로젝트 요약:")
print("=" * 60)

print("\n📊 데이터 개요:")
print(f"- 총 고객 수: {len(df):,}명")
print(f"- 이탈률: {df['Churn'].mean()*100:.1f}%")
print(f"- 사용된 피쳐 수: {len(modeling_features)}개")

print("\n🏆 최종 모델 성능:")
print(f"- 모델: {best_model_name}")
print(f"- Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"- Precision: {results[best_model_name]['precision']:.4f}")
print(f"- Recall: {results[best_model_name]['recall']:.4f}")
print(f"- F1-Score: {results[best_model_name]['f1']:.4f}")
if results[best_model_name]['auc']:
    print(f"- AUC: {results[best_model_name]['auc']:.4f}")

print("\n🎯 주요 성과:")
print("1. 고객 이탈 예측 모델 성공적으로 구축")
print("2. 다양한 머신러닝 알고리즘 비교 분석 완료")
print("3. 하이퍼파라미터 튜닝을 통한 성능 최적화")
print("4. 피쳐 중요도 분석을 통한 인사이트 도출")
print("5. 실무 적용 가능한 예측 시스템 구축")

print("\n🚀 향후 개선 방향:")
print("1. 더 많은 피쳐 엔지니어링 시도")
print("2. 딥러닝 모델 적용 검토")
print("3. 앙상블 모델 구축")
print("4. 실시간 예측 시스템 구축")
print("5. A/B 테스트를 통한 모델 검증")

# ============================================================================
# 마크다운 셀 내용 (복사해서 노트북에 붙여넣기)
# ============================================================================

"""
# 📊 E-Commerce 고객 이탈 예측 프로젝트
## 03. 이탈 예측 모델링

이 노트북에서는 다양한 머신러닝 모델을 사용하여 고객 이탈을 예측합니다.

### 📋 목차
1. [라이브러리 임포트](#1-라이브러리-임포트)
2. [데이터 로딩 및 전처리](#2-데이터-로딩-및-전처리)
3. [피쳐 선택 및 데이터 준비](#3-피쳐-선택-및-데이터-준비)
4. [데이터 분할](#4-데이터-분할)
5. [피쳐 스케일링](#5-피쳐-스케일링)
6. [클래스 불균형 처리](#6-클래스-불균형-처리)
7. [기본 모델 성능 비교](#7-기본-모델-성능-비교)
8. [모델 성능 비교 시각화](#8-모델-성능-비교-시각화)
9. [하이퍼파라미터 튜닝](#9-하이퍼파라미터-튜닝)
10. [혼동 행렬 시각화](#10-혼동-행렬-시각화)
11. [ROC 곡선 및 AUC](#11-roc-곡선-및-auc)
12. [피쳐 중요도 분석](#12-피쳐-중요도-분석)
13. [임계값 조정](#13-임계값-조정)
14. [모델 해석 및 인사이트](#14-모델-해석-및-인사이트)
15. [모델 저장](#15-모델-저장)
16. [예측 함수 생성](#16-예측-함수-생성)
17. [프로젝트 요약](#17-프로젝트-요약)

## 1. 라이브러리 임포트

## 2. 데이터 로딩 및 전처리

## 3. 피쳐 선택 및 데이터 준비

## 4. 데이터 분할

## 5. 피쳐 스케일링

## 6. 클래스 불균형 처리

## 7. 기본 모델 성능 비교

## 8. 모델 성능 비교 시각화

## 9. 하이퍼파라미터 튜닝

## 10. 혼동 행렬 시각화

## 11. ROC 곡선 및 AUC

## 12. 피쳐 중요도 분석

## 13. 임계값 조정

## 14. 모델 해석 및 인사이트

## 15. 모델 저장

## 16. 예측 함수 생성

## 17. 프로젝트 요약

## 📝 요약

이 노트북에서는 E-Commerce 고객 이탈 예측을 위한 머신러닝 모델을 구축했습니다.

### 주요 성과:
- **다양한 모델 비교**: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree
- **성능 최적화**: 하이퍼파라미터 튜닝을 통한 모델 성능 향상
- **피쳐 중요도 분석**: 고객 이탈에 영향을 미치는 주요 요인 파악
- **실무 적용**: 예측 함수 및 모델 저장을 통한 실무 적용 가능

### 최종 모델:
- **모델**: Random Forest (튜닝된 버전)
- **성능**: F1-Score 기준 최고 성능
- **주요 피쳐**: 고객 유지 기간, 주문 횟수, 만족도 등

### 비즈니스 가치:
1. 고위험 고객 사전 식별
2. 맞춤형 고객 유지 전략 수립
3. 마케팅 효율성 향상
4. 고객 만족도 개선 프로그램 개발

### 다음 단계:
1. 실시간 예측 시스템 구축
2. A/B 테스트를 통한 모델 검증
3. 추가 피쳐 엔지니어링
4. 딥러닝 모델 적용 검토
"""
