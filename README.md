# Football Transfer Prediction Project

## 📋 프로젝트 개요

이 프로젝트는 머신러닝을 활용하여 축구 선수의 이적 가능성을 예측하는 시스템입니다. 8개의 고급 피처 엔지니어링과 다양한 머신러닝 모델을 비교하여 최적의 예측 모델을 구축했습니다.

## 🎯 주요 목표

- **이적 예측**: 23/24 시즌 선수들의 이적 가능성 예측
- **모델 비교**: 8개 머신러닝 모델의 성능 비교 및 최적 모델 선택
- **피처 엔지니어링**: 8개의 고급 피처를 통한 예측 성능 향상
- **오버피팅 분석**: 학습 곡선과 검증 곡선을 통한 모델 안정성 평가

## 📊 데이터 구조

### `data/raw/` (원본 데이터)

```
├── players.csv              # 선수 기본 정보 (이름, 생년월일, 포지션, 키 등)
├── clubs.csv                # 클럽 정보 (이름, 국가, 평균 나이 등)
├── games.csv                # 경기 정보 (날짜, 홈/어웨이, 결과)
├── transfers.csv            # 이적 정보 (이적 여부, 시기, 금액)
├── appearances.csv          # 출전 정보 (출전 시간, 골, 어시스트, 카드)
├── game_events.csv          # 경기 이벤트 (골, 교체, 카드 등)
├── game_lineups.csv         # 라인업 정보 (선발/교체 선수)
├── player_valuations.csv    # 선수 가치 정보 (시장 가치 변화)
└── competitions.csv         # 대회 정보 (리그, 컵 대회)
```

### `data/curated/` (전처리된 데이터)

```
├── train_df.csv             # 원본 훈련 데이터 (14,873 rows × 27 columns)
├── test_df.csv              # 원본 테스트 데이터 (1,442 rows × 27 columns)
└── README.md                # 데이터 설명서
```

### 자동 생성되는 임시 파일들

- 피처 엔지니어링은 모델링 파이프라인 내에서 실시간 적용
- 원본 데이터 보존, 중간 파일 생성 최소화

## 🔧 피처 엔지니어링

### 기본 피처 (23개)

- 선수 기본 정보: `age_at_season`, `height_in_cm`, `position`, `sub_position`
- 성과 지표: `goals`, `assists`, `yellow_cards`, `red_cards`, `season_avg_minutes`
- 시장 가치: `player_market_value_in_eur`, `log_market_value`
- 클럽 정보: `club_squad_size`, `club_average_age`, `club_foreigners_percentage`

### 고급 피처 (8개)

1. **`minutes_vs_club_avg`**: 선수 출전시간 / 클럽 평균 출전시간
2. **`age_difference`**: 선수 나이 - 클럽 평균 나이
3. **`attack_contribution`**: (골 + 어시스트) × 시즌 승수
4. **`is_foreigner`**: 외국인 선수 여부 (England 기준)
5. **`height_vs_position`**: 선수 키 / 포지션별 평균 키
6. **`cards_per_minute`**: (옐로카드 + 레드카드) / 출전시간
7. **`club_tenure_seasons`**: 클럽 재직 시즌 수
8. **`position_competition`**: 포지션별 경쟁 선수 수

## 🤖 모델 성능

### 최고 성능 모델: Logistic Regression

- **복합 점수**: 0.5594
- **선택 기준**: 복합 점수 (가중 평균)
- **가중 공식**: Accuracy(0.2) + Precision(0.2) + Recall(0.2) + F1(0.2) + AUC(0.2)

### 모델 순위

1. 🥇 **Logistic Regression**: 0.5594
2. 🥈 **SVM**: 0.5248
3. 🥉 **LightGBM**: 0.5122
4. Gradient Boosting: 0.4506
5. XGBoost: 0.4368
6. Decision Tree: 0.4127
7. KNN: 0.4059
8. Random Forest: 0.3998

## 📈 평가 지표

### 복합 점수 (Composite Score)

```
복합 점수 = Accuracy × 0.2 + Precision × 0.2 + Recall × 0.2 + F1-Score × 0.2 + AUC × 0.2
```

- **Accuracy (20%)**: 전체 예측 정확도
- **Precision (20%)**: 예측한 이적 중 실제 이적 비율
- **Recall (20%)**: 실제 이적 중 예측한 이적 비율
- **F1-Score (20%)**: Precision과 Recall의 조화평균
- **AUC (20%)**: 전체적인 분류 성능

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 통합 실행 (`main.py` 권장)

```bash
# 기본 모델링 (피처 엔지니어링 자동 적용)
python main.py --mode train

# 예측만 실행 (기본 모델링에 포함)
python main.py --mode predict

# 하이퍼파라미터 튜닝
python main.py --mode tune

# 정규화 강화 (오버피팅 완화)
python main.py --mode regularize

# 앙상블 모델링
python main.py --mode ensemble

# 전체 파이프라인 실행
python main.py --mode all
```

### 3. 개별 스크립트 실행 (고급 사용자)

```bash
# 기본 모델링
python scripts/run_final_modeling.py

# 하이퍼파라미터 최적화
python scripts/hyperparameter_tuning.py

# 정규화 강화
python scripts/regularization_improvement.py

# 앙상블 모델
python scripts/ensemble_modeling.py

# 모델 성능 저장
python scripts/save_model_performance.py
```

## 📁 프로젝트 구조

### 핵심 모듈

#### `src/features/feature_engineering.py`

```
├── CustomLabelEncoder          # 파이프라인용 라벨 인코더
│   ├── 새로운 라벨 처리 (-1 매핑)
│   └── sklearn Pipeline 호환
├── FootballFeatureEngineer     # 11개 피처 생성 + 전처리
│   ├── 기본 피처 (시즌 연도, 나이, 로그 시장가치)
│   ├── 고급 피처 (클럽 비교, 포지션 분석, 경쟁도)
│   ├── 피처 타입 자동 분류 (수치형/순서형/명목형)
│   └── 하이브리드 전처리 파이프라인
├── DataLeakageChecker          # 데이터 누수 검사
│   ├── 시간적 누수 검사
│   ├── 피처 누수 검사 (높은 상관관계)
│   └── 데이터 품질 검사
└── OverfittingChecker          # 오버피팅 검사
    ├── 학습 곡선 분석
    └── 교차검증 일관성 검사
```

#### `src/models/football_modeling.py`

```
└── FootballModelTrainer        # 통합 모델링 파이프라인
    ├── 8개 모델 비교 (Logistic, RF, GB, SVM, KNN, DT, XGB, LGBM)
    ├── 자동 피처 엔지니어링 적용
    ├── 하이브리드 인코딩 (라벨 + 원핫)
    ├── 복합 점수 기반 모델 선택
    ├── SHAP 분석 (피처 중요도 해석)
    └── 23/24 시즌 예측 생성
```

#### `src/visualization/plotter.py`

```
└── ModelVisualizer             # 시각화 통합 클래스
    ├── 모델 성능 비교 차트
    ├── 혼동 행렬 히트맵
    ├── ROC 곡선 분석
    ├── 피처 중요도 그래프 (상위 30개)
    ├── SHAP 분석 (요약 + 바 플롯, 상위 20개, 가로 확장)
    └── 예측 분포 히스토그램
```

#### `scripts/` 디렉토리 (개별 실행 스크립트들)

```
├── run_final_modeling.py       # 기본 모델링 실행
├── hyperparameter_tuning.py    # 하이퍼파라미터 최적화
├── regularization_improvement.py  # 정규화 강화
├── ensemble_modeling.py        # 앙상블 모델 구축
└── save_model_performance.py   # 모델 성능 상세 저장
```

### 실행 방법

#### 중앙 집중식 실행 (`main.py`)

```bash
python main.py --mode train      # 기본 모델링
python main.py --mode tune       # 하이퍼파라미터 튜닝
python main.py --mode regularize # 정규화 강화
python main.py --mode ensemble   # 앙상블 모델링
python main.py --mode all        # 전체 파이프라인
```

#### 개별 스크립트 실행

```bash
python scripts/run_final_modeling.py
python scripts/hyperparameter_tuning.py
python scripts/regularization_improvement.py
python scripts/ensemble_modeling.py
python scripts/save_model_performance.py
```

### 출력 파일

#### 모델 결과

- `outputs/model.pkl`: 최종 모델
- `outputs/preprocessor.pkl`: 전처리기
- `outputs/model_results.pkl`: 전체 모델 결과

#### 성능 분석

- `outputs/detailed_model_performance.csv`: 상세 성능 지표
- `outputs/model_adoption_info.csv`: 최고 모델 선택 정보

#### 시각화

- `outputs/model_comparison.png`: 모델 성능 비교
- `outputs/confusion_matrix.png`: 혼동 행렬
- `outputs/roc_curve.png`: ROC 곡선
- `outputs/feature_importance.png`: 피처 중요도 (상위 30개)
- `outputs/shap_summary.png`: SHAP 요약 플롯 (상위 20개)
- `outputs/shap_bar.png`: SHAP 바 플롯 (상위 20개)

#### 예측 결과

- `outputs/23_24_transfer_predictions.csv`: 23/24 시즌 예측 결과
- `outputs/prediction_distribution.png`: 예측 분포

## ⚠️ 오버피팅 분석 결과

현재 모델들은 심각한 오버피팅을 보이고 있습니다:

- **Random Forest**: 훈련 점수 1.0000, 검증 점수 0.0475 (차이: 0.9525)
- **Logistic Regression**: 훈련 점수 0.4715, 검증 점수 0.2353 (차이: 0.2362)
- **Gradient Boosting**: 훈련 점수 0.4462, 검증 점수 0.0328 (차이: 0.4134)

## 🎯 성능 향상 전략

### 1. 하이퍼파라미터 튜닝

```python
# Random Forest
n_estimators: [50, 100, 200, 300]
max_depth: [5, 10, 15, None]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]

# Gradient Boosting
n_estimators: [100, 200, 300]
learning_rate: [0.01, 0.1, 0.2]
max_depth: [3, 5, 7]
subsample: [0.8, 0.9, 1.0]

# LightGBM
n_estimators: [100, 200, 300]
learning_rate: [0.01, 0.1, 0.2]
num_leaves: [31, 50, 100]
max_depth: [5, 10, 15]
```

### 2. 정규화 강화

- **L1/L2 정규화**: Logistic Regression, SVM
- **Early Stopping**: Gradient Boosting, LightGBM
- **Dropout**: 신경망 모델 (추가 구현 필요)

### 3. 데이터 증강

- **SMOTE**: 소수 클래스 오버샘플링
- **더 많은 데이터 수집**: 추가 시즌 데이터
- **피처 선택**: 상관관계가 높은 피처 제거

### 4. 앙상블 방법

- **Voting Classifier**: 여러 모델의 투표
- **Stacking**: 메타 모델을 통한 예측
- **Bagging**: 배깅을 통한 안정성 향상

## 📊 현재 예측 결과

- **총 선수**: 1,442명
- **예측 이적**: 183명 (12.7%)
- **고위험 선수 (60%+)**: 109명 (7.6%)

## 🔧 기술 스택

- **Python**: 3.9+
- **Pandas**: 데이터 처리
- **Scikit-learn**: 머신러닝
- **XGBoost**: 그래디언트 부스팅
- **LightGBM**: 그래디언트 부스팅
- **SHAP**: 모델 해석
- **Matplotlib/Seaborn**: 시각화

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🤝 기여

프로젝트 개선을 위한 제안이나 버그 리포트는 언제든 환영합니다.

---

**최종 업데이트**: 2025년 9월 5일
