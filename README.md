# Football Transfer Prediction Project

## 📋 프로젝트 개요

- [ ]

## 🎯 주요 목표

- **이적 예측**: 23/24 시즌 선수들의 이적 가능성 예측
- **모델 비교**: 8개 머신러닝 모델의 성능 비교 및 최적 모델 선택
- **피처 엔지니어링**: 13개의 고급 피처를 통한 예측 성능 향상
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

---

## 🔧 피처 엔지니어링 및 전처리 파이프라인

### 📈 피처 엔지니어링 프로세스

**1단계: 기본 피처 (27개)**

- 선수 기본 정보: `age_at_season`, `height_in_cm`, `position`, `sub_position`
- 성과 지표: `goals`, `assists`, `yellow_cards`, `red_cards` (3배 가중치), `season_avg_minutes`
- 시장 가치: `player_market_value_in_eur`, `log_market_value`
- 클럽 정보: `club_squad_size`, `club_average_age`, `club_foreigners_percentage`

**2단계: 고급 피처 생성 (13개 추가)**

1. **`season_start_year`**: 시즌을 연도로 변환 (예: '21/22' → 2021)
2. **`age_at_season`**: 시즌 시작 시점 선수 나이
3. **`log_market_value`**: 시장가치 로그 변환 (log1p, €25,000~€180,000,000 → 10.1~19.0, 왜도 2.716→-0.439)
4. **`is_foreigner`**: 외국인 선수 여부 (England 기준)
5. **`minutes_vs_club_avg`**: 선수 출전시간 / 클럽 평균 출전시간
6. **`age_difference`**: 선수 나이 - 클럽 평균 나이
7. **`attack_contribution`**: (골 + 어시스트) × 시즌 승수
8. **`height_vs_position`**: 선수 키 / 포지션별 평균 키
9. **`weighted_cards`**: 옐로카드 + (레드카드 × 3) - **가중치 적용**
10. **`cards_per_minute`**: 가중 카드 점수 / 출전시간
11. **`discipline_level`**: 징계 수준 (0: 없음, 1: 경고만, 2: 퇴장 포함)
12. **`club_tenure_seasons`**: 클럽 재적 기간 (시즌 수)
13. **`position_competition`**: 포지션별 경쟁 선수 수

### 🛠️ 데이터 전처리 파이프라인

**3단계 전처리 프로세스:**

1. **피처 엔지니어링**: `create_engineered_features(df)`

   - 27개 기본 피처 → 40개 피처로 확장
   - 13개 고급 피처 생성 (시즌 연도, 나이, 로그 변환, 비율 등)

2. **피처 타입 분류**: `get_feature_types(df_processed)`

   - 자동으로 수치형/명목형 분류
   - ID/메타데이터 7개 컬럼 제외 후 33개 모델링 피처 선별

3. **전처리기 생성**: `create_preprocessor(feature_types)`

   - ColumnTransformer로 피처 타입별 파이프라인 구성
   - 수치형(28개) + 명목형(5개) = 2개 파이프라인 생성
   - 각 파이프라인은 결측치 처리 → 변환 순서로 2단계 구성

**전처리기 구성 (ColumnTransformer)**

1. **수치형 피처 (28개)**:

   - 포함: `goals`, `assists`, `age_at_season`, `log_market_value`, `minutes_vs_club_avg` 등

   ```python
   Pipeline([
       ('imputer', SimpleImputer(strategy='median')),    # 중앙값으로 결측치 대체
       ('scaler', StandardScaler(with_mean=True, with_std=True))  # 평균0, 분산1 정규화
   ])
   ```

   **효과**: DataFrame → 중앙값 결측치 처리 → 표준화 → numpy array

2. **명목형 피처 (5개)**: `['club_name', 'country_of_birth', 'foot', 'position', 'sub_position']`

   ```python
   Pipeline([
       ('imputer', SimpleImputer(strategy='most_frequent')),      # 최빈값으로 결측치 대체
       ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # 원핫 인코딩
   ])
   ```

   **효과**: DataFrame → 최빈값 결측치 처리 → 더미변수 생성 → numpy array

**🔄 전처리 결과**

- **입력**: 33개 피처 DataFrame (수치형 28개 + 명목형 5개)
- **출력**: 통일된 numpy array (모든 피처가 수치형으로 변환)
- **차원 변화**: 명목형 원핫 인코딩으로 인해 최종 피처 수 증가
- **데이터 타입**: 모든 값이 float64로 통일되어 ML 모델 입력 준비 완료

**⚠️ 데이터 누수 방지**

- 전처리기는 **train 데이터로만 학습** (`preprocessor.fit(X_train)`)
- validation 및 23/24 예측 데이터는 **transform만 적용**
- 결측치 통계(중앙값, 최빈값), 표준화 통계(평균, 표준편차) 모두 train 데이터 기준

---

## 📊 데이터 분할 전략

| 시계열 기반 분할 구조                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 데이터 분할 플로우차트                                                                                   |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **전체 데이터 구조:**`<br><br>`전체 데이터 (12/13-23/24 시즌)`<br>`├── Train Pool (12/13-22/23)`<br>`│ ├── Train (12/13-21/22) ← 모델 학습 + 전처리기 학습 `<br>`│ └── Validation (22/23) ← 성능 평가 + 하이퍼파라미터 튜닝 `<br>`└── Prediction (23/24) ← 최종 예측 대상 (답 없음)`<br><br>`**핵심 원칙:**`<br>`• **시간적 순서 준수**: 과거 → 현재 → 미래 순서 유지 `<br>`• **22/23 시즌**: Validation으로 사용 (모델 성능 평가)`<br>`• **23/24 시즌**: 실제 예측 대상 (target 없는 미래 데이터)`<br>`• **데이터 누수 방지**: 전처리기는 train 데이터로만 학습 | ![데이터 분할 전략](assets/images/Data_Split.png) `<br>`_시계열 데이터의 시간적 순서를 고려한 분할 전략_ |

- **시계열 순서 유지**: 과거 데이터로 미래 예측

**데이터 크기:**

- Train: 11,951 rows (11-21 시즌)
- Validation: 1,480 rows (22/23 시즌)
- Prediction: 2,884 rows (23/24 시즌) → 중복 제거 후 570 rows

**Prediction이 Validation보다 큰 이유:**

- 23/24 시즌 데이터에는 **동일 선수의 여러 기록**이 포함 (시장가치 변동, 클럽 이적 등)
- 최종 예측 시 `player_name` 기준으로 **중복 제거** 후 최고 확률만 유지

---

## 🤖 모델 성능

### 모델 순위 (기본 성능)

1. 🥇 **SVM**: 0.4695 (Precision 중심 복합점수)
2. 🥈 **Logistic Regression**: 0.4651 (Precision 중심 복합점수)
3. 🥉 **LightGBM**: 0.4626 (Precision 중심 복합점수)
4. Gradient Boosting: 0.4124
5. Random Forest: 0.3951
6. XGBoost: 0.3873
7. KNN: 0.3846
8. Decision Tree: 0.3646

### 최고 성능 모델: SVM

- **복합 점수**: 0.4695 (Precision 중심 가중평균)
- **선택 기준**: 복합 점수 (Precision 중심 가중평균)
- **가중 공식**: Precision×0.4 + F1×0.3 + Accuracy×0.2 + Recall×0.1

**💡 Precision 중심 가중공식 채택 이유:**

- **Precision 개선 목표**: Confusion Matrix 분석 결과 Precision이 낮아 개선 필요
- **실용적 중요도**: 이적 예측에서 False Positive 최소화가 중요
- **비즈니스 요구사항 반영**: 이적 예측에서는 정확도, 정밀도, 재현율이 모두 중요
- **클래스 불균형 고려**: AUC는 불균형 데이터에서 안정적인 성능 지표 제공
- **동등한 가중치**: 모든 지표를 20%씩 동일하게 반영하여 편향 방지
- **하이퍼파라미터 튜닝 일관성**: 모든 고급 기법에서 동일한 평가 기준 사용

---

## 📈 평가 지표

### 복합 점수 (Composite Score)

```
복합 점수 = Precision×0.4 + F1×0.3 + Accuracy×0.2 + Recall×0.1
```

- **Precision (40%)**: 예측한 이적 중 실제 이적 비율 (가장 중요)
- **F1-Score (30%)**: Precision과 Recall의 조화평균 (균형 지표)
- **Accuracy (20%)**: 전체 예측 정확도
- **Recall (10%)**: 실제 이적 중 예측한 이적 비율

**💡 가중치 설정 근거:**

- **Precision 최우선**: False Positive 예측률이 낮아, 값 최소화를 위하여 가중치를 높힘
- **F1-Score 중시**: 균형잡힌 성능 평가
- **GridSearchCV 일관성**: 모든 튜닝 과정에서 동일한 기준 적용

---

## 🎯 성능 향상 전략

**기본 모델 → 고급 기법 적용:**

1. **하이퍼파라미터 튜닝**: GridSearchCV로 최적 파라미터 탐색
2. **정규화 강화**: L1/L2 정규화, Early Stopping으로 오버피팅 완화
3. **앙상블 모델**: Voting, Stacking, Bagging으로 성능 향상
4. **자동 선택**: 각 단계에서 더 좋은 모델만 최종 채택

---

## 📊 성능 향상 기록

### 🏁 기준 성능 (Baseline)

**기본 모델링 결과 (튜닝 전):**

| 순위 | 모델                    | 복합 점수  | Accuracy | Precision | Recall | F1-Score | AUC    |
| ---- | ----------------------- | ---------- | -------- | --------- | ------ | -------- | ------ |
| 🥇   | **SVM**                 | **0.4695** | 0.5649   | 0.3688    | 0.6659 | 0.4747   | 0.6139 |
| 🥈   | **Logistic Regression** | **0.4651** | 0.4736   | 0.3434    | 0.8581 | 0.4905   | 0.6365 |
| 🥉   | **LightGBM**            | **0.4626** | 0.6020   | 0.3801    | 0.5515 | 0.4500   | 0.6238 |

### 📈 성능 향상 추적

#### **1️⃣ 하이퍼파라미터 튜닝 결과**

| 모델                | 튜닝 전 | 튜닝 후    | 향상도      | 최적 파라미터                                                        |
| ------------------- | ------- | ---------- | ----------- | -------------------------------------------------------------------- |
| Logistic Regression | 0.5604  | **0.5655** | **+0.0051** | {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}... (이전 최고값 유지) |
| SVM                 | 0.5376  | **0.5574** | **+0.0198** | {'C': 0.005, 'kernel': 'linear'}...                                  |
| LightGBM            | 0.5215  | **0.5056** | **-0.0159** | {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5}...       |

#### **2️⃣ 정규화 개선 결과**

| 모델                     | 정규화 전 | 정규화 후 | 향상도  | 적용 기법             |
| ------------------------ | --------- | --------- | ------- | --------------------- |
| Logistic Regression (L1) | **TBD**   | **TBD**   | **TBD** | L1 Regularization     |
| Logistic Regression (L2) | **TBD**   | **TBD**   | **TBD** | L2 Regularization     |
| SVM (RBF)                | **TBD**   | **TBD**   | **TBD** | C Parameter Tuning    |
| LightGBM (Regularized)   | **TBD**   | **TBD**   | **TBD** | reg_alpha, reg_lambda |

#### **3️⃣ 앙상블 모델링 결과**

| 앙상블 방법   | 개별 모델 최고점수 | 앙상블 점수 | 향상도  | 구성 모델 |
| ------------- | ------------------ | ----------- | ------- | --------- |
| Voting (Soft) | **TBD**            | **TBD**     | **TBD** | **TBD**   |
| Voting (Hard) | **TBD**            | **TBD**     | **TBD** | **TBD**   |
| Stacking      | **TBD**            | **TBD**     | **TBD** | **TBD**   |
| Bagging       | **TBD**            | **TBD**     | **TBD** | **TBD**   |

### 🏆 최종 성능 요약

| 단계                    | 최고 모델 | 복합 점수 | 전 단계 대비 향상 | 누적 향상도 |
| ----------------------- | --------- | --------- | ----------------- | ----------- |
| **기준선**              | SVM       | 0.4695    | -                 | -           |
| **하이퍼파라미터 튜닝** | **TBD**   | **TBD**   | **TBD**           | **TBD**     |
| **정규화 개선**         | **TBD**   | **TBD**   | **TBD**           | **TBD**     |
| **앙상블 모델링**       | **TBD**   | **TBD**   | **TBD**           | **TBD**     |

> **📝 수동 업데이트**: 성능 기록은 실험 완료 후 수동으로 업데이트됩니다

---

## 📊 현재 예측 결과

- **총 선수**: 1,442명
- **예측 이적**: 183명 (12.7%)
- **고위험 선수 (60%+)**: 109명 (7.6%)

---

## 📁 프로젝트 구조

### 핵심 모듈

#### `src/features/feature_engineering.py`

```
└── FootballFeatureEngineer        # 피처 엔지니어링 클래스
    ├── 🔧 기본 피처 정제 (27개 → 40개)
    ├── 📊 13개 고급 피처 생성
    │   ├── weighted_cards (Red Card 3배 가중치)
    │   ├── discipline_level (징계 심각도: 0/1/2)
    │   ├── cards_per_minute (분당 카드 수)
    │   ├── club_tenure_seasons (클럽 재적 기간)
    │   ├── position_competition (포지션 경쟁도)
    │   ├── season_start_year (시즌 시작 연도)
    │   ├── age_at_season (시즌 시작 나이)
    │   ├── log_market_value (로그 변환 시장가치)
    │   ├── is_foreigner (외국인 여부)
    │   ├── minutes_vs_club_avg (클럽 평균 대비 출전시간)
    │   ├── age_difference (나이 차이)
    │   ├── attack_contribution (공격 기여도)
    │   └── height_vs_position (포지션별 키 비교)
    ├── 🗂️ 피처 타입 자동 분류 (수치형/명목형)
    └── ⚙️ 전처리 파이프라인 생성
```

**핵심 전처리 파이프라인:**

- **수치형 (28개)**: 중앙값 결측치 처리 → 표준화
- **명목형 (5개)**: 최빈값 결측치 처리 → 원핫 인코딩

#### `src/models/football_modeling.py`

```
└── FootballModelTrainer        # 통합 모델링 파이프라인
    ├── 📊 데이터 품질 및 누수 검사 (DataLeakageChecker)
    ├── 🔧 자동 피처 엔지니어링 적용 (FootballFeatureEngineer)
    ├── 🤖 8개 모델 비교 (Logistic, RF, GB, SVM, KNN, DT, XGB, LGBM)
    ├── 📈 복합 점수 기반 모델 선택
    ├── ⚠️ 오버피팅 검사 (OverfittingChecker)
    ├── 🔍 SHAP 분석 (피처 중요도 해석)
    └── 📋 23/24 시즌 예측 생성
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

### 정리된 프로젝트 구조

```
SKN18-2nd-4Team/
├── main.py                     # 🎯 중앙 실행 파일
├── data/
│   ├── curated/                # 정제된 데이터
│   │   ├── train_df.csv        # 훈련 데이터 (12/13-22/23)
│   │   └── test_df.csv         # 테스트 데이터 (23/24)
│   └── raw/                    # 원본 데이터 (9개 CSV 파일)
├── src/                        # 핵심 모듈
│   ├── features/
│   │   └── feature_engineering.py  # 피처 엔지니어링 + 전처리
│   ├── models/
│   │   └── football_modeling.py    # 모델 훈련 + 평가
│   ├── data/
│   │   └── data_loader_new.py       # 데이터 로딩
│   ├── utils/
│   │   └── config.py               # 설정 관리
│   └── visualization/
│       └── plotter.py              # 시각화 (ModelVisualizer)
├── scripts/                        # 실행 스크립트
│   ├── run_final_modeling.py       # 기본 모델링
│   ├── hyperparameter_tuning.py    # 하이퍼파라미터 튜닝
│   ├── regularization_improvement.py # 정규화 강화
│   ├── ensemble_modeling.py        # 앙상블 모델링
│   └── save_model_performance.py   # 성능 저장
└── outputs/                        # 결과 파일 (자동 생성)
    ├── *.png                       # 시각화 결과
    └── *.csv                       # 예측 결과 + 성능 지표
```

---

## 🚀 **전체 파이프라인 (`--mode all`) 실행 과정**

```
🚀 전체 파이프라인 실행 시작

1단계: 기본 모델링
├── 데이터 로딩 (train_df + test_df)
├── 피처 엔지니어링 (13개 새 피처 생성)
├── 8개 모델 학습 (LR, RF, XGB, LGBM, SVM, KNN, NB, DT)
├── 최고 성능 모델 선택 (가중 복합 점수 기준)
├── SHAP 분석 + 시각화
├── 23/24 시즌 예측 생성
└── 📁 outputs/ 저장 (기본 모델)

2단계: 하이퍼파라미터 튜닝
├── GridSearchCV로 최적 파라미터 탐색
├── 성능 비교: 기존 vs 튜닝
└── 더 좋으면 📁 outputs/ 업데이트 (튜닝 모델)

3단계: 정규화 강화
├── L1/L2 정규화, Early Stopping 적용
├── 성능 비교: 기존 vs 정규화
└── 더 좋으면 📁 outputs/ 업데이트 (정규화 모델)

4단계: 앙상블 모델
├── Voting, Stacking, Bagging 앙상블
├── 성능 비교: 기존 vs 앙상블
└── 더 좋으면 📁 outputs/ 업데이트 (앙상블 모델)

5단계: 최종 예측 결과 업데이트
├── 최종 최고 성능 모델로 SHAP 재생성
├── 23/24 예측 재실행 (개선된 모델 기반)
└── 📁 outputs/ 최종 저장

✅ 전체 파이프라인 완료!

```

## 🚀 실행 방법

```bash
# 전체 파이프라인 실행 (추천) - 최고 성능 모델 자동 선택
python main.py --mode all

# 개별 실행
python main.py --mode train        # 기본 모델링 (피처 엔지니어링 자동 적용)
python main.py --mode tune         # 하이퍼파라미터 튜닝
python main.py --mode regularize   # 정규화 강화 (오버피팅 완화)
python main.py --mode ensemble     # 앙상블 모델링

# 고급 옵션
python main.py --mode train --force-retrain   # 강제 재학습 (개선된 모델 무시)
```

### 출력 파일

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

## 🔧 기술 스택

### 핵심 라이브러리

- **Python**: 3.9+
- **Pandas**: 데이터 처리 및 조작
- **NumPy**: 수치 연산
- **Scikit-learn**: 머신러닝 파이프라인
  - `ColumnTransformer`: 피처 타입별 전처리
  - `SimpleImputer`: 결측치 처리 (중앙값/최빈값)
  - `StandardScaler`: 수치형 피처 정규화
  - `OneHotEncoder`: 범주형 피처 인코딩
  - `Pipeline`: 전처리-모델 파이프라인 구축

### 머신러닝 모델

- **Linear Models**: Logistic Regression, SVM
- **Tree-based**: Decision Tree, Random Forest
- **Boosting**: Gradient Boosting, XGBoost, LightGBM
- **Instance-based**: K-Nearest Neighbors

### 모델 해석 및 시각화

- **SHAP**: 모델 해석 (TreeExplainer, LinearExplainer, KernelExplainer)
- **Matplotlib**: 기본 시각화
- **Seaborn**: 통계 시각화

### 개발 도구

- **Joblib**: 모델 직렬화
- **Logging**: 실행 로그 관리
- **Pathlib**: 파일 경로 관리

---

**최종 업데이트**: 2025년 9월 6일
