# E-Commerce 데이터셋 EDA 및 이탈 예측 모델링

## 📊 프로젝트 개요

이 프로젝트는 E-Commerce 고객 데이터에 대한 탐색적 데이터 분석(EDA)을 수행하고, 고객 이탈(Churn) 예측을 위한 머신러닝 모델을 개발합니다. 모듈화된 구조로 설계되어 재사용 가능하고 확장 가능한 분석 파이프라인을 제공합니다.

## 📁 프로젝트 구조

```
SKN18-2nd-4Team/
├── 📊 E Commerce Dataset.xlsx          # 원본 데이터
├── 📁 src/                             # 모듈화된 소스 코드
│   ├── 📁 data/                        # 데이터 로딩 및 탐색
│   │   ├── data_loader.py              # 데이터 로딩 클래스
│   │   └── data_explorer.py            # 데이터 탐색 클래스
│   ├── 📁 visualization/               # 시각화 모듈
│   │   └── plotter.py                  # 데이터 시각화 클래스
│   ├── 📁 features/                    # 피쳐 엔지니어링 (향후 확장)
│   ├── 📁 models/                      # 머신러닝 모델 (향후 확장)
│   └── 📁 utils/                       # 유틸리티 함수 (향후 확장)
├── 📁 notes/                           # 쥬피터 노트북
│   ├── 01_데이터_로딩_및_기본_EDA.ipynb    # 기본 EDA
│   ├── 02_상세_피쳐_분석.ipynb             # 상세 피쳐 분석
│   └── 03_이탈_예측_모델링.ipynb          # 모델링
├── 📄 eda_analysis.py                  # 기본 EDA 스크립트
├── 📄 detailed_eda.py                  # 상세 EDA 스크립트
├── 📄 requirements.txt                 # Python 패키지
├── 📄 README.md                        # 프로젝트 설명서
└── 📊 시각화 파일들
    ├── eda_visualization.png           # 기본 시각화
    ├── correlation_heatmap.png         # 상관관계 히트맵
    └── feature_distributions.png       # 피쳐 분포
```

## 🔍 데이터 개요

- **데이터 크기**: 5,630행 × 20열
- **시트 구성**:
  - `E Comm`: 실제 고객 데이터
  - `Data Dict`: 데이터 딕셔너리

## 📋 주요 피쳐 분석

### 기본 정보

- **CustomerID**: 고객 고유 ID (5,630개 unique 값)
- **Churn**: 이탈 여부 (0: 유지, 1: 이탈)
- **Tenure**: 고객 유지 기간 (개월)

### 인구통계학적 특성

- **Gender**: 성별 (Male: 60.1%, Female: 39.9%)
- **MaritalStatus**: 결혼상태 (Married: 53.0%, Single: 31.9%, Divorced: 15.1%)
- **CityTier**: 도시 등급 (Tier 1: 65.1%, Tier 3: 30.6%, Tier 2: 4.3%)

### 행동 특성

- **PreferredLoginDevice**: 선호 로그인 기기
  - Mobile Phone: 49.1%
  - Computer: 29.0%
  - Phone: 21.9%
- **PreferredPaymentMode**: 선호 결제 방식
  - Debit Card: 41.1%
  - Credit Card: 26.7%
  - E wallet: 10.9%
- **PreferedOrderCat**: 선호 주문 카테고리
  - Laptop & Accessory: 36.4%
  - Mobile Phone: 22.6%
  - Fashion: 14.7%

### 서비스 이용 특성

- **HourSpendOnApp**: 앱 사용 시간 (평균 2.9시간)
- **SatisfactionScore**: 만족도 점수 (1-5점, 평균 3.1점)
- **OrderCount**: 주문 횟수 (평균 3.0회)
- **CashbackAmount**: 캐시백 금액 (평균 177.2원)

## 🎯 주요 발견사항

### 이탈률 분석

- **전체 이탈률**: 16.8% (948명)
- **성별별 이탈률**:
  - Male: 17.7%
  - Female: 15.5%
- **결혼상태별 이탈률**:
  - Single: 26.7% (가장 높음)
  - Divorced: 14.6%
  - Married: 11.5% (가장 낮음)
- **도시 등급별 이탈률**:
  - Tier 3: 21.4% (가장 높음)
  - Tier 2: 19.8%
  - Tier 1: 14.5% (가장 낮음)

### 상관관계 분석

Churn과 가장 높은 상관관계를 보이는 피쳐들:

1. **Tenure** (고객 유지 기간): -0.349 (음의 상관관계)
2. **DaySinceLastOrder** (마지막 주문 후 경과일): -0.161
3. **CashbackAmount** (캐시백 금액): -0.154
4. **NumberOfDeviceRegistered** (등록된 기기 수): 0.108
5. **SatisfactionScore** (만족도 점수): 0.105

### 고위험 고객 프로필

- 평균 만족도: 3.4점
- 평균 주문 횟수: 2.8회
- 평균 앱 사용 시간: 3.0시간
- 불만 제기 비율: 53.6%

### 충성 고객 프로필

- 평균 만족도: 3.0점
- 평균 주문 횟수: 3.0회
- 평균 앱 사용 시간: 2.9시간
- 불만 제기 비율: 23.4%

## 💡 비즈니스 인사이트

### 1. 이탈 예방 전략

- **신규 고객 관리**: Tenure가 낮은 고객의 이탈률이 높음
- **만족도 향상**: 만족도가 낮은 고객의 이탈 가능성이 높음
- **불만 해결**: 불만을 제기한 고객의 이탈률이 53.6%로 매우 높음

### 2. 타겟 마케팅

- **Single 고객**: 이탈률이 26.7%로 가장 높아 특별 관리 필요
- **Tier 3 도시 고객**: 이탈률이 21.4%로 높음
- **낮은 주문 빈도 고객**: 주문 횟수가 적은 고객의 이탈 가능성이 높음

### 3. 서비스 개선 포인트

- **앱 사용성**: 앱 사용 시간이 적은 고객의 이탈 가능성이 높음
- **캐시백 정책**: 캐시백 금액이 높을수록 이탈률이 낮음
- **배송 서비스**: WarehouseToHome 거리가 이탈과 양의 상관관계

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 쥬피터 노트북 사용 (권장)

```bash
# 쥬피터 노트북 실행
jupyter notebook

# 또는 쥬피터 랩 실행
jupyter lab
```

**노트북 실행 순서:**

1. `01_데이터_로딩_및_기본_EDA.ipynb` - 기본 데이터 탐색
2. `02_상세_피쳐_분석.ipynb` - 상세 피쳐 분석
3. `03_이탈_예측_모델링.ipynb` - 머신러닝 모델링

### 3. 스크립트 실행

```bash
# 기본 EDA 분석
python3 eda_analysis.py

# 상세 EDA 분석
python3 detailed_eda.py
```

### 4. 모듈화된 코드 사용

```python
# 데이터 로딩
from src.data import DataLoader, DataExplorer
data_loader = DataLoader()
df = data_loader.load_data()

# 데이터 탐색
explorer = DataExplorer(df)
feature_summary = explorer.get_feature_summary()

# 시각화
from src.visualization import DataPlotter
plotter = DataPlotter(df)
plotter.plot_feature_summary()
```

## 📊 생성된 시각화 파일

1. **eda_visualization.png**: 기본 데이터 시각화
2. **correlation_heatmap.png**: 피쳐 간 상관관계 히트맵
3. **feature_distributions.png**: 주요 피쳐 분포 분석

## 🔧 기술 스택

- **Python 3.13**
- **pandas**: 데이터 처리
- **numpy**: 수치 계산
- **matplotlib**: 기본 시각화
- **seaborn**: 고급 시각화
- **scikit-learn**: 머신러닝
- **jupyter**: 노트북 환경
- **plotly**: 인터랙티브 시각화
- **openpyxl**: Excel 파일 읽기

## 🏗️ 모듈 구조

### DataLoader 클래스

- Excel 파일 로딩
- 시트 정보 조회
- 데이터 딕셔너리 로딩
- 기본 정보 제공

### DataExplorer 클래스

- 피쳐 요약 정보
- Unique 값 분석
- 수치형/범주형 통계
- 상관관계 분석
- 이탈 패턴 분석

### DataPlotter 클래스

- 피쳐 요약 시각화
- 상관관계 히트맵
- 이탈 분석 시각화
- 피쳐 분포 시각화

## 📈 다음 단계

1. **머신러닝 모델링**: 이탈 예측 모델 개발
2. **피쳐 엔지니어링**: 새로운 피쳐 생성
3. **모델 최적화**: 하이퍼파라미터 튜닝
4. **A/B 테스트**: 이탈 예방 전략 검증
5. **실시간 예측**: 프로덕션 환경 배포
