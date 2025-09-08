# 사용법 가이드

## 📁 프로젝트 구조

### 루트 디렉토리 (메인 파일들)

- `main.py` - **메인 실행 파일**
- `config_final.yaml` - 설정 파일
- `README.md` - 프로젝트 문서
- `USAGE.md` - 사용법 가이드

### scripts/ 디렉토리 (개별 실행 스크립트들)

- `run_final_modeling.py` - 기본 모델링
- `hyperparameter_tuning.py` - 하이퍼파라미터 튜닝
- `regularization_improvement.py` - 정규화 강화
- `ensemble_modeling.py` - 앙상블 모델
- `save_model_performance.py` - 모델 성능 상세 저장

### src/ 디렉토리 (함수/클래스들)

- `src/features/` - 피처 엔지니어링 함수들
- `src/models/` - 모델링 클래스들
- `src/data/` - 데이터 로딩 함수들
- `src/visualization/` - 시각화 함수들
- `src/utils/` - 유틸리티 함수들

## 🚀 실행 방법

### 1. 기본 사용법 (추천)

```bash
# 전체 파이프라인 실행 (기본 모델링 → 고급 기법)
python main.py --mode all

# 개별 실행
python main.py --mode train                   # 기본 모델링 (기존 개선된 모델 재사용)
python main.py --mode train --force-retrain   # 강제 재학습 (개선된 모델 무시)
python main.py --mode tune                    # 하이퍼파라미터 튜닝
python main.py --mode regularize              # 정규화 강화
python main.py --mode ensemble                # 앙상블 모델
```

### 2. 단계별 실행

```bash
# 1단계: 기본 모델링 (피처 엔지니어링 자동 적용)
python main.py --mode train

# 2단계: 고급 모델링 (선택사항)
python main.py --mode tune        # 하이퍼파라미터 튜닝
python main.py --mode regularize  # 정규화 강화
python main.py --mode ensemble    # 앙상블 모델
```

### 3. 개별 파일 실행 (고급 사용자)

```bash
# scripts/ 폴더에서 개별 실행
python scripts/run_final_modeling.py
python scripts/hyperparameter_tuning.py
python scripts/regularization_improvement.py
python scripts/ensemble_modeling.py
python scripts/save_model_performance.py
```

## 📊 출력 파일

### 기본 출력

- `outputs/24_25_transfer_predictions.csv` - 예측 결과
- `outputs/model.pkl` - 최종 모델
- `outputs/preprocessor.pkl` - 전처리기

### 시각화

- `outputs/model_comparison.png` - 모델 성능 비교
- `outputs/confusion_matrix.png` - 혼동 행렬
- `outputs/roc_curve.png` - ROC 곡선
- `outputs/feature_importance.png` - 피처 중요도 (상위 30개)
- `outputs/shap_summary.png` - SHAP 분석 (가로 확장, 상위 20개)
- `outputs/shap_bar.png` - SHAP 바 플롯 (상위 20개)
- `outputs/prediction_distribution.png` - 예측 분포

### 성능 분석

- `outputs/detailed_model_performance.csv` - 상세 모델 성능 순위
- `outputs/model_performance_scores.csv` - 기본 성능 점수

### 고급 모델링 결과

- `outputs/tuned_model_performance.csv` - 튜닝된 모델 성능
- `outputs/regularized_model_performance.csv` - 정규화된 모델 성능
- `outputs/ensemble_model_performance.csv` - 앙상블 모델 성능

## 💡 권장 사용법

### 처음 사용자

```bash
python main.py --mode all
```

### 개발자/연구자

```bash
# 단계별 실행으로 각 단계 확인
python main.py --mode train
python main.py --mode tune
python main.py --mode regularize
```

### 프로덕션 환경

```bash
# 기본 모델링만 (가장 빠름)
python main.py --mode train
```

## 🔧 설정 변경

`config_final.yaml` 파일에서 설정을 변경할 수 있습니다:

- 데이터 경로
- 모델 파라미터
- 출력 경로
- 피처 설정

## ❓ 문제 해결

### 모듈 오류

```bash
# 프로젝트 루트에서 실행
cd /path/to/SKN18-2nd-4Team
python main.py --mode all
```

### 성능 향상

```bash
# 현재 기본 모델링에 8개 모델 비교 포함
python main.py --mode train

# 고급 기법들 (현재 수정 중)
python main.py --mode tune        # 하이퍼파라미터 튜닝 (수정 필요)
python main.py --mode regularize  # 정규화 강화 (수정 필요)
python main.py --mode ensemble    # 앙상블 모델링 (수정 필요)
```

## 💡 **모델 성능 향상을 위한 반복 실행**

고급 기법들로 모델을 개선한 후, 최종 결과(SHAP, 확률값, 예측)를 확인하려면:

```bash
# 1단계: 고급 기법들 실행 (순서 무관, 각각 독립적)
python main.py --mode tune        # 하이퍼파라미터 최적화
python main.py --mode regularize  # 정규화 강화로 오버피팅 완화
python main.py --mode ensemble    # 앙상블 모델로 성능 향상

# 2단계: 개선된 결과 확인
python main.py --mode train       # 향상된 모델의 SHAP 분석 및 24/25 예측 (개선된 모델 재사용)
```

> **스마트 모델 관리**:
>
> - `train` 모드는 **기존 개선된 모델을 자동으로 재사용**합니다
> - 강제로 처음부터 학습하려면 `--force-retrain` 옵션 사용
> - 각 고급 기법은 더 좋은 성능일 때만 최종 모델을 업데이트합니다
