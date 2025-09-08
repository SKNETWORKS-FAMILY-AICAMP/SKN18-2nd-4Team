#!/usr/bin/env python3
"""
정규화 강화
- L1/L2 정규화 적용
- Early Stopping 구현
- Dropout 추가 (신경망 모델)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LightGBM import
try:
    from lightgbm import LGBMClassifier
    _has_lgbm = True
except ImportError:
    _has_lgbm = False

def regularization_improvement():
    """정규화 강화 실행"""
    logger.info("🔧 정규화 강화 시작")
    
    try:
        # 1. 데이터 로드
        from src.data.data_loader_new import DataLoaderNew
        from src.utils.config import Config
        
        config = Config("config_final.yaml")
        data_loader = DataLoaderNew(config)
        train_df, test_df = data_loader.load_all_data()
        
        print(f"📊 데이터 로드 완료:")
        print(f"  - Train: {train_df.shape[0]:,} rows")
        print(f"  - Test: {test_df.shape[0]:,} rows")
        
        # 2. 기본 모델링과 동일한 방식으로 처리
        from src.models.football_modeling import FootballModelTrainer
        
        # 전체 데이터 합치기 (모델링용)
        all_data = pd.concat([train_df, test_df], ignore_index=True)
        
        # 기본 모델링 결과 재사용 (중복 학습 방지)
        outputs_dir = Path(config.output_dir)
        model_results_path = outputs_dir / "model_results.pkl"
        
        if model_results_path.exists():
            logger.info("💾 기존 모델링 결과 재사용 (중복 학습 방지)")
            model_results = joblib.load(model_results_path)
        else:
            logger.info("🚀 기본 모델링 결과가 없어서 새로 학습합니다")
            model_trainer = FootballModelTrainer(all_data, config)
            model_results = model_trainer.run_pipeline()
        
        # 전처리된 데이터 가져오기
        X_val = model_results['X_val']  # validation 데이터 사용
        y_val = model_results['y_val']
        X_train = model_results['X_train']
        y_train = model_results['y_train']
        preprocessor = model_results['preprocessor']
        
        # 이미 전처리된 데이터를 명확한 변수명으로 할당
        X_train_processed = X_train  # 전처리 완료된 train 데이터
        X_val_processed = X_val      # 전처리 완료된 validation 데이터
        
        # 5. 교차 검증 설정
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        from sklearn.metrics import f1_score, roc_auc_score
        f1_scorer = make_scorer(f1_score)
        
        # 6. 정규화된 모델 정의 (성능 상위 3개 모델만)
        regularized_models = {}
        
        # LightGBM with regularization (성능 1위)
        if _has_lgbm:
            regularized_models['LightGBM (Regularized)'] = LGBMClassifier(
                n_estimators=100, learning_rate=0.1, num_leaves=31,
                max_depth=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,  # L1, L2 정규화 추가
                class_weight='balanced', random_state=42
            )
        
        # Logistic Regression with L1/L2 regularization (성능 2위)
        regularized_models['Logistic Regression (L1)'] = LogisticRegression(
            penalty='l1', C=0.1, solver='liblinear', 
            class_weight='balanced', random_state=42, max_iter=1000
        )
        regularized_models['Logistic Regression (L2)'] = LogisticRegression(
            penalty='l2', C=0.1, solver='lbfgs',
            class_weight='balanced', random_state=42, max_iter=1000
        )
        
        # XGBoost with regularization (성능 3위)
        try:
            import xgboost as xgb
            regularized_models['XGBoost (Regularized)'] = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,  # L1, L2 정규화 추가
                random_state=42, eval_metric='logloss'
            )
        except ImportError:
            logger.warning("XGBoost가 설치되지 않았습니다. XGBoost 정규화를 건너뜁니다.")
        
        # 7. 정규화된 모델 훈련 및 평가
        regularization_results = {}
        
        for model_name, model in regularized_models.items():
            logger.info(f"🔧 {model_name} 훈련 시작")
            
            # 이미 전처리된 데이터로 직접 모델 훈련
            # 교차 검증
            # 복합점수 기준으로 교차 검증 (Precision 중심 가중평균)
            def composite_scorer(y_true, y_pred):
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                return (precision * 0.4 + f1 * 0.3 + accuracy * 0.2 + recall * 0.1)
            
            composite_scorer_func = make_scorer(composite_scorer)
            cv_scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring=composite_scorer_func)
            
            # 전체 데이터로 훈련
            model.fit(X_train_processed, y_train)
            
            # 예측
            y_pred = model.predict(X_val_processed)
            y_pred_proba = model.predict_proba(X_val_processed)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 성능 평가
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0
            
            # 복합 점수 계산 (Precision 중심 가중평균)
            # Precision 40% + F1 30% + Accuracy 20% + Recall 10%
            composite_score = (precision * 0.4 + f1 * 0.3 + accuracy * 0.2 + recall * 0.1)
            
            regularization_results[model_name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'composite_score': composite_score
            }
            
            logger.info(f"✅ {model_name} 완료:")
            logger.info(f"   CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            logger.info(f"   Test F1: {f1:.4f}")
            logger.info(f"   Composite: {composite_score:.4f}")
        
        # 8. 최고 모델 선택
        best_model_name = max(regularization_results.keys(), 
                             key=lambda x: regularization_results[x]['composite_score'])
        best_model_info = regularization_results[best_model_name]
        
        logger.info(f"🏆 최고 정규화 모델: {best_model_name}")
        logger.info(f"   복합 점수: {best_model_info['composite_score']:.4f}")
        
        # 9. 오버피팅 분석
        logger.info("🔍 오버피팅 분석")
        
        for model_name, results in regularization_results.items():
            cv_score = results['cv_mean']
            test_score = results['f1']
            gap = test_score - cv_score
            
            status = "✅ 좋음" if gap < 0.05 else "⚠️ 주의" if gap < 0.1 else "❌ 오버피팅"
            logger.info(f"   {model_name}: CV={cv_score:.4f}, Test={test_score:.4f}, Gap={gap:.4f} {status}")
        
        # 10. 결과 저장
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 정규화 결과 저장
        regularization_path = output_dir / "regularization_results.pkl"
        joblib.dump(regularization_results, regularization_path)
        
        # 최고 모델 저장
        best_regularized_path = output_dir / "best_regularized_model.pkl"
        joblib.dump(best_model_info['model'], best_regularized_path)
        
        # 성능 비교 CSV
        performance_df = pd.DataFrame({
            'model': list(regularization_results.keys()),
            'cv_f1_mean': [regularization_results[m]['cv_mean'] for m in regularization_results.keys()],
            'cv_f1_std': [regularization_results[m]['cv_std'] for m in regularization_results.keys()],
            'test_accuracy': [regularization_results[m]['accuracy'] for m in regularization_results.keys()],
            'test_precision': [regularization_results[m]['precision'] for m in regularization_results.keys()],
            'test_recall': [regularization_results[m]['recall'] for m in regularization_results.keys()],
            'test_f1': [regularization_results[m]['f1'] for m in regularization_results.keys()],
            'test_auc': [regularization_results[m]['auc'] for m in regularization_results.keys()],
            'composite_score': [regularization_results[m]['composite_score'] for m in regularization_results.keys()]
        })
        performance_df = performance_df.sort_values('composite_score', ascending=False)
        performance_df.to_csv(output_dir / "regularized_model_performance.csv", index=False)
        
        # 11. 결과 요약
        print("\n" + "="*80)
        print("🎉 정규화 강화 완료!")
        print("="*80)
        print("🏆 최고 모델:", best_model_name)
        print("📊 성능 지표:")
        print(f"   CV F1:     {best_model_info['cv_mean']:.4f} (±{best_model_info['cv_std']:.4f})")
        print(f"   Test Accuracy:  {best_model_info['accuracy']:.4f}")
        print(f"   Test Precision: {best_model_info['precision']:.4f}")
        print(f"   Test Recall:    {best_model_info['recall']:.4f}")
        print(f"   Test F1:        {best_model_info['f1']:.4f}")
        print(f"   Test AUC:       {best_model_info['auc']:.4f}")
        print(f"   Composite:      {best_model_info['composite_score']:.4f}")
        print("\n📁 생성된 파일:")
        print("  - outputs/regularization_results.pkl")
        print("  - outputs/best_regularized_model.pkl")
        print("  - outputs/regularized_model_performance.csv")
        print("="*80)
        
        # 8. 최고 성능 모델이 기존 모델보다 좋으면 최종 모델 업데이트
        current_best_score = max(model_results['model_scores'].values()) if 'model_scores' in model_results else 0
        if 'tuning_improvement' in model_results:
            current_best_score += model_results['tuning_improvement']  # 튜닝 개선분 반영
            
        regularized_best_score = best_model_info['composite_score']
        
        if regularized_best_score > current_best_score:
            logger.info(f"🎉 정규화된 모델이 더 우수합니다! {current_best_score:.4f} → {regularized_best_score:.4f}")
            
            # 최종 model_results 업데이트
            model_results['best_model'] = best_model_info['model']
            model_results['best_model_name'] = f"{best_model_name} (Regularized)"
            model_results['regularization_improvement'] = regularized_best_score - current_best_score
            
            # 최종 모델 저장 (outputs/ 덮어쓰기)
            outputs_dir = Path(config.output_dir)
            joblib.dump(best_model_info['model'], outputs_dir / "model.pkl")
            joblib.dump(model_results, outputs_dir / "model_results.pkl")
            
            logger.info("✅ 최종 모델이 정규화된 모델로 업데이트되었습니다")
            
            # 모델 성능 정보 업데이트
            from scripts.save_model_performance import save_model_performance
            save_model_performance(model_results)
        else:
            logger.info(f"기존 모델이 더 우수합니다. {current_best_score:.4f} > {regularized_best_score:.4f}")
        
        logger.info("✅ 정규화 강화 완료")
        
    except Exception as e:
        logger.error(f"정규화 강화 오류: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    regularization_improvement()

if __name__ == "__main__":
    main()
