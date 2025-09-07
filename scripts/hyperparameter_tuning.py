#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝
- 성능 상위 3개 모델 최적화: Logistic Regression, SVM, LightGBM
- GridSearchCV를 통한 체계적 탐색
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LightGBM import
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    _has_lgbm = True
except ImportError:
    _has_lgbm = False

def hyperparameter_tuning():
    """하이퍼파라미터 튜닝 실행"""
    logger.info("🔧 하이퍼파라미터 튜닝 시작")
    
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
        
        # 2. 피처 준비
        target_col = config.target_column
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
        X_val = model_results['X_val']  # 전처리된 데이터
        y_val = model_results['y_val']
        X_train = model_results['X_train']
        y_train = model_results['y_train']
        preprocessor = model_results['preprocessor']
        
        # 이미 전처리된 데이터를 명확한 변수명으로 할당
        X_train_processed = X_train  # 전처리 완료된 train 데이터
        X_val_processed = X_val      # 전처리 완료된 validation 데이터
        
        # 5. 교차 검증 설정
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 6. 평가 지표 설정 - 복합점수와 일치시킴
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
        
        def composite_scorer(y_true, y_pred):
            """복합점수 계산: Precision 중심 가중평균 (Precision 40% + F1 30% + Accuracy 20% + Recall 10%)"""
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            # Precision을 중시하는 가중평균
            composite = (precision * 0.4 + f1 * 0.3 + accuracy * 0.2 + recall * 0.1)
            return composite
        
        composite_scorer_func = make_scorer(composite_scorer)
        f1_scorer = make_scorer(f1_score)
        auc_scorer = make_scorer(roc_auc_score)
        
        # 7. 하이퍼파라미터 그리드 정의 (성능 상위 3개 모델만)
        param_grids = {}
        
        # 2단계: 최적 구간 주변 세밀 탐색 (3개 모델 모두)
        # Logistic Regression - 1.0 주변 세밀 탐색
        param_grids['Logistic Regression'] = {
            'C': [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],  # 1.0 주변 세밀하게
            'penalty': ['l2'],  # 성공한 penalty 유지
            'max_iter': [1000]
        }
        
        # SVM - 0.1 주변 세밀 탐색 (linear kernel)
        param_grids['SVM'] = {
            'C': [0.05, 0.08, 0.1, 0.12, 0.15, 0.2],  # 0.1 주변 세밀하게
            'kernel': ['linear'],  # 성공한 kernel 유지
            'gamma': ['scale']
        }
        
        # LightGBM - 기본값 주변 세밀 탐색 (튜닝 실패했으므로 기본값 기준)
        if _has_lgbm:
            param_grids['LightGBM'] = {
                'n_estimators': [50, 100, 150, 200, 250],  # 기본값 100 주변
                'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],  # 기본값 0.1 주변
                'max_depth': [3, 4, 5, 6, 7]  # 기본값 3 주변
            }
        
        # 8. 모델 정의 (성능 상위 3개 모델만)
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
        }
        
        if _has_lgbm:
            models['LightGBM'] = LGBMClassifier(random_state=42, class_weight='balanced')
        
        # 파라미터 그리드에서 classifier__ 접두사 제거 (Pipeline 사용하지 않음)
        for model_name in param_grids:
            new_params = {}
            for param, values in param_grids[model_name].items():
                # classifier__ 접두사 제거
                new_param = param.replace('classifier__', '')
                new_params[new_param] = values
            param_grids[model_name] = new_params
        
        # 9. 하이퍼파라미터 튜닝 실행 (전처리된 데이터에 직접 적용)
        tuning_results = {}
        
        for model_name, model in models.items():
            if model_name not in param_grids:
                continue
                
            logger.info(f"🔧 {model_name} 하이퍼파라미터 튜닝 시작")
            
            # 파라미터 호환성 처리
            current_params = param_grids[model_name].copy()
            
            # Logistic Regression의 penalty-solver 호환성 처리
            if model_name == 'Logistic Regression':
                # penalty에 따라 적절한 solver 자동 선택
                compatible_params = []
                for penalty in current_params['penalty']:
                    param_combo = current_params.copy()
                    param_combo['penalty'] = [penalty]
                    
                    # penalty에 따른 solver 자동 선택
                    if penalty == 'l1':
                        param_combo['solver'] = ['liblinear']  # l1은 liblinear 사용
                        param_combo.pop('l1_ratio', None)     # l1은 l1_ratio 불필요
                    elif penalty == 'l2':
                        param_combo['solver'] = ['lbfgs']      # l2는 lbfgs 사용 (빠름)
                        param_combo.pop('l1_ratio', None)     # l2는 l1_ratio 불필요
                    elif penalty == 'elasticnet':
                        param_combo['solver'] = ['saga']       # elasticnet은 saga 사용
                        # l1_ratio는 유지 (elasticnet 필수 파라미터)
                    
                    compatible_params.append(param_combo)
                
                # 각 호환 조합별로 개별 튜닝
                best_score = -1
                best_params = None
                best_estimator = None
                
                for param_combo in compatible_params:
                    grid_search = GridSearchCV(
                        model,
                        param_combo,
                        cv=cv,
                        scoring=composite_scorer_func,
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train_processed, y_train)
                    
                    if grid_search.best_score_ > best_score:
                        best_score = grid_search.best_score_
                        best_params = grid_search.best_params_
                        best_estimator = grid_search.best_estimator_
                
                # 결과 저장
                tuning_results[model_name] = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'best_model': best_estimator
                }
                
            else:
                # SVM과 LightGBM은 기존 방식
                grid_search = GridSearchCV(
                    model,
                    current_params,
                    cv=cv,
                    scoring=composite_scorer_func,
                    n_jobs=-1,
                    verbose=1
                )
                
                # 하이퍼파라미터 튜닝 실행 (이미 전처리된 데이터 사용)
                grid_search.fit(X_train_processed, y_train)
                
                # 결과 저장
                tuning_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'best_model': grid_search.best_estimator_
                }
            
            logger.info(f"✅ {model_name} 튜닝 완료")
            logger.info(f"   최고 점수: {tuning_results[model_name]['best_score']:.4f}")
            logger.info(f"   최적 파라미터: {tuning_results[model_name]['best_params']}")
        
        # 10. 최적 모델 평가
        logger.info("📊 최적 모델 평가 시작")
        
        best_models = {}
        for model_name, results in tuning_results.items():
            best_model = results['best_model']
            
            # 예측
            y_pred = best_model.predict(X_val_processed)
            y_pred_proba = best_model.predict_proba(X_val_processed)[:, 1]
            
            # 성능 평가
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # 복합 점수 계산 (균등 가중)
            # Precision 중심 가중평균 (GridSearchCV와 일치)
            composite_score = (precision * 0.4 + f1 * 0.3 + accuracy * 0.2 + recall * 0.1)
            
            best_models[model_name] = {
                'model': best_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'composite_score': composite_score,
                'best_params': results['best_params']
            }
            
            logger.info(f"✅ {model_name} 평가 완료:")
            logger.info(f"   Accuracy: {accuracy:.4f}")
            logger.info(f"   Precision: {precision:.4f}")
            logger.info(f"   Recall: {recall:.4f}")
            logger.info(f"   F1-Score: {f1:.4f}")
            logger.info(f"   AUC: {auc:.4f}")
            logger.info(f"   Composite: {composite_score:.4f}")
        
        # 11. 최고 모델 선택
        best_model_name = max(best_models.keys(), key=lambda x: best_models[x]['composite_score'])
        best_model_info = best_models[best_model_name]
        
        logger.info(f"🏆 최고 모델: {best_model_name}")
        logger.info(f"   복합 점수: {best_model_info['composite_score']:.4f}")
        
        # 12. 결과 저장
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 튜닝 결과 저장
        tuning_results_path = output_dir / "hyperparameter_tuning_results.pkl"
        joblib.dump(tuning_results, tuning_results_path)
        
        # 최고 모델 저장
        best_model_path = output_dir / "best_tuned_model.pkl"
        joblib.dump(best_model_info['model'], best_model_path)
        
        # 성능 비교 CSV
        performance_df = pd.DataFrame({
            'model': list(best_models.keys()),
            'accuracy': [best_models[m]['accuracy'] for m in best_models.keys()],
            'precision': [best_models[m]['precision'] for m in best_models.keys()],
            'recall': [best_models[m]['recall'] for m in best_models.keys()],
            'f1_score': [best_models[m]['f1'] for m in best_models.keys()],
            'auc': [best_models[m]['auc'] for m in best_models.keys()],
            'composite_score': [best_models[m]['composite_score'] for m in best_models.keys()]
        })
        performance_df = performance_df.sort_values('composite_score', ascending=False)
        performance_df.to_csv(output_dir / "tuned_model_performance.csv", index=False)
        
        # 13. 결과 요약
        print("\n" + "="*80)
        print("🎉 하이퍼파라미터 튜닝 완료!")
        print("="*80)
        print("🏆 최고 모델:", best_model_name)
        print("📊 성능 지표:")
        print(f"   Accuracy:  {best_model_info['accuracy']:.4f}")
        print(f"   Precision: {best_model_info['precision']:.4f}")
        print(f"   Recall:    {best_model_info['recall']:.4f}")
        print(f"   F1-Score:  {best_model_info['f1']:.4f}")
        print(f"   AUC:       {best_model_info['auc']:.4f}")
        print(f"   Composite: {best_model_info['composite_score']:.4f}")
        print("\n📁 생성된 파일:")
        print("  - outputs/hyperparameter_tuning_results.pkl")
        print("  - outputs/best_tuned_model.pkl")
        print("  - outputs/tuned_model_performance.csv")
        print("="*80)
        
        # 8. 최고 성능 모델이 기존 모델보다 좋으면 최종 모델 업데이트
        original_best_score = max(model_results['model_scores'].values()) if 'model_scores' in model_results else 0
        tuned_best_score = best_model_info['composite_score']
        
        if tuned_best_score > original_best_score:
            logger.info(f"🎉 튜닝된 모델이 더 우수합니다! {original_best_score:.4f} → {tuned_best_score:.4f}")
            
            # 최종 model_results 업데이트
            model_results['best_model'] = best_model_info['model']
            model_results['best_model_name'] = f"{best_model_name} (Tuned)"
            model_results['tuning_improvement'] = tuned_best_score - original_best_score
            
            # 최종 모델 저장 (outputs/ 덮어쓰기)
            outputs_dir = Path(config.output_dir)
            joblib.dump(best_model_info['model'], outputs_dir / "model.pkl")
            joblib.dump(model_results, outputs_dir / "model_results.pkl")
            
            logger.info("✅ 최종 모델이 튜닝된 모델로 업데이트되었습니다")
            
            # 모델 성능 정보 업데이트
            from scripts.save_model_performance import save_model_performance
            save_model_performance(model_results)
        else:
            logger.info(f"기존 모델이 더 우수합니다. {original_best_score:.4f} > {tuned_best_score:.4f}")
        
        logger.info("✅ 하이퍼파라미터 튜닝 완료")
        
    except Exception as e:
        logger.error(f"하이퍼파라미터 튜닝 오류: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    hyperparameter_tuning()

if __name__ == "__main__":
    main()
