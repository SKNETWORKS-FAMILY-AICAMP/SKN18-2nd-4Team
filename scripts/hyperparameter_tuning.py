#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝
- Random Forest, Gradient Boosting, LightGBM 최적화
- GridSearchCV를 통한 체계적 탐색
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import joblib

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

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
        
        # 모델 훈련 (기본 파이프라인 실행)
        model_trainer = FootballModelTrainer(all_data, config)
        model_results = model_trainer.run_pipeline()
        
        # 전처리된 데이터 가져오기
        X_train = model_results['X_test']  # 이미 전처리됨
        y_train = model_results['y_test']
        preprocessor = model_results['preprocessor']
        
        # 간단하게 처리 (이미 전처리된 데이터 사용)
        X_train_processed = X_train  # 이미 전처리됨
        X_test_processed = X_train   # 동일한 데이터
        
        # 5. 교차 검증 설정
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 6. 평가 지표 설정
        from sklearn.metrics import f1_score, roc_auc_score
        f1_scorer = make_scorer(f1_score)
        auc_scorer = make_scorer(roc_auc_score)
        
        # 7. 하이퍼파라미터 그리드 정의
        param_grids = {}
        
        # Random Forest
        param_grids['Random Forest'] = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Gradient Boosting
        param_grids['Gradient Boosting'] = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # LightGBM
        if _has_lgbm:
            param_grids['LightGBM'] = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'max_depth': [5, 10, 15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        # 8. 모델 정의
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        if _has_lgbm:
            models['LightGBM'] = LGBMClassifier(random_state=42, class_weight='balanced')
        
        # 9. 하이퍼파라미터 튜닝 실행
        tuning_results = {}
        
        for model_name, model in models.items():
            logger.info(f"🔧 {model_name} 하이퍼파라미터 튜닝 시작")
            
            # Pipeline 생성
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # GridSearchCV
            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=cv,
                scoring=f1_scorer,
                n_jobs=-1,
                verbose=1
            )
            
            # 훈련
            grid_search.fit(X_train, y_train)
            
            # 결과 저장
            tuning_results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_model': grid_search.best_estimator_
            }
            
            logger.info(f"✅ {model_name} 튜닝 완료")
            logger.info(f"   최고 점수: {grid_search.best_score_:.4f}")
            logger.info(f"   최적 파라미터: {grid_search.best_params_}")
        
        # 10. 최적 모델 평가
        logger.info("📊 최적 모델 평가 시작")
        
        best_models = {}
        for model_name, results in tuning_results.items():
            best_model = results['best_model']
            
            # 예측
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # 성능 평가
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # 복합 점수 계산
            composite_score = auc * 0.4 + f1 * 0.3 + precision * 0.2 + recall * 0.1
            
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
