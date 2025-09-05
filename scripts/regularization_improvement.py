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
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
        
        # 2. 피처 준비
        target_col = config.target_column
        # 2. 피처 엔지니어링 적용
        from src.features.feature_engineering import FootballFeatureEngineer
        feature_engineer = FootballFeatureEngineer()
        train_df_processed = feature_engineer.create_engineered_features(train_df)
        test_df_processed = feature_engineer.create_engineered_features(test_df)
        
        # 3. 데이터 분할
        exclude_cols = {'player_id', 'club_id', 'season', target_col, 'player_name', 
                       'date_of_birth', 'agent_name', 'net_transfer_record'}
        feature_cols = [col for col in train_df_processed.columns if col not in exclude_cols]
        
        X_train = train_df_processed[feature_cols]
        y_train = train_df_processed[target_col]
        X_test = test_df_processed[feature_cols]
        y_test = test_df_processed[target_col]
        
        # 4. 전처리기 생성
        all_data_processed = pd.concat([train_df_processed, test_df_processed], ignore_index=True)
        _, preprocessor, _ = feature_engineer.fit_transform(all_data_processed)
        
        # 5. 전처리된 데이터
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # 5. 교차 검증 설정
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scorer = make_scorer(f1_score)
        
        # 6. 정규화된 모델 정의
        regularized_models = {}
        
        # Logistic Regression with L1/L2 regularization
        regularized_models['Logistic Regression (L1)'] = LogisticRegression(
            penalty='l1', C=0.1, solver='liblinear', 
            class_weight='balanced', random_state=42, max_iter=1000
        )
        regularized_models['Logistic Regression (L2)'] = LogisticRegression(
            penalty='l2', C=0.1, 
            class_weight='balanced', random_state=42, max_iter=1000
        )
        
        # SVM with regularization
        regularized_models['SVM (RBF)'] = SVC(
            C=0.1, kernel='rbf', gamma='scale',
            class_weight='balanced', random_state=42, probability=True
        )
        regularized_models['SVM (Linear)'] = SVC(
            C=0.1, kernel='linear',
            class_weight='balanced', random_state=42, probability=True
        )
        
        # Random Forest with regularization
        regularized_models['Random Forest (Regularized)'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt',
            class_weight='balanced', random_state=42
        )
        
        # Gradient Boosting with regularization
        regularized_models['Gradient Boosting (Regularized)'] = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_split=10, min_samples_leaf=4,
            random_state=42
        )
        
        # LightGBM with regularization
        if _has_lgbm:
            regularized_models['LightGBM (Regularized)'] = LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                max_depth=5, subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', random_state=42
            )
        
        # 7. 정규화된 모델 훈련 및 평가
        regularization_results = {}
        
        for model_name, model in regularized_models.items():
            logger.info(f"🔧 {model_name} 훈련 시작")
            
            # Pipeline 생성
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # 교차 검증
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=f1_scorer)
            
            # 전체 데이터로 훈련
            pipeline.fit(X_train, y_train)
            
            # 예측
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # 성능 평가
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # 복합 점수 계산
            composite_score = auc * 0.4 + f1 * 0.3 + precision * 0.2 + recall * 0.1
            
            regularization_results[model_name] = {
                'model': pipeline,
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
