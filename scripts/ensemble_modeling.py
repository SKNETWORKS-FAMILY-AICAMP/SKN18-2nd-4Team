#!/usr/bin/env python3
"""
앙상블 모델 구축
- Voting Classifier
- Stacking Classifier
- Bagging 방법
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

def ensemble_modeling():
    """앙상블 모델 구축"""
    logger.info("🤝 앙상블 모델 구축 시작")
    
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
        
        # 4. 교차 검증 설정
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scorer = make_scorer(f1_score)
        
        # 5. 기본 모델들 정의
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, 
                                       min_samples_split=10, min_samples_leaf=4,
                                       class_weight='balanced', random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                            max_depth=5, subsample=0.8, random_state=42)),
            ('lr', LogisticRegression(C=0.1, class_weight='balanced', 
                                    random_state=42, max_iter=1000))
        ]
        
        if _has_lgbm:
            base_models.append(('lgb', LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                                    num_leaves=31, max_depth=5,
                                                    class_weight='balanced', random_state=42)))
        
        # 6. 앙상블 모델 정의
        ensemble_models = {}
        
        # Voting Classifier (Hard)
        ensemble_models['Voting (Hard)'] = VotingClassifier(
            estimators=base_models,
            voting='hard'
        )
        
        # Voting Classifier (Soft)
        ensemble_models['Voting (Soft)'] = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        # Stacking Classifier
        ensemble_models['Stacking'] = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(C=0.1, class_weight='balanced', 
                                             random_state=42, max_iter=1000),
            cv=3
        )
        
        # Bagging with Random Forest
        ensemble_models['Bagging (RF)'] = BaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, max_depth=8,
                                                class_weight='balanced', random_state=42),
            n_estimators=10,
            random_state=42
        )
        
        # Bagging with Gradient Boosting
        ensemble_models['Bagging (GB)'] = BaggingClassifier(
            base_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                    max_depth=4, random_state=42),
            n_estimators=10,
            random_state=42
        )
        
        # 7. 앙상블 모델 훈련 및 평가
        ensemble_results = {}
        
        for model_name, model in ensemble_models.items():
            logger.info(f"🤝 {model_name} 훈련 시작")
            
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
            
            ensemble_results[model_name] = {
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
        
        # 8. 최고 앙상블 모델 선택
        best_ensemble_name = max(ensemble_results.keys(), 
                                key=lambda x: ensemble_results[x]['composite_score'])
        best_ensemble_info = ensemble_results[best_ensemble_name]
        
        logger.info(f"🏆 최고 앙상블 모델: {best_ensemble_name}")
        logger.info(f"   복합 점수: {best_ensemble_info['composite_score']:.4f}")
        
        # 9. 앙상블 효과 분석
        logger.info("📊 앙상블 효과 분석")
        
        # 기본 모델들과 비교
        base_model_scores = []
        for name, _ in base_models:
            base_model_scores.append(ensemble_results.get(name, {}).get('composite_score', 0))
        
        avg_base_score = np.mean(base_model_scores)
        best_ensemble_score = best_ensemble_info['composite_score']
        improvement = best_ensemble_score - avg_base_score
        
        logger.info(f"   평균 기본 모델 점수: {avg_base_score:.4f}")
        logger.info(f"   최고 앙상블 점수: {best_ensemble_score:.4f}")
        logger.info(f"   개선도: {improvement:.4f} ({improvement/avg_base_score*100:.1f}%)")
        
        # 10. 결과 저장
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 앙상블 결과 저장
        ensemble_path = output_dir / "ensemble_results.pkl"
        joblib.dump(ensemble_results, ensemble_path)
        
        # 최고 앙상블 모델 저장
        best_ensemble_path = output_dir / "best_ensemble_model.pkl"
        joblib.dump(best_ensemble_info['model'], best_ensemble_path)
        
        # 성능 비교 CSV
        performance_df = pd.DataFrame({
            'model': list(ensemble_results.keys()),
            'cv_f1_mean': [ensemble_results[m]['cv_mean'] for m in ensemble_results.keys()],
            'cv_f1_std': [ensemble_results[m]['cv_std'] for m in ensemble_results.keys()],
            'test_accuracy': [ensemble_results[m]['accuracy'] for m in ensemble_results.keys()],
            'test_precision': [ensemble_results[m]['precision'] for m in ensemble_results.keys()],
            'test_recall': [ensemble_results[m]['recall'] for m in ensemble_results.keys()],
            'test_f1': [ensemble_results[m]['f1'] for m in ensemble_results.keys()],
            'test_auc': [ensemble_results[m]['auc'] for m in ensemble_results.keys()],
            'composite_score': [ensemble_results[m]['composite_score'] for m in ensemble_results.keys()]
        })
        performance_df = performance_df.sort_values('composite_score', ascending=False)
        performance_df.to_csv(output_dir / "ensemble_model_performance.csv", index=False)
        
        # 11. 결과 요약
        print("\n" + "="*80)
        print("🎉 앙상블 모델 구축 완료!")
        print("="*80)
        print("🏆 최고 앙상블 모델:", best_ensemble_name)
        print("📊 성능 지표:")
        print(f"   CV F1:     {best_ensemble_info['cv_mean']:.4f} (±{best_ensemble_info['cv_std']:.4f})")
        print(f"   Test Accuracy:  {best_ensemble_info['accuracy']:.4f}")
        print(f"   Test Precision: {best_ensemble_info['precision']:.4f}")
        print(f"   Test Recall:    {best_ensemble_info['recall']:.4f}")
        print(f"   Test F1:        {best_ensemble_info['f1']:.4f}")
        print(f"   Test AUC:       {best_ensemble_info['auc']:.4f}")
        print(f"   Composite:      {best_ensemble_info['composite_score']:.4f}")
        print(f"\n📈 앙상블 효과:")
        print(f"   평균 기본 모델: {avg_base_score:.4f}")
        print(f"   최고 앙상블:    {best_ensemble_score:.4f}")
        print(f"   개선도:         {improvement:.4f} ({improvement/avg_base_score*100:.1f}%)")
        print("\n📁 생성된 파일:")
        print("  - outputs/ensemble_results.pkl")
        print("  - outputs/best_ensemble_model.pkl")
        print("  - outputs/ensemble_model_performance.csv")
        print("="*80)
        
        logger.info("✅ 앙상블 모델 구축 완료")
        
    except Exception as e:
        logger.error(f"앙상블 모델 구축 오류: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    ensemble_modeling()

if __name__ == "__main__":
    main()
