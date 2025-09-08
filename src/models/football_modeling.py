"""
Football Transfer Prediction - Modeling Pipeline
실제 결과 파일들을 생성하는 모델링 클래스
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# 피처 엔지니어링 추가
from src.features.feature_engineering import FootballFeatureEngineer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, 
                           roc_curve, precision_recall_curve)
from sklearn.metrics import auc as sklearn_auc
from sklearn.utils.class_weight import compute_class_weight

# Optional libraries
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    _has_xgb = True
except ImportError:
    _has_xgb = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    _has_lgbm = True
except ImportError:
    _has_lgbm = False

try:
    import shap
    _has_shap = True
except ImportError:
    _has_shap = False

logger = logging.getLogger(__name__)

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """파이프라인에서 사용할 수 있는 라벨 인코더"""
    
    def __init__(self):
        self.label_encoders = {}
        self.is_fitted = False
    
    def fit(self, X, y=None):
        X = X.copy()
        for i in range(X.shape[1]):
            le = LabelEncoder()
            le.fit(X[:, i].astype(str))
            self.label_encoders[i] = le
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("LabelEncoder must be fitted before transform")
        X = X.copy()
        X_encoded = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            le = self.label_encoders[i]
            X_encoded[:, i] = le.transform([
                x if pd.notna(x) and x != '' else 'unknown'
                for x in X[:, i].astype(str)
            ])
        return X_encoded.astype(float)

class FootballModelTrainer:
    """Football Transfer Prediction 모델 훈련 클래스"""
    
    def __init__(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, pred_data: pd.DataFrame, config):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.pred_data = pred_data
        self.config = config
        self.target_col = config.target_column
        self.ordinal_features = config.features_ordinal
        self.nominal_features = config.features_nominal
        
        # 결과 저장용
        self.model_results = {}
        self.best_model = None
        self.preprocessor = None
        self.output_dir = Path(config.output_dir)
        
    def run_pipeline(self) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info("🚀 모델링 파이프라인 시작")
        
        # 0. 데이터 품질 및 누수 검사
        from src.features.feature_engineering import DataLeakageChecker
        
        # 데이터 품질 검사
        quality_results = DataLeakageChecker.check_data_quality(self.train_data)
        logger.info("🔍 데이터 품질 검사 완료")
        if quality_results['high_missing_features']:
            logger.warning(f"높은 결측치 피처: {list(quality_results['high_missing_features'].keys())}")
        if quality_results['constant_features']:
            logger.warning(f"상수 피처: {quality_results['constant_features']}")
        if quality_results['duplicate_rows'] > 0:
            logger.info(f"ℹ️ 중복 행: {quality_results['duplicate_rows']}개 (정상적인 데이터 특성)")
        
        # 시간적 누수 검사
        if 'season' in self.train_data.columns:
            temporal_results = DataLeakageChecker.check_temporal_leakage(
                self.train_data, 'season', self.target_col
            )
            logger.info("🔍 시간적 데이터 누수 검사 완료")
        
        # 피처 누수 검사
        feature_leakage = DataLeakageChecker.check_feature_leakage(self.train_data, self.target_col)
        if feature_leakage['suspicious_features']:
            logger.warning(f"의심스러운 피처: {feature_leakage['suspicious_features']}")
        else:
            logger.info("✅ 피처 누수 없음")
        
        # 1. 피처 엔지니어링 (각 데이터에 개별 적용)
        feature_engineer = FootballFeatureEngineer()
        self.train_data = feature_engineer.create_engineered_features(self.train_data)
        self.valid_data = feature_engineer.create_engineered_features(self.valid_data)
        self.test_data = feature_engineer.create_engineered_features(self.test_data)
        self.pred_data = feature_engineer.create_engineered_features(self.pred_data)
        logger.info(f"✅ 피처 엔지니어링 완료: train {self.train_data.shape}, valid {self.valid_data.shape}, test {self.test_data.shape}, pred {self.pred_data.shape}")
        
        # 2. 데이터 분할
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data()
        
        # 24/25 예측용 데이터 준비
        X_2425 = self._prepare_prediction_data()
        
        # 3. 전처리기 생성 및 학습 (train 데이터만 사용 - 데이터 누수 방지)
        self.preprocessor = self._create_preprocessor()
        self.preprocessor.fit(X_train)
        logger.info("✅ 전처리기 학습 완료")
        
        # 전처리 적용
        X_train = self.preprocessor.transform(X_train)
        X_val = self.preprocessor.transform(X_val)
        X_test = self.preprocessor.transform(X_test)
        X_2425 = self.preprocessor.transform(X_2425)
        logger.info(f"✅ 전처리 적용 완료: train {X_train.shape}, valid {X_val.shape}, test {X_test.shape}, pred {X_2425.shape}")
        
        # 4. 8개 모델 학습 및 평가
        models = self._define_models()
        model_scores, model_details = self._train_and_evaluate_models(models, X_train, y_train, X_val, y_val)
        
        # 5. 최고 성능 모델 선택
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = models[best_model_name]
        logger.info(f"🏆 최고 성능 모델: {best_model_name} (점수: {model_scores[best_model_name]:.4f})")
        
        # 6. 최종 평가
        final_results = self._final_evaluation(X_val, y_val)
        
        # 7. 24/25 예측
        predictions_2425 = self._predict_2425(X_2425)
        
        # 8. 피처 중요도 처리
        feature_importance = None
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self._get_processed_feature_names()
            # 피처 수가 맞는지 확인
            if len(self.best_model.feature_importances_) == len(feature_names):
                feature_importance = pd.Series(
                    self.best_model.feature_importances_,
                    index=feature_names
                ).sort_values(ascending=True)
            else:
                # 피처 수가 맞지 않으면 기본 이름 사용
                feature_importance = pd.Series(
                    self.best_model.feature_importances_,
                    index=[f"feature_{i}" for i in range(len(self.best_model.feature_importances_))]
                ).sort_values(ascending=True)
        
        # 9. 오버피팅 분석 (상위 3개 모델)
        logger.info("🔍 오버피팅 분석 시작")
        from src.features.feature_engineering import OverfittingChecker
        
        # 상위 3개 모델에 대해 오버피팅 분석
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        learning_curve_results = {}
        
        for model_name, score in top_models:
            if model_name in model_details:
                model = model_details[model_name]['model']
                try:
                    # 오버피팅 분석 수행
                    lc_result = OverfittingChecker.check_learning_curves(
                        model, X_train_processed, y_train, X_val_processed, y_val
                    )
                    learning_curve_results[model_name] = lc_result
                    logger.info(f"✅ {model_name} 오버피팅 분석 완료")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} 오버피팅 분석 실패: {e}")
        
        # 10. 결과 저장
        self.model_results = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'X_2425': X_2425,
            'predictions_2425': predictions_2425,
            'model_scores': model_scores,
            'model_details': model_details,
            'best_model_name': best_model_name,
            'best_model': self.best_model,
            'preprocessor': self.preprocessor,
            'final_results': final_results,
            'feature_importance': feature_importance,
            'learning_curve_results': learning_curve_results
        }
        
        # 11. 시각화 (SHAP, 피처 중요도, 학습 곡선)
        try:
            from src.visualization.plotter import ModelVisualizer
            visualizer = ModelVisualizer(self.model_results, self.output_dir)
            
            # 모든 시각화 생성 (학습 곡선 포함)
            visualizer.create_all_plots()
            
            logger.info("✅ 시각화 완료")
        except Exception as e:
            logger.error(f"시각화 오류: {e}")
        
        logger.info("✅ 모델링 파이프라인 완료")
        return self.model_results
    
    def _get_numeric_features(self) -> List[str]:
        """수치형 피처 추출 (ID 변수 제외)"""
        all_features = set(self.train_data.columns) - {self.target_col}
        categorical_features = set(self.ordinal_features + self.nominal_features)
        # 제외할 ID 변수들
        exclude_cols = {'player_id', 'club_id', 'season'}
        return [col for col in all_features if col not in categorical_features and 
                col not in exclude_cols and
                pd.api.types.is_numeric_dtype(self.train_data[col])]
    
    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
        """데이터 분할 (4개 데이터셋 사용)"""
        # Train 데이터
        X_train = self.train_data.drop(columns=[self.target_col])
        y_train = self.train_data[self.target_col]
        
        # Valid 데이터
        X_val = self.valid_data.drop(columns=[self.target_col])
        y_val = self.valid_data[self.target_col]
        
        # Test 데이터
        X_test = self.test_data.drop(columns=[self.target_col])
        y_test = self.test_data[self.target_col]
        
        logger.info(f"📊 데이터 분할 완료: train {X_train.shape}, valid {X_val.shape}, test {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _prepare_prediction_data(self) -> pd.DataFrame:
        """24/25 예측용 데이터 준비"""
        X_2425 = self.pred_data.copy()
        # target 컬럼이 있다면 제거 (pred 데이터는 target이 없을 수 있음)
        if self.target_col in X_2425.columns:
            X_2425 = X_2425.drop(columns=[self.target_col])
        logger.info(f"📊 24/25 예측 데이터 준비: {X_2425.shape}")
        return X_2425
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """전처리기 생성"""
        # 수치형 피처 전처리
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 범주형 피처 전처리
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 순서형 피처 전처리
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('label', CustomLabelEncoder())
        ])
        
        # 컬럼 분류
        exclude_features = self.ordinal_features + self.nominal_features + [self.target_col] + self.config.features_numeric_exclude
        numeric_features = [col for col in self.train_data.columns 
                          if col not in exclude_features]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, self.nominal_features),
                ('ord', ordinal_transformer, self.ordinal_features)
            ]
        )
        
        return preprocessor
    
    def _get_processed_feature_names(self) -> List[str]:
        """전처리된 피처 이름들 반환"""
        feature_names = []
        
        # 수치형 피처
        exclude_features = self.ordinal_features + self.nominal_features + [self.target_col] + self.config.features_numeric_exclude
        numeric_features = [col for col in self.train_data.columns if col not in exclude_features]
        feature_names.extend(numeric_features)
        
        # 범주형 피처 (원핫 인코딩 후)
        if hasattr(self.preprocessor, 'named_transformers_'):
            cat_transformer = self.preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'get_feature_names_out'):
                cat_features = cat_transformer.get_feature_names_out(self.nominal_features)
                feature_names.extend(cat_features)
            else:
                # 원핫 인코딩된 피처 이름 생성
                for feature in self.nominal_features:
                    if feature in self.train_data.columns:
                        unique_values = self.train_data[feature].dropna().unique()
                        for value in unique_values:
                            feature_names.append(f"{feature}_{value}")
        
        # 순서형 피처
        feature_names.extend(self.ordinal_features)
        
        return feature_names
    
    def _define_models(self) -> Dict[str, Any]:
        """모델 정의"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
        }
        
        # 클래스 불균형 보정 값 계산
        pos_weight = (self.train_data[self.target_col].shape[0] - self.train_data[self.target_col].sum()) / max(1, self.train_data[self.target_col].sum()) if self.train_data[self.target_col].sum() > 0 else 1.0
        
        # XGBoost
        if _has_xgb:
            models['XGBoost'] = XGBClassifier(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric='logloss',
                tree_method='hist',
                scale_pos_weight=float(pos_weight),
                n_jobs=-1
            )
        
        # LightGBM
        if _has_lgbm:
            models['LightGBM'] = LGBMClassifier(
                random_state=42,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight='balanced',
                n_jobs=-1
            )
        
        return models
    
    def _train_and_evaluate_models(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                                  y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """모델 학습 및 평가"""
        logger.info("🤖 8개 모델 학습 및 평가 시작")
        
        model_scores = {}
        model_details = {}
        
        # 교차 검증 설정
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"🔄 {name} 학습 중...")
            
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 성능 지표 계산
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0
            
            # 교차 검증
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
            
            # 복합 점수 계산 (Precision 중심 가중평균)
            # Precision 40% + F1 30% + Accuracy 20% + Recall 10%
            composite_score = (precision * 0.4 + f1 * 0.3 + accuracy * 0.2 + recall * 0.1)
            model_scores[name] = composite_score
            
            # 상세 성능 지표 저장
            model_details[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'composite_score': composite_score,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std()
            }
            
            logger.info(f"✅ {name}: AUC={auc:.3f}, F1={f1:.3f}, Composite={composite_score:.3f}")
        
        return model_scores, model_details
    
    def _final_evaluation(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """최종 평가"""
        logger.info("📊 최종 평가 수행")
        
        # 최고 성능 모델로 예측
        y_pred = self.best_model.predict(X_val)
        y_pred_proba = self.best_model.predict_proba(X_val)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # 성능 지표
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0
        
        # 분류 리포트
        report = classification_report(y_val, y_pred, output_dict=True)
        
        # 혼동 행렬
        cm = confusion_matrix(y_val, y_pred)
        
        # ROC 곡선
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba) if y_pred_proba is not None else ([], [], [])
        roc_auc_value = sklearn_auc(fpr, tpr) if len(fpr) > 0 else 0
        
        # Precision-Recall 곡선
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba) if y_pred_proba is not None else ([], [], [])
        pr_auc_score = sklearn_auc(recall_curve, precision_curve) if len(precision_curve) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'roc_auc': roc_auc_value,
            'pr_auc': pr_auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        }
        
        logger.info(f"📊 최종 성능: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        return results
    
    def _predict_2425(self, X_2425: pd.DataFrame) -> np.ndarray:
        """24/25 시즌 예측"""
        logger.info("🔮 24/25 시즌 예측 수행")
        
        predictions = self.best_model.predict(X_2425)
        prediction_proba = self.best_model.predict_proba(X_2425)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        logger.info(f"✅ 24/25 예측 완료: {len(predictions)}개 선수, 이적 예상 {predictions.sum()}명")
        
        return {
            'predictions': predictions,
            'probabilities': prediction_proba
        }
    
    def predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        """24/25 시즌 예측 (pred_df 사용)"""
        logger.info("🔮 24/25 시즌 예측 시작")
        
        # 피처 엔지니어링 적용
        feature_engineer = FootballFeatureEngineer()
        pred_data_processed = feature_engineer.create_engineered_features(pred_data)
        
        # target 컬럼 제거
        if self.target_col in pred_data_processed.columns:
            X_pred = pred_data_processed.drop(columns=[self.target_col])
        else:
            X_pred = pred_data_processed
        
        # 전처리 적용
        X_pred_processed = self.preprocessor.transform(X_pred)
        
        # 예측
        predictions_binary = self.best_model.predict(X_pred_processed)
        predictions_proba = self.best_model.predict_proba(X_pred_processed)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # 결과 DataFrame 생성
        result_df = pred_data_processed[['player_name', 'position', 'club_name']].copy()
        result_df['transfer_prediction'] = predictions_binary
        if predictions_proba is not None:
            result_df['transfer_probability'] = predictions_proba
        else:
            result_df['transfer_probability'] = predictions_binary.astype(float)
        
        logger.info(f"✅ 24/25 예측 완료: {len(result_df)}개 선수, 이적 예상 {predictions_binary.sum()}명")
        return result_df
    
    def save_model(self, output_dir: Path):
        """모델 및 전처리기 저장"""
        logger.info("💾 모델 및 전처리기 저장")
        
        # 모델 저장
        model_path = output_dir / "best_tuned_model.pkl"
        joblib.dump(self.best_model, model_path)
        logger.info(f"✅ 모델 저장 완료: {model_path}")
        
        # 전처리기 저장
        preprocessor_path = output_dir / "preprocessor.pkl"
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"✅ 전처리기 저장 완료: {preprocessor_path}")
        
        # 모델 결과 저장
        results_path = output_dir / "model_results.pkl"
        joblib.dump(self.model_results, results_path)
        logger.info(f"✅ 모델 결과 저장 완료: {results_path}")