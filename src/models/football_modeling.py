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
                           roc_curve, precision_recall_curve, auc)
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
        self.label_encoders = {}
        if hasattr(X, 'iloc'):
            for i in range(X.shape[1]):
                le = LabelEncoder()
                le.fit(X.iloc[:, i].astype(str))
                self.label_encoders[i] = le
        else:
            for i in range(X.shape[1]):
                le = LabelEncoder()
                le.fit(X[:, i].astype(str))
                self.label_encoders[i] = le
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("LabelEncoder가 fit되지 않았습니다.")
        
        X_encoded = np.zeros_like(X, dtype=float)
        
        if hasattr(X, 'iloc'):
            for i in range(X.shape[1]):
                try:
                    X_encoded[:, i] = self.label_encoders[i].transform(X.iloc[:, i].astype(str))
                except ValueError:
                    # 새로운 라벨이 있는 경우 -1로 매핑
                    X_encoded[:, i] = np.array([
                        self.label_encoders[i].transform([x])[0] if x in self.label_encoders[i].classes_ else -1
                        for x in X.iloc[:, i].astype(str)
                    ])
        else:
            for i in range(X.shape[1]):
                try:
                    X_encoded[:, i] = self.label_encoders[i].transform(X[:, i].astype(str))
                except ValueError:
                    X_encoded[:, i] = np.array([
                        self.label_encoders[i].transform([x])[0] if x in self.label_encoders[i].classes_ else -1
                        for x in X[:, i].astype(str)
                    ])
        return X_encoded.astype(float)

class FootballModelTrainer:
    """Football Transfer Prediction 모델 훈련 클래스"""
    
    def __init__(self, df_model: pd.DataFrame, config):
        self.df_model = df_model
        self.config = config
        self.target_col = config.target_column
        self.ordinal_features = config.features_ordinal
        self.nominal_features = config.features_nominal
        
        # 결과 저장용
        self.model_results = {}
        self.best_model = None
        self.preprocessor = None
        
    def run_pipeline(self) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info("🚀 모델링 파이프라인 시작")
        
        # 0. 데이터 품질 및 누수 검사
        from src.features.feature_engineering import DataLeakageChecker
        
        # 데이터 품질 검사
        quality_results = DataLeakageChecker.check_data_quality(self.df_model)
        logger.info("🔍 데이터 품질 검사 완료")
        if quality_results['high_missing_features']:
            logger.warning(f"높은 결측치 피처: {list(quality_results['high_missing_features'].keys())}")
        if quality_results['constant_features']:
            logger.warning(f"상수 피처: {quality_results['constant_features']}")
        if quality_results['duplicate_rows'] > 0:
            logger.info(f"ℹ️ 중복 행: {quality_results['duplicate_rows']}개 (정상적인 데이터 특성)")
        
        # 시간적 누수 검사
        if 'season' in self.df_model.columns:
            temporal_results = DataLeakageChecker.check_temporal_leakage(
                self.df_model, 'season', self.target_col
            )
            logger.info("🔍 시간적 데이터 누수 검사 완료")
        
        # 피처 누수 검사
        feature_leakage = DataLeakageChecker.check_feature_leakage(self.df_model, self.target_col)
        if feature_leakage['suspicious_features']:
            logger.warning(f"의심스러운 피처: {feature_leakage['suspicious_features']}")
        else:
            logger.info("✅ 피처 누수 없음")
        
        # 1. 피처 엔지니어링 (전체 데이터에 적용)
        feature_engineer = FootballFeatureEngineer()
        self.df_model = feature_engineer.create_engineered_features(self.df_model)
        logger.info(f"✅ 피처 엔지니어링 완료: {self.df_model.shape}")
        
        # 2. 데이터 분할 (22/23을 validation으로 사용)
        X_train, X_val, y_train, y_val, X_2324 = self._split_data()
        
        # 3. 전처리기 생성 및 학습 (train 데이터만 사용 - 데이터 누수 방지)
        feature_types = feature_engineer.get_feature_types(self.df_model)
        self.preprocessor = feature_engineer.create_preprocessor(feature_types)
        self.feature_types = feature_types
        
        # train 데이터로만 전처리기 학습
        self.preprocessor.fit(X_train)
        logger.info("✅ 전처리기 학습 완료 (train 데이터만 사용)")
        
        # 전처리 적용
        X_train = self.preprocessor.transform(X_train)
        X_val = self.preprocessor.transform(X_val)
        logger.info(f"✅ 전처리 적용 완료: train {X_train.shape}, validation {X_val.shape}")
        
        # 4. 모델 정의
        models = self._define_models()
        
        # 5. 모델 훈련 및 평가 (validation 데이터로 평가)
        model_scores, model_details = self._train_and_evaluate_models(models, X_train, y_train, X_val, y_val)
        
        # 6. 최적 모델 선택
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = models[best_model_name]
        
        # 6.5. 오버피팅 검사
        from src.features.feature_engineering import OverfittingChecker
        
        # 학습 곡선 분석
        learning_curve_results = OverfittingChecker.check_learning_curves(
            self.best_model, X_train, y_train, X_val, y_val
        )
        if learning_curve_results['is_overfitting']:
            logger.warning(f"⚠️ 오버피팅 감지! 최종 갭: {learning_curve_results['final_gap']:.3f}, 최대 갭: {learning_curve_results['max_gap']:.3f}")
        else:
            logger.info(f"✅ 오버피팅 없음 (최종 갭: {learning_curve_results['final_gap']:.3f})")
        
        # 교차검증 일관성 검사
        cv_results = OverfittingChecker.check_cv_consistency(self.best_model, X_train, y_train)
        if cv_results['is_stable']:
            logger.info(f"✅ 교차검증 안정성: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}")
        else:
            logger.warning(f"⚠️ 교차검증 불안정: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}")
        
        # 7. 최종 평가 (validation 데이터로 평가)
        final_results = self._final_evaluation(X_val, y_val)
        
        # 8. SHAP 분석 (validation 데이터로 분석)
        shap_results = self._shap_analysis(X_val, y_val)
        
        # 9. 결과 저장
        self.model_results = {
            'model_scores': model_scores,
            'model_details': model_details,  # 상세 성능 지표 추가
            'model_comparison': model_scores,  # Plotter에서 사용
            'best_model_name': best_model_name,
            'best_model': self.best_model,
            'preprocessor': self.preprocessor,
            'final_results': final_results,
            'shap_results': shap_results,
            'X_train': X_train,      # 훈련 데이터 추가
            'y_train': y_train,      # 훈련 타겟 추가
            'X_val': X_val,   # validation 데이터
            'y_val': y_val,   # validation 타겟
            'X_2324': X_2324,
            # 검사 결과 추가
            'data_quality_results': quality_results,
            'feature_leakage_results': feature_leakage,
            'learning_curve_results': learning_curve_results,
            'cv_consistency_results': cv_results
        }
        
        # 9. 시각화 (SHAP, 피처 중요도, 학습 곡선)
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
        all_features = set(self.df_model.columns) - {self.target_col}
        categorical_features = set(self.ordinal_features + self.nominal_features)
        # 제외할 ID 변수들
        exclude_cols = {'player_id', 'club_id', 'season'}
        
        return [col for col in all_features if col not in categorical_features and 
                col not in exclude_cols and
                pd.api.types.is_numeric_dtype(self.df_model[col])]
    
    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """데이터 분할 (22/23을 validation으로 사용)"""
        # 피처와 타겟 분리 (ID 변수 제외)
        exclude_cols = {'player_id', 'club_id', 'season', self.target_col}
        feature_cols = [col for col in self.df_model.columns if col not in exclude_cols]
        X = self.df_model[feature_cols]
        y = self.df_model[self.target_col]
        
        # 23/24 데이터 분리 (예측용 데이터)
        X_2324 = None
        if 'season' in self.df_model.columns:
            mask_2324 = self.df_model['season'] == '23/24'
            X_2324 = self.df_model[mask_2324][feature_cols].copy()
            X = X[~mask_2324].copy()
            y = y[~mask_2324].copy()
        
        # 22/23을 validation으로 사용
        if 'season' in self.df_model.columns and '22/23' in self.df_model['season'].values:
            validation_mask = self.df_model['season'] == '22/23'
            # 22/23을 제외한 나머지를 train으로, 22/23을 validation으로
            X_train, X_val = X[~validation_mask], X[validation_mask]
            y_train, y_val = y[~validation_mask], y[validation_mask]
            
            logger.info(f"📊 데이터 분할 완료:")
            logger.info(f"  - Train: {X_train.shape[0]:,} rows (11-21 시즌)")
            logger.info(f"  - Validation: {X_val.shape[0]:,} rows (22/23 시즌)")
            logger.info(f"  - Prediction: {X_2324.shape[0] if X_2324 is not None else 0:,} rows (23/24 시즌)")
        else:
            # season 컬럼이 없는 경우 일반적인 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"📊 데이터 분할 완료 (랜덤 분할):")
            logger.info(f"  - Train: {X_train.shape[0]:,} rows")
            logger.info(f"  - Validation: {X_val.shape[0]:,} rows")
        
        return X_train, X_val, y_train, y_val, X_2324
    
    def _create_preprocessor(self, numeric_features: List[str]) -> ColumnTransformer:
        """전처리기 생성"""
        # 수치형 변수 전처리
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 순서형 변수 전처리 (라벨 인코딩)
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('label_encoder', CustomLabelEncoder())
        ])
        
        # 명목형 변수 전처리 (원핫 인코딩)
        nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        # 컬럼 변환기
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('ord', ordinal_transformer, self.ordinal_features),
                ('nom', nominal_transformer, self.nominal_features)
            ]
        )
        
        return preprocessor
    
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
        
        if _has_xgb:
            models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
        
        if _has_lgbm:
            models['LightGBM'] = LGBMClassifier(random_state=42, class_weight='balanced')
        
        return models
    
    def _train_and_evaluate_models(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                                  y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """모델 훈련 및 평가"""
        model_scores = {}
        model_details = {}  # 상세 성능 지표 저장
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"🔧 {name} 모델 훈련 중...")
            
            # 이미 전처리된 데이터이므로 모델만 사용
            # 교차 검증 (전처리된 데이터로 직접 수행)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
            
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 성능 지표 계산
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0
            
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
        """최종 모델 평가"""
        # 이미 훈련된 모델 사용 (전처리된 데이터)
        # 예측
        y_pred = self.best_model.predict(X_val)
        y_pred_proba = self.best_model.predict_proba(X_val)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # 성능 지표
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # ROC 곡선
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        
        # 혼동 행렬
        cm = confusion_matrix(y_val, y_pred)
        
        # 피처 중요도 (Random Forest인 경우)
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr),
            'feature_importance': feature_importance
        }
    
    def _shap_analysis(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """SHAP 분석 (일관성 보장)"""
        if not _has_shap:
            logger.warning("SHAP가 설치되지 않았습니다.")
            return {}
        
        try:
            # 이미 훈련된 모델과 전처리기 사용 (재훈련 방지)
            # X_val는 이미 전처리된 상태
            X_val_processed = X_val

            # SHAP Explainer (모델 타입에 따라 선택)
            model_name = type(self.best_model).__name__
            
            if hasattr(self.best_model, 'feature_importances_'):
                # Tree-based 모델 (RandomForest, GradientBoosting, XGBoost, LightGBM 등)
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(X_val_processed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # 이진 분류의 positive class
            elif 'Linear' in model_name or 'Logistic' in model_name:
                # Linear 모델 (LogisticRegression, LinearRegression 등)
                explainer = shap.LinearExplainer(self.best_model, X_val_processed)
                shap_values = explainer.shap_values(X_val_processed)
            else:
                # 기타 모델 (SVM, KNN 등) - KernelExplainer 사용 (느림)
                background = shap.kmeans(X_val_processed, 50)  # 배경 데이터 샘플링
                explainer = shap.KernelExplainer(self.best_model.predict_proba, background)
                shap_values = explainer.shap_values(X_val_processed[:100])  # 샘플만 계산
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            
            # 피처 이름 생성 (전처리 후 피처명)
            feature_names = self._get_processed_feature_names()
            
            return {
                'shap_values': shap_values,
                'feature_names': feature_names,
                'X_val_processed': X_val_processed
            }
            
        except Exception as e:
            logger.error(f"SHAP 분석 오류: {e}")
            return {}
    
    def _get_feature_names(self) -> List[str]:
        """피처 이름 추출"""
        try:
            # 전처리기에서 피처 이름 추출
            feature_names = []
            
            # 수치형 피처
            numeric_features = self._get_numeric_features()
            feature_names.extend(numeric_features)
            
            # 순서형 피처
            feature_names.extend(self.ordinal_features)
            
            # 명목형 피처 (원핫 인코딩된 이름들)
            for feature in self.nominal_features:
                unique_values = self.df_model[feature].unique()
                for value in unique_values:
                    feature_names.append(f"{feature}_{value}")
            
            return feature_names
            
        except Exception as e:
            logger.error(f"피처 이름 추출 오류: {e}")
            return [f"feature_{i}" for i in range(100)]  # 기본 이름
    
    def _get_processed_feature_names(self) -> List[str]:
        """전처리 후 피처명 생성"""
        try:
            feature_names = []
            
            # 수치형 피처
            numeric_features = self._get_numeric_features()
            feature_names.extend(numeric_features)
            
            # 순서형 피처
            feature_names.extend(self.ordinal_features)
            
            # 명목형 피처 (원핫 인코딩된 이름들)
            for feature in self.nominal_features:
                if feature in self.df_model.columns:
                    unique_values = self.df_model[feature].unique()
                    for value in unique_values:
                        feature_names.append(f"{feature}_{value}")
            
            # 실제 전처리된 데이터 차원에 맞게 조정
            try:
                # 테스트 데이터로 실제 차원 확인 (ID 컬럼 제외)
                exclude_cols = {'player_id', 'club_id', 'season', self.target_col, 'player_name', 
                               'date_of_birth', 'agent_name', 'net_transfer_record'}
                test_cols = [col for col in self.df_model.columns if col not in exclude_cols]
                test_data = self.df_model[test_cols].head(1)
                processed = self.preprocessor.transform(test_data)
                actual_dim = processed.shape[1]
                
                if len(feature_names) != actual_dim:
                    # 차원이 맞지 않으면 실제 차원에 맞게 조정
                    if len(feature_names) > actual_dim:
                        feature_names = feature_names[:actual_dim]
                    else:
                        # 부족한 피처명은 기본 이름으로 채움
                        for i in range(len(feature_names), actual_dim):
                            feature_names.append(f"feature_{i}")
                
                logger.info(f"✅ 실제 피처명 생성: {len(feature_names)}개 (차원: {actual_dim})")
                
            except Exception as e:
                logger.warning(f"실제 차원 확인 실패: {e}")
            
            return feature_names
            
        except Exception as e:
            logger.error(f"전처리 후 피처명 생성 오류: {e}")
            return [f"feature_{i}" for i in range(100)]
    
    
    def save_model(self, output_dir: Path):
        """모델 저장"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        joblib.dump(self.best_model, output_dir / 'model.pkl')
        joblib.dump(self.preprocessor, output_dir / 'preprocessor.pkl')
        
        # 결과 저장
        joblib.dump(self.model_results, output_dir / 'model_results.pkl')
        
        logger.info(f"💾 모델 저장 완료: {output_dir}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """23/24 시즌 예측"""
        # 피처 엔지니어링 적용
        from src.features.feature_engineering import FootballFeatureEngineer
        feature_engineer = FootballFeatureEngineer()
        df_processed = feature_engineer.create_engineered_features(df)
        
        # 모델링 피처 선택 (ID 변수 제외)
        exclude_cols = {'player_id', 'club_id', 'season', 'transfer', 'player_name', 
                       'date_of_birth', 'agent_name', 'net_transfer_record'}
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        X_pred = df_processed[feature_cols]
        
        # 전처리 및 예측
        X_pred_processed = self.preprocessor.transform(X_pred)
        predictions = self.best_model.predict(X_pred_processed)
        probabilities = self.best_model.predict_proba(X_pred_processed)[:, 1]
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame({
            'player_name': df['player_name'],
            'club_name': df['club_name'],
            'position': df['position'],
            'predicted_transfer': predictions,
            'transfer_probability': probabilities,
            'transfer_probability_percent': (probabilities * 100).round(1)
        })
        
        return result_df
    
    @classmethod
    def load_model(cls, output_dir: Path):
        """모델 로드"""
        model = joblib.load(output_dir / 'model.pkl')
        preprocessor = joblib.load(output_dir / 'preprocessor.pkl')
        model_results = joblib.load(output_dir / 'model_results.pkl')
        
        # 새로운 인스턴스 생성
        instance = cls.__new__(cls)
        instance.best_model = model
        instance.preprocessor = preprocessor
        instance.model_results = model_results
        
        return instance