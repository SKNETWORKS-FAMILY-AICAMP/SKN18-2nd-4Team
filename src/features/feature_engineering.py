"""
Football Transfer Prediction - Feature Engineering Module
피쳐 엔지니어링 및 데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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
            raise ValueError("Must fit before transform")
            
        if hasattr(X, 'iloc'):
            X_encoded = X.copy()
            for i in range(X.shape[1]):
                try:
                    X_encoded.iloc[:, i] = self.label_encoders[i].transform(X.iloc[:, i].astype(str))
                except ValueError:
                    # 새로운 라벨을 -1로 처리
                    unique_labels = set(X.iloc[:, i].astype(str))
                    known_labels = set(self.label_encoders[i].classes_)
                    new_labels = unique_labels - known_labels
                    if new_labels:
                        logger.warning(f"새로운 라벨 발견: {new_labels}, -1로 처리합니다.")
                    X_encoded.iloc[:, i] = X.iloc[:, i].astype(str).apply(
                        lambda x: self.label_encoders[i].transform([x])[0] if x in self.label_encoders[i].classes_ else -1
                    )
            return X_encoded.astype(float)
        else:
            X_encoded = X.copy()
            for i in range(X.shape[1]):
                try:
                    X_encoded[:, i] = self.label_encoders[i].transform(X[:, i].astype(str))
                except ValueError:
                    unique_labels = set(X[:, i].astype(str))
                    known_labels = set(self.label_encoders[i].classes_)
                    new_labels = unique_labels - known_labels
                    if new_labels:
                        logger.warning(f"새로운 라벨 발견: {new_labels}, -1로 처리합니다.")
                    X_encoded[:, i] = np.array([
                        self.label_encoders[i].transform([x])[0] if x in self.label_encoders[i].classes_ else -1
                        for x in X[:, i].astype(str)
                    ])
            return X_encoded.astype(float)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class FootballFeatureEngineer:
    """Football 데이터 전용 피쳐 엔지니어링 클래스"""
    
    def __init__(self):
        self.feature_config = {
            'ordinal_features': [],  # season 제거 (모델링에서 제외)
            'nominal_features': ['club_name', 'country_of_birth', 'foot', 'position', 'sub_position'],
            'target_col': 'transfer'
        }
        self.position_avg_height = {}
        self.club_avg_minutes = {}
        self.is_fitted = False
    
    def detect_season_column(self, df: pd.DataFrame) -> Optional[str]:
        """시즌 컬럼 자동 탐지"""
        candidate_cols = [c for c in df.columns if 'season' in c.lower()]
        for c in candidate_cols:
            try:
                if df[c].astype(str).str.contains(r"^\d{2}/\d{2}$", na=False).any():
                    return c
            except Exception:
                continue
        return None
    
    def season_start_year(self, s: str) -> float:
        """시즌 문자열을 시작 연도로 변환"""
        try:
            s = str(s)
            if '/' in s:
                yy = int(s.split('/')[0])
                return 2000 + yy
        except Exception:
            return np.nan
        return np.nan
    
    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """11개 피처 엔지니어링 적용"""
        logger.info("🔧 피처 엔지니어링 적용 중...")
        df_fe = df.copy()
        
        # 통계 계산 (fit 단계)
        if not self.is_fitted:
            if 'position' in df_fe.columns and 'height_in_cm' in df_fe.columns:
                self.position_avg_height = df_fe.groupby('position')['height_in_cm'].mean().to_dict()
            
            if 'club_name' in df_fe.columns and 'season_avg_minutes' in df_fe.columns:
                self.club_avg_minutes = df_fe.groupby('club_name')['season_avg_minutes'].mean().to_dict()
            
            self.is_fitted = True
        
        # 1. 시즌 시작 연도
        if 'season' in df_fe.columns:
            df_fe['season_start_year'] = df_fe['season'].apply(
                lambda x: 2000 + int(x.split('/')[0]) if pd.notna(x) and '/' in str(x) else np.nan
            )
        
        # 2. 나이 계산
        if 'date_of_birth' in df_fe.columns and 'season_start_year' in df_fe.columns:
            birth_years = df_fe['date_of_birth'].astype(str).str.extract(r"^(\d{4})")[0]
            birth_years = pd.to_numeric(birth_years, errors='coerce')
            df_fe['age_at_season'] = (df_fe['season_start_year'] - birth_years).astype('float')
        
        # 3. 로그 시장가치
        if 'market_value_in_eur' in df_fe.columns:
            df_fe['log_market_value'] = np.log1p(pd.to_numeric(df_fe['market_value_in_eur'], errors='coerce'))
        
        # 4. 외국인 여부
        if 'country_of_birth' in df_fe.columns:
            df_fe['is_foreigner'] = (df_fe['country_of_birth'] != 'England').astype(int)
        
        # 5. 클럽 평균 대비 출전시간
        if 'season_avg_minutes' in df_fe.columns and 'club_name' in df_fe.columns:
            df_fe['minutes_vs_club_avg'] = df_fe.apply(
                lambda row: row['season_avg_minutes'] / (self.club_avg_minutes.get(row['club_name'], 1) + 1e-6)
                if pd.notna(row['season_avg_minutes']) and pd.notna(row['club_name']) else np.nan,
                axis=1
            )
        
        # 6. 클럽 평균 연령 대비 차이
        if 'age_at_season' in df_fe.columns and 'club_average_age' in df_fe.columns:
            df_fe['age_difference'] = df_fe['age_at_season'] - df_fe['club_average_age']
        
        # 7. 공격 기여도
        if 'goals' in df_fe.columns and 'assists' in df_fe.columns:
            df_fe['attack_contribution'] = df_fe['goals'] + df_fe['assists']
        
        # 8. 포지션별 평균 키 대비 비율
        if 'height_in_cm' in df_fe.columns and 'position' in df_fe.columns:
            df_fe['height_vs_position'] = df_fe.apply(
                lambda row: row['height_in_cm'] / (self.position_avg_height.get(row['position'], 180) + 1e-6)
                if pd.notna(row['height_in_cm']) and pd.notna(row['position']) else np.nan,
                axis=1
            )
        
        # 9. 분당 카드 수
        if 'yellow_cards' in df_fe.columns and 'season_avg_minutes' in df_fe.columns:
            df_fe['cards_per_minute'] = df_fe['yellow_cards'] / (df_fe['season_avg_minutes'] + 1e-6)
        
        # 10. 클럽 재적 기간
        if 'season' in df_fe.columns and 'player_id' in df_fe.columns:
            season_counts = df_fe.groupby('player_id')['season'].nunique()
            df_fe['club_tenure_seasons'] = df_fe['player_id'].map(season_counts).fillna(1)
        
        # 11. 포지션 내 경쟁 강도
        if 'position' in df_fe.columns and 'club_name' in df_fe.columns:
            position_counts = df_fe.groupby(['club_name', 'position']).size()
            df_fe['position_competition'] = df_fe.apply(
                lambda row: position_counts.get((row['club_name'], row['position']), 1)
                if pd.notna(row['club_name']) and pd.notna(row['position']) else 1,
                axis=1
            )
        
        logger.info(f"✅ 피처 엔지니어링 완료: {df_fe.shape[1] - df.shape[1]}개 피처 추가")
        return df_fe
    
    def get_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """피쳐 타입 분류"""
        # ID 변수 및 제외할 변수들
        exclude_cols = {
            'player_id', 'club_id', 'season', 'player_name', 'club_name',
            'date_of_birth', 'agent_name', 'net_transfer_record',
            self.feature_config['target_col']
        }
        
        # 수치형 변수 자동 탐지
        numeric_features = [
            c for c in df.columns 
            if c not in self.feature_config['ordinal_features'] + 
               self.feature_config['nominal_features'] + 
               list(exclude_cols)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        
        # 존재하는 피쳐만 선택
        ordinal_features = [c for c in self.feature_config['ordinal_features'] if c in df.columns]
        nominal_features = [c for c in self.feature_config['nominal_features'] if c in df.columns]
        
        return {
            'numeric': numeric_features,
            'ordinal': ordinal_features,
            'nominal': nominal_features
        }
    
    def create_preprocessor(self, feature_types: Dict[str, List[str]]) -> ColumnTransformer:
        """전처리 파이프라인 생성"""
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
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 하이브리드 전처리기
        transformers = [('num', numeric_transformer, feature_types['numeric'])]
        
        if feature_types['ordinal']:
            transformers.append(('ord', ordinal_transformer, feature_types['ordinal']))
        
        if feature_types['nominal']:
            transformers.append(('nom', nominal_transformer, feature_types['nominal']))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer, Dict[str, List[str]]]:
        """전체 피쳐 엔지니어링 및 전처리 파이프라인"""
        # 피쳐 엔지니어링 적용
        df_processed = self.create_engineered_features(df)
        
        # 피쳐 타입 분류
        feature_types = self.get_feature_types(df_processed)
        
        # 전처리기 생성
        preprocessor = self.create_preprocessor(feature_types)
        
        return df_processed, preprocessor, feature_types


class DataLeakageChecker:
    """데이터 누수 검사 클래스"""
    
    @staticmethod
    def check_temporal_leakage(df: pd.DataFrame, time_col: str, target_col: str) -> Dict[str, bool]:
        """시간적 데이터 누수 검사"""
        results = {}
        
        if time_col in df.columns:
            unique_times = sorted(df[time_col].unique())
            results['has_future_data'] = len(unique_times) > 1
            
            time_target = df.groupby(time_col)[target_col].mean()
            results['temporal_consistency'] = len(time_target.unique()) > 1
        
        return results
    
    @staticmethod
    def check_feature_leakage(df: pd.DataFrame, target_col: str) -> Dict[str, List[str]]:
        """피쳐 누수 검사"""
        suspicious_features = []
        
        # 타겟과 완벽한 상관관계
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != target_col:
                corr = abs(df[col].corr(df[target_col]))
                if corr > 0.95:
                    suspicious_features.append(f"{col} (correlation: {corr:.3f})")
        
        # 타겟과 동일한 분포
        for col in df.columns:
            if col != target_col and df[col].nunique() == df[target_col].nunique():
                if set(df[col].unique()) == set(df[target_col].unique()):
                    suspicious_features.append(f"{col} (identical distribution)")
        
        return {'suspicious_features': suspicious_features}
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
        """데이터 품질 검사"""
        results = {}
        
        # 결측치 비율
        missing_ratio = df.isnull().sum() / len(df)
        results['high_missing_features'] = missing_ratio[missing_ratio > 0.5].to_dict()
        
        # 중복 행
        results['duplicate_rows'] = df.duplicated().sum()
        
        # 상수 피쳐
        constant_features = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        results['constant_features'] = constant_features
        
        return results


class OverfittingChecker:
    """오버피팅 검사 클래스"""
    
    @staticmethod
    def check_learning_curves(model, X_train, y_train, X_val, y_val, 
                             train_sizes: List[float] = None) -> Dict[str, any]:
        """학습 곡선 분석"""
        if train_sizes is None:
            train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        train_scores = []
        val_scores = []
        
        for size in train_sizes:
            n_samples = int(len(X_train) * size)
            X_subset = X_train[:n_samples]
            y_subset = y_train[:n_samples]
            
            model.fit(X_subset, y_subset)
            
            train_score = model.score(X_subset, y_subset)
            val_score = model.score(X_val, y_val)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        # 오버피팅 지표
        final_gap = train_scores[-1] - val_scores[-1]
        max_gap = max([t - v for t, v in zip(train_scores, val_scores)])
        
        return {
            'train_scores': train_scores,
            'val_scores': val_scores,
            'final_gap': final_gap,
            'max_gap': max_gap,
            'is_overfitting': final_gap > 0.1 or max_gap > 0.15
        }
    
    @staticmethod
    def check_cv_consistency(model, X, y, cv_folds: int = 5) -> Dict[str, any]:
        """교차검증 일관성 검사"""
        from sklearn.model_selection import cross_val_score
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='f1')
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_range': cv_scores.max() - cv_scores.min(),
            'is_stable': cv_scores.std() < 0.05
        }
