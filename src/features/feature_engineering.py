"""
Football Transfer Prediction - Feature Engineering Module
피쳐 엔지니어링 및 데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


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
                    unique_labels = set(X.iloc[:, i].astype(str))
                    known_labels = set(self.label_encoders[i].classes_)
                    new_labels = unique_labels - known_labels
                    if new_labels:
                        print(f"⚠️ 새로운 라벨 발견: {new_labels}, -1로 처리합니다.")
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
                        print(f"⚠️ 새로운 라벨 발견: {new_labels}, -1로 처리합니다.")
                    X_encoded[:, i] = np.array([
                        self.label_encoders[i].transform([x])[0] if x in self.label_encoders[i].classes_ else -1
                        for x in X[:, i].astype(str)
                    ])
            return X_encoded.astype(float)


class FootballFeatureEngineer:
    """Football 데이터 전용 피쳐 엔지니어링 클래스"""
    
    def __init__(self):
        self.feature_config = {
            'ordinal_features': ['season', 'position', 'sub_position'],
            'nominal_features': ['club_name', 'country_of_birth', 'foot'],
            'target_col': 'transfer'
        }
        self.numeric_features = []
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
        for c in df.columns:
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
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 피쳐 생성"""
        df = df.copy()
        
        # 시즌 시작 연도
        season_col = self.detect_season_column(df)
        if season_col is not None:
            df['season_start_year'] = df[season_col].apply(self.season_start_year)
        
        # 나이 계산
        if 'date_of_birth' in df.columns and 'season_start_year' in df.columns:
            by = df['date_of_birth'].astype(str).str.extract(r"^(\d{4})")[0]
            birth_year = pd.to_numeric(by, errors='coerce')
            df['age_at_season'] = (df['season_start_year'] - birth_year).astype('float')
        
        # 시장가치 관련 피쳐
        if 'player_market_value_in_eur' in df.columns:
            df['log_market_value'] = np.log1p(pd.to_numeric(df['player_market_value_in_eur'], errors='coerce'))
        
        if 'player_highest_market_value_in_eur' in df.columns and 'player_market_value_in_eur' in df.columns:
            mv = pd.to_numeric(df['player_market_value_in_eur'], errors='coerce')
            mv_hi = pd.to_numeric(df['player_highest_market_value_in_eur'], errors='coerce')
            df['value_growth'] = (mv_hi - mv)
            df['negotiation_proxy'] = 0.6 * mv + 0.4 * mv_hi
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 피쳐 엔지니어링"""
        df = df.copy()
        
        # 1. 시즌 평균 출전시간 / 클럽 시즌 평균 러닝타임
        if 'season_avg_minutes' in df.columns and 'club_average_age' in df.columns:
            # 클럽별 시즌 평균 러닝타임 계산
            club_running_time = df.groupby(['club_name', 'season'])['season_avg_minutes'].mean().reset_index()
            club_running_time.columns = ['club_name', 'season', 'club_season_avg_minutes']
            df = df.merge(club_running_time, on=['club_name', 'season'], how='left')
            df['minutes_vs_club_avg'] = df['season_avg_minutes'] / (df['club_season_avg_minutes'] + 1e-6)
        
        # 2. 나이 차이 (선수 나이 - 클럽 평균 나이)
        if 'age_at_season' in df.columns and 'club_average_age' in df.columns:
            df['age_difference'] = df['age_at_season'] - df['club_average_age']
            df['age_relative_position'] = df['age_difference'] / (df['club_average_age'] + 1e-6)
        
        # 3. 공격 기여도 vs 팀 성과
        if 'goals' in df.columns and 'assists' in df.columns and 'season_win_count' in df.columns:
            df['attack_contribution'] = df['goals'] + df['assists']
            df['attack_vs_team_success'] = df['attack_contribution'] * df['season_win_count']
            df['attack_efficiency'] = df['attack_contribution'] / (df['season_win_count'] + 1e-6)
        
        # 4. 외국인 선수 여부 및 비율
        if 'country_of_birth' in df.columns and 'club_foreigners_percentage' in df.columns:
            df['is_foreigner'] = (df['country_of_birth'] != 'England').astype(int)
            df['foreigner_vs_club_ratio'] = df['is_foreigner'] * df['club_foreigners_percentage']
            df['is_foreigner_advantage'] = (df['is_foreigner'] == 1) & (df['club_foreigners_percentage'] > 50)
        
        # 5. 포지션별 키 적합성
        if 'position' in df.columns and 'height_in_cm' in df.columns:
            # 포지션별 평균 키 계산
            position_height = df.groupby('position')['height_in_cm'].mean().reset_index()
            position_height.columns = ['position', 'position_avg_height']
            df = df.merge(position_height, on='position', how='left')
            df['height_vs_position'] = df['height_in_cm'] - df['position_avg_height']
            df['height_advantage'] = df['height_vs_position'] / (df['position_avg_height'] + 1e-6)
        
        # 6. 경고장과 출전시간의 관계
        if 'yellow_cards' in df.columns and 'season_avg_minutes' in df.columns:
            df['cards_per_minute'] = df['yellow_cards'] / (df['season_avg_minutes'] + 1e-6)
            df['discipline_score'] = 1 / (df['cards_per_minute'] + 1e-6)
        
        # 7. 클럽 재적 기간 (시즌별)
        if 'season' in df.columns and 'club_name' in df.columns:
            # 선수별 클럽별 첫 시즌 찾기
            player_club_first_season = df.groupby(['player_name', 'club_name'])['season'].min().reset_index()
            player_club_first_season.columns = ['player_name', 'club_name', 'first_season']
            df = df.merge(player_club_first_season, on=['player_name', 'club_name'], how='left')
            
            # 시즌을 숫자로 변환하여 재적 기간 계산
            season_order = ['12/13', '13/14', '14/15', '15/16', '16/17', '17/18', '18/19', '19/20', '20/21', '21/22', '22/23']
            season_to_num = {s: i for i, s in enumerate(season_order)}
            df['season_num'] = df['season'].map(season_to_num)
            df['first_season_num'] = df['first_season'].map(season_to_num)
            df['club_tenure_seasons'] = df['season_num'] - df['first_season_num'] + 1
            df['club_tenure_seasons'] = df['club_tenure_seasons'].fillna(1)  # 첫 시즌은 1
        
        # 8. 포지션별 테이블 순위 (간단한 버전)
        if 'position' in df.columns and 'club_name' in df.columns:
            # 클럽별 포지션별 선수 수 계산
            position_club_count = df.groupby(['position', 'club_name']).size().reset_index(name='position_club_count')
            df = df.merge(position_club_count, on=['position', 'club_name'], how='left')
            df['position_competition'] = df['position_club_count'] - 1  # 경쟁자 수
        
        return df
    
    def get_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """피쳐 타입 분류"""
        # 수치형 변수 자동 탐지
        numeric_features = [
            c for c in df.columns 
            if c not in self.feature_config['ordinal_features'] + 
               self.feature_config['nominal_features'] + 
               [self.feature_config['target_col']]
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].dtype in ['int64', 'float64', 'int32', 'float32']
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
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, feature_types['numeric']),
                ('ord', ordinal_transformer, feature_types['ordinal']),
                ('nom', nominal_transformer, feature_types['nominal'])
            ]
        )
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer, Dict[str, List[str]]]:
        """전체 피쳐 엔지니어링 및 전처리"""
        # 기본 피쳐 생성
        df_processed = self.create_basic_features(df)
        
        # 고급 피쳐 생성
        df_processed = self.create_advanced_features(df_processed)
        
        # 피쳐 타입 분류
        feature_types = self.get_feature_types(df_processed)
        
        # 전처리기 생성
        preprocessor = self.create_preprocessor(feature_types)
        
        # 타겟 제외한 피쳐만 선택
        modeling_features = (feature_types['numeric'] + 
                           feature_types['ordinal'] + 
                           feature_types['nominal'])
        
        X = df_processed[modeling_features]
        
        self.is_fitted = True
        
        return df_processed, preprocessor, feature_types


class DataLeakageChecker:
    """데이터 누수 검사 클래스"""
    
    @staticmethod
    def check_temporal_leakage(df: pd.DataFrame, time_col: str, target_col: str) -> Dict[str, bool]:
        """시간적 데이터 누수 검사"""
        results = {}
        
        # 1. 미래 데이터 포함 여부
        if time_col in df.columns:
            unique_times = sorted(df[time_col].unique())
            results['has_future_data'] = len(unique_times) > 1
            
            # 2. 시간 순서와 타겟 분포의 관계
            time_target = df.groupby(time_col)[target_col].mean()
            results['temporal_consistency'] = len(time_target.unique()) > 1
        
        return results
    
    @staticmethod
    def check_feature_leakage(df: pd.DataFrame, target_col: str) -> Dict[str, List[str]]:
        """피쳐 누수 검사"""
        suspicious_features = []
        
        # 1. 타겟과 완벽한 상관관계
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != target_col:
                corr = abs(df[col].corr(df[target_col]))
                if corr > 0.95:
                    suspicious_features.append(f"{col} (correlation: {corr:.3f})")
        
        # 2. 타겟과 동일한 분포
        for col in df.columns:
            if col != target_col and df[col].nunique() == df[target_col].nunique():
                if set(df[col].unique()) == set(df[target_col].unique()):
                    suspicious_features.append(f"{col} (identical distribution)")
        
        return {'suspicious_features': suspicious_features}
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
        """데이터 품질 검사"""
        results = {}
        
        # 1. 결측치 비율
        missing_ratio = df.isnull().sum() / len(df)
        results['high_missing_features'] = missing_ratio[missing_ratio > 0.5].to_dict()
        
        # 2. 중복 행
        results['duplicate_rows'] = df.duplicated().sum()
        
        # 3. 상수 피쳐
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
