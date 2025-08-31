import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataExplorer:
    """데이터 탐색을 담당하는 클래스"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): 탐색할 데이터
        """
        self.data = data
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def get_feature_summary(self) -> pd.DataFrame:
        """
        모든 피쳐의 요약 정보를 반환합니다.
        
        Returns:
            pd.DataFrame: 피쳐 요약 테이블
        """
        summary_data = []
        
        for col in self.data.columns:
            dtype = str(self.data[col].dtype)
            unique_count = self.data[col].nunique()
            missing_count = self.data[col].isnull().sum()
            missing_percent = (missing_count / len(self.data)) * 100
            
            # 데이터 타입별 추가 정보
            if col in self.numeric_cols:
                data_type = 'Numeric'
                if unique_count <= 20:  # 범주형으로 간주할 수 있는 수치형
                    data_type = 'Numeric (Categorical)'
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                mean_val = self.data[col].mean()
                additional_info = f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}"
            else:
                data_type = 'Categorical'
                most_common = self.data[col].mode().iloc[0] if not self.data[col].mode().empty else "N/A"
                additional_info = f"Most Common: {most_common}"
            
            summary_data.append({
                'Feature': col,
                'Data Type': data_type,
                'Dtype': dtype,
                'Unique Values': unique_count,
                'Missing Count': missing_count,
                'Missing %': f"{missing_percent:.1f}%",
                'Additional Info': additional_info
            })
        
        return pd.DataFrame(summary_data)
    
    def get_unique_values_analysis(self, max_display: int = 10) -> Dict[str, pd.DataFrame]:
        """
        각 피쳐의 unique 값 분석을 반환합니다.
        
        Args:
            max_display (int): 표시할 최대 unique 값 개수
            
        Returns:
            Dict[str, pd.DataFrame]: 피쳐별 unique 값 분석 결과
        """
        results = {}
        
        for col in self.data.columns:
            unique_count = self.data[col].nunique()
            value_counts = self.data[col].value_counts()
            
            if unique_count <= max_display:
                # 모든 값 표시
                analysis_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(self.data) * 100).round(2)
                })
            else:
                # 상위 값들만 표시
                top_values = value_counts.head(max_display)
                analysis_df = pd.DataFrame({
                    'Value': list(top_values.index) + [f'... 기타 {unique_count - max_display}개 값'],
                    'Count': list(top_values.values) + [value_counts.iloc[max_display:].sum()],
                    'Percentage': list((top_values.values / len(self.data) * 100).round(2)) + 
                                [((value_counts.iloc[max_display:].sum() / len(self.data)) * 100).round(2)]
                })
            
            results[col] = analysis_df
        
        return results
    
    def get_numeric_statistics(self) -> pd.DataFrame:
        """
        수치형 피쳐들의 기술통계를 반환합니다.
        
        Returns:
            pd.DataFrame: 기술통계 테이블
        """
        if not self.numeric_cols:
            return pd.DataFrame()
        
        stats = self.data[self.numeric_cols].describe()
        
        # 추가 통계 계산
        additional_stats = pd.DataFrame({
            'Skewness': self.data[self.numeric_cols].skew(),
            'Kurtosis': self.data[self.numeric_cols].kurtosis(),
            'Missing Count': self.data[self.numeric_cols].isnull().sum(),
            'Missing %': (self.data[self.numeric_cols].isnull().sum() / len(self.data) * 100).round(2)
        })
        
        # 통계 테이블 결합
        combined_stats = pd.concat([stats.T, additional_stats], axis=1)
        
        return combined_stats
    
    def get_categorical_statistics(self) -> pd.DataFrame:
        """
        범주형 피쳐들의 통계를 반환합니다.
        
        Returns:
            pd.DataFrame: 범주형 통계 테이블
        """
        if not self.categorical_cols:
            return pd.DataFrame()
        
        cat_stats = []
        
        for col in self.categorical_cols:
            value_counts = self.data[col].value_counts()
            
            cat_stats.append({
                'Feature': col,
                'Unique Count': len(value_counts),
                'Most Common': value_counts.index[0],
                'Most Common Count': value_counts.iloc[0],
                'Most Common %': (value_counts.iloc[0] / len(self.data) * 100).round(2),
                'Missing Count': self.data[col].isnull().sum(),
                'Missing %': (self.data[col].isnull().sum() / len(self.data) * 100).round(2)
            })
        
        return pd.DataFrame(cat_stats)
    
    def get_correlation_analysis(self, target_col: str = 'Churn') -> pd.DataFrame:
        """
        수치형 피쳐들과 타겟 변수 간의 상관관계를 분석합니다.
        
        Args:
            target_col (str): 타겟 변수명
            
        Returns:
            pd.DataFrame: 상관관계 분석 결과
        """
        if target_col not in self.data.columns:
            print(f"❌ 타겟 변수 '{target_col}'를 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 수치형 컬럼만 선택 (타겟 변수 포함)
        numeric_data = self.data[self.numeric_cols + [target_col]]
        
        # 결측값이 있는 컬럼 제외
        numeric_data = numeric_data.dropna()
        
        if len(numeric_data) == 0:
            print("❌ 결측값이 너무 많아 상관관계 분석을 수행할 수 없습니다.")
            return pd.DataFrame()
        
        # 상관관계 계산
        corr_matrix = numeric_data.corr()
        
        # 타겟 변수와의 상관관계만 추출
        target_corr = corr_matrix[target_col].sort_values(ascending=False)
        
        # 상관관계 강도 분류
        def classify_correlation(corr):
            if abs(corr) >= 0.7:
                return "Very Strong"
            elif abs(corr) >= 0.5:
                return "Strong"
            elif abs(corr) >= 0.3:
                return "Moderate"
            elif abs(corr) >= 0.1:
                return "Weak"
            else:
                return "Very Weak"
        
        correlation_df = pd.DataFrame({
            'Feature': target_corr.index,
            'Correlation': target_corr.values,
            'Abs_Correlation': abs(target_corr.values),
            'Strength': [classify_correlation(corr) for corr in target_corr.values]
        })
        
        return correlation_df
    
    def get_churn_analysis(self, target_col: str = 'Churn') -> Dict[str, pd.DataFrame]:
        """
        이탈(Churn) 관련 분석을 수행합니다.
        
        Args:
            target_col (str): 타겟 변수명
            
        Returns:
            Dict[str, pd.DataFrame]: 이탈 분석 결과
        """
        if target_col not in self.data.columns:
            print(f"❌ 타겟 변수 '{target_col}'를 찾을 수 없습니다.")
            return {}
        
        results = {}
        
        # 전체 이탈률
        total_churn_rate = self.data[target_col].mean() * 100
        churn_counts = self.data[target_col].value_counts()
        
        results['overall_churn'] = pd.DataFrame({
            'Metric': ['Total Customers', 'Churned Customers', 'Retained Customers', 'Churn Rate'],
            'Value': [
                len(self.data),
                churn_counts.get(1, 0),
                churn_counts.get(0, 0),
                f"{total_churn_rate:.1f}%"
            ]
        })
        
        # 범주형 변수별 이탈률
        categorical_churn = {}
        for col in self.categorical_cols:
            if col != target_col:
                churn_by_cat = self.data.groupby(col)[target_col].agg(['count', 'sum', 'mean'])
                churn_by_cat.columns = ['Total', 'Churned', 'Churn_Rate']
                churn_by_cat['Churn_Rate_Percent'] = (churn_by_cat['Churn_Rate'] * 100).round(2)
                categorical_churn[col] = churn_by_cat
        
        results['categorical_churn'] = categorical_churn
        
        return results
