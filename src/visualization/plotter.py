import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class DataPlotter:
    """데이터 시각화를 담당하는 클래스"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): 시각화할 데이터
        """
        self.data = data
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def plot_feature_summary(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        피쳐 요약 정보를 시각화합니다.
        
        Args:
            figsize (Tuple[int, int]): 그래프 크기
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('데이터 피쳐 요약', fontsize=16, fontweight='bold')
        
        # 1. 데이터 타입 분포
        data_types = []
        for col in self.data.columns:
            if col in self.numeric_cols:
                if self.data[col].nunique() <= 20:
                    data_types.append('Numeric (Categorical)')
                else:
                    data_types.append('Numeric')
            else:
                data_types.append('Categorical')
        
        type_counts = pd.Series(data_types).value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('데이터 타입 분포')
        
        # 2. 결측값 분포
        missing_counts = self.data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            axes[0, 1].bar(range(len(missing_counts)), missing_counts.values)
            axes[0, 1].set_title('결측값 개수')
            axes[0, 1].set_xlabel('피쳐')
            axes[0, 1].set_ylabel('결측값 개수')
            axes[0, 1].set_xticks(range(len(missing_counts)))
            axes[0, 1].set_xticklabels(missing_counts.index, rotation=45, ha='right')
        else:
            axes[0, 1].text(0.5, 0.5, '결측값 없음', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title('결측값 현황')
        
        # 3. 수치형 피쳐 분포 (히스토그램)
        if len(self.numeric_cols) > 0:
            for i, col in enumerate(self.numeric_cols[:4]):  # 최대 4개만
                row, col_idx = i // 2, i % 2
                if row == 1:  # 두 번째 행
                    axes[row, col_idx].hist(self.data[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
                    axes[row, col_idx].set_title(f'{col} 분포')
                    axes[row, col_idx].set_xlabel(col)
                    axes[row, col_idx].set_ylabel('빈도')
        
        # 4. 범주형 피쳐 분포 (막대그래프)
        if len(self.categorical_cols) > 0:
            col = self.categorical_cols[0]
            value_counts = self.data[col].value_counts().head(10)
            axes[1, 1].bar(range(len(value_counts)), value_counts.values)
            axes[1, 1].set_title(f'{col} 상위 10개 값')
            axes[1, 1].set_xlabel('값')
            axes[1, 1].set_ylabel('개수')
            axes[1, 1].set_xticks(range(len(value_counts)))
            axes[1, 1].set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, target_col: str = 'Churn', figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        상관관계 히트맵을 그립니다.
        
        Args:
            target_col (str): 타겟 변수명
            figsize (Tuple[int, int]): 그래프 크기
        """
        if target_col not in self.data.columns:
            print(f"❌ 타겟 변수 '{target_col}'를 찾을 수 없습니다.")
            return
        
        # 수치형 컬럼만 선택
        numeric_data = self.data[self.numeric_cols + [target_col]]
        numeric_data = numeric_data.dropna()
        
        if len(numeric_data) == 0:
            print("❌ 결측값이 너무 많아 상관관계 분석을 수행할 수 없습니다.")
            return
        
        # 상관관계 계산
        corr_matrix = numeric_data.corr()
        
        # 히트맵 그리기
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('수치형 피쳐 상관관계 히트맵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_churn_analysis(self, target_col: str = 'Churn', figsize: Tuple[int, int] = (15, 12)) -> None:
        """
        이탈 분석을 시각화합니다.
        
        Args:
            target_col (str): 타겟 변수명
            figsize (Tuple[int, int]): 그래프 크기
        """
        if target_col not in self.data.columns:
            print(f"❌ 타겟 변수 '{target_col}'를 찾을 수 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('이탈(Churn) 분석', fontsize=16, fontweight='bold')
        
        # 1. 전체 이탈률
        churn_counts = self.data[target_col].value_counts()
        axes[0, 0].pie(churn_counts.values, labels=['유지', '이탈'], autopct='%1.1f%%')
        axes[0, 0].set_title('전체 이탈률')
        
        # 2. 성별별 이탈률
        if 'Gender' in self.data.columns:
            gender_churn = pd.crosstab(self.data['Gender'], self.data[target_col])
            gender_churn.plot(kind='bar', ax=axes[0, 1], color=['lightblue', 'lightcoral'])
            axes[0, 1].set_title('성별별 이탈률')
            axes[0, 1].set_xlabel('성별')
            axes[0, 1].set_ylabel('고객 수')
            axes[0, 1].legend(['유지', '이탈'])
        
        # 3. 결혼상태별 이탈률
        if 'MaritalStatus' in self.data.columns:
            marital_churn = pd.crosstab(self.data['MaritalStatus'], self.data[target_col])
            marital_churn.plot(kind='bar', ax=axes[0, 2], color=['lightgreen', 'lightcoral'])
            axes[0, 2].set_title('결혼상태별 이탈률')
            axes[0, 2].set_xlabel('결혼상태')
            axes[0, 2].set_ylabel('고객 수')
            axes[0, 2].legend(['유지', '이탈'])
        
        # 4. 도시 등급별 이탈률
        if 'CityTier' in self.data.columns:
            city_churn = pd.crosstab(self.data['CityTier'], self.data[target_col])
            city_churn.plot(kind='bar', ax=axes[1, 0], color=['lightyellow', 'lightcoral'])
            axes[1, 0].set_title('도시 등급별 이탈률')
            axes[1, 0].set_xlabel('도시 등급')
            axes[1, 0].set_ylabel('고객 수')
            axes[1, 0].legend(['유지', '이탈'])
        
        # 5. 만족도별 이탈률
        if 'SatisfactionScore' in self.data.columns:
            satisfaction_churn = pd.crosstab(self.data['SatisfactionScore'], self.data[target_col])
            satisfaction_churn.plot(kind='bar', ax=axes[1, 1], color=['lightpink', 'lightcoral'])
            axes[1, 1].set_title('만족도별 이탈률')
            axes[1, 1].set_xlabel('만족도 점수')
            axes[1, 1].set_ylabel('고객 수')
            axes[1, 1].legend(['유지', '이탈'])
        
        # 6. 주문 횟수별 이탈률
        if 'OrderCount' in self.data.columns:
            order_churn = pd.crosstab(self.data['OrderCount'], self.data[target_col])
            order_churn.plot(kind='bar', ax=axes[1, 2], color=['lightcyan', 'lightcoral'])
            axes[1, 2].set_title('주문 횟수별 이탈률')
            axes[1, 2].set_xlabel('주문 횟수')
            axes[1, 2].set_ylabel('고객 수')
            axes[1, 2].legend(['유지', '이탈'])
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distributions(self, figsize: Tuple[int, int] = (18, 15)) -> None:
        """
        주요 피쳐들의 분포를 시각화합니다.
        
        Args:
            figsize (Tuple[int, int]): 그래프 크기
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('주요 피쳐 분포 분석', fontsize=16, fontweight='bold')
        
        # 수치형 피쳐들
        numeric_features = ['Tenure', 'SatisfactionScore', 'HourSpendOnApp', 
                          'OrderCount', 'CashbackAmount', 'NumberOfAddress']
        
        for i, feature in enumerate(numeric_features):
            if feature in self.data.columns:
                row, col = i // 3, i % 3
                
                if feature in ['Tenure', 'CashbackAmount']:
                    # 히스토그램
                    axes[row, col].hist(self.data[feature].dropna(), bins=20, alpha=0.7, edgecolor='black')
                    axes[row, col].set_title(f'{feature} 분포')
                    axes[row, col].set_xlabel(feature)
                    axes[row, col].set_ylabel('빈도')
                else:
                    # 막대그래프
                    value_counts = self.data[feature].value_counts().sort_index()
                    axes[row, col].bar(value_counts.index, value_counts.values)
                    axes[row, col].set_title(f'{feature} 분포')
                    axes[row, col].set_xlabel(feature)
                    axes[row, col].set_ylabel('고객 수')
        
        # 범주형 피쳐들
        categorical_features = ['PreferredPaymentMode', 'PreferedOrderCat', 'Gender']
        
        for i, feature in enumerate(categorical_features):
            if feature in self.data.columns:
                row, col = (i + 6) // 3, (i + 6) % 3
                value_counts = self.data[feature].value_counts()
                axes[row, col].bar(range(len(value_counts)), value_counts.values)
                axes[row, col].set_title(f'{feature} 분포')
                axes[row, col].set_xlabel('값')
                axes[row, col].set_ylabel('고객 수')
                axes[row, col].set_xticks(range(len(value_counts)))
                axes[row, col].set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
