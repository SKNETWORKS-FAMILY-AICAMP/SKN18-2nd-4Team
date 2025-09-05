"""
새로운 데이터용 데이터 로더
train/test가 이미 분리된 데이터 처리
"""

import pandas as pd
from pathlib import Path
from src.utils.config import Config

class DataLoaderNew:
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = self.config.data_curated_dir
        self.train_csv_name = self.config.train_csv_name
        self.test_csv_name = self.config.test_csv_name

    def load_train_data(self) -> pd.DataFrame:
        """훈련 데이터 로드"""
        dataset_path = self.data_dir / self.train_csv_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Train dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path, low_memory=True)
        print(f"📊 Train data loaded from {dataset_path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def load_test_data(self) -> pd.DataFrame:
        """테스트 데이터 로드"""
        dataset_path = self.data_dir / self.test_csv_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path, low_memory=True)
        print(f"📊 Test data loaded from {dataset_path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def prepare_model_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """모델링용 데이터 준비"""
        df_model = df.copy()
        target_col = self.config.target_column

        # 타겟 컬럼 확인 및 변환
        if target_col not in df_model.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        df_model[target_col] = pd.to_numeric(df_model[target_col], errors='coerce').fillna(0).astype(int)
        print(f"✅ Target column '{target_col}' set. Positive ratio: {df_model[target_col].mean()*100:.1f}%")
        
        # 컬럼명 통일 (market_value_in_eur -> player_market_value_in_eur)
        if 'market_value_in_eur' in df_model.columns:
            df_model['player_market_value_in_eur'] = df_model['market_value_in_eur']
            df_model = df_model.drop(columns=['market_value_in_eur'])
            print("✅ Column name unified: market_value_in_eur -> player_market_value_in_eur")
        
        return df_model

    def load_all_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """모든 데이터 로드 (train + test)"""
        train_df = self.load_train_data()
        test_df = self.load_test_data()
        
        # 데이터 준비
        train_df = self.prepare_model_data(train_df)
        test_df = self.prepare_model_data(test_df)
        
        return train_df, test_df
