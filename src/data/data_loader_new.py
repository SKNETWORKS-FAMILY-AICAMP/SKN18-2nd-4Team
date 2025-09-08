"""
새로운 데이터용 데이터 로더
train/valid/test/pred가 이미 분리된 데이터 처리
"""

import pandas as pd
from pathlib import Path
from src.utils.config import Config

class DataLoaderNew:
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = self.config.data_curated_dir
        self.train_csv_name = self.config.train_csv_name
        self.valid_csv_name = self.config.valid_csv_name
        self.test_csv_name = self.config.test_csv_name
        self.pred_csv_name = self.config.pred_csv_name

    def load_train_data(self) -> pd.DataFrame:
        """훈련 데이터 로드"""
        dataset_path = self.data_dir / self.train_csv_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Train dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path, low_memory=True)
        print(f"📊 Train data loaded from {dataset_path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def load_valid_data(self) -> pd.DataFrame:
        """검증 데이터 로드"""
        dataset_path = self.data_dir / self.valid_csv_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Valid dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path, low_memory=True)
        print(f"📊 Valid data loaded from {dataset_path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def load_test_data(self) -> pd.DataFrame:
        """테스트 데이터 로드"""
        dataset_path = self.data_dir / self.test_csv_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path, low_memory=True)
        print(f"📊 Test data loaded from {dataset_path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def load_pred_data(self) -> pd.DataFrame:
        """예측 데이터 로드"""
        dataset_path = self.data_dir / self.pred_csv_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Prediction dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path, low_memory=True)
        print(f"📊 Pred data loaded from {dataset_path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def prepare_model_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """모델링용 데이터 준비"""
        df_model = df.copy()
        target_col = self.config.target_column

        # 타겟 컬럼 확인 및 변환 (pred_df는 target이 없을 수 있음)
        if target_col in df_model.columns:
            df_model[target_col] = pd.to_numeric(df_model[target_col], errors='coerce').fillna(0).astype(int)
            print(f"✅ Target column '{target_col}' set. Positive ratio: {df_model[target_col].mean()*100:.1f}%")
        else:
            print(f"ℹ️ Target column '{target_col}' not found (prediction data)")
        
        # 두 컬럼 모두 유지 (player_highest_market_value_in_eur, market_value_in_eur)
        # 컬럼명 변환 없이 원본 컬럼명 유지
        
        return df_model

    def load_all_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """모든 데이터 로드 (train + valid + test + pred)"""
        train_df = self.load_train_data()
        valid_df = self.load_valid_data()
        test_df = self.load_test_data()
        pred_df = self.load_pred_data()
        
        # 데이터 준비
        train_df = self.prepare_model_data(train_df)
        valid_df = self.prepare_model_data(valid_df)
        test_df = self.prepare_model_data(test_df)
        pred_df = self.prepare_model_data(pred_df) # Now applies to pred_df
        
        return train_df, valid_df, test_df, pred_df