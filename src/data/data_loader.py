import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """데이터 로딩을 담당하는 클래스"""
    
    def __init__(self, data_path: str = "E Commerce Dataset.xlsx"):
        """
        Args:
            data_path (str): 데이터 파일 경로
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.data_dict = None
        
    def load_data(self, sheet_name: str = 'E Comm') -> pd.DataFrame:
        """
        Excel 파일에서 데이터를 로드합니다.
        
        Args:
            sheet_name (str): 로드할 시트 이름
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        try:
            self.raw_data = pd.read_excel(self.data_path, sheet_name=sheet_name)
            print(f"✅ 데이터 로드 성공: {self.raw_data.shape[0]:,}행 × {self.raw_data.shape[1]}열")
            return self.raw_data
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None
    
    def load_data_dict(self) -> pd.DataFrame:
        """
        데이터 딕셔너리를 로드합니다.
        
        Returns:
            pd.DataFrame: 데이터 딕셔너리
        """
        try:
            self.data_dict = pd.read_excel(self.data_path, sheet_name='Data Dict')
            print(f"✅ 데이터 딕셔너리 로드 성공: {self.data_dict.shape[0]}행")
            return self.data_dict
        except Exception as e:
            print(f"❌ 데이터 딕셔너리 로드 실패: {e}")
            return None
    
    def get_sheet_info(self) -> dict:
        """
        Excel 파일의 시트 정보를 반환합니다.
        
        Returns:
            dict: 시트 정보
        """
        try:
            excel_file = pd.ExcelFile(self.data_path)
            sheet_info = {}
            
            for sheet_name in excel_file.sheet_names:
                df_temp = pd.read_excel(self.data_path, sheet_name=sheet_name, nrows=5)
                sheet_info[sheet_name] = {
                    'shape': df_temp.shape,
                    'columns': list(df_temp.columns),
                    'sample_data': df_temp.head(3)
                }
            
            return sheet_info
        except Exception as e:
            print(f"❌ 시트 정보 조회 실패: {e}")
            return {}
    
    def get_basic_info(self) -> dict:
        """
        데이터의 기본 정보를 반환합니다.
        
        Returns:
            dict: 기본 정보
        """
        if self.raw_data is None:
            print("❌ 데이터가 로드되지 않았습니다.")
            return {}
        
        info = {
            'shape': self.raw_data.shape,
            'dtypes': self.raw_data.dtypes.to_dict(),
            'memory_usage': self.raw_data.memory_usage(deep=True).sum() / 1024**2,
            'missing_info': self.raw_data.isnull().sum().to_dict(),
            'missing_percent': (self.raw_data.isnull().sum() / len(self.raw_data) * 100).to_dict()
        }
        
        return info
