"""
Configuration management for Football Transfer Prediction
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List

class Config:
    """설정 관리 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"설정 파일 파싱 오류: {e}")
    
    @property
    def data_config(self) -> Dict[str, str]:
        """데이터 설정"""
        return self.config.get('data', {})
    
    @property
    def features_config(self) -> Dict[str, Any]:
        """피처 설정"""
        return self.config.get('features', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """모델 설정"""
        return self.config.get('model', {})
    
    @property
    def evaluation_config(self) -> Dict[str, Any]:
        """평가 설정"""
        return self.config.get('evaluation', {})
    
    @property
    def shap_config(self) -> Dict[str, Any]:
        """SHAP 설정"""
        return self.config.get('shap', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    # 데이터 관련 속성들
    @property
    def data_raw_dir(self): return Path(self.get('data.raw_dir'))
    @property
    def data_curated_dir(self): return Path(self.get('data.curated_dir'))
    @property
    def data_processed_dir(self): return Path(self.get('data.processed_dir'))
    @property
    def target_csv_name(self): return self.get('data.target_csv_name')
    @property
    def train_csv_name(self): return self.get('data.train_csv_name')
    @property
    def test_csv_name(self): return self.get('data.test_csv_name')
    @property
    def valid_csv_name(self): return self.get('data.valid_csv_name')
    @property
    def pred_csv_name(self): return self.get('data.pred_csv_name')
    @property
    def target_column(self): return self.get('data.target_column')
    
    # 모델 관련 속성들
    @property
    def model_random_state(self): return self.get('model.random_state')
    @property
    def model_test_season(self): return self.get('model.test_season')
    @property
    def model_prediction_season(self): return self.get('model.prediction_season')
    @property
    def model_composite_weights(self): return self.get('model.composite_weights')
    @property
    def output_model_path(self): return Path(self.get('model.output_model_path'))
    @property
    def output_preprocessor_path(self): return Path(self.get('model.output_preprocessor_path'))
    
    # 피처 관련 속성들
    @property
    def features_ordinal(self): return self.get('features.ordinal')
    @property
    def features_nominal(self): return self.get('features.nominal')
    @property
    def features_numeric_exclude(self): return self.get('features.numeric_exclude')
    
    # 출력 관련 속성들
    @property
    def output_dir(self): return Path(self.get('output.output_dir'))
    @property
    def output_predictions_csv(self): return Path(self.get('output.predictions_csv'))
    @property
    def output_model_comparison_plot(self): return Path(self.get('output.model_comparison_plot'))
    @property
    def output_shap_summary_plot(self): return Path(self.get('output.shap_summary_plot'))
    @property
    def output_feature_importance_plot(self): return Path(self.get('output.feature_importance_plot'))
    @property
    def output_prediction_distribution_plot(self): return Path(self.get('output.prediction_distribution_plot'))
