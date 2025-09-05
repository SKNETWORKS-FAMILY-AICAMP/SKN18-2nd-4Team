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
