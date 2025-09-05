#!/usr/bin/env python3
"""
SHAP 피처명 디버그 스크립트
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_shap_features():
    """SHAP 피처명 디버그"""
    logger.info("🔍 SHAP 피처명 디버그 시작")
    
    try:
        # 저장된 모델 결과 로드
        model_results = joblib.load("outputs/model_results.pkl")
        
        if 'shap_results' in model_results and model_results['shap_results']:
            shap_results = model_results['shap_results']
            
            print("📊 SHAP 결과 정보:")
            print(f"- SHAP values shape: {np.array(shap_results['shap_values']).shape}")
            print(f"- X_test_processed shape: {shap_results['X_test_processed'].shape}")
            print(f"- Feature names count: {len(shap_results.get('feature_names', []))}")
            
            if 'feature_names' in shap_results:
                print("\n🏷️ 피처명 목록 (처음 20개):")
                for i, name in enumerate(shap_results['feature_names'][:20]):
                    print(f"  {i}: {name}")
                
                print(f"\n📈 총 피처 수: {len(shap_results['feature_names'])}")
            else:
                print("❌ 피처명이 없습니다!")
                
        else:
            print("❌ SHAP 결과가 없습니다!")
            
    except Exception as e:
        logger.error(f"디버그 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_shap_features()
