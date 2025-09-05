#!/usr/bin/env python3
"""
SHAP 이미지만 다시 생성하는 스크립트
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

def regenerate_shap():
    """SHAP 이미지만 다시 생성"""
    logger.info("🔄 SHAP 이미지 재생성 시작")
    
    try:
        # 저장된 모델 결과 로드
        model_results = joblib.load("outputs/model_results.pkl")
        
        # 시각화 클래스 생성
        from src.visualization.plotter import ModelVisualizer
        output_dir = Path("outputs")
        visualizer = ModelVisualizer(model_results, output_dir)
        
        # SHAP 분석만 다시 실행
        if 'shap_results' in model_results and model_results['shap_results']:
            logger.info("📊 SHAP 그래프 재생성 중...")
            visualizer.plot_shap_analysis()
            logger.info("✅ SHAP 그래프 재생성 완료")
        else:
            logger.warning("SHAP 결과가 없습니다. 전체 모델링을 다시 실행해주세요.")
            
    except Exception as e:
        logger.error(f"SHAP 재생성 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    regenerate_shap()
