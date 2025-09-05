#!/usr/bin/env python3
"""
모델 성능 결과를 CSV로 저장하는 스크립트
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

def save_model_results():
    """모델 성능 결과를 CSV로 저장"""
    logger.info("💾 모델 성능 결과 저장 시작")
    
    try:
        # 저장된 모델 결과 로드
        model_results = joblib.load("outputs/model_results.pkl")
        
        # 모델 점수 데이터프레임 생성
        if 'model_scores' in model_results:
            model_scores = model_results['model_scores']
            best_model_name = model_results.get('best_model_name', 'Unknown')
            
            # 데이터프레임 생성
            results_data = []
            for model_name, score in model_scores.items():
                is_best = "Yes" if model_name == best_model_name else "No"
                results_data.append({
                    'Model': model_name,
                    'Composite_Score': score,
                    'Rank': 0,  # 나중에 설정
                    'Is_Best': is_best
                })
            
            # 점수 순으로 정렬하고 순위 설정
            results_df = pd.DataFrame(results_data)
            results_df = results_df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
            results_df['Rank'] = range(1, len(results_df) + 1)
            
            # CSV 저장
            output_path = Path("outputs/model_performance_results.csv")
            results_df.to_csv(output_path, index=False)
            
            print("📊 모델 성능 결과:")
            print(results_df.to_string(index=False))
            print(f"\n💾 결과가 {output_path}에 저장되었습니다.")
            
        else:
            print("❌ 모델 점수 데이터가 없습니다!")
            
    except Exception as e:
        logger.error(f"모델 결과 저장 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    save_model_results()
