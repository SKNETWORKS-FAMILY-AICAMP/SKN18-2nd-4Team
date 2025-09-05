#!/usr/bin/env python3
"""
모델 성능 결과 확인 스크립트
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

def check_model_performance():
    """모델 성능 결과 확인"""
    logger.info("📊 모델 성능 결과 확인 시작")
    
    try:
        # 저장된 모델 결과 로드
        model_results = joblib.load("outputs/model_results.pkl")
        
        print("🏆 모델 성능 결과")
        print("=" * 60)
        
        # 모델 점수 확인
        if 'model_scores' in model_results:
            model_scores = model_results['model_scores']
            best_model_name = model_results.get('best_model_name', 'Unknown')
            
            print(f"🥇 최고 성능 모델: {best_model_name}")
            print(f"📈 최고 점수: {model_scores[best_model_name]:.4f}")
            print("\n📊 전체 모델 성능 순위:")
            print("-" * 60)
            
            # 점수 순으로 정렬
            sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (model_name, score) in enumerate(sorted_scores, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                print(f"{medal} {i:2d}. {model_name:<20} : {score:.4f}")
        
        # 최종 평가 결과 확인
        if 'final_results' in model_results:
            final_results = model_results['final_results']
            print(f"\n📋 최종 평가 결과 ({best_model_name}):")
            print("-" * 60)
            print(f"Accuracy  : {final_results.get('accuracy', 0):.4f}")
            print(f"Precision : {final_results.get('precision', 0):.4f}")
            print(f"Recall    : {final_results.get('recall', 0):.4f}")
            print(f"F1-Score  : {final_results.get('f1', 0):.4f}")
            print(f"AUC       : {final_results.get('auc', 0):.4f}")
        
        # 복합 점수 가중치 확인
        print(f"\n⚖️ 복합 점수 가중치:")
        print("-" * 60)
        print("AUC      : 40%")
        print("F1-Score : 30%")
        print("Precision: 20%")
        print("Recall   : 10%")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"모델 성능 확인 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_performance()
