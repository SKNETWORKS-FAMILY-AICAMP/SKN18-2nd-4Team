#!/usr/bin/env python3
"""
Football Transfer Prediction - Main Entry Point
축구 선수 이적 예측 프로젝트 메인 실행 파일
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Football Transfer Prediction')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'tune', 'regularize', 'ensemble', 'all'],
                       help='실행 모드 선택')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 기본 모델링
        from scripts.run_final_modeling import run_final_modeling
        run_final_modeling()
        
    elif args.mode == 'predict':
        # 예측만 실행 (기존 모델 사용)
        print("예측 모드는 기본 모델링에 포함되어 있습니다.")
        print("python main.py --mode train 을 실행하세요.")
        
    elif args.mode == 'tune':
        # 하이퍼파라미터 튜닝
        from scripts.hyperparameter_tuning import hyperparameter_tuning
        hyperparameter_tuning()
        
    elif args.mode == 'regularize':
        # 정규화 강화
        from scripts.regularization_improvement import regularization_improvement
        regularization_improvement()
        
    elif args.mode == 'ensemble':
        # 앙상블 모델
        from scripts.ensemble_modeling import ensemble_modeling
        ensemble_modeling()
        
    elif args.mode == 'all':
        # 전체 파이프라인
        print("🚀 전체 파이프라인 실행 시작")
        
        # 1. 기본 모델링
        print("🤖 1단계: 기본 모델링")
        from scripts.run_final_modeling import run_final_modeling
        run_final_modeling()
        
        # 2. 하이퍼파라미터 튜닝
        print("🔧 2단계: 하이퍼파라미터 튜닝")
        from scripts.hyperparameter_tuning import hyperparameter_tuning
        hyperparameter_tuning()
        
        # 3. 정규화 강화
        print("🔧 3단계: 정규화 강화")
        from scripts.regularization_improvement import regularization_improvement
        regularization_improvement()
        
        # 4. 앙상블 모델
        print("🤝 4단계: 앙상블 모델")
        from scripts.ensemble_modeling import ensemble_modeling
        ensemble_modeling()
        
        print("✅ 전체 파이프라인 완료!")

if __name__ == "__main__":
    main()
