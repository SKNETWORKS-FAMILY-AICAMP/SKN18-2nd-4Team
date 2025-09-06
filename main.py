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
                       choices=['train', 'tune', 'regularize', 'ensemble', 'all'],
                       help='실행 모드 선택')
    parser.add_argument('--force-retrain', action='store_true',
                       help='강제로 모델 재학습 (기존 개선된 모델 무시)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 기본 모델링 (기존 개선된 모델 재사용 가능)
        from scripts.run_final_modeling import run_final_modeling
        run_final_modeling(force_retrain=args.force_retrain)
        
        # 성능 기록 자동 업데이트
        print("📊 성능 기록 업데이트...")
        print("📊 성능 기록 업데이트는 수동으로 진행됩니다.")
        
    elif args.mode == 'tune':
        # 하이퍼파라미터 튜닝
        from scripts.hyperparameter_tuning import hyperparameter_tuning
        hyperparameter_tuning()
        
        # 성능 기록 자동 업데이트
        print("📊 성능 기록 업데이트...")
        print("📊 성능 기록 업데이트는 수동으로 진행됩니다.")
        
    elif args.mode == 'regularize':
        # 정규화 강화
        from scripts.regularization_improvement import regularization_improvement
        regularization_improvement()
        
        # 성능 기록 자동 업데이트
        print("📊 성능 기록 업데이트...")
        print("📊 성능 기록 업데이트는 수동으로 진행됩니다.")
        
    elif args.mode == 'ensemble':
        # 앙상블 모델
        from scripts.ensemble_modeling import ensemble_modeling
        ensemble_modeling()
        
        # 성능 기록 자동 업데이트
        print("📊 성능 기록 업데이트...")
        print("📊 성능 기록 업데이트는 수동으로 진행됩니다.")
        
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
        
        # 5. 최종 예측 결과 업데이트 (개선된 모델로)
        print("📊 5단계: 최종 예측 결과 업데이트")
        from scripts.run_final_modeling import run_final_modeling
        run_final_modeling()  # 개선된 최종 모델로 예측 재실행
        
        print("✅ 전체 파이프라인 완료!")
        print("🎉 모든 개선 기법이 적용된 최고 성능 모델로 최종 결과가 저장되었습니다!")

if __name__ == "__main__":
    main()
