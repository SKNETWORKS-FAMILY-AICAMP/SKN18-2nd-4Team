#!/usr/bin/env python3
"""
완전한 모델링 파이프라인 실행
1. 하이퍼파라미터 튜닝
2. 정규화 개선
3. 앙상블 모델링
4. 최종 결과 생성
"""

import sys
import logging
from pathlib import Path
import subprocess


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """완전한 모델링 파이프라인 실행"""
    
    print("="*80)
    print("🚀 완전한 모델링 파이프라인 시작")
    print("="*80)
    
    try:
        # 1단계: 하이퍼파라미터 튜닝
        print("\n🔧 1단계: 하이퍼파라미터 튜닝")
        logger.info("하이퍼파라미터 튜닝 시작")
        
        from hyperparameter_tuning import hyperparameter_tuning
        hyperparameter_tuning()
        
        print("✅ 하이퍼파라미터 튜닝 완료")
        
        # 성능 기록 업데이트
        print("📊 성능 기록 업데이트...")
        print("📊 성능 기록 업데이트는 수동으로 진행됩니다.")
        
        # 2단계: 정규화 개선
        print("\n🛡️ 2단계: 정규화 개선")
        logger.info("정규화 개선 시작")
        
        from regularization_improvement import regularization_improvement
        regularization_improvement()
        
        print("✅ 정규화 개선 완료")
        
        # 성능 기록 업데이트
        print("📊 성능 기록 업데이트...")
        print("📊 성능 기록 업데이트는 수동으로 진행됩니다.")
        
        # 3단계: 앙상블 모델링
        print("\n🎯 3단계: 앙상블 모델링")
        logger.info("앙상블 모델링 시작")
        
        from ensemble_modeling import ensemble_modeling
        ensemble_modeling()
        
        print("✅ 앙상블 모델링 완료")
        
        # 성능 기록 업데이트
        print("📊 성능 기록 업데이트...")
        print("📊 성능 기록 업데이트는 수동으로 진행됩니다.")
        
        # 4단계: 최종 결과 생성
        print("\n📊 4단계: 최종 결과 생성")
        logger.info("최종 결과 생성 시작")
        
        from run_final_modeling import run_final_modeling
        run_final_modeling()
        
        print("✅ 최종 결과 생성 완료")
        
        # 5단계: 결과 요약
        print("\n📋 5단계: 결과 요약")
        summarize_results()
        
        print("\n" + "="*80)
        print("🎉 완전한 모델링 파이프라인 완료!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"파이프라인 실행 오류: {e}")
        import traceback
        traceback.print_exc()

def summarize_results():
    """결과 요약"""
    import pandas as pd
    import joblib
    from pathlib import Path
    
    output_dir = Path("outputs")
    
    print("📁 생성된 파일들:")
    files_to_check = [
        "hyperparameter_tuning_results.pkl",
        "best_tuned_model.pkl", 
        "tuned_model_performance.csv",
        "regularization_results.pkl",
        "best_regularized_model.pkl",
        "regularized_model_performance.csv",
        "ensemble_results.pkl",
        "best_ensemble_model.pkl",
        "ensemble_model_performance.csv",
        "23_24_transfer_predictions.csv",
        "model.pkl"
    ]
    
    for file in files_to_check:
        file_path = output_dir / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    # 성능 비교
    try:
        print("\n📊 최종 성능 비교:")
        
        # 튜닝 성능
        if (output_dir / "tuned_model_performance.csv").exists():
            tuned_df = pd.read_csv(output_dir / "tuned_model_performance.csv")
            print(f"🔧 튜닝 최고 성능: {tuned_df.iloc[0]['composite_score']:.4f} ({tuned_df.iloc[0]['model']})")
        
        # 정규화 성능
        if (output_dir / "regularized_model_performance.csv").exists():
            reg_df = pd.read_csv(output_dir / "regularized_model_performance.csv")
            print(f"🛡️ 정규화 최고 성능: {reg_df.iloc[0]['composite_score']:.4f} ({reg_df.iloc[0]['model']})")
        
        # 앙상블 성능
        if (output_dir / "ensemble_model_performance.csv").exists():
            ens_df = pd.read_csv(output_dir / "ensemble_model_performance.csv")
            print(f"🎯 앙상블 최고 성능: {ens_df.iloc[0]['composite_score']:.4f} ({ens_df.iloc[0]['model']})")
        
        # 예측 결과
        if (output_dir / "23_24_transfer_predictions.csv").exists():
            pred_df = pd.read_csv(output_dir / "23_24_transfer_predictions.csv")
            print(f"⚽ 23/24 이적 예측: {len(pred_df)}명의 선수")
            print(f"  - 최고 확률: {pred_df.iloc[0]['transfer_probability']:.4f} ({pred_df.iloc[0]['player_name']})")
            
    except Exception as e:
        print(f"성능 비교 중 오류: {e}")

if __name__ == "__main__":
    run_complete_pipeline()
