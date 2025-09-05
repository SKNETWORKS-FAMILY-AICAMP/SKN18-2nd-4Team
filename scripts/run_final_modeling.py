#!/usr/bin/env python3
"""
최종 데이터로 모델링 실행
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_final_modeling():
    """최종 데이터로 모델링 실행"""
    logger.info("🚀 최종 모델링 시작")
    
    try:
        # 1. 설정 로드
        from src.utils.config import Config
        config = Config("config_final.yaml")
        
        # 2. 데이터 로드
        from src.data.data_loader_new import DataLoaderNew
        data_loader = DataLoaderNew(config)
        train_df, test_df = data_loader.load_all_data()
        
        print(f"📊 최종 데이터 로드 완료:")
        print(f"  - Train: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
        print(f"  - Test: {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
        print(f"  - Train 이적률: {train_df['transfer'].mean()*100:.1f}%")
        print(f"  - Test 이적률: {test_df['transfer'].mean()*100:.1f}%")
        
        # 3. 모델링 실행
        from src.models.football_modeling import FootballModelTrainer
        
        # 전체 데이터 합치기 (모델링용)
        all_data = pd.concat([train_df, test_df], ignore_index=True)
        
        # 모델 훈련
        model_trainer = FootballModelTrainer(all_data, config)
        model_results = model_trainer.run_pipeline()
        
        # 4. 시각화
        from src.visualization.plotter import ModelVisualizer
        output_dir = Path("outputs")
        visualizer = ModelVisualizer(model_results, output_dir)
        visualizer.create_all_plots()
        
        # 5. 모델 저장
        model_trainer.save_model(output_dir)
        
        # 6. 23/24 예측 (test 데이터)
        predictions = model_trainer.predict(test_df)
        
        # 중복 제거 (선수별로 최고 확률만 유지)
        predictions_dedup = predictions.loc[predictions.groupby('player_name')['transfer_probability'].idxmax()]
        
        # 확률이 높은 순으로 내림차순 정렬
        predictions_sorted = predictions_dedup.sort_values('transfer_probability', ascending=False)
        predictions_sorted.to_csv(output_dir / "23_24_transfer_predictions.csv", index=False)
        
        # 7. 예측 분포 그래프
        visualizer.plot_prediction_distribution(predictions)
        
        # 8. 모델 성능 점수 저장 및 출력
        try:
            from scripts.save_model_performance import save_model_performance
            save_model_performance()
        except ImportError:
            logger.info("모델 성능 저장 모듈이 없습니다. 건너뜁니다.")
        
        logger.info("✅ 최종 모델링 완료")
        
        # 9. 결과 요약
        print(f"\n📊 최종 결과:")
        print(f"  - 최고 모델: {model_results['best_model_name']}")
        print(f"  - 예측 선수 수: {len(predictions):,}명")
        print(f"  - 예측 이적: {predictions['predicted_transfer'].sum()}명")
        print(f"  - 고위험 선수 (60%+): {len(predictions[predictions['transfer_probability_percent'] >= 60])}명")
        
    except Exception as e:
        logger.error(f"최종 모델링 오류: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    run_final_modeling()

if __name__ == "__main__":
    main()
