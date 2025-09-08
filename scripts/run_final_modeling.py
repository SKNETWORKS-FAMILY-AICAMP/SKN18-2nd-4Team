#!/usr/bin/env python3
"""
최종 데이터로 모델링 실행
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_final_modeling(force_retrain=False):
    """최종 데이터로 모델링 실행
    
    Args:
        force_retrain (bool): True면 강제로 재학습, False면 기존 개선된 모델 활용
    """
    logger.info("🚀 최종 모델링 시작")
    
    try:
        # 1. 설정 로드
        from src.utils.config import Config
        config = Config("config_final.yaml")
        
        # 2. 기존 개선된 모델 확인
        import joblib
        outputs_dir = Path(config.output_dir)
        model_results_path = outputs_dir / "model_results.pkl"
        
        if not force_retrain and model_results_path.exists():
            logger.info("💾 기존 개선된 모델 결과 발견 - 재사용합니다")
            model_results = joblib.load(model_results_path)
            
            # 기존 모델이 개선된 모델인지 확인
            improvements = []
            if 'tuning_improvement' in model_results:
                improvements.append(f"튜닝(+{model_results['tuning_improvement']:.4f})")
            if 'regularization_improvement' in model_results:
                improvements.append(f"정규화(+{model_results['regularization_improvement']:.4f})")
            if 'ensemble_improvement' in model_results:
                improvements.append(f"앙상블(+{model_results['ensemble_improvement']:.4f})")
                
            if improvements:
                logger.info(f"🎉 적용된 개선 기법: {', '.join(improvements)}")
                logger.info(f"📊 최종 모델: {model_results.get('best_model_name', 'Unknown')}")
            else:
                logger.info("📊 기본 모델 사용 중")
                
        else:
            if force_retrain:
                logger.info("🔄 강제 재학습 모드")
            else:
                logger.info("🆕 새로운 모델 학습")
            model_results = None
        
        # 3. 데이터 로드
        from src.data.data_loader_new import DataLoaderNew
        data_loader = DataLoaderNew(config)
        train_df, valid_df, test_df, pred_df = data_loader.load_all_data()
        
        print(f"📊 최종 데이터 로드 완료:")
        print(f"  - Train: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
        print(f"  - Valid: {valid_df.shape[0]:,} rows x {valid_df.shape[1]} columns")
        print(f"  - Test: {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
        print(f"  - Pred: {pred_df.shape[0]:,} rows x {pred_df.shape[1]} columns")
        print(f"  - Train 이적률: {train_df['transfer'].mean()*100:.1f}%")
        print(f"  - Valid 이적률: {valid_df['transfer'].mean()*100:.1f}%")
        print(f"  - Test 이적률: {test_df['transfer'].mean()*100:.1f}%")
        
        # 4. 모델링 실행
        from src.models.football_modeling import FootballModelTrainer
        
        # 모델 훈련 (기존 결과가 있으면 재사용)
        if model_results is None:
            logger.info("🔥 새로운 모델 학습 시작")
            model_trainer = FootballModelTrainer(train_df, valid_df, test_df, pred_df, config)
            model_results = model_trainer.run_pipeline()
        else:
            logger.info("♻️ 기존 개선된 모델 재사용")
            # 예측만을 위해 trainer 초기화 (학습 없이)
            model_trainer = FootballModelTrainer(train_df, valid_df, test_df, pred_df, config)
            model_trainer.model_results = model_results
            # 기존 결과에서 필요한 속성들 복원
            model_trainer.best_model = model_results.get('best_model')
            model_trainer.preprocessor = model_results.get('preprocessor')
            model_trainer.best_model_name = model_results.get('best_model_name')
            
            # 최종 모델로 SHAP 분석 재실행 (5단계에서)
            logger.info("🔍 최종 모델로 SHAP 분석 재실행")
            try:
                # 검증 데이터 준비 (전처리)
                X_val = model_results.get('X_val')
                y_val = model_results.get('y_val')
                if X_val is not None and y_val is not None:
                    shap_results = model_trainer._shap_analysis(X_val, y_val)
                    if shap_results:
                        model_results['shap_results'] = shap_results
                        logger.info("✅ SHAP 분석 완료 (최종 모델)")
                    else:
                        logger.warning("SHAP 분석 실패")
                else:
                    logger.warning("검증 데이터가 없어 SHAP 분석을 건너뜁니다")
            except Exception as e:
                logger.warning(f"SHAP 분석 중 오류: {e}")
        
        # 4. 시각화
        from src.visualization.plotter import ModelVisualizer
        output_dir = Path("outputs")
        visualizer = ModelVisualizer(model_results, output_dir)
        visualizer.create_all_plots()
        
        # 5. 모델 저장
        model_trainer.save_model(output_dir)
        
        # 6. 24/25 예측 (pred 데이터)
        predictions = model_trainer.predict(pred_df)
        
        # 중복 제거 (선수별로 최고 확률만 유지)
        predictions_dedup = predictions.loc[predictions.groupby('player_name')['transfer_probability'].idxmax()]
        
        # 확률이 높은 순으로 내림차순 정렬
        predictions_sorted = predictions_dedup.sort_values('transfer_probability', ascending=False)
        predictions_sorted.to_csv(output_dir / "24_25_transfer_predictions.csv", index=False)
        
        # 7. 예측 분포 그래프
        visualizer.plot_prediction_distribution(predictions)
        
        # 8. 모델 성능 점수 저장 및 출력
        try:
            from scripts.save_model_performance import save_model_performance
            save_model_performance(model_results)
        except ImportError:
            logger.info("모델 성능 저장 모듈이 없습니다. 건너뜁니다.")
        except Exception as e:
            logger.error(f"모델 성능 저장 중 오류: {e}")
        
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
