#!/usr/bin/env python3
"""
Football Transfer Prediction - Main Execution Script
실무용 메인 실행 스크립트

Usage:
    python main.py --mode train    # 모델 훈련
    python main.py --mode predict  # 23/24 예측
    python main.py --mode evaluate # 모델 평가
"""

import argparse
import sys
from pathlib import Path
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.football_modeling import FootballModelTrainer
from src.visualization.plotter import ModelVisualizer
from src.utils.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/football_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Football Transfer Prediction')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], 
                       default='train', help='실행 모드')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='설정 파일 경로')
    parser.add_argument('--output-dir', type=str, default='outputs', 
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = Config(args.config)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        if args.mode == 'train':
            train_model(config, output_dir)
        elif args.mode == 'predict':
            predict_transfers(config, output_dir)
        elif args.mode == 'evaluate':
            evaluate_model(config, output_dir)
            
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)

def train_model(config, output_dir):
    """모델 훈련"""
    logger.info("🚀 모델 훈련 시작")
    
    # 1. 데이터 로드
    data_loader = DataLoader(config)
    df_raw = data_loader.load_raw_data()
    logger.info(f"✅ Raw 데이터 로드: {df_raw.shape}")
    
    # 2. 피처 엔지니어링
    feature_engineer = FeatureEngineer(df_raw)
    df_processed = feature_engineer.run_feature_engineering()
    logger.info(f"✅ 피처 엔지니어링 완료: {df_processed.shape}")
    
    # 3. 모델 훈련
    model_trainer = FootballModelTrainer(df_processed, config)
    model_results = model_trainer.run_pipeline()
    logger.info("✅ 모델 훈련 완료")
    
    # 4. 시각화
    visualizer = ModelVisualizer(model_results, output_dir)
    visualizer.create_all_plots()
    logger.info("✅ 시각화 완료")
    
    # 5. 결과 저장
    model_trainer.save_model(output_dir)
    logger.info(f"💾 모델 저장: {output_dir}")

def predict_transfers(config, output_dir):
    """23/24 시즌 이적 예측"""
    logger.info("🔮 23/24 시즌 이적 예측 시작")
    
    # 1. 모델 로드
    model_trainer = FootballModelTrainer.load_model(output_dir)
    
    # 2. 23/24 데이터 로드 및 전처리
    data_loader = DataLoader(config)
    df_2324 = data_loader.load_2324_data()
    
    feature_engineer = FeatureEngineer(df_2324)
    df_2324_processed = feature_engineer.run_feature_engineering()
    
    # 3. 예측 수행
    predictions = model_trainer.predict(df_2324_processed)
    
    # 4. 결과 저장
    output_path = output_dir / "23_24_transfer_predictions.csv"
    predictions.to_csv(output_path, index=False)
    logger.info(f"💾 예측 결과 저장: {output_path}")

def evaluate_model(config, output_dir):
    """모델 평가"""
    logger.info("📊 모델 평가 시작")
    
    # 모델 로드 및 평가
    model_trainer = FootballModelTrainer.load_model(output_dir)
    evaluation_results = model_trainer.evaluate_model()
    
    # 평가 결과 저장
    evaluation_path = output_dir / "model_evaluation.json"
    evaluation_results.save(evaluation_path)
    logger.info(f"💾 평가 결과 저장: {evaluation_path}")

if __name__ == "__main__":
    main()
