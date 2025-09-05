#!/usr/bin/env python3
"""
간단한 테스트 실행 스크립트
실제로 모든 결과 파일들이 생성되는지 확인
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

def create_sample_data():
    """샘플 데이터 생성"""
    logger.info("📊 샘플 데이터 생성 중...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # 기본 피처들
    data = {
        'player_name': [f'Player_{i}' for i in range(n_samples)],
        'club_name': np.random.choice(['Chelsea', 'Arsenal', 'Liverpool', 'Man City', 'Tottenham'], n_samples),
        'position': np.random.choice(['Attack', 'Midfield', 'Defense', 'Goalkeeper'], n_samples),
        'season': np.random.choice(['20/21', '21/22', '22/23', '23/24'], n_samples),
        'age_at_season': np.random.randint(18, 35, n_samples),
        'player_market_value_in_eur': np.random.exponential(1000000, n_samples),
        'season_avg_minutes': np.random.randint(0, 3000, n_samples),
        'goals': np.random.poisson(5, n_samples),
        'assists': np.random.poisson(3, n_samples),
        'yellow_cards': np.random.poisson(3, n_samples),
        'red_cards': np.random.poisson(0.2, n_samples),
        'height_in_cm': np.random.normal(180, 10, n_samples),
        'country_of_birth': np.random.choice(['England', 'Spain', 'France', 'Germany', 'Brazil'], n_samples),
        'foot': np.random.choice(['Right', 'Left', 'Both'], n_samples),
        'sub_position': np.random.choice(['ST', 'LW', 'RW', 'CM', 'CDM', 'CB', 'LB', 'RB', 'GK'], n_samples),
        'club_squad_size': np.random.randint(20, 30, n_samples),
        'club_average_age': np.random.uniform(24, 28, n_samples),
        'club_foreigners_percentage': np.random.uniform(0.3, 0.8, n_samples),
        'season_win_count': np.random.randint(10, 30, n_samples),
        'transfer': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 15% 이적률
    }
    
    df = pd.DataFrame(data)
    
    # 고급 피처 생성
    df['log_market_value'] = np.log1p(df['player_market_value_in_eur'])
    df['minutes_vs_club_avg'] = df['season_avg_minutes'] / df.groupby(['club_name', 'season'])['season_avg_minutes'].transform('mean')
    df['age_difference'] = df['age_at_season'] - df.groupby(['club_name', 'season'])['age_at_season'].transform('mean')
    df['attack_contribution'] = (df['goals'] + df['assists']) * df['season_win_count']
    df['is_foreigner'] = (df['country_of_birth'] != 'England').astype(int)
    df['height_vs_position'] = df['height_in_cm'] / df.groupby('position')['height_in_cm'].transform('mean')
    df['cards_per_minute'] = (df['yellow_cards'] + df['red_cards']) / (df['season_avg_minutes'] + 1)
    df['club_tenure_seasons'] = df.groupby('player_name')['season'].transform('count')
    df['position_competition'] = df.groupby(['club_name', 'season', 'position'])['player_name'].transform('count')
    
    logger.info(f"✅ 샘플 데이터 생성 완료: {df.shape}")
    return df

def test_modeling():
    """모델링 테스트"""
    logger.info("🚀 모델링 테스트 시작")
    
    # 1. 샘플 데이터 생성
    df = create_sample_data()
    
    # 2. 설정 생성
    from src.utils.config import Config
    config = Config("config.yaml")
    
    # 3. 모델 훈련
    from src.models.football_modeling import FootballModelTrainer
    model_trainer = FootballModelTrainer(df, config)
    model_results = model_trainer.run_pipeline()
    
    # 4. 시각화 생성
    from src.visualization.plotter import ModelVisualizer
    output_dir = Path("outputs")
    visualizer = ModelVisualizer(model_results, output_dir)
    visualizer.create_all_plots()
    
    # 5. 모델 저장
    model_trainer.save_model(output_dir)
    
    # 6. 23/24 예측
    df_2324 = df[df['season'] == '23/24'].copy()
    if len(df_2324) > 0:
        predictions = model_trainer.predict(df_2324)
        predictions.to_csv(output_dir / "23_24_transfer_predictions.csv", index=False)
        
        # 예측 분포 그래프
        visualizer.plot_prediction_distribution(predictions)
        
        logger.info(f"✅ 23/24 예측 완료: {len(predictions)}명")
    
    logger.info("✅ 모델링 테스트 완료")

def check_outputs():
    """출력 파일 확인"""
    logger.info("📁 출력 파일 확인 중...")
    
    output_dir = Path("outputs")
    expected_files = [
        "model.pkl",
        "preprocessor.pkl", 
        "model_results.pkl",
        "model_comparison.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "feature_importance.png",
        "shap_summary.png",
        "shap_bar.png",
        "23_24_transfer_predictions.csv",
        "prediction_distribution.png"
    ]
    
    missing_files = []
    for file in expected_files:
        if not (output_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"❌ 누락된 파일들: {missing_files}")
    else:
        logger.info("✅ 모든 출력 파일이 생성되었습니다!")
    
    # 파일 목록 출력
    logger.info("📊 생성된 파일들:")
    for file in output_dir.glob("*"):
        logger.info(f"  - {file.name}")

if __name__ == "__main__":
    try:
        test_modeling()
        check_outputs()
        print("\n🎉 테스트 완료! 모든 결과 파일이 생성되었습니다.")
        print("📁 outputs/ 디렉토리를 확인해보세요.")
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
