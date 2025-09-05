"""
Data loading and preprocessing for Football Transfer Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """데이터 로딩 및 전처리 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.data_config['raw_dir'])
        self.curated_dir = Path(config.data_config['curated_dir'])
        self.processed_dir = Path(config.data_config['processed_dir'])
        
    def load_raw_data(self) -> pd.DataFrame:
        """Raw 데이터 로드 및 통합"""
        logger.info("📁 Raw 데이터 로드 시작")
        
        # 1. 선수 기본 정보
        players = pd.read_csv(self.raw_dir / "players.csv")
        logger.info(f"✅ players.csv: {players.shape}")
        
        # 2. 클럽 정보
        clubs = pd.read_csv(self.raw_dir / "clubs.csv")
        logger.info(f"✅ clubs.csv: {clubs.shape}")
        
        # 3. 출전 기록
        appearances = pd.read_csv(self.raw_dir / "appearances.csv")
        logger.info(f"✅ appearances.csv: {appearances.shape}")
        
        # 4. 시장가치 변동
        valuations = pd.read_csv(self.raw_dir / "player_valuations.csv")
        logger.info(f"✅ player_valuations.csv: {valuations.shape}")
        
        # 5. 이적 기록
        transfers = pd.read_csv(self.raw_dir / "transfers.csv")
        logger.info(f"✅ transfers.csv: {transfers.shape}")
        
        # 6. 경기 정보
        games = pd.read_csv(self.raw_dir / "games.csv")
        logger.info(f"✅ games.csv: {games.shape}")
        
        # 7. 클럽별 경기 기록
        club_games = pd.read_csv(self.raw_dir / "club_games.csv")
        logger.info(f"✅ club_games.csv: {club_games.shape}")
        
        # 데이터 통합 (기존 로직 사용)
        df_integrated = self._integrate_data(players, clubs, appearances, 
                                           valuations, transfers, games, club_games)
        
        logger.info(f"✅ 통합 데이터: {df_integrated.shape}")
        return df_integrated
    
    def load_curated_data(self) -> pd.DataFrame:
        """Curated 데이터 로드"""
        curated_path = self.curated_dir / "player_final.csv"
        
        if not curated_path.exists():
            logger.warning("Curated 데이터가 없습니다. Raw 데이터를 통합합니다.")
            return self.load_raw_data()
        
        df = pd.read_csv(curated_path)
        logger.info(f"✅ Curated 데이터 로드: {df.shape}")
        return df
    
    def load_2324_data(self) -> pd.DataFrame:
        """23/24 시즌 데이터 로드"""
        df = self.load_curated_data()
        df_2324 = df[df['season'] == '23/24'].copy()
        logger.info(f"✅ 23/24 시즌 데이터: {df_2324.shape}")
        return df_2324
    
    def _integrate_data(self, players, clubs, appearances, valuations, 
                       transfers, games, club_games) -> pd.DataFrame:
        """데이터 통합 (기존 로직)"""
        # 기존 1번 파일의 통합 로직을 여기에 구현
        # 간단한 예시
        df_integrated = players.copy()
        
        # 클럽 정보 통합
        df_integrated = df_integrated.merge(
            clubs[['club_id', 'name']], 
            left_on='current_club_id', 
            right_on='club_id', 
            how='left'
        )
        
        # 시장가치 통합
        latest_valuations = valuations.groupby('player_id').last().reset_index()
        df_integrated = df_integrated.merge(
            latest_valuations[['player_id', 'market_value_in_eur']],
            on='player_id',
            how='left'
        )
        
        return df_integrated
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """전처리된 데이터 저장"""
        output_path = self.processed_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"💾 전처리 데이터 저장: {output_path}")