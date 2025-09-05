#!/usr/bin/env python3
"""
23/24 시즌 이적 예측 실행 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import predict_transfers
from src.utils.config import Config

if __name__ == "__main__":
    print("🔮 Football Transfer Prediction - 23/24 시즌 예측 시작")
    
    # 설정 로드
    config = Config("config.yaml")
    
    # 출력 디렉토리
    output_dir = Path("outputs")
    
    # 23/24 시즌 예측 실행
    predict_transfers(config, output_dir)
    
    print("✅ 23/24 시즌 예측 완료!")
    print(f"📁 결과 저장 위치: {output_dir}")
    print("\n📊 생성된 파일들:")
    print("- 23_24_transfer_predictions.csv: 예측 결과")
    print("- prediction_distribution.png: 예측 분포")
    
    # 예측 결과 요약
    predictions_path = output_dir / "23_24_transfer_predictions.csv"
    if predictions_path.exists():
        import pandas as pd
        df = pd.read_csv(predictions_path)
        
        high_risk = len(df[df['transfer_probability_percent'] >= 60])
        predicted_transfers = len(df[df['predicted_transfer'] == 1])
        
        print(f"\n📈 예측 결과 요약:")
        print(f"- 총 선수 수: {len(df)}명")
        print(f"- 예측 이적 선수: {predicted_transfers}명 ({predicted_transfers/len(df)*100:.1f}%)")
        print(f"- 고위험 선수 (≥60%): {high_risk}명 ({high_risk/len(df)*100:.1f}%)")
        
        print(f"\n🚨 상위 5명 고위험 선수:")
        top_5 = df.head(5)
        for idx, row in top_5.iterrows():
            print(f"  {row['player_name']} ({row['position']}, {row['club_name']}) - {row['transfer_probability_percent']:.1f}%")
