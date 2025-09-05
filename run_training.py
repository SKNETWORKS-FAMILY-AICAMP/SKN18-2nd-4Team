#!/usr/bin/env python3
"""
모델 훈련 실행 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import train_model
from src.utils.config import Config

if __name__ == "__main__":
    print("🚀 Football Transfer Prediction - 모델 훈련 시작")
    
    # 설정 로드
    config = Config("config.yaml")
    
    # 출력 디렉토리
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 모델 훈련 실행
    train_model(config, output_dir)
    
    print("✅ 모델 훈련 완료!")
    print(f"📁 결과 저장 위치: {output_dir}")
    print("\n📊 생성된 파일들:")
    print("- model.pkl: 훈련된 모델")
    print("- preprocessor.pkl: 전처리기")
    print("- model_comparison.png: 모델 성능 비교")
    print("- confusion_matrix.png: 혼동 행렬")
    print("- roc_curve.png: ROC 곡선")
    print("- feature_importance.png: 피처 중요도")
    print("- shap_summary.png: SHAP 요약")
    print("- shap_bar.png: SHAP 막대 그래프")
