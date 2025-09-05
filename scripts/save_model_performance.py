#!/usr/bin/env python3
"""
모델 성능 점수를 상세히 저장하고 출력하는 스크립트
"""

import pandas as pd
import joblib
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model_performance():
    """모델 성능 점수를 상세히 저장하고 출력"""
    try:
        # 경로 설정
        outputs_dir = Path("outputs")
        model_results_path = outputs_dir / "model_results.pkl"
        
        if not model_results_path.exists():
            logger.error("model_results.pkl 파일이 없습니다.")
            return
        
        # 모델 결과 로드
        model_results = joblib.load(model_results_path)
        
        if 'model_comparison' not in model_results:
            logger.error("모델 비교 결과가 없습니다.")
            return
        
        model_comparison = model_results['model_comparison']
        
        # 상세 성능 데이터프레임 생성
        detailed_performance = []
        
        for model_name, metrics in model_comparison.items():
            # metrics가 dict가 아닌 경우 처리
            if isinstance(metrics, (int, float)):
                # 단순 점수인 경우 Composite_Score로 처리
                detailed_performance.append({
                    'Model': model_name,
                    'Accuracy': 0,
                    'Precision': 0,
                    'Recall': 0,
                    'F1_Score': 0,
                    'AUC': 0,
                    'Composite_Score': round(metrics, 4)
                })
            else:
                # dict인 경우 정상 처리
                detailed_performance.append({
                    'Model': model_name,
                    'Accuracy': round(metrics.get('Accuracy', 0), 4),
                    'Precision': round(metrics.get('Precision', 0), 4),
                    'Recall': round(metrics.get('Recall', 0), 4),
                    'F1_Score': round(metrics.get('F1_Score', 0), 4),
                    'AUC': round(metrics.get('AUC', 0), 4),
                    'Composite_Score': round(metrics.get('Composite_Score', 0), 4)
                })
        
        # 데이터프레임 생성 및 정렬
        df_performance = pd.DataFrame(detailed_performance)
        df_performance = df_performance.sort_values('Composite_Score', ascending=False)
        df_performance['Rank'] = range(1, len(df_performance) + 1)
        
        # 순위별 이모지 추가
        rank_emoji = {1: '🥇', 2: '🥈', 3: '🥉'}
        df_performance['Medal'] = df_performance['Rank'].map(lambda x: rank_emoji.get(x, ''))
        
        # 컬럼 순서 재정렬
        df_performance = df_performance[['Rank', 'Medal', 'Model', 'Composite_Score', 
                                       'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']]
        
        # CSV 저장
        performance_csv_path = outputs_dir / "detailed_model_performance.csv"
        df_performance.to_csv(performance_csv_path, index=False, encoding='utf-8-sig')
        
        # 콘솔 출력
        print("\n" + "="*80)
        print("📊 **모델 성능 상세 결과**")
        print("="*80)
        
        for idx, row in df_performance.iterrows():
            print(f"\n{row['Medal']} **{row['Rank']}위: {row['Model']}**")
            print(f"   • 종합점수: {row['Composite_Score']:.4f}")
            print(f"   • 정확도  : {row['Accuracy']:.4f}")
            print(f"   • 정밀도  : {row['Precision']:.4f}")
            print(f"   • 재현율  : {row['Recall']:.4f}")
            print(f"   • F1점수  : {row['F1_Score']:.4f}")
            print(f"   • AUC     : {row['AUC']:.4f}")
        
        print("\n" + "="*80)
        print(f"✅ 상세 성능 결과가 {performance_csv_path}에 저장되었습니다.")
        
        # 채택 모델 정보
        best_model_info = {
            'Best_Model': df_performance.iloc[0]['Model'],
            'Best_Score': df_performance.iloc[0]['Composite_Score'],
            'Selection_Criteria': 'Composite Score (Weighted Average)',
            'Weight_Formula': 'Accuracy(0.2) + Precision(0.2) + Recall(0.2) + F1(0.2) + AUC(0.2)'
        }
        
        # 채택 모델 정보 CSV 저장
        adoption_csv_path = outputs_dir / "model_adoption_info.csv"
        pd.DataFrame([best_model_info]).to_csv(adoption_csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"✅ 모델 성능 저장 완료: {len(df_performance)}개 모델")
        
    except Exception as e:
        logger.error(f"모델 성능 저장 중 오류: {e}")

def main():
    save_model_performance()

if __name__ == "__main__":
    main()