"""
Visualization module for Football Transfer Prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# SHAP import (optional)
try:
    import shap
    _has_shap = True
except ImportError:
    _has_shap = False

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """모델 시각화 클래스"""
    
    def __init__(self, model_results: Dict[str, Any], output_dir: Path):
        self.model_results = model_results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        
    def create_all_plots(self):
        """모든 시각화 생성"""
        logger.info("📊 시각화 생성 시작")
        
        # 1. 모델 성능 비교
        self.plot_model_comparison()
        
        # 2. 혼동 행렬
        self.plot_confusion_matrix()
        
        # 3. ROC 곡선
        self.plot_roc_curve()
        
        # 4. 피처 중요도
        self.plot_feature_importance()
        
        # 5. SHAP 분석
        if _has_shap and 'shap_results' in self.model_results:
            self.plot_shap_analysis()
        
        # 6. 학습 곡선 (오버피팅 분석)
        if 'learning_curve_results' in self.model_results:
            self.plot_learning_curves()
        
        logger.info("✅ 시각화 완료")
    
    def plot_model_comparison(self):
        """모델 성능 비교 그래프"""
        if 'model_scores' not in self.model_results:
            logger.warning("모델 성능 점수가 없습니다.")
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(self.model_results['model_scores'].keys())
        scores = list(self.model_results['model_scores'].values())
        
        bars = ax.bar(models, scores, color='skyblue', alpha=0.7)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Composite Score', fontsize=12)
        ax.set_xlabel('Models', fontsize=12)
        
        # 값 표시
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ 모델 성능 비교 그래프 저장 완료")
    
    def plot_confusion_matrix(self):
        """혼동 행렬"""
        if 'final_results' not in self.model_results or 'confusion_matrix' not in self.model_results['final_results']:
            logger.warning("혼동 행렬 데이터가 없습니다.")
            return
            
        cm = self.model_results['final_results']['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ 혼동 행렬 그래프 저장 완료")
    
    def plot_roc_curve(self):
        """ROC 곡선"""
        if 'final_results' not in self.model_results or 'roc_curve' not in self.model_results['final_results']:
            logger.warning("ROC 곡선 데이터가 없습니다.")
            return
            
        fpr, tpr = self.model_results['final_results']['roc_curve']
        auc = self.model_results['final_results']['auc']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ ROC 곡선 그래프 저장 완료")
    
    def plot_feature_importance(self):
        """피처 중요도 (상위 30개)"""
        if 'final_results' not in self.model_results or 'feature_importance' not in self.model_results['final_results']:
            logger.warning("피처 중요도 데이터가 없습니다.")
            return
            
        importance = self.model_results['final_results']['feature_importance']
        if importance is None:
            logger.warning("피처 중요도를 계산할 수 없습니다.")
            return
        
        # 상위 30개만 선택
        top_importance = importance.tail(30)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        top_importance.plot(kind='barh', ax=ax, color='lightcoral')
        ax.set_title('Feature Importance (Top 30)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ 피처 중요도 그래프 저장 완료 (상위 30개)")
    
    def plot_shap_analysis(self):
        """SHAP 분석"""
        if not _has_shap or 'shap_results' not in self.model_results:
            logger.warning("SHAP 분석을 위한 데이터가 없습니다.")
            return
            
        try:
            shap_results = self.model_results['shap_results']
            if not shap_results:
                logger.warning("SHAP 결과가 비어있습니다.")
                return
                
            shap_values = shap_results['shap_values']
            X_val_processed = shap_results['X_val_processed']
            feature_names = shap_results.get('feature_names', [])
            
            # 피처명이 없으면 기본 이름 생성
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(X_val_processed.shape[1])]
            
            # DataFrame으로 변환하여 피처명 설정
            X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
            
            # SHAP summary plot (가로로 길게, 피처명 명확히)
            plt.figure(figsize=(20, 12))  # 더 가로로 길게
            shap.summary_plot(
                shap_values,
                X_val_df,
                max_display=20,
                show=False,
                plot_size=(20, 12)  # SHAP 내부 크기 설정
            )
            plt.title('SHAP Feature Importance Distribution (Top 20)', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('SHAP value (impact on model output)', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=11)  # 피처명 폰트 크기
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # SHAP bar plot (가로로 길게)
            plt.figure(figsize=(16, 10))  # 가로로 길게
            shap.summary_plot(
                shap_values,
                X_val_df,
                plot_type="bar",
                max_display=20,
                show=False,
                plot_size=(16, 10)
            )
            plt.title('SHAP Feature Importance Ranking (Top 20)', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Mean |SHAP value|', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=11)  # 피처명 폰트 크기
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_bar.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info("✅ SHAP 분석 완료")
            
        except Exception as e:
            logger.error(f"SHAP 분석 오류: {e}")
    
    def plot_prediction_distribution(self, predictions: pd.DataFrame):
        """예측 결과 분포"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 이적 확률 분포 (퍼센트로 변환)
        if 'transfer_probability_percent' in predictions.columns:
            prob_col = 'transfer_probability_percent'
        else:
            prob_col = 'transfer_probability'
            # 0-1 범위를 0-100으로 변환
            predictions = predictions.copy()
            predictions['transfer_probability_percent'] = predictions[prob_col] * 100
            prob_col = 'transfer_probability_percent'
        
        ax1.hist(predictions[prob_col], bins=30, alpha=0.7, color='skyblue')
        ax1.set_title('Transfer Probability Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Transfer Probability (%)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.axvline(x=60, color='red', linestyle='--', label='High Risk Threshold (60%)')
        ax1.legend()
        
        # 포지션별 이적 확률
        position_risk = predictions.groupby('position')['transfer_probability_percent'].mean().sort_values(ascending=True)
        position_risk.plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title('Average Transfer Probability by Position', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Average Transfer Probability (%)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curves(self):
        """학습 곡선 (오버피팅 분석)"""
        if 'learning_curve_results' not in self.model_results:
            logger.warning("학습 곡선 데이터가 없습니다.")
            return
        
        from src.features.feature_engineering import OverfittingChecker
        
        # 학습 곡선 데이터 추출
        lc_results = self.model_results['learning_curve_results']
        train_scores = lc_results['train_scores']
        val_scores = lc_results['val_scores']
        final_gap = lc_results['final_gap']
        max_gap = lc_results['max_gap']
        is_overfitting = lc_results['is_overfitting']
        
        # 훈련 크기 계산 (기본값 사용)
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_samples = [int(len(train_scores) * size * 10) for size in train_sizes]  # 근사치
        
        # 시각화
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 학습 곡선
        ax1.plot(n_samples, train_scores, 'o-', color='blue', label='Training Score', linewidth=2, markersize=6)
        ax1.plot(n_samples, val_scores, 'o-', color='red', label='Validation Score', linewidth=2, markersize=6)
        ax1.fill_between(n_samples, train_scores, val_scores, alpha=0.2, color='gray', label='Gap')
        ax1.set_xlabel('Training Samples', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Learning Curves', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. 오버피팅 분석
        gaps = [t - v for t, v in zip(train_scores, val_scores)]
        ax2.plot(n_samples, gaps, 'o-', color='purple', linewidth=2, markersize=6)
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Warning (0.1)')
        ax2.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Severe (0.15)')
        ax2.set_xlabel('Training Samples', fontsize=12)
        ax2.set_ylabel('Score Gap (Train - Val)', fontsize=12)
        ax2.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. 최종 성능 비교
        categories = ['Training', 'Validation']
        scores = [train_scores[-1], val_scores[-1]]
        colors = ['blue', 'red']
        bars = ax3.bar(categories, scores, color=colors, alpha=0.7)
        ax3.set_ylabel('Final Score', fontsize=12)
        ax3.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1)
        
        # 점수 값 표시
        for bar, score in zip(bars, scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 오버피팅 지표 요약
        ax4.axis('off')
        
        status_color = 'red' if is_overfitting else 'green'
        status_text = 'OVERFITTING DETECTED' if is_overfitting else 'NO OVERFITTING'
        
        info_text = f"""OVERFITTING ANALYSIS SUMMARY

Status: {status_text}

Final Gap: {final_gap:.3f}
Max Gap: {max_gap:.3f}

Thresholds:
• Warning: > 0.1
• Severe: > 0.15

Training Samples: {len(train_scores) * 10:,}
Validation Samples: {len(val_scores) * 10:,}

Recommendation:
{'⚠️ Consider regularization or more data' if is_overfitting else '✅ Model appears stable'}"""
        
        ax4.text(0.05, 0.5, info_text, fontsize=11, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Model Overfitting Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("✅ 학습 곡선 그래프 저장 완료")