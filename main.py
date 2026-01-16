#!/usr/bin/env python3
"""
Liquidity Risk ML Predictor
============================
Main execution script for liquidity risk prediction.

Author: Avni Derashree
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import create_liquidity_dataset, get_default_dataset
from src.liquidity_features import LiquidityFeatureEngineer, calculate_feature_importance
from src.ml_models import (
    train_all_models, create_model_comparison, get_best_model, cross_validate_model
)
from src.liquidity_scorer import LiquidityScoringEngine, create_liquidity_report
from src.visualization import (
    plot_liquidity_timeseries, plot_liquidity_comparison, plot_risk_distribution,
    plot_feature_importance, plot_prediction_vs_actual, plot_model_comparison,
    plot_liquidity_heatmap, plot_correlation_matrix
)


def print_header(text: str, char: str = "="):
    """Print formatted section header."""
    print(f"\n{char * 70}")
    print(f" {text}")
    print(f"{char * 70}")


def main():
    """Main execution function."""
    
    print_header("LIQUIDITY RISK ML PREDICTOR", "=")
    print("\nThis analysis performs:")
    print("  1. Liquidity metrics calculation (Amihud, Spread, Impact)")
    print("  2. Feature engineering for ML prediction")
    print("  3. Multiple ML model training and comparison")
    print("  4. Liquidity risk scoring and classification")
    print("  5. Visualization of results")
    
    # =========================================================================
    # STEP 1: Load and Prepare Data
    # =========================================================================
    print_header("STEP 1: Data Loading & Liquidity Metrics", "-")
    
    config = get_default_dataset()
    tickers = config['tickers']
    
    print(f"\nAssets: {', '.join(tickers)}")
    
    # Create liquidity dataset
    df = create_liquidity_dataset(
        tickers=tickers,
        start_date="2020-01-01"
    )
    
    print(f"\nDataset shape: {df.shape}")
    
    # Handle date column - could be regular column or index
    if 'date' in df.columns:
        date_min = df['date'].min()
        date_max = df['date'].max()
    else:
        date_min = df.index.min()
        date_max = df.index.max()
    print(f"Date range: {date_min} to {date_max}")
    
    # Display sample metrics
    print("\nSample Liquidity Metrics:")
    sample_cols = ['ticker', 'spread_proxy', 'amihud', 'volume_ratio', 'price_impact']
    sample_cols = [c for c in sample_cols if c in df.columns]
    if 'date' in df.columns:
        sample_cols = ['date'] + sample_cols
    print(df[sample_cols].head(10).to_string(index=False))
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print_header("STEP 2: Feature Engineering", "-")
    
    engineer = LiquidityFeatureEngineer(
        target_col='spread_proxy',
        scale_features=True
    )
    
    feature_set = engineer.fit_transform(df, forecast_horizon=1)
    
    print(f"\nFeature matrix shape: {feature_set.X.shape}")
    print(f"Target shape: {feature_set.y.shape}")
    print(f"Number of features: {len(feature_set.feature_names)}")
    
    print("\nTop 15 Features Created:")
    for i, name in enumerate(feature_set.feature_names[:15], 1):
        print(f"  {i:2}. {name}")
    
    # =========================================================================
    # STEP 3: Train/Test Split
    # =========================================================================
    print_header("STEP 3: Train/Test Split", "-")
    
    # Time-series split (no shuffle)
    X = feature_set.X
    y = feature_set.y
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Train/Test ratio: 80/20 (time-based split)")
    
    # =========================================================================
    # STEP 4: Model Training
    # =========================================================================
    print_header("STEP 4: Model Training", "-")
    
    # Train all models
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n📊 Model Comparison:")
    comparison_df = create_model_comparison(results)
    print(comparison_df.to_string(index=False))
    
    # Best model
    best = get_best_model(results)
    print(f"\n🏆 Best Model: {best.model_name}")
    print(f"   Test RMSE: {best.test_rmse:.6f}")
    print(f"   Test R²: {best.test_r2:.4f}")
    print(f"   Test MAE: {best.test_mae:.6f}")
    print(f"   MAPE: {best.mape:.2%}")
    
    # =========================================================================
    # STEP 5: Cross-Validation
    # =========================================================================
    print_header("STEP 5: Time Series Cross-Validation", "-")
    
    print(f"\nPerforming 5-fold time series CV on {best.model_name}...")
    cv_scores = cross_validate_model(best.model, X, y, n_splits=5)
    
    print(f"\nCross-Validation Results:")
    print(f"  RMSE: {cv_scores['cv_rmse_mean']:.6f} (±{cv_scores['cv_rmse_std']:.6f})")
    print(f"  MAE:  {cv_scores['cv_mae_mean']:.6f} (±{cv_scores['cv_mae_std']:.6f})")
    print(f"  R²:   {cv_scores['cv_r2_mean']:.4f} (±{cv_scores['cv_r2_std']:.4f})")
    
    # =========================================================================
    # STEP 6: Feature Importance
    # =========================================================================
    print_header("STEP 6: Feature Importance", "-")
    
    if best.feature_importance is not None:
        print("\nTop 15 Most Important Features:")
        top_features = best.feature_importance.head(15)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"  {i:2}. {row['feature']}: {row['importance']:.4f}")
    
    # =========================================================================
    # STEP 7: Liquidity Scoring
    # =========================================================================
    print_header("STEP 7: Liquidity Scoring", "-")
    
    scorer = LiquidityScoringEngine()
    scored_df = scorer.calculate_composite_score(df)
    
    # Display summary
    print("\nLiquidity Score Summary by Asset:")
    summary = scorer.get_score_summary(scored_df)
    print(summary.to_string())
    
    print("\nRisk Level Distribution:")
    risk_dist = scorer.get_risk_distribution(scored_df)
    for level, row in risk_dist.iterrows():
        print(f"  {level}: {row['Count']:,} ({row['Percentage']:.1f}%)")
    
    # =========================================================================
    # STEP 8: Generate Visualizations
    # =========================================================================
    print_header("STEP 8: Generating Visualizations", "-")
    
    os.makedirs('output', exist_ok=True)
    
    print("\nSaving charts to ./output/ directory...")
    
    # Chart 1: Liquidity time series for first ticker
    fig1 = plot_liquidity_timeseries(df, tickers[0])
    fig1.savefig('output/liquidity_timeseries.png', dpi=150, bbox_inches='tight')
    print("  ✓ liquidity_timeseries.png")
    
    # Chart 2: Liquidity comparison
    fig2 = plot_liquidity_comparison(scored_df, metric='composite_score')
    fig2.savefig('output/liquidity_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ liquidity_comparison.png")
    
    # Chart 3: Risk distribution
    fig3 = plot_risk_distribution(scored_df)
    fig3.savefig('output/risk_distribution.png', dpi=150, bbox_inches='tight')
    print("  ✓ risk_distribution.png")
    
    # Chart 4: Feature importance
    if best.feature_importance is not None:
        fig4 = plot_feature_importance(best.feature_importance, top_n=15)
        fig4.savefig('output/feature_importance.png', dpi=150, bbox_inches='tight')
        print("  ✓ feature_importance.png")
    
    # Chart 5: Prediction vs actual
    fig5 = plot_prediction_vs_actual(
        y_test.values, best.predictions, best.model_name
    )
    fig5.savefig('output/prediction_vs_actual.png', dpi=150, bbox_inches='tight')
    print("  ✓ prediction_vs_actual.png")
    
    # Chart 6: Model comparison
    fig6 = plot_model_comparison(comparison_df, metric='Test RMSE')
    fig6.savefig('output/model_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ model_comparison.png")
    
    # Chart 7: Correlation matrix
    fig7 = plot_correlation_matrix(df)
    fig7.savefig('output/correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("  ✓ correlation_matrix.png")
    
    # Chart 8: Liquidity heatmap
    fig8 = plot_liquidity_heatmap(scored_df, metric='composite_score')
    fig8.savefig('output/liquidity_heatmap.png', dpi=150, bbox_inches='tight')
    print("  ✓ liquidity_heatmap.png")
    
    plt.close('all')
    
    # =========================================================================
    # STEP 9: Save Models and Results
    # =========================================================================
    print_header("STEP 9: Saving Models & Results", "-")
    
    os.makedirs('models', exist_ok=True)
    
    # Save best model
    model_path = f'models/best_model_{best.model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(best.model, model_path)
    print(f"  ✓ Saved best model to {model_path}")
    
    # Save feature engineer
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    print("  ✓ Saved feature engineer to models/feature_engineer.pkl")
    
    # Save scorer
    joblib.dump(scorer, 'models/liquidity_scorer.pkl')
    print("  ✓ Saved liquidity scorer to models/liquidity_scorer.pkl")
    
    # Save results
    results_dict = {
        'model_name': best.model_name,
        'test_rmse': best.test_rmse,
        'test_r2': best.test_r2,
        'test_mae': best.test_mae,
        'mape': best.mape,
        'cv_rmse': cv_scores['cv_rmse_mean'],
        'n_features': len(feature_set.feature_names),
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    joblib.dump(results_dict, 'output/training_results.pkl')
    print("  ✓ Saved training results to output/training_results.pkl")
    
    # Save comparison table
    comparison_df.to_csv('output/model_comparison.csv', index=False)
    print("  ✓ Saved model comparison to output/model_comparison.csv")
    
    # Save scored data
    scored_df.to_csv('output/liquidity_scores.csv', index=False)
    print("  ✓ Saved liquidity scores to output/liquidity_scores.csv")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("ANALYSIS COMPLETE", "=")
    
    print("\n📊 Key Results:")
    print(f"\n  Data:")
    print(f"    • Assets analyzed: {len(tickers)}")
    print(f"    • Total observations: {len(df):,}")
    print(f"    • Features engineered: {len(feature_set.feature_names)}")
    
    print(f"\n  Best Model ({best.model_name}):")
    print(f"    • Test RMSE: {best.test_rmse:.6f}")
    print(f"    • Test R²: {best.test_r2:.4f}")
    print(f"    • MAPE: {best.mape:.2%}")
    print(f"    • CV RMSE: {cv_scores['cv_rmse_mean']:.6f} (±{cv_scores['cv_rmse_std']:.6f})")
    
    print(f"\n  Liquidity Scoring:")
    print(f"    • Average Score: {scored_df['composite_score'].mean():.1f}")
    high_risk_pct = (scored_df['risk_level'].isin(['High Risk', 'Very High Risk'])).mean() * 100
    print(f"    • High Risk %: {high_risk_pct:.1f}%")
    
    print("\n📁 Output files saved to ./output/")
    print("📁 Models saved to ./models/")
    
    print("\nDone! ✅")
    
    return results, scored_df, best


if __name__ == "__main__":
    results, scored_df, best_model = main()
