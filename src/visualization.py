"""
Visualization Module
Charts and plots for liquidity risk analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_liquidity_timeseries(
    df: pd.DataFrame,
    ticker: str,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot liquidity metrics over time for a single ticker.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Liquidity data with 'date' and 'ticker' columns
    ticker : str
        Ticker to plot
    metrics : List[str], optional
        Metrics to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
    """
    if metrics is None:
        metrics = ['spread_proxy', 'amihud_ma20', 'volume_ratio', 'realized_vol']
    
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('date')
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric in ticker_data.columns:
            axes[i].plot(
                ticker_data['date'], 
                ticker_data[metric], 
                color=color, 
                linewidth=1
            )
            axes[i].fill_between(
                ticker_data['date'],
                ticker_data[metric],
                alpha=0.3,
                color=color
            )
            axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.suptitle(f'Liquidity Metrics: {ticker}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_liquidity_comparison(
    df: pd.DataFrame,
    metric: str = 'composite_score',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare liquidity across multiple tickers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Liquidity data with scores
    metric : str
        Metric to compare
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Box plot by ticker
    tickers = df['ticker'].unique()
    data_by_ticker = [df[df['ticker'] == t][metric].dropna() for t in tickers]
    
    bp = ax.boxplot(data_by_ticker, labels=tickers, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_xlabel('Ticker', fontsize=11)
    ax.set_title(f'Liquidity Comparison: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_risk_distribution(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot risk level distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Liquidity data with 'risk_level' column
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    risk_order = ['Very High Risk', 'High Risk', 'Moderate Risk', 'Low Risk', 'Very Low Risk']
    risk_colors = ['#FF4444', '#FF9944', '#FFD700', '#90EE90', '#32CD32']
    
    counts = df['risk_level'].value_counts()
    counts = counts.reindex(risk_order, fill_value=0)
    
    bars = ax.bar(counts.index, counts.values, color=risk_colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords='offset points',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xlabel('Risk Level', fontsize=11)
    ax.set_title('Liquidity Risk Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance_df.head(top_n).sort_values('importance')
    
    colors = plt.cm.RdYlGn(
        np.linspace(0.2, 0.8, len(top_features))
    )
    
    bars = ax.barh(
        top_features['feature'], 
        top_features['importance'],
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance (Top Features)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_prediction_vs_actual(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    model_name: str = 'Model',
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predicted vs actual values.
    
    Parameters:
    -----------
    y_actual : np.ndarray
        Actual values
    y_predicted : np.ndarray
        Predicted values
    model_name : str
        Model name for title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_actual, y_predicted, alpha=0.5, s=10, c='steelblue')
    
    # Perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax1.set_xlabel('Actual', fontsize=11)
    ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title(f'{model_name}: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals distribution
    ax2 = axes[1]
    residuals = y_actual - y_predicted
    ax2.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'Test RMSE',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model comparison bar chart.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Model comparison table
    metric : str
        Metric to compare
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = comparison_df['Model'].values
    
    # Handle string format of metrics
    values = comparison_df[metric].values
    if isinstance(values[0], str):
        values = [float(v.replace('%', '')) for v in values]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords='offset points',
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_liquidity_heatmap(
    df: pd.DataFrame,
    metric: str = 'composite_score',
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot liquidity heatmap over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Liquidity data
    metric : str
        Metric for heatmap
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot data
    pivot = df.pivot_table(
        index='ticker', 
        columns='date', 
        values=metric,
        aggfunc='mean'
    )
    
    # Resample to weekly for visibility if too many dates
    if len(pivot.columns) > 52:
        pivot.columns = pd.to_datetime(pivot.columns)
        pivot = pivot.T.resample('W').mean().T
    
    sns.heatmap(
        pivot, 
        cmap='RdYlGn', 
        center=50,
        ax=ax,
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    
    ax.set_title(f'Liquidity Heatmap: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Asset', fontsize=11)
    
    # Reduce x-tick labels
    n_ticks = 10
    step = max(1, len(pivot.columns) // n_ticks)
    ax.set_xticks(range(0, len(pivot.columns), step))
    ax.set_xticklabels(
        [str(pivot.columns[i])[:10] for i in range(0, len(pivot.columns), step)],
        rotation=45,
        ha='right'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix of liquidity metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Liquidity data
    columns : List[str], optional
        Columns to include
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if columns is None:
        columns = [
            'spread_proxy', 'amihud', 'volume_ratio', 'price_impact',
            'realized_vol', 'intraday_range', 'dollar_volume'
        ]
    
    columns = [c for c in columns if c in df.columns]
    corr = df[columns].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap='RdYlBu_r',
        center=0, 
        square=True, 
        linewidths=1, 
        ax=ax,
        vmin=-1, 
        vmax=1,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_title('Liquidity Metrics Correlation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print("Available functions:")
    print("  - plot_liquidity_timeseries()")
    print("  - plot_liquidity_comparison()")
    print("  - plot_risk_distribution()")
    print("  - plot_feature_importance()")
    print("  - plot_prediction_vs_actual()")
    print("  - plot_model_comparison()")
    print("  - plot_liquidity_heatmap()")
    print("  - plot_correlation_matrix()")
