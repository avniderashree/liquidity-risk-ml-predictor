"""
Liquidity Scorer Module
Composite liquidity scoring and risk classification.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class LiquidityScore:
    """Container for liquidity score results."""
    ticker: str
    date: str
    composite_score: float
    risk_level: str
    component_scores: Dict[str, float]
    percentile_rank: float
    
    def __str__(self) -> str:
        return (
            f"{self.ticker} ({self.date}): "
            f"Score={self.composite_score:.2f} ({self.risk_level})"
        )


class LiquidityScoringEngine:
    """
    Composite liquidity scoring engine.
    
    Combines multiple liquidity metrics into a single score
    and classifies assets by liquidity risk level.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        risk_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize scoring engine.
        
        Parameters:
        -----------
        weights : Dict[str, float], optional
            Weights for each liquidity component
        risk_thresholds : Dict[str, float], optional
            Thresholds for risk classification
        """
        # Default weights for liquidity components
        self.weights = weights or {
            'spread_score': 0.25,
            'volume_score': 0.25,
            'amihud_score': 0.20,
            'impact_score': 0.15,
            'volatility_score': 0.15
        }
        
        # Risk thresholds (higher score = better liquidity)
        self.risk_thresholds = risk_thresholds or {
            'Very High Risk': 20,
            'High Risk': 40,
            'Moderate Risk': 60,
            'Low Risk': 80,
            'Very Low Risk': 100
        }
        
        # Fitted parameters
        self.percentile_cache: Dict[str, np.ndarray] = {}
        self.fitted = False
    
    def _normalize_metric(
        self, 
        values: np.ndarray, 
        higher_is_worse: bool = True
    ) -> np.ndarray:
        """
        Normalize metric to 0-100 scale.
        
        Parameters:
        -----------
        values : np.ndarray
            Raw metric values
        higher_is_worse : bool
            If True, lower values get higher scores
        
        Returns:
        --------
        np.ndarray
            Normalized scores (0-100)
        """
        # Handle edge cases
        if len(values) == 0:
            return values
        
        # Clip outliers at 1st and 99th percentile
        p1, p99 = np.nanpercentile(values, [1, 99])
        clipped = np.clip(values, p1, p99)
        
        # Min-max scaling
        min_val, max_val = np.nanmin(clipped), np.nanmax(clipped)
        
        if max_val == min_val:
            return np.full_like(values, 50.0)
        
        normalized = (clipped - min_val) / (max_val - min_val) * 100
        
        if higher_is_worse:
            normalized = 100 - normalized
        
        return normalized
    
    def calculate_spread_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate bid-ask spread score.
        
        Lower spread = better liquidity = higher score.
        """
        if 'spread_proxy' in df.columns:
            return self._normalize_metric(
                df['spread_proxy'].values, 
                higher_is_worse=True
            )
        return np.full(len(df), 50.0)
    
    def calculate_volume_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate volume score.
        
        Higher volume = better liquidity = higher score.
        """
        if 'dollar_volume' in df.columns:
            log_volume = np.log1p(df['dollar_volume'].values)
            return self._normalize_metric(log_volume, higher_is_worse=False)
        elif 'volume' in df.columns:
            log_volume = np.log1p(df['volume'].values)
            return self._normalize_metric(log_volume, higher_is_worse=False)
        return np.full(len(df), 50.0)
    
    def calculate_amihud_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate Amihud illiquidity score.
        
        Lower Amihud = better liquidity = higher score.
        """
        if 'amihud_ma20' in df.columns:
            return self._normalize_metric(
                df['amihud_ma20'].values, 
                higher_is_worse=True
            )
        elif 'amihud' in df.columns:
            return self._normalize_metric(
                df['amihud'].values, 
                higher_is_worse=True
            )
        return np.full(len(df), 50.0)
    
    def calculate_impact_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate price impact score.
        
        Lower impact = better liquidity = higher score.
        """
        if 'price_impact' in df.columns:
            return self._normalize_metric(
                df['price_impact'].values, 
                higher_is_worse=True
            )
        return np.full(len(df), 50.0)
    
    def calculate_volatility_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate volatility-based score.
        
        Lower volatility = better liquidity conditions = higher score.
        """
        if 'realized_vol' in df.columns:
            return self._normalize_metric(
                df['realized_vol'].values, 
                higher_is_worse=True
            )
        return np.full(len(df), 50.0)
    
    def classify_risk(self, score: float) -> str:
        """
        Classify risk level based on composite score.
        
        Parameters:
        -----------
        score : float
            Composite liquidity score (0-100)
        
        Returns:
        --------
        str
            Risk level classification
        """
        for level, threshold in self.risk_thresholds.items():
            if score < threshold:
                return level
        return 'Very Low Risk'
    
    def calculate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite liquidity scores for all rows.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Liquidity metrics data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with composite scores and risk levels
        """
        result = df.copy()
        
        # Calculate component scores
        result['spread_score'] = self.calculate_spread_score(df)
        result['volume_score'] = self.calculate_volume_score(df)
        result['amihud_score'] = self.calculate_amihud_score(df)
        result['impact_score'] = self.calculate_impact_score(df)
        result['volatility_score'] = self.calculate_volatility_score(df)
        
        # Calculate weighted composite score
        component_cols = [
            'spread_score', 'volume_score', 'amihud_score', 
            'impact_score', 'volatility_score'
        ]
        
        result['composite_score'] = sum(
            result[col] * self.weights.get(col, 0.2)
            for col in component_cols
        )
        
        # Classify risk levels
        result['risk_level'] = result['composite_score'].apply(self.classify_risk)
        
        # Calculate percentile rank
        result['percentile_rank'] = result.groupby('ticker')['composite_score'].transform(
            lambda x: x.rank(pct=True) * 100
        )
        
        self.fitted = True
        
        return result
    
    def get_score_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics by ticker.
        
        Parameters:
        -----------
        scored_df : pd.DataFrame
            DataFrame with composite scores
        
        Returns:
        --------
        pd.DataFrame
            Summary by ticker
        """
        summary = scored_df.groupby('ticker').agg({
            'composite_score': ['mean', 'std', 'min', 'max'],
            'risk_level': lambda x: x.value_counts().index[0]  # Mode
        }).round(2)
        
        summary.columns = [
            'Avg Score', 'Std Dev', 'Min Score', 'Max Score', 'Typical Risk'
        ]
        
        return summary.sort_values('Avg Score', ascending=False)
    
    def get_risk_distribution(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get distribution of risk levels.
        
        Parameters:
        -----------
        scored_df : pd.DataFrame
            DataFrame with composite scores
        
        Returns:
        --------
        pd.DataFrame
            Risk level distribution
        """
        dist = scored_df['risk_level'].value_counts()
        dist_pct = (dist / len(scored_df) * 100).round(2)
        
        return pd.DataFrame({
            'Count': dist,
            'Percentage': dist_pct
        })
    
    def identify_illiquid_periods(
        self, 
        scored_df: pd.DataFrame,
        threshold: float = 30
    ) -> pd.DataFrame:
        """
        Identify periods of low liquidity.
        
        Parameters:
        -----------
        scored_df : pd.DataFrame
            DataFrame with composite scores
        threshold : float
            Score below which is considered illiquid
        
        Returns:
        --------
        pd.DataFrame
            Illiquid periods
        """
        illiquid = scored_df[scored_df['composite_score'] < threshold].copy()
        return illiquid.sort_values('composite_score')
    
    def score_single_observation(
        self,
        metrics: Dict[str, float],
        ticker: str,
        date: str
    ) -> LiquidityScore:
        """
        Score a single observation.
        
        Parameters:
        -----------
        metrics : Dict[str, float]
            Liquidity metrics
        ticker : str
            Asset ticker
        date : str
            Date
        
        Returns:
        --------
        LiquidityScore
            Score object
        """
        # Create single-row DataFrame
        df = pd.DataFrame([metrics])
        
        # Calculate scores
        component_scores = {
            'spread_score': float(self.calculate_spread_score(df)[0]),
            'volume_score': float(self.calculate_volume_score(df)[0]),
            'amihud_score': float(self.calculate_amihud_score(df)[0]),
            'impact_score': float(self.calculate_impact_score(df)[0]),
            'volatility_score': float(self.calculate_volatility_score(df)[0])
        }
        
        composite = sum(
            score * self.weights.get(name, 0.2)
            for name, score in component_scores.items()
        )
        
        return LiquidityScore(
            ticker=ticker,
            date=date,
            composite_score=composite,
            risk_level=self.classify_risk(composite),
            component_scores=component_scores,
            percentile_rank=50.0  # Placeholder
        )


def create_liquidity_report(
    scored_df: pd.DataFrame,
    engine: LiquidityScoringEngine
) -> str:
    """
    Create a text report of liquidity analysis.
    
    Parameters:
    -----------
    scored_df : pd.DataFrame
        Scored liquidity data
    engine : LiquidityScoringEngine
        Scoring engine used
    
    Returns:
    --------
    str
        Formatted report
    """
    lines = []
    lines.append("=" * 60)
    lines.append(" LIQUIDITY RISK REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Overall statistics
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total Observations: {len(scored_df):,}")
    lines.append(f"Date Range: {scored_df['date'].min()} to {scored_df['date'].max()}")
    lines.append(f"Assets Analyzed: {scored_df['ticker'].nunique()}")
    lines.append(f"Average Liquidity Score: {scored_df['composite_score'].mean():.1f}")
    lines.append("")
    
    # Risk distribution
    lines.append("RISK DISTRIBUTION")
    lines.append("-" * 40)
    dist = engine.get_risk_distribution(scored_df)
    for level, row in dist.iterrows():
        lines.append(f"  {level}: {row['Count']:,} ({row['Percentage']:.1f}%)")
    lines.append("")
    
    # By ticker
    lines.append("LIQUIDITY BY ASSET")
    lines.append("-" * 40)
    summary = engine.get_score_summary(scored_df)
    for ticker, row in summary.iterrows():
        lines.append(
            f"  {ticker}: Score={row['Avg Score']:.1f} "
            f"(±{row['Std Dev']:.1f}) - {row['Typical Risk']}"
        )
    lines.append("")
    
    # Illiquid periods
    illiquid = engine.identify_illiquid_periods(scored_df, threshold=30)
    if len(illiquid) > 0:
        lines.append("ILLIQUID PERIODS (Score < 30)")
        lines.append("-" * 40)
        for _, row in illiquid.head(5).iterrows():
            lines.append(
                f"  {row['ticker']} on {row['date']}: "
                f"Score={row['composite_score']:.1f}"
            )
        if len(illiquid) > 5:
            lines.append(f"  ... and {len(illiquid) - 5} more")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Testing Liquidity Scorer...")
    
    # Create sample data
    from data_loader import create_liquidity_dataset
    
    df = create_liquidity_dataset(
        tickers=['SPY', 'AAPL', 'MSFT'],
        start_date="2022-01-01"
    )
    
    # Score liquidity
    engine = LiquidityScoringEngine()
    scored = engine.calculate_composite_score(df)
    
    # Print report
    report = create_liquidity_report(scored, engine)
    print(report)
