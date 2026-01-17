"""
Liquidity Features Module
Feature engineering for liquidity risk prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass


@dataclass
class FeatureSet:
    """Container for feature engineering results."""
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]
    scaler: Optional[RobustScaler]
    
    def describe(self) -> pd.DataFrame:
        """Get feature statistics."""
        return self.X.describe()


class LiquidityFeatureEngineer:
    """
    Feature engineering pipeline for liquidity prediction.
    """
    
    def __init__(
        self, 
        target_col: str = 'spread_proxy',
        scale_features: bool = True
    ):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        target_col : str
            Target column name
        scale_features : bool
            Whether to scale features
        """
        self.target_col = target_col
        self.scale_features = scale_features
        self.scaler = RobustScaler() if scale_features else None
        self.feature_names: List[str] = []
        self.fitted = False
    
    def create_lagged_features(
        self, 
        df: pd.DataFrame, 
        cols: List[str], 
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged features for time series.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        cols : List[str]
            Columns to lag
        lags : List[int]
            Lag periods
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with lagged features
        """
        result = df.copy()
        
        for col in cols:
            for lag in lags:
                result[f'{col}_lag{lag}'] = df.groupby('ticker')[col].shift(lag)
        
        return result
    
    def create_rolling_features(
        self, 
        df: pd.DataFrame, 
        cols: List[str], 
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        cols : List[str]
            Columns to compute rolling stats for
        windows : List[int]
            Window sizes
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling features
        """
        result = df.copy()
        
        for col in cols:
            for window in windows:
                # Rolling mean
                result[f'{col}_ma{window}'] = (
                    df.groupby('ticker')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                
                # Rolling std
                result[f'{col}_std{window}'] = (
                    df.groupby('ticker')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )
                
                # Rolling max
                result[f'{col}_max{window}'] = (
                    df.groupby('ticker')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).max())
                )
                
                # Rolling skew (for larger windows)
                if window >= 10:
                    result[f'{col}_skew{window}'] = (
                        df.groupby('ticker')[col]
                        .transform(lambda x: x.rolling(window, min_periods=5).skew())
                    )
        
        return result
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between liquidity metrics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with interaction features
        """
        result = df.copy()
        
        # Volume-volatility interaction
        if 'volume_ratio' in df.columns and 'realized_vol' in df.columns:
            result['vol_volume_interaction'] = (
                df['volume_ratio'] * df['realized_vol']
            )
        
        # Amihud relative to moving average
        if 'amihud' in df.columns and 'amihud_ma20' in df.columns:
            result['amihud_ratio'] = (
                df['amihud'] / df['amihud_ma20'].replace(0, np.nan)
            )
        
        # Price impact normalized by volume
        if 'price_impact' in df.columns and 'volume' in df.columns:
            result['normalized_impact'] = (
                df['price_impact'] * np.log1p(df['volume'])
            )
        
        # Spread relative to volatility
        if 'spread_proxy' in df.columns and 'realized_vol' in df.columns:
            result['spread_vol_ratio'] = (
                df['spread_proxy'] / df['realized_vol'].replace(0, np.nan)
            )
        
        return result
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical/bucketed features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with categorical features
        """
        result = df.copy()
        
        # Volume regime
        if 'volume_ratio' in df.columns:
            result['volume_regime'] = pd.cut(
                df['volume_ratio'],
                bins=[0, 0.5, 1.0, 1.5, np.inf],
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        # Volatility regime
        if 'realized_vol' in df.columns:
            vol_percentiles = df.groupby('ticker')['realized_vol'].transform(
                lambda x: pd.qcut(x, q=4, labels=[0, 1, 2, 3], duplicates='drop')
            )
            result['vol_regime'] = vol_percentiles.astype(float)
        
        # Day of week
        if 'date' in df.columns:
            result['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        
        # Month
        if 'date' in df.columns:
            result['month'] = pd.to_datetime(df['date']).dt.month
        
        return result
    
    def create_target_features(
        self, 
        df: pd.DataFrame, 
        forecast_horizon: int = 1
    ) -> pd.DataFrame:
        """
        Create forward-looking target variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        forecast_horizon : int
            Days ahead to predict
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with target variable
        """
        result = df.copy()
        
        # Forward target
        result['target'] = (
            df.groupby('ticker')[self.target_col]
            .shift(-forecast_horizon)
        )
        
        # Direction (binary classification target)
        current = df[self.target_col]
        future = result['target']
        result['target_direction'] = (future > current).astype(int)
        
        # Change magnitude
        result['target_change'] = (future - current) / current.replace(0, np.nan)
        
        return result
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        forecast_horizon: int = 1
    ) -> FeatureSet:
        """
        Fit the feature engineer and transform data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input liquidity data
        forecast_horizon : int
            Days ahead to predict
        
        Returns:
        --------
        FeatureSet
            Complete feature set for ML
        """
        # Step 1: Create lagged features
        lag_cols = ['spread_proxy', 'amihud', 'volume_ratio', 'realized_vol']
        lag_cols = [c for c in lag_cols if c in df.columns]
        featured = self.create_lagged_features(df, lag_cols, lags=[1, 2, 3, 5])
        
        # Step 2: Create rolling features
        roll_cols = ['spread_proxy', 'volume_ratio', 'abs_returns']
        roll_cols = [c for c in roll_cols if c in df.columns]
        featured = self.create_rolling_features(featured, roll_cols, windows=[5, 10, 20])
        
        # Step 3: Create interaction features
        featured = self.create_interaction_features(featured)
        
        # Step 4: Create categorical features
        featured = self.create_categorical_features(featured)
        
        # Step 5: Create target
        featured = self.create_target_features(featured, forecast_horizon)
        
        # Drop rows with NaN target
        featured = featured.dropna(subset=['target'])
        
        # Select feature columns
        exclude_cols = [
            'date', 'ticker', 'target', 'target_direction', 'target_change',
            'returns', self.target_col
        ]
        
        self.feature_names = [
            col for col in featured.columns 
            if col not in exclude_cols and featured[col].dtype in ['float64', 'int64', 'float32']
        ]
        
        X = featured[self.feature_names].copy()
        y = featured['target'].copy()
        
        # Handle infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        for col in X.columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
        
        # Scale features
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        self.fitted = True
        
        return FeatureSet(
            X=X,
            y=y,
            feature_names=self.feature_names,
            scaler=self.scaler
        )
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted engineer.
        
        Parameters:
        -----------
        df : pd.DataFrame
            New data
        
        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        if not self.fitted:
            raise ValueError("Must call fit_transform first")
        
        # Apply same transformations
        lag_cols = ['spread_proxy', 'amihud', 'volume_ratio', 'realized_vol']
        lag_cols = [c for c in lag_cols if c in df.columns]
        featured = self.create_lagged_features(df, lag_cols, lags=[1, 2, 3, 5])
        
        roll_cols = ['spread_proxy', 'volume_ratio', 'abs_returns']
        roll_cols = [c for c in roll_cols if c in featured.columns]
        featured = self.create_rolling_features(featured, roll_cols, windows=[5, 10, 20])
        
        featured = self.create_interaction_features(featured)
        featured = self.create_categorical_features(featured)
        
        # Select features
        X = featured[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        
        for col in X.columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
        
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        return X


def calculate_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Create feature importance DataFrame.
    
    Parameters:
    -----------
    feature_names : List[str]
        List of feature names
    importances : np.ndarray
        Importance values
    top_n : int
        Number of top features to return
    
    Returns:
    --------
    pd.DataFrame
        Sorted feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df['cumulative'] = (
        importance_df['importance'].cumsum() / importance_df['importance'].sum()
    )
    
    return importance_df.head(top_n)


if __name__ == "__main__":
    print("Testing Feature Engineering...")
    
    # Create sample data
    from data_loader import create_liquidity_dataset
    
    df = create_liquidity_dataset(
        tickers=['SPY', 'AAPL'],
        start_date="2022-01-01"
    )
    
    # Engineer features
    engineer = LiquidityFeatureEngineer(target_col='spread_proxy')
    feature_set = engineer.fit_transform(df, forecast_horizon=1)
    
    print(f"\nFeature matrix shape: {feature_set.X.shape}")
    print(f"Target shape: {feature_set.y.shape}")
    print(f"\nTop 10 features:")
    for i, name in enumerate(feature_set.feature_names[:10], 1):
        print(f"  {i}. {name}")
