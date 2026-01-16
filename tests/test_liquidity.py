"""
Unit Tests for Liquidity Risk ML Predictor
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


class TestDataLoader:
    """Tests for data_loader module."""
    
    def test_generate_synthetic_ohlcv(self):
        """Test synthetic OHLCV generation."""
        from src.data_loader import generate_synthetic_ohlcv
        
        tickers = ['SPY', 'AAPL']
        data = generate_synthetic_ohlcv(
            tickers=tickers,
            start_date="2023-01-01",
            end_date="2023-03-01",
            random_state=42
        )
        
        # Check structure
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.columns, pd.MultiIndex)
        
        # Check OHLCV columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert col in data.columns.get_level_values(0)
        
        # Check all tickers present
        for ticker in tickers:
            assert ticker in data['Close'].columns
        
        # Check OHLC consistency
        for ticker in tickers:
            assert (data['High'][ticker] >= data['Low'][ticker]).all()
            assert (data['High'][ticker] >= data['Open'][ticker]).all()
            assert (data['High'][ticker] >= data['Close'][ticker]).all()
            assert (data['Low'][ticker] <= data['Open'][ticker]).all()
            assert (data['Low'][ticker] <= data['Close'][ticker]).all()
    
    def test_calculate_liquidity_metrics(self):
        """Test liquidity metrics calculation."""
        from src.data_loader import generate_synthetic_ohlcv, calculate_liquidity_metrics
        
        ohlcv = generate_synthetic_ohlcv(
            tickers=['SPY'],
            start_date="2023-01-01",
            end_date="2023-06-01",
            random_state=42
        )
        
        metrics = calculate_liquidity_metrics(ohlcv)
        
        # Check required columns
        required_cols = [
            'ticker', 'date', 'returns', 'realized_vol', 
            'volume', 'amihud', 'spread_proxy', 'price_impact'
        ]
        for col in required_cols:
            assert col in metrics.columns, f"Missing column: {col}"
        
        # Check no NaN in key metrics (after dropna)
        assert not metrics['spread_proxy'].isna().all()
        assert not metrics['amihud'].isna().all()
    
    def test_create_liquidity_dataset(self):
        """Test full dataset creation."""
        from src.data_loader import create_liquidity_dataset
        
        df = create_liquidity_dataset(
            tickers=['SPY', 'AAPL'],
            start_date="2023-01-01",
            end_date="2023-03-01"
        )
        
        assert len(df) > 0
        assert 'ticker' in df.columns
        assert df['ticker'].nunique() == 2


class TestLiquidityFeatures:
    """Tests for liquidity_features module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample liquidity data."""
        np.random.seed(42)
        n = 200
        
        dates = pd.date_range('2023-01-01', periods=n, freq='B')
        
        df = pd.DataFrame({
            'date': np.tile(dates, 2),
            'ticker': ['SPY'] * n + ['AAPL'] * n,
            'spread_proxy': np.random.uniform(0.001, 0.01, n * 2),
            'amihud': np.random.uniform(0.0001, 0.001, n * 2),
            'amihud_ma20': np.random.uniform(0.0001, 0.001, n * 2),
            'volume_ratio': np.random.uniform(0.5, 2.0, n * 2),
            'realized_vol': np.random.uniform(0.1, 0.3, n * 2),
            'abs_returns': np.random.uniform(0, 0.03, n * 2),
            'price_impact': np.random.uniform(0.0001, 0.01, n * 2),
            'intraday_range': np.random.uniform(0.01, 0.03, n * 2),
            'volume': np.random.randint(1000000, 10000000, n * 2),
            'dollar_volume': np.random.uniform(1e8, 1e10, n * 2),
            'turnover': np.random.uniform(0.5, 2, n * 2)
        })
        
        return df
    
    def test_feature_engineer_fit_transform(self, sample_data):
        """Test feature engineering pipeline."""
        from src.liquidity_features import LiquidityFeatureEngineer
        
        engineer = LiquidityFeatureEngineer(
            target_col='spread_proxy',
            scale_features=True
        )
        
        feature_set = engineer.fit_transform(sample_data, forecast_horizon=1)
        
        # Check output
        assert feature_set.X is not None
        assert feature_set.y is not None
        assert len(feature_set.X) == len(feature_set.y)
        assert len(feature_set.feature_names) > 0
        assert engineer.fitted
    
    def test_lagged_features(self, sample_data):
        """Test lagged feature creation."""
        from src.liquidity_features import LiquidityFeatureEngineer
        
        engineer = LiquidityFeatureEngineer()
        result = engineer.create_lagged_features(
            sample_data, 
            cols=['spread_proxy'], 
            lags=[1, 2]
        )
        
        assert 'spread_proxy_lag1' in result.columns
        assert 'spread_proxy_lag2' in result.columns
    
    def test_rolling_features(self, sample_data):
        """Test rolling feature creation."""
        from src.liquidity_features import LiquidityFeatureEngineer
        
        engineer = LiquidityFeatureEngineer()
        result = engineer.create_rolling_features(
            sample_data, 
            cols=['volume_ratio'], 
            windows=[5]
        )
        
        assert 'volume_ratio_ma5' in result.columns
        assert 'volume_ratio_std5' in result.columns


class TestMLModels:
    """Tests for ml_models module."""
    
    @pytest.fixture
    def train_test_data(self):
        """Create sample train/test data."""
        np.random.seed(42)
        n = 500
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        y = pd.Series(
            0.01 + X['feature_0'] * 0.001 + np.random.randn(n) * 0.001
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_train_ridge(self, train_test_data):
        """Test Ridge regression training."""
        from src.ml_models import train_ridge_model
        
        X_train, X_test, y_train, y_test = train_test_data
        result = train_ridge_model(X_train, y_train, X_test, y_test)
        
        assert result.model is not None
        assert result.train_rmse >= 0
        assert result.test_rmse >= 0
        assert len(result.predictions) == len(y_test)
    
    def test_train_random_forest(self, train_test_data):
        """Test Random Forest training."""
        from src.ml_models import train_random_forest
        
        X_train, X_test, y_train, y_test = train_test_data
        result = train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=10, max_depth=5
        )
        
        assert result.model is not None
        assert result.feature_importance is not None
        assert len(result.feature_importance) == X_train.shape[1]
    
    def test_train_gradient_boosting(self, train_test_data):
        """Test Gradient Boosting training."""
        from src.ml_models import train_gradient_boosting
        
        X_train, X_test, y_train, y_test = train_test_data
        result = train_gradient_boosting(
            X_train, y_train, X_test, y_test,
            n_estimators=10, max_depth=3
        )
        
        assert result.model is not None
        assert result.test_r2 is not None
    
    def test_train_all_models(self, train_test_data):
        """Test training all models."""
        from src.ml_models import train_all_models, get_best_model
        
        X_train, X_test, y_train, y_test = train_test_data
        results = train_all_models(X_train, y_train, X_test, y_test)
        
        assert len(results) >= 3  # At least Ridge, RF, GB
        
        best = get_best_model(results)
        assert best.model is not None
    
    def test_model_comparison(self, train_test_data):
        """Test model comparison table."""
        from src.ml_models import train_all_models, create_model_comparison
        
        X_train, X_test, y_train, y_test = train_test_data
        results = train_all_models(X_train, y_train, X_test, y_test)
        
        comparison = create_model_comparison(results)
        
        assert 'Model' in comparison.columns
        assert 'Test RMSE' in comparison.columns
        assert len(comparison) == len(results)


class TestLiquidityScorer:
    """Tests for liquidity_scorer module."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics data."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n, freq='B'),
            'ticker': 'SPY',
            'spread_proxy': np.random.uniform(0.001, 0.01, n),
            'dollar_volume': np.random.uniform(1e9, 1e10, n),
            'amihud_ma20': np.random.uniform(0.0001, 0.001, n),
            'price_impact': np.random.uniform(0.0001, 0.01, n),
            'realized_vol': np.random.uniform(0.1, 0.3, n)
        })
    
    def test_composite_score_calculation(self, sample_metrics):
        """Test composite score calculation."""
        from src.liquidity_scorer import LiquidityScoringEngine
        
        engine = LiquidityScoringEngine()
        scored = engine.calculate_composite_score(sample_metrics)
        
        assert 'composite_score' in scored.columns
        assert 'risk_level' in scored.columns
        assert scored['composite_score'].between(0, 100).all()
    
    def test_risk_classification(self):
        """Test risk level classification."""
        from src.liquidity_scorer import LiquidityScoringEngine
        
        engine = LiquidityScoringEngine()
        
        assert engine.classify_risk(10) == 'Very High Risk'
        assert engine.classify_risk(30) == 'High Risk'
        assert engine.classify_risk(50) == 'Moderate Risk'
        assert engine.classify_risk(70) == 'Low Risk'
        assert engine.classify_risk(90) == 'Very Low Risk'
    
    def test_score_summary(self, sample_metrics):
        """Test score summary by ticker."""
        from src.liquidity_scorer import LiquidityScoringEngine
        
        engine = LiquidityScoringEngine()
        scored = engine.calculate_composite_score(sample_metrics)
        summary = engine.get_score_summary(scored)
        
        assert 'Avg Score' in summary.columns
        assert 'SPY' in summary.index


class TestVisualization:
    """Tests for visualization module."""
    
    def test_visualization_imports(self):
        """Test that visualization functions can be imported."""
        from src.visualization import (
            plot_liquidity_timeseries,
            plot_liquidity_comparison,
            plot_risk_distribution,
            plot_feature_importance,
            plot_prediction_vs_actual,
            plot_model_comparison,
            plot_liquidity_heatmap,
            plot_correlation_matrix
        )
        
        # All functions should be callable
        assert callable(plot_liquidity_timeseries)
        assert callable(plot_risk_distribution)
        assert callable(plot_feature_importance)


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full pipeline from data to scoring."""
        from src.data_loader import create_liquidity_dataset
        from src.liquidity_features import LiquidityFeatureEngineer
        from src.ml_models import train_random_forest
        from src.liquidity_scorer import LiquidityScoringEngine
        
        # Create dataset
        df = create_liquidity_dataset(
            tickers=['SPY'],
            start_date="2023-01-01",
            end_date="2023-03-01"
        )
        
        # Feature engineering
        engineer = LiquidityFeatureEngineer()
        feature_set = engineer.fit_transform(df, forecast_horizon=1)
        
        assert len(feature_set.X) > 0
        
        # Train model
        split_idx = int(len(feature_set.X) * 0.8)
        X_train = feature_set.X.iloc[:split_idx]
        X_test = feature_set.X.iloc[split_idx:]
        y_train = feature_set.y.iloc[:split_idx]
        y_test = feature_set.y.iloc[split_idx:]
        
        if len(X_train) > 10 and len(X_test) > 0:
            result = train_random_forest(
                X_train, y_train, X_test, y_test,
                n_estimators=10
            )
            assert result.model is not None
        
        # Scoring
        scorer = LiquidityScoringEngine()
        scored = scorer.calculate_composite_score(df)
        
        assert 'composite_score' in scored.columns
        assert 'risk_level' in scored.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
