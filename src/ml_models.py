"""
ML Models Module
Machine learning models for liquidity risk prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@dataclass
class ModelResult:
    """Container for model training results."""
    model_name: str
    model: object
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    mape: float
    feature_importance: Optional[pd.DataFrame]
    predictions: np.ndarray
    
    def summary(self) -> Dict:
        """Return summary metrics."""
        return {
            'Model': self.model_name,
            'Train RMSE': f"{self.train_rmse:.6f}",
            'Test RMSE': f"{self.test_rmse:.6f}",
            'Train MAE': f"{self.train_mae:.6f}",
            'Test MAE': f"{self.test_mae:.6f}",
            'Train R²': f"{self.train_r2:.4f}",
            'Test R²': f"{self.test_r2:.4f}",
            'MAPE': f"{self.mape:.2%}"
        }


def train_ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float = 1.0
) -> ModelResult:
    """
    Train Ridge regression model.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    alpha : Regularization strength
    
    Returns:
    --------
    ModelResult
        Training results
    """
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Feature importance from coefficients
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        model_name='Ridge Regression',
        model=model,
        train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
        test_rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
        train_mae=mean_absolute_error(y_train, train_pred),
        test_mae=mean_absolute_error(y_test, test_pred),
        train_r2=r2_score(y_train, train_pred),
        test_r2=r2_score(y_test, test_pred),
        mape=mean_absolute_percentage_error(y_test, test_pred),
        feature_importance=importance_df,
        predictions=test_pred
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = 10
) -> ModelResult:
    """
    Train Random Forest regressor.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    n_estimators : Number of trees
    max_depth : Maximum tree depth
    
    Returns:
    --------
    ModelResult
        Training results
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        model_name='Random Forest',
        model=model,
        train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
        test_rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
        train_mae=mean_absolute_error(y_train, train_pred),
        test_mae=mean_absolute_error(y_test, test_pred),
        train_r2=r2_score(y_train, train_pred),
        test_r2=r2_score(y_test, test_pred),
        mape=mean_absolute_percentage_error(y_test, test_pred),
        feature_importance=importance_df,
        predictions=test_pred
    )


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1
) -> ModelResult:
    """
    Train Gradient Boosting regressor.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    n_estimators : Number of boosting rounds
    max_depth : Maximum tree depth
    learning_rate : Learning rate
    
    Returns:
    --------
    ModelResult
        Training results
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        model_name='Gradient Boosting',
        model=model,
        train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
        test_rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
        train_mae=mean_absolute_error(y_train, train_pred),
        test_mae=mean_absolute_error(y_test, test_pred),
        train_r2=r2_score(y_train, train_pred),
        test_r2=r2_score(y_test, test_pred),
        mape=mean_absolute_percentage_error(y_test, test_pred),
        feature_importance=importance_df,
        predictions=test_pred
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1
) -> ModelResult:
    """
    Train XGBoost regressor.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    n_estimators : Number of boosting rounds
    max_depth : Maximum tree depth
    learning_rate : Learning rate
    
    Returns:
    --------
    ModelResult
        Training results
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        model_name='XGBoost',
        model=model,
        train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
        test_rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
        train_mae=mean_absolute_error(y_train, train_pred),
        test_mae=mean_absolute_error(y_test, test_pred),
        train_r2=r2_score(y_train, train_pred),
        test_r2=r2_score(y_test, test_pred),
        mape=mean_absolute_percentage_error(y_test, test_pred),
        feature_importance=importance_df,
        predictions=test_pred
    )


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1
) -> ModelResult:
    """
    Train LightGBM regressor.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    n_estimators : Number of boosting rounds
    max_depth : Maximum tree depth
    learning_rate : Learning rate
    
    Returns:
    --------
    ModelResult
        Training results
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        model_name='LightGBM',
        model=model,
        train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
        test_rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
        train_mae=mean_absolute_error(y_train, train_pred),
        test_mae=mean_absolute_error(y_test, test_pred),
        train_r2=r2_score(y_train, train_pred),
        test_r2=r2_score(y_test, test_pred),
        mape=mean_absolute_percentage_error(y_test, test_pred),
        feature_importance=importance_df,
        predictions=test_pred
    )


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, ModelResult]:
    """
    Train all available models.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    
    Returns:
    --------
    Dict[str, ModelResult]
        Results for each model
    """
    results = {}
    
    # Ridge Regression
    print("Training Ridge Regression...")
    results['ridge'] = train_ridge_model(X_train, y_train, X_test, y_test)
    
    # Random Forest
    print("Training Random Forest...")
    results['random_forest'] = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    results['gradient_boosting'] = train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    # XGBoost
    if HAS_XGBOOST:
        print("Training XGBoost...")
        results['xgboost'] = train_xgboost(X_train, y_train, X_test, y_test)
    
    # LightGBM
    if HAS_LIGHTGBM:
        print("Training LightGBM...")
        results['lightgbm'] = train_lightgbm(X_train, y_train, X_test, y_test)
    
    return results


def create_model_comparison(results: Dict[str, ModelResult]) -> pd.DataFrame:
    """
    Create comparison table of all models.
    
    Parameters:
    -----------
    results : Dict[str, ModelResult]
        Model results
    
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    rows = []
    for name, result in results.items():
        rows.append(result.summary())
    
    df = pd.DataFrame(rows)
    return df.sort_values('Test RMSE')


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Perform time series cross-validation.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to validate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    n_splits : int
        Number of CV folds
    
    Returns:
    --------
    Dict[str, float]
        CV scores
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_rmse = []
    cv_mae = []
    cv_r2 = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_cv, y_train_cv)
        
        preds = model_clone.predict(X_val_cv)
        
        cv_rmse.append(np.sqrt(mean_squared_error(y_val_cv, preds)))
        cv_mae.append(mean_absolute_error(y_val_cv, preds))
        cv_r2.append(r2_score(y_val_cv, preds))
    
    return {
        'cv_rmse_mean': np.mean(cv_rmse),
        'cv_rmse_std': np.std(cv_rmse),
        'cv_mae_mean': np.mean(cv_mae),
        'cv_mae_std': np.std(cv_mae),
        'cv_r2_mean': np.mean(cv_r2),
        'cv_r2_std': np.std(cv_r2)
    }


def get_best_model(results: Dict[str, ModelResult]) -> ModelResult:
    """
    Get the best model by test RMSE.
    
    Parameters:
    -----------
    results : Dict[str, ModelResult]
        Model results
    
    Returns:
    --------
    ModelResult
        Best model
    """
    best_name = min(results.keys(), key=lambda k: results[k].test_rmse)
    return results[best_name]


if __name__ == "__main__":
    print("Testing ML Models...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(
        0.01 + X['feature_0'] * 0.001 + X['feature_1'] * 0.0005 + np.random.randn(n_samples) * 0.001
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Train all models
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Compare
    print("\nModel Comparison:")
    print(create_model_comparison(results).to_string(index=False))
    
    # Best model
    best = get_best_model(results)
    print(f"\nBest Model: {best.model_name}")
    print(f"Test RMSE: {best.test_rmse:.6f}")
