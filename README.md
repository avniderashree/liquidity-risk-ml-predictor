# Liquidity Risk ML Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green.svg)](https://lightgbm.readthedocs.io/)

A comprehensive **machine learning framework** for predicting liquidity risk metrics including bid-ask spreads, market impact, and composite liquidity scores. This project implements industry-standard liquidity measures used by market makers, trading desks, quantitative analysts, and risk managers at investment banks and hedge funds.

---

## 📋 Table of Contents

1. [What Is This Project?](#-what-is-this-project)
2. [Why Liquidity Risk Matters](#-why-liquidity-risk-matters)
3. [Key Concepts Explained](#-key-concepts-explained)
4. [Features](#-features)
5. [Installation](#-installation)
6. [How to Run](#-how-to-run)
7. [Understanding the Output](#-understanding-the-output)
8. [Project Structure](#-project-structure)
9. [Module Documentation](#-module-documentation)
10. [Methodology Deep Dive](#-methodology-deep-dive)
11. [Liquidity Metrics Explained](#-liquidity-metrics-explained)
12. [Sample Results](#-sample-results)
13. [Code Examples](#-code-examples)
14. [Customization Guide](#-customization-guide)
15. [Troubleshooting](#-troubleshooting)
16. [References](#-references)

---

## 🎯 What Is This Project?

This project uses **machine learning** to predict liquidity risk in financial markets. It answers critical questions that traders and risk managers face daily:

| Question | What This Project Provides |
|----------|----------------------------|
| *"How expensive will it be to trade this asset tomorrow?"* | **Spread Prediction** |
| *"What is the market impact of a large order?"* | **Price Impact Estimation** |
| *"Which assets in my portfolio have liquidity risk?"* | **Liquidity Scoring** |
| *"When are the most illiquid periods?"* | **Regime Detection** |
| *"What drives liquidity in my universe?"* | **Feature Importance** |

### The Core Problem

```
You want to sell $10 million of stock XYZ.

Scenario A (Liquid Market):
  • Bid-Ask Spread: 0.01%
  • Your cost: $1,000
  • Execution: Instant

Scenario B (Illiquid Market):
  • Bid-Ask Spread: 0.50%
  • Your cost: $50,000
  • Execution: Market moves against you

→ 50x cost difference! This is liquidity risk.
```

### Real-World Applications

| Industry | Use Case |
|----------|----------|
| **Market Makers** | Bid-ask spread forecasting for optimal quoting |
| **Trading Desks** | Transaction Cost Analysis (TCA), optimal execution |
| **Asset Managers** | Portfolio liquidity monitoring, stress testing |
| **Risk Management** | Liquidity VaR, regulatory reporting (Basel III) |
| **Hedge Funds** | Position sizing based on liquidity constraints |
| **Regulators** | Systemic liquidity risk monitoring |

---

## 🤔 Why Liquidity Risk Matters

### The 2008 Financial Crisis

During the 2008 crisis, liquidity evaporated almost overnight:

```
Before Crisis          During Crisis
─────────────          ─────────────
Bid-Ask: 0.02%    →    Bid-Ask: 2-5%
Volume: Normal    →    Volume: -80%
Market Depth: OK  →    No bids at all

Result: Fire sales, cascade failures, $trillions in losses
```

### The 2020 COVID Crash

Even Treasury markets (the "safest" assets) experienced severe liquidity stress:

```
March 9-18, 2020:
  • Treasury bid-ask spreads: 10x normal
  • VIX: Spiked to 82 (all-time high)
  • Fed intervention: $1.5 trillion in repos
```

### Why ML for Liquidity?

Traditional approaches assume liquidity is constant or uses simple moving averages. But liquidity is:

- **Non-linear**: Small changes in volatility cause large liquidity changes
- **Regime-dependent**: Crash periods behave completely differently
- **Cross-sectional**: Correlations between stocks matter
- **Time-varying**: Yesterday's liquidity ≠ tomorrow's

**Machine learning** can capture these complex patterns.

---

## 📚 Key Concepts Explained

### What is Liquidity?

**Liquidity** is the ability to buy or sell an asset quickly without significantly affecting its price.

```
High Liquidity:             Low Liquidity:
┌─────────────────┐         ┌─────────────────┐
│ Tight spreads   │         │ Wide spreads    │
│ High volume     │         │ Low volume      │
│ Low impact      │         │ High impact     │
│ Fast execution  │         │ Slow execution  │
└─────────────────┘         └─────────────────┘
     SPY, AAPL                Small-cap, OTC
```

### Two Types of Liquidity Risk

1. **Market Liquidity Risk**: The risk that you cannot execute a trade at fair value
   - Wide bid-ask spreads
   - Low trading volume
   - High price impact

2. **Funding Liquidity Risk**: The risk that you cannot meet cash obligations
   - Margin calls
   - Redemptions
   - Debt rollovers

This project focuses on **Market Liquidity Risk**.

### The Bid-Ask Spread

The **spread** is the difference between the best bid (buy) and ask (sell) prices:

```
           Order Book
    ┌───────────────────────┐
    │  ASK: $100.05 (sell)  │  ← You buy here
    │  ─────────────────    │
    │  BID: $100.00 (buy)   │  ← You sell here
    └───────────────────────┘
    
    Spread = $100.05 - $100.00 = $0.05 (0.05%)
```

**Spread as Cost:**
```
If you buy and immediately sell:
  Buy at:  $100.05
  Sell at: $100.00
  Loss:    $0.05 per share (the spread)
```

### Market Impact

**Market impact** is how much your trade moves the price:

```
Small Order (10,000 shares):
  Price before: $100.00
  Price after:  $100.01
  Impact: 0.01%

Large Order (1,000,000 shares):
  Price before: $100.00
  Price after:  $100.50
  Impact: 0.50%
```

**Kyle's Lambda (λ)** measures this:
```
Price Change = λ × √(Order Size)
```

---

## ✨ Features

### Liquidity Metrics (10 Industry-Standard Measures)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Bid-Ask Spread** | Corwin-Schultz | Trading cost |
| **Roll Spread** | 2√(-Cov(rₜ,rₜ₋₁)) | Implied spread |
| **Amihud Ratio** | \|r\|/DollarVolume | Price impact |
| **Kyle's Lambda** | Δp/√Volume | Market impact |
| **Dollar Volume** | Price × Volume | Trading activity |
| **Turnover** | Volume/AvgVolume | Relative activity |
| **Realized Volatility** | σ(returns) | Risk proxy |
| **Intraday Range** | (H-L)/C | Daily volatility |
| **Volume Ratio** | Vol/MA(Vol) | Volume surprise |
| **High-Low Vol** | Parkinson estimator | Range-based vol |

### Feature Engineering (68 Features)

The ML pipeline creates sophisticated features:

| Category | # Features | Examples |
|----------|------------|----------|
| **Base Metrics** | 10 | spread, amihud, volume_ratio |
| **Lagged (1,2,3,5 days)** | 16 | spread_lag1, amihud_lag5 |
| **Rolling Mean (5,10,20)** | 9 | spread_ma5, volume_ratio_ma20 |
| **Rolling Std** | 9 | spread_std10, abs_returns_std20 |
| **Rolling Max** | 9 | volume_ratio_max5 |
| **Rolling Skew** | 6 | abs_returns_skew20 |
| **Interactions** | 4 | vol_volume_interaction |
| **Categorical** | 5 | volume_regime, vol_regime |

### ML Models (5 Algorithms)

| Model | Type | When to Use |
|-------|------|-------------|
| **Ridge Regression** | Linear | Interpretability, baseline |
| **Random Forest** | Ensemble | Feature importance, robustness |
| **Gradient Boosting** | Boosting | Complex patterns |
| **XGBoost** | Gradient Boosting | Speed + accuracy |
| **LightGBM** | Gradient Boosting | Large datasets, fastest |

### Liquidity Scoring (5 Risk Levels)

| Score | Risk Level | Description | Color |
|-------|------------|-------------|-------|
| 80-100 | Very Low Risk | Extremely liquid, no concerns | 🟢 |
| 60-80 | Low Risk | Liquid, safe to trade | 🟡 |
| 40-60 | Moderate Risk | Some liquidity concerns | 🟠 |
| 20-40 | High Risk | Illiquid, trade carefully | 🔴 |
| 0-20 | Very High Risk | Severely illiquid, avoid | ⚫ |

### Visualizations (8 Publication-Quality Charts)

1. **Liquidity Time Series** - Track liquidity over time
2. **Asset Comparison** - Box plots comparing assets
3. **Risk Distribution** - Bar chart of risk levels
4. **Feature Importance** - Top predictive features
5. **Prediction vs Actual** - Model accuracy scatter
6. **Model Comparison** - RMSE by algorithm
7. **Correlation Matrix** - Metric relationships
8. **Liquidity Heatmap** - Asset × Time visualization

---

## 🛠️ Installation

### Prerequisites

- **Python 3.8+** (tested on 3.8, 3.9, 3.10, 3.11, 3.12)
- **pip** package manager
- **Git** for cloning

### Step 1: Clone the Repository

```bash
git clone https://github.com/avniderashree/liquidity-risk-ml-predictor.git
cd liquidity-risk-ml-predictor
```

### Step 2: Create Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `numpy`, `pandas` - Data manipulation
- `scipy` - Statistical functions
- `scikit-learn` - ML framework
- `xgboost`, `lightgbm` - Gradient boosting
- `matplotlib`, `seaborn` - Visualization
- `yfinance` - Market data (optional)
- `joblib` - Model persistence
- `pytest` - Testing

### Step 4: Verify Installation

```bash
python -c "import sklearn; import xgboost; import lightgbm; print('✓ All packages installed successfully')"
```

---

## 🚀 How to Run

### Option 1: Run Main Script (Recommended First-Time)

```bash
python main.py
```

**What happens (9 steps):**

1. **Data Loading**: Fetches 5+ years of OHLCV data for 6 assets
2. **Liquidity Metrics**: Calculates 10 industry-standard metrics
3. **Feature Engineering**: Creates 68 predictive features
4. **Train/Test Split**: 80/20 time-based split
5. **Model Training**: Trains 5 ML algorithms
6. **Cross-Validation**: 5-fold time series CV
7. **Liquidity Scoring**: Scores all observations
8. **Visualization**: Generates 8 charts
9. **Model Saving**: Saves best model + results

**Expected runtime:** 30-60 seconds

### Option 2: Run Unit Tests

```bash
pytest tests/ -v
```

### Option 3: Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Option 4: Import as Library

```python
from src.data_loader import create_liquidity_dataset
from src.liquidity_features import LiquidityFeatureEngineer
from src.ml_models import train_all_models, get_best_model
from src.liquidity_scorer import LiquidityScoringEngine

# Load data
df = create_liquidity_dataset(
    tickers=['AAPL', 'MSFT', 'SPY'],
    start_date='2022-01-01'
)

# Engineer features
engineer = LiquidityFeatureEngineer()
feature_set = engineer.fit_transform(df)

# Train models
results = train_all_models(X_train, y_train, X_test, y_test)
best = get_best_model(results)

# Score liquidity
scorer = LiquidityScoringEngine()
scored = scorer.calculate_composite_score(df)
```

---

## 📊 Understanding the Output

### Console Output (Step-by-Step)

When you run `python main.py`, you'll see:

```
======================================================================
 LIQUIDITY RISK ML PREDICTOR
======================================================================

This analysis performs:
  1. Liquidity metrics calculation (Amihud, Spread, Impact)
  2. Feature engineering for ML prediction
  3. Multiple ML model training and comparison
  4. Liquidity risk scoring and classification
  5. Visualization of results

----------------------------------------------------------------------
 STEP 1: Data Loading & Liquidity Metrics
----------------------------------------------------------------------

Assets: AAPL, MSFT, GOOGL, AMZN, SPY, QQQ
Fetched 1518 days of data for 6 assets
Created liquidity dataset: 7602 rows

Dataset shape: (7602, 17)
Date range: 2020-01-02 to 2026-01-14

Sample Liquidity Metrics:
ticker  spread_proxy   amihud  volume_ratio  price_impact
  AAPL      0.000000 0.006820      0.920632      0.008719
  AAPL      0.000000 0.006040      0.941521      0.007767
  ...

----------------------------------------------------------------------
 STEP 2: Feature Engineering
----------------------------------------------------------------------

Feature matrix shape: (7596, 68)
Target shape: (7596,)
Number of features: 68

Top 15 Features Created:
   1. abs_returns
   2. realized_vol
   3. high_low_vol
   4. volume
   5. volume_ma20
   ...

----------------------------------------------------------------------
 STEP 4: Model Training
----------------------------------------------------------------------
Training Ridge Regression...
Training Random Forest...
Training Gradient Boosting...
Training XGBoost...
Training LightGBM...

📊 Model Comparison:
            Model Train RMSE Test RMSE Train R² Test R²
    Random Forest   0.004853  0.003778   0.4166  0.2895
         LightGBM   0.004674  0.003846   0.4589  0.2637
 Ridge Regression   0.005957  0.003904   0.1210  0.2414
Gradient Boosting   0.004373  0.003930   0.5263  0.2314
          XGBoost   0.004210  0.004145   0.5610  0.1446

🏆 Best Model: Random Forest
   Test RMSE: 0.003778
   Test R²: 0.2895
   Test MAE: 0.002973

----------------------------------------------------------------------
 STEP 5: Time Series Cross-Validation
----------------------------------------------------------------------

Performing 5-fold time series CV on Random Forest...

Cross-Validation Results:
  RMSE: 0.005599 (±0.001315)
  MAE:  0.004518 (±0.001096)
  R²:   0.1033 (±0.0947)

----------------------------------------------------------------------
 STEP 6: Feature Importance
----------------------------------------------------------------------

Top 15 Most Important Features:
   1. intraday_range: 0.1432
   2. high_low_vol: 0.1303
   3. volume_ratio_ma20: 0.0281
   4. amihud_lag2: 0.0265
   5. volume_ratio_lag1: 0.0248
   ...

----------------------------------------------------------------------
 STEP 7: Liquidity Scoring
----------------------------------------------------------------------

Liquidity Score Summary by Asset:
        Avg Score  Std Dev  Min Score  Max Score   Typical Risk
SPY         88.87     5.45      49.11      98.98  Very Low Risk
QQQ         79.43     7.26      46.37      95.85  Very Low Risk
AAPL        71.81     7.33      35.69      93.01       Low Risk
MSFT        66.17     8.80      25.58      87.04       Low Risk
AMZN        62.84    10.64      24.55      85.94       Low Risk
GOOGL       52.94    10.78       9.21      79.80  Moderate Risk

Risk Level Distribution:
  Low Risk: 3,840 (50.5%)
  Very Low Risk: 2,007 (26.4%)
  Moderate Risk: 1,542 (20.3%)
  High Risk: 207 (2.7%)
  Very High Risk: 6 (0.1%)

----------------------------------------------------------------------
 ANALYSIS COMPLETE
----------------------------------------------------------------------

📊 Key Results:

  Data:
    • Assets analyzed: 6
    • Total observations: 7,602
    • Features engineered: 68

  Best Model (Random Forest):
    • Test RMSE: 0.003778
    • Test R²: 0.2895
    • CV RMSE: 0.005599 (±0.001315)

  Liquidity Scoring:
    • Average Score: 70.3
    • High Risk %: 2.8%

📁 Output files saved to ./output/
📁 Models saved to ./models/

Done! ✅
```

### Generated Files

**`./output/` folder (8 Charts + Data):**

| File | Description |
|------|-------------|
| `liquidity_timeseries.png` | 4-panel time series of liquidity metrics |
| `liquidity_comparison.png` | Box plots comparing assets by liquidity score |
| `risk_distribution.png` | Bar chart of risk level distribution |
| `feature_importance.png` | Top 15 predictive features |
| `prediction_vs_actual.png` | Scatter plot + residual histogram |
| `model_comparison.png` | RMSE by model bar chart |
| `correlation_matrix.png` | Triangular heatmap of metric correlations |
| `liquidity_heatmap.png` | Asset × Time liquidity heatmap |
| `liquidity_scores.csv` | Full scored dataset |
| `model_comparison.csv` | Model performance table |
| `training_results.pkl` | Saved training metrics |

**`./models/` folder (Serialized Models):**

| File | Description |
|------|-------------|
| `best_model_random_forest.pkl` | Trained best model |
| `feature_engineer.pkl` | Feature transformation pipeline |
| `liquidity_scorer.pkl` | Scoring engine |

---

## 📁 Project Structure

```
liquidity-risk-ml-predictor/
│
├── main.py                      # 🚀 Main execution script
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
├── .gitignore                   # Git ignore rules
│
├── src/                         # Core Python modules
│   ├── __init__.py              # Package marker
│   ├── data_loader.py           # OHLCV fetching + liquidity metrics
│   ├── liquidity_features.py    # Feature engineering pipeline
│   ├── ml_models.py             # 5 ML algorithms + training
│   ├── liquidity_scorer.py      # Composite scoring engine
│   └── visualization.py         # 8 chart functions
│
├── tests/
│   └── test_liquidity.py        # Unit tests (25+ tests)
│
├── output/                      # Generated charts & results
│   ├── *.png                    # Visualization outputs
│   ├── *.csv                    # Data exports
│   └── *.pkl                    # Serialized results
│
├── models/                      # Saved models
│   ├── best_model_*.pkl
│   ├── feature_engineer.pkl
│   └── liquidity_scorer.pkl
│
├── notebooks/                   # Jupyter notebooks (optional)
│
└── data/                        # Data files (optional)
```

---

## 📖 Module Documentation

### `src/data_loader.py`

**Purpose:** Fetch market data and calculate liquidity metrics.

**Key Functions:**

```python
# Fetch real market data (auto-fallback to synthetic)
from src.data_loader import fetch_market_data

prices = fetch_market_data(
    tickers=['SPY', 'AAPL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2024-12-31'  # Optional, defaults to today
)
```

```python
# Generate synthetic OHLCV data
from src.data_loader import generate_synthetic_ohlcv

prices = generate_synthetic_ohlcv(
    tickers=['STOCK_A', 'STOCK_B'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    random_state=42  # For reproducibility
)
```

```python
# Calculate all liquidity metrics
from src.data_loader import calculate_liquidity_metrics

metrics = calculate_liquidity_metrics(prices)
# Returns: DataFrame with spread_proxy, amihud, volume_ratio, etc.
```

```python
# One-liner to get complete dataset
from src.data_loader import create_liquidity_dataset

df = create_liquidity_dataset(
    tickers=['SPY', 'AAPL'],
    start_date='2022-01-01'
)
# Returns: DataFrame with 17 columns ready for ML
```

---

### `src/liquidity_features.py`

**Purpose:** Feature engineering for ML prediction.

**Key Class:**

```python
from src.liquidity_features import LiquidityFeatureEngineer

# Initialize
engineer = LiquidityFeatureEngineer(
    target_col='spread_proxy',  # What to predict
    scale_features=True         # Normalize features
)

# Fit and transform
feature_set = engineer.fit_transform(
    df,                    # Liquidity data
    forecast_horizon=1     # Predict 1 day ahead
)

# Access results
X = feature_set.X              # Features DataFrame (68 columns)
y = feature_set.y              # Target Series
names = feature_set.feature_names  # List of feature names
scaler = feature_set.scaler    # Fitted scaler for new data

# Transform new data
X_new = engineer.transform(new_df)
```

**Feature Categories:**

```python
# Lagged features
engineer.create_lagged_features(df, cols=['spread'], lags=[1, 2, 3])
# Creates: spread_lag1, spread_lag2, spread_lag3

# Rolling features
engineer.create_rolling_features(df, cols=['volume'], windows=[5, 10, 20])
# Creates: volume_ma5, volume_std5, volume_max5, volume_ma10, ...

# Interaction features
engineer.create_interaction_features(df)
# Creates: vol_volume_interaction, amihud_ratio, normalized_impact

# Categorical features
engineer.create_categorical_features(df)
# Creates: volume_regime (0-3), vol_regime (0-3), day_of_week, month
```

---

### `src/ml_models.py`

**Purpose:** Train and compare ML models.

**Training Functions:**

```python
from src.ml_models import (
    train_ridge_model,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    train_lightgbm,
    train_all_models
)

# Train single model
result = train_random_forest(
    X_train, y_train, X_test, y_test,
    n_estimators=100,
    max_depth=10
)

print(f"RMSE: {result.test_rmse:.4f}")
print(f"R²: {result.test_r2:.4f}")
print(result.feature_importance.head(10))

# Train all models at once
results = train_all_models(X_train, y_train, X_test, y_test)
# Returns: Dict with 5 ModelResult objects
```

**Model Comparison:**

```python
from src.ml_models import create_model_comparison, get_best_model

# Create comparison table
comparison_df = create_model_comparison(results)
print(comparison_df)

# Get best model by test RMSE
best = get_best_model(results)
print(f"Best: {best.model_name}")
```

**Cross-Validation:**

```python
from src.ml_models import cross_validate_model

cv_scores = cross_validate_model(
    model=best.model,
    X=X,
    y=y,
    n_splits=5  # 5-fold time series CV
)

print(f"CV RMSE: {cv_scores['cv_rmse_mean']:.4f} (±{cv_scores['cv_rmse_std']:.4f})")
```

---

### `src/liquidity_scorer.py`

**Purpose:** Composite liquidity scoring and risk classification.

**Key Class:**

```python
from src.liquidity_scorer import LiquidityScoringEngine

# Initialize with default weights
scorer = LiquidityScoringEngine()

# Or customize weights and thresholds
scorer = LiquidityScoringEngine(
    weights={
        'spread_score': 0.30,    # Increase spread importance
        'volume_score': 0.30,
        'amihud_score': 0.15,
        'impact_score': 0.15,
        'volatility_score': 0.10
    },
    risk_thresholds={
        'Very High Risk': 15,
        'High Risk': 35,
        'Moderate Risk': 55,
        'Low Risk': 75,
        'Very Low Risk': 100
    }
)

# Score data
scored_df = scorer.calculate_composite_score(df)

# Access scores
print(scored_df['composite_score'].describe())
print(scored_df['risk_level'].value_counts())
```

**Analysis Methods:**

```python
# Summary by ticker
summary = scorer.get_score_summary(scored_df)
print(summary)

# Risk distribution
risk_dist = scorer.get_risk_distribution(scored_df)
print(risk_dist)

# Find illiquid periods
illiquid = scorer.identify_illiquid_periods(scored_df, threshold=30)
print(illiquid)

# Score single observation
from src.liquidity_scorer import LiquidityScore

score = scorer.score_single_observation(
    metrics={'spread_proxy': 0.005, 'amihud': 0.01, 'volume': 1e8},
    ticker='AAPL',
    date='2024-01-15'
)
print(score)  # AAPL (2024-01-15): Score=72.5 (Low Risk)
```

---

### `src/visualization.py`

**Purpose:** Generate publication-quality charts.

**Available Functions:**

```python
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

# Time series
fig = plot_liquidity_timeseries(
    df, 
    ticker='AAPL',
    metrics=['spread_proxy', 'amihud', 'volume_ratio'],
    save_path='output/aapl_liquidity.png'
)

# Asset comparison
fig = plot_liquidity_comparison(
    scored_df,
    metric='composite_score',
    save_path='output/comparison.png'
)

# Feature importance
fig = plot_feature_importance(
    importance_df,
    top_n=15,
    save_path='output/features.png'
)

# Prediction scatter
fig = plot_prediction_vs_actual(
    y_actual=y_test.values,
    y_predicted=predictions,
    model_name='Random Forest'
)
```

---

## 🔬 Methodology Deep Dive

### Data Pipeline

```
┌─────────────────┐
│ 1. Data Source  │ ← yfinance or synthetic
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. OHLCV Data   │ ← Open, High, Low, Close, Volume
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────┐
│ 3. Liquidity Metrics (10)                   │
│   • Bid-Ask Spread (Corwin-Schultz)         │
│   • Roll Spread (Covariance-based)          │
│   • Amihud Illiquidity Ratio                │
│   • Price Impact (Kyle's Lambda proxy)      │
│   • Dollar Volume, Turnover                 │
│   • Realized Vol, Intraday Range            │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────┐
│ 4. Feature Engineering (68 features)        │
│   • Lagged: 1, 2, 3, 5-day                  │
│   • Rolling: 5, 10, 20-day mean/std/max     │
│   • Interactions: vol×volume, ratios        │
│   • Categorical: regimes, calendar          │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────┐
│ 5. Train/Test Split (80/20 time-based)      │
│   • Training: First 80% of dates            │
│   • Testing: Last 20% of dates              │
│   • NO shuffle (preserves time order)       │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────┐
│ 6. Model Training                           │
│   • Ridge Regression                        │
│   • Random Forest                           │
│   • Gradient Boosting                       │
│   • XGBoost                                 │
│   • LightGBM                                │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────┐
│ 7. Cross-Validation (5-fold time series)    │
│   Fold 1: Train[───] Test[─]                │
│   Fold 2: Train[─────] Test[─]              │
│   Fold 3: Train[───────] Test[─]            │
│   Fold 4: Train[─────────] Test[─]          │
│   Fold 5: Train[───────────] Test[─]        │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────┐
│ 8. Best Model Selection (by Test RMSE)      │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────┐
│ 9. Liquidity Scoring                        │
│   • Normalize each metric to 0-100          │
│   • Weight components (configurable)        │
│   • Classify into 5 risk levels             │
└─────────────────────────────────────────────┘
```

### Liquidity Score Calculation

```python
# Step 1: Normalize each metric to 0-100
spread_score = normalize(spread, higher_is_worse=True)
volume_score = normalize(log(volume), higher_is_worse=False)
amihud_score = normalize(amihud, higher_is_worse=True)
impact_score = normalize(impact, higher_is_worse=True)
vol_score = normalize(volatility, higher_is_worse=True)

# Step 2: Calculate weighted composite
composite = (
    0.25 * spread_score +
    0.25 * volume_score +
    0.20 * amihud_score +
    0.15 * impact_score +
    0.15 * vol_score
)

# Step 3: Classify risk level
if composite < 20:
    risk = "Very High Risk"
elif composite < 40:
    risk = "High Risk"
elif composite < 60:
    risk = "Moderate Risk"
elif composite < 80:
    risk = "Low Risk"
else:
    risk = "Very Low Risk"
```

---

## 📈 Liquidity Metrics Explained

### 1. Corwin-Schultz Bid-Ask Spread Estimator

Estimates spread from daily high/low prices:

```
β = [ln(H₁/L₁)]² + [ln(H₂/L₂)]²
γ = [ln(H(1,2)/L(1,2))]²

α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))

Spread = 2(e^α - 1) / (1 + e^α)
```

**Intuition:** If the true spread is wide, high-low ranges will be larger.

### 2. Amihud Illiquidity Ratio

```
Amihud = |Daily Return| / Dollar Volume × 10^10

Example:
  Return: 2%
  Dollar Volume: $500 million
  
  Amihud = 0.02 / 500,000,000 × 10^10 = 40
```

**Interpretation:**
- Higher Amihud = More price impact per dollar traded = Less liquid
- SPY: Amihud ≈ 0.001 (very liquid)
- Small-cap: Amihud ≈ 50-500 (illiquid)

### 3. Roll Spread Estimator

Based on negative autocorrelation in returns:

```
Roll Spread = 2 × √(-Cov(rₜ, rₜ₋₁))
```

**Intuition:** Bid-ask bounce creates negative autocorrelation.

### 4. Kyle's Lambda (Price Impact)

```
λ ≈ |Return| / √Volume × 10^4
```

Higher λ = trades move prices more = less liquid.

---

## 📊 Sample Results

### Model Performance

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
|-------|------------|-----------|----------|---------|
| Random Forest | 0.004853 | 0.003778 | 0.4166 | 0.2895 |
| LightGBM | 0.004674 | 0.003846 | 0.4589 | 0.2637 |
| Ridge | 0.005957 | 0.003904 | 0.1210 | 0.2414 |
| Gradient Boosting | 0.004373 | 0.003930 | 0.5263 | 0.2314 |
| XGBoost | 0.004210 | 0.004145 | 0.5610 | 0.1446 |

### Top Features

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | intraday_range | 14.32% | High-low as % of close |
| 2 | high_low_vol | 13.03% | Parkinson volatility |
| 3 | volume_ratio_ma20 | 2.81% | 20-day avg volume ratio |
| 4 | amihud_lag2 | 2.65% | Amihud 2 days ago |
| 5 | volume_ratio_lag1 | 2.48% | Volume ratio yesterday |

### Liquidity by Asset

| Asset | Avg Score | Std Dev | Risk Level |
|-------|-----------|---------|------------|
| SPY | 88.87 | 5.45 | Very Low Risk |
| QQQ | 79.43 | 7.26 | Very Low Risk |
| AAPL | 71.81 | 7.33 | Low Risk |
| MSFT | 66.17 | 8.80 | Low Risk |
| AMZN | 62.84 | 10.64 | Low Risk |
| GOOGL | 52.94 | 10.78 | Moderate Risk |

---

## 💻 Code Examples

### Example 1: Quick Start

```python
from src.data_loader import create_liquidity_dataset
from src.liquidity_scorer import LiquidityScoringEngine

# Get data and score
df = create_liquidity_dataset(['AAPL', 'TSLA', 'NVDA'])
scorer = LiquidityScoringEngine()
scored = scorer.calculate_composite_score(df)

# Print summary
print(scorer.get_score_summary(scored))
```

### Example 2: Custom Portfolio Analysis

```python
# Define your portfolio
my_tickers = ['AMZN', 'GOOGL', 'META', 'NFLX', 'MSFT']

# Get recent data
df = create_liquidity_dataset(
    tickers=my_tickers,
    start_date='2023-01-01'
)

# Score with custom weights (emphasize volume)
scorer = LiquidityScoringEngine(
    weights={
        'spread_score': 0.15,
        'volume_score': 0.40,
        'amihud_score': 0.20,
        'impact_score': 0.15,
        'volatility_score': 0.10
    }
)

scored = scorer.calculate_composite_score(df)

# Find least liquid days
illiquid = scorer.identify_illiquid_periods(scored, threshold=40)
print(illiquid[['date', 'ticker', 'composite_score']].head(10))
```

### Example 3: Train Your Own Model

```python
from src.data_loader import create_liquidity_dataset
from src.liquidity_features import LiquidityFeatureEngineer
from src.ml_models import train_xgboost

# Get data
df = create_liquidity_dataset(['SPY', 'QQQ', 'IWM'])

# Feature engineering
engineer = LiquidityFeatureEngineer()
fs = engineer.fit_transform(df, forecast_horizon=5)  # 5-day ahead

# Split
split = int(len(fs.X) * 0.8)
X_train, X_test = fs.X.iloc[:split], fs.X.iloc[split:]
y_train, y_test = fs.y.iloc[:split], fs.y.iloc[split:]

# Train with custom params
result = train_xgboost(
    X_train, y_train, X_test, y_test,
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05
)

print(f"Test R²: {result.test_r2:.4f}")
print(f"Top features:\n{result.feature_importance.head(10)}")
```

---

## ⚙️ Customization Guide

### Use Your Own Data

```python
import pandas as pd
from src.liquidity_features import LiquidityFeatureEngineer
from src.ml_models import train_all_models

# Load your CSV (must have these columns)
df = pd.read_csv('my_data.csv')
# Required: date, ticker, open, high, low, close, volume

# Calculate liquidity metrics
from src.data_loader import calculate_liquidity_metrics

# Create OHLCV MultiIndex format
ohlcv = ...  # Your data in yfinance format

metrics = calculate_liquidity_metrics(ohlcv)
```

### Add New Features

Edit `src/liquidity_features.py`:

```python
def create_my_custom_features(self, df):
    result = df.copy()
    
    # Your custom feature
    result['my_feature'] = df['volume'] / df['realized_vol']
    
    return result
```

### Change Scoring Weights

```python
scorer = LiquidityScoringEngine(
    weights={
        'spread_score': 0.40,    # Increase spread weight
        'volume_score': 0.20,
        'amihud_score': 0.15,
        'impact_score': 0.15,
        'volatility_score': 0.10
    }
)
```

---

## 🔧 Troubleshooting

### Common Issues

**yfinance download fails:**
```
Error: No data found for ticker XYZ
```
**Solution:** Project auto-falls back to synthetic data. To force real data:
```bash
pip install --upgrade yfinance
```

**XGBoost/LightGBM import error:**
```bash
pip install xgboost lightgbm
```

**Memory error with large datasets:**
Reduce date range or number of tickers.

**Matplotlib backend issues on macOS:**
```python
import matplotlib
matplotlib.use('Agg')
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Specific test class
pytest tests/test_liquidity.py::TestMLModels -v
```

---

## 📚 References

### Academic Papers

1. **Amihud, Y.** (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects." *Journal of Financial Markets* 5, 31-56.

2. **Corwin, S. & Schultz, P.** (2012). "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices." *Journal of Finance* 67(2), 719-760.

3. **Roll, R.** (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market." *Journal of Finance* 39(4), 1127-1139.

4. **Kyle, A.S.** (1985). "Continuous Auctions and Insider Trading." *Econometrica* 53(6), 1315-1335.

5. **Pastor, L. & Stambaugh, R.** (2003). "Liquidity Risk and Expected Stock Returns." *Journal of Political Economy* 111(3), 642-685.

### Books

- Harris, L. (2003). *Trading and Exchanges: Market Microstructure for Practitioners*
- Hasbrouck, J. (2007). *Empirical Market Microstructure*
- Foucault, T., Pagano, M., & Röell, A. (2013). *Market Liquidity*

### Regulatory Documents

- Basel III: Liquidity Coverage Ratio (LCR)
- Basel III: Net Stable Funding Ratio (NSFR)
- SEC Rule 15c3-5 Market Access

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Avni Derashree**  
Quantitative Risk Analyst | Machine Learning | Market Microstructure

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/avniderashree/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/avniderashree)

---

## 🔗 Related Projects

| Project | Description |
|---------|-------------|
| [Portfolio VaR Calculator](https://github.com/avniderashree/portfolio-var-calculator) | VaR using Historical, Parametric, Monte Carlo |
| [GARCH Volatility Forecaster](https://github.com/avniderashree/garch-volatility-forecaster) | GARCH/EGARCH volatility prediction |
| [Credit Risk PD/LGD Model](https://github.com/avniderashree/credit-risk-pd-lgd-model) | PD and LGD modeling with XGBoost |
| [Monte Carlo Stress Testing](https://github.com/avniderashree/monte-carlo-stress-testing) | Portfolio stress testing framework |
| **Liquidity Risk ML Predictor** | ← You are here! |

---

*Last updated: January 2026*
