# Liquidity Risk ML Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.readthedocs.io/)

A machine learning framework for **predicting liquidity risk metrics** including bid-ask spreads, market impact, and composite liquidity scores. This project implements industry-standard liquidity measures used by market makers, trading desks, and risk managers.

---

## 📋 Table of Contents

1. [What Is This Project?](#-what-is-this-project)
2. [Key Concepts](#-key-concepts)
3. [Features](#-features)
4. [Installation](#-installation)
5. [How to Run](#-how-to-run)
6. [Understanding the Output](#-understanding-the-output)
7. [Project Structure](#-project-structure)
8. [Module Documentation](#-module-documentation)
9. [Methodology](#-methodology)
10. [Sample Results](#-sample-results)
11. [Code Examples](#-code-examples)
12. [Customization](#-customization)
13. [Troubleshooting](#-troubleshooting)
14. [References](#-references)

---

## 🎯 What Is This Project?

This project predicts **liquidity risk** using machine learning. It answers critical questions:

- *"How expensive will it be to trade this asset tomorrow?"* → **Spread Prediction**
- *"What is the market impact of a large order?"* → **Price Impact Estimation**
- *"Which assets have liquidity risk?"* → **Liquidity Scoring**

### Real-World Applications

| Industry | Use Case |
|----------|----------|
| **Market Makers** | Bid-ask spread forecasting |
| **Trading Desks** | Transaction cost analysis (TCA) |
| **Asset Managers** | Portfolio liquidity monitoring |
| **Risk Management** | Liquidity stress testing |
| **Regulators** | Systemic liquidity risk |

---

## 📚 Key Concepts

### What is Liquidity Risk?

Liquidity risk is the risk that an asset cannot be traded quickly enough without significantly impacting its price.

**Two Types:**
1. **Market Liquidity Risk**: Difficulty trading due to market conditions
2. **Funding Liquidity Risk**: Inability to meet financial obligations

### Key Liquidity Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Bid-Ask Spread** | (Ask - Bid) / Mid | Cost of immediate execution |
| **Amihud Ratio** | \|Return\| / Dollar Volume | Price impact per dollar traded |
| **Roll Spread** | 2 × √(-Cov(rₜ, rₜ₋₁)) | Implied spread from price reversals |
| **Kyle's Lambda** | ΔPrice / √Volume | Market impact coefficient |
| **Turnover** | Volume / Avg Volume | Trading activity relative to normal |

### The Amihud Illiquidity Ratio

```
Amihud = |Daily Return| / Daily Dollar Volume × 10⁶

Higher Amihud = More Illiquid
Lower Amihud = More Liquid
```

**Example:**
- SPY: Amihud ≈ 0.001 (very liquid)
- Small-cap: Amihud ≈ 1.0 (illiquid)

### Corwin-Schultz Spread Estimator

Estimates bid-ask spread from daily high/low prices using the relationship between price range and spread.

---

## ✨ Features

### Liquidity Metrics (10 Measures)

| Metric | Description |
|--------|-------------|
| Bid-Ask Spread Proxy | Corwin-Schultz estimator |
| Roll Spread | Implied spread from return autocorrelation |
| Amihud Ratio | Price impact per dollar |
| Kyle's Lambda | Market impact coefficient |
| Dollar Volume | Trading value |
| Turnover | Relative trading activity |
| Realized Volatility | 20-day rolling volatility |
| Intraday Range | (High - Low) / Close |
| Volume Ratio | Volume / 20-day MA |
| High-Low Volatility | Parkinson volatility |

### Feature Engineering (60+ Features)

| Type | Features |
|------|----------|
| Lagged | 1, 2, 3, 5-day lags |
| Rolling | 5, 10, 20-day mean, std, max, skew |
| Interaction | Vol × Volume, Amihud ratio |
| Categorical | Volume regime, volatility regime |

### ML Models (5 Algorithms)

| Model | Type | Strength |
|-------|------|----------|
| Ridge Regression | Linear | Interpretable |
| Random Forest | Ensemble | Feature importance |
| Gradient Boosting | Boosting | Non-linear patterns |
| XGBoost | Gradient Boosting | Speed + accuracy |
| LightGBM | Gradient Boosting | Large datasets |

### Liquidity Scoring

| Score Range | Risk Level | Description |
|-------------|------------|-------------|
| 80-100 | Very Low Risk | Highly liquid |
| 60-80 | Low Risk | Liquid |
| 40-60 | Moderate Risk | Some concerns |
| 20-40 | High Risk | Illiquid |
| 0-20 | Very High Risk | Severely illiquid |

### Visualizations (8 Charts)

1. Liquidity time series
2. Asset comparison box plots
3. Risk distribution bar chart
4. Feature importance
5. Prediction vs actual scatter
6. Model comparison
7. Correlation matrix
8. Liquidity heatmap

---

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **pip** package manager
- **Git**

### Step 1: Clone Repository

```bash
git clone https://github.com/avniderashree/liquidity-risk-ml-predictor.git
cd liquidity-risk-ml-predictor
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import sklearn; import xgboost; print('✓ All packages installed')"
```

---

## 🚀 How to Run

### Option 1: Run Main Script

```bash
python main.py
```

**What happens:**
1. Fetches market data for 6 assets
2. Calculates 10 liquidity metrics
3. Engineers 60+ features
4. Trains 5 ML models
5. Performs time series cross-validation
6. Scores liquidity risk
7. Generates 8 visualizations
8. Saves models and results

**Expected runtime:** 30-60 seconds

### Option 2: Run Tests

```bash
pytest tests/ -v
```

### Option 3: Import as Library

```python
from src.data_loader import create_liquidity_dataset
from src.ml_models import train_all_models
from src.liquidity_scorer import LiquidityScoringEngine

# Create dataset
df = create_liquidity_dataset(tickers=['SPY', 'AAPL'])

# Score liquidity
scorer = LiquidityScoringEngine()
scored = scorer.calculate_composite_score(df)
```

---

## 📊 Understanding the Output

### Console Output

```
======================================================================
 LIQUIDITY RISK ML PREDICTOR
======================================================================

📊 Model Comparison:
            Model  Test RMSE  Test R²
    Random Forest   0.003778   0.2895
         LightGBM   0.003846   0.2637
 Ridge Regression   0.003904   0.2414

🏆 Best Model: Random Forest
   Test RMSE: 0.003778
   Test R²: 0.2895

Liquidity Score Summary by Asset:
        Avg Score    Typical Risk
SPY         88.87   Very Low Risk
QQQ         79.43   Very Low Risk
AAPL        71.81        Low Risk
MSFT        66.17        Low Risk
GOOGL       52.94   Moderate Risk

Risk Level Distribution:
  Low Risk: 3,840 (50.5%)
  Very Low Risk: 2,007 (26.4%)
  Moderate Risk: 1,542 (20.3%)
```

### Generated Files

**`./output/` folder:**

| File | Description |
|------|-------------|
| `liquidity_timeseries.png` | Time series of liquidity metrics |
| `liquidity_comparison.png` | Box plots comparing assets |
| `risk_distribution.png` | Distribution of risk levels |
| `feature_importance.png` | Top features for prediction |
| `prediction_vs_actual.png` | Model accuracy scatter |
| `model_comparison.png` | Model RMSE comparison |
| `correlation_matrix.png` | Metric correlations |
| `liquidity_heatmap.png` | Liquidity over time by asset |
| `liquidity_scores.csv` | All scored data |

**`./models/` folder:**

| File | Description |
|------|-------------|
| `best_model_*.pkl` | Trained best model |
| `feature_engineer.pkl` | Feature transformer |
| `liquidity_scorer.pkl` | Scoring engine |

---

## 📁 Project Structure

```
liquidity-risk-ml-predictor/
│
├── main.py                      # Main entry point
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
├── .gitignore                   # Git ignore rules
│
├── src/                         # Core modules
│   ├── __init__.py
│   ├── data_loader.py           # Data fetching & metrics
│   ├── liquidity_features.py    # Feature engineering
│   ├── ml_models.py             # ML model training
│   ├── liquidity_scorer.py      # Risk scoring
│   └── visualization.py         # Charts
│
├── tests/
│   └── test_liquidity.py        # Unit tests
│
├── models/                      # Saved models
├── output/                      # Generated charts/results
└── notebooks/                   # Jupyter notebooks
```

---

## 📖 Module Documentation

### `src/data_loader.py`

**Purpose:** Fetch market data and calculate liquidity metrics.

```python
from src.data_loader import create_liquidity_dataset

df = create_liquidity_dataset(
    tickers=['SPY', 'AAPL', 'MSFT'],
    start_date='2020-01-01'
)
```

### `src/liquidity_features.py`

**Purpose:** Feature engineering for ML prediction.

```python
from src.liquidity_features import LiquidityFeatureEngineer

engineer = LiquidityFeatureEngineer(target_col='spread_proxy')
feature_set = engineer.fit_transform(df, forecast_horizon=1)
```

### `src/ml_models.py`

**Purpose:** Train and compare ML models.

```python
from src.ml_models import train_all_models, get_best_model

results = train_all_models(X_train, y_train, X_test, y_test)
best = get_best_model(results)
```

### `src/liquidity_scorer.py`

**Purpose:** Composite liquidity scoring and risk classification.

```python
from src.liquidity_scorer import LiquidityScoringEngine

scorer = LiquidityScoringEngine()
scored = scorer.calculate_composite_score(df)
summary = scorer.get_score_summary(scored)
```

---

## 🔬 Methodology

### Data Pipeline

```
Raw OHLCV Data → Liquidity Metrics → Feature Engineering → 
Train/Test Split → Model Training → Cross-Validation → 
Best Model → Liquidity Scoring
```

### Liquidity Score Calculation

```python
composite_score = (
    0.25 × spread_score +      # Lower spread = higher score
    0.25 × volume_score +      # Higher volume = higher score
    0.20 × amihud_score +      # Lower Amihud = higher score
    0.15 × impact_score +      # Lower impact = higher score
    0.15 × volatility_score    # Lower vol = higher score
)
```

---

## 📊 Sample Results

### Model Performance

| Model | Test RMSE | Test R² |
|-------|-----------|---------|
| Random Forest | 0.003778 | 0.2895 |
| LightGBM | 0.003846 | 0.2637 |
| Ridge | 0.003904 | 0.2414 |

### Top Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | intraday_range | 0.1432 |
| 2 | high_low_vol | 0.1303 |
| 3 | volume_ratio_ma20 | 0.0281 |

---

## 💻 Code Examples

### Example: Score Custom Portfolio

```python
from src.data_loader import create_liquidity_dataset
from src.liquidity_scorer import LiquidityScoringEngine

df = create_liquidity_dataset(['NVDA', 'AMD', 'INTC'])
scorer = LiquidityScoringEngine()
scored = scorer.calculate_composite_score(df)
print(scorer.get_score_summary(scored))
```

---

## ⚙️ Customization

### Custom Scoring Weights

```python
custom_weights = {
    'spread_score': 0.15,
    'volume_score': 0.40,  # Emphasize volume
    'amihud_score': 0.20,
    'impact_score': 0.15,
    'volatility_score': 0.10
}

scorer = LiquidityScoringEngine(weights=custom_weights)
```

---

## 🔧 Troubleshooting

**yfinance fails:** Project auto-falls back to synthetic data.

**XGBoost import error:** `pip install xgboost lightgbm`

**Run tests:** `pytest tests/ -v`

---

## 📚 References

1. Amihud, Y. (2002). "Illiquidity and stock returns"
2. Corwin, S. & Schultz, P. (2012). "Bid-Ask Spread Estimation"
3. Roll, R. (1984). "Implicit Bid-Ask Spread Measure"
4. Kyle, A. (1985). "Continuous Auctions and Insider Trading"

---

## 👤 Author

**Avni Derashree**  
Quantitative Risk Analyst | Machine Learning | Market Microstructure

---

*Last updated: January 2026*
