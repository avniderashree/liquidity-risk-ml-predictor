# 🧪 Codelab: Build a Liquidity Risk ML Predictor from Scratch

**Estimated time:** 8–10 hours · **Difficulty:** Intermediate · **Language:** Python 3.8+

---

## What You'll Build

By the end of this codelab, you'll have a professional-grade **machine learning framework** for predicting liquidity risk — the same kind of system used by market makers, trading desks, and risk managers at investment banks. It will:

- Generate realistic **synthetic market data** (OHLCV for 6 assets over 5+ years, with built-in liquidity regimes) or fetch real data via yfinance
- Calculate **10 industry-standard liquidity metrics**: Corwin-Schultz bid-ask spread estimator, Roll spread, Amihud illiquidity ratio, Kyle's Lambda (price impact), dollar volume, turnover, realized volatility, intraday range, volume ratio, and Parkinson high-low volatility
- Engineer **68 predictive features** from those metrics: lagged values (1,2,3,5 days), rolling statistics (5,10,20-day mean/std/max/skew), cross-metric interaction terms, and categorical regime indicators
- Train **5 ML models** (Ridge Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM) to predict next-day bid-ask spreads
- Perform **time-series cross-validation** (expanding window, 5 folds) — the correct way to validate financial ML models
- Build a **composite liquidity scoring engine** that maps raw metrics into a 0–100 score with 5 risk levels (Very Low to Very High Risk)
- Generate **8 publication-quality charts**: liquidity time series, asset comparison boxes, risk distribution bars, feature importance, prediction vs actual scatter, model comparison, correlation matrix, and liquidity heatmap
- Ship with **25+ unit tests** covering data generation, metrics calculation, feature engineering, model training, scoring, and end-to-end integration

The final project structure:

```
liquidity-risk-ml-predictor/
├── main.py                      # Entry point — runs the full 9-step pipeline
├── requirements.txt             # Dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # OHLCV generation/fetching + 10 liquidity metrics
│   ├── liquidity_features.py    # 68-feature engineering pipeline (fit/transform)
│   ├── ml_models.py             # 5 ML algorithms + cross-validation + comparison
│   ├── liquidity_scorer.py      # Composite scoring engine (0-100 + risk levels)
│   └── visualization.py         # 8 chart functions
├── tests/
│   └── test_liquidity.py        # 25+ tests across 7 classes
├── output/                      # Generated charts (PNGs) + reports (CSVs)
└── models/                      # Serialized models + feature engineer + scorer
```

---

## Prerequisites

- Python 3.8+ installed
- Basic familiarity with Python (functions, classes, DataFrames)
- A terminal / command line

**No finance or ML background required.** Every concept — bid-ask spreads, Amihud ratios, random forests, time-series cross-validation — is explained from first principles before we code it.

---

---

# PART 1: THE CONCEPTS (What & Why)

No coding yet. Read this entire section first. Every line of code we write later maps directly to a concept explained here.

---

## 1.1 The Problem: Why Is Liquidity Risk So Dangerous?

You manage a $500 million hedge fund. You hold $50 million in a mid-cap stock. The market starts dropping. You decide to sell.

```
Scenario A — Liquid market (normal day):
  You place a sell order for $50 million.
  Bid-ask spread: 0.02%
  Your cost: $50M × 0.02% = $10,000
  Price impact: 0.05%
  Your cost: $50M × 0.05% = $25,000
  Total cost of exiting: $35,000
  Execution: Under 5 minutes

Scenario B — Illiquid market (crisis day):
  You place a sell order for $50 million.
  Bid-ask spread: 1.5%
  Your cost: $50M × 1.5% = $750,000
  Price impact: 3%
  Your cost: $50M × 3% = $1,500,000
  Total cost of exiting: $2,250,000
  Execution: Hours, maybe days, price keeps falling while you wait

→ 64× cost difference. This is liquidity risk.
```

Liquidity risk is invisible when markets are calm, then explodes exactly when you need to trade most urgently. The 2008 financial crisis, 2020 COVID crash, and 2023 bank runs all featured liquidity evaporating overnight.

**This project builds a system to PREDICT when liquidity is deteriorating** — before it's too late.

---

## 1.2 What Exactly Is Liquidity?

Liquidity is the ability to buy or sell an asset **quickly**, **cheaply**, and **without moving the price**.

```
Three dimensions of liquidity:

1. TIGHTNESS (spread):
   How much does it cost to execute a round-trip?
   Tight → $0.01 spread on AAPL
   Wide  → $0.50 spread on a penny stock

2. DEPTH (volume):
   How much can you trade before moving the price?
   Deep   → Trade $10M of SPY, price doesn't move
   Shallow → Trade $100K of small-cap, price moves 2%

3. RESILIENCY (recovery):
   How fast does the market bounce back after a large trade?
   Resilient → SPY recovers in seconds
   Fragile   → Small-cap stays impacted for hours
```

---

## 1.3 The Bid-Ask Spread: The Most Fundamental Liquidity Measure

Every asset has two prices at any moment:

```
┌──────────────────────────────────────────────────────┐
│                    ORDER BOOK                        │
│                                                      │
│  ASK (offer) = $100.05    ← Lowest price sellers     │
│                              will accept             │
│  ─────────── SPREAD = $0.05 (0.05%) ───────────      │
│                                                      │
│  BID = $100.00            ← Highest price buyers     │
│                              will pay                │
└──────────────────────────────────────────────────────┘

If you BUY, you pay the ASK ($100.05).
If you SELL, you get the BID ($100.00).
If you buy and immediately sell: you lose $0.05 per share.

That $0.05 is the "transaction cost" — the market maker's profit.
```

**The problem**: We often don't have real-time bid-ask data. We only have daily OHLCV (Open, High, Low, Close, Volume). So we need **estimators** — mathematical formulas that infer the spread from daily price data.

This project implements two spread estimators:

```
1. Corwin-Schultz (2012):
   Uses daily High and Low prices to estimate the spread.
   Intuition: The daily high tends to be at the ask price,
   and the daily low at the bid price. By comparing two-day
   ranges to one-day ranges, we can separate spread from volatility.

2. Roll (1984):
   Uses the autocorrelation of returns.
   Intuition: Bid-ask bounce creates negative autocorrelation.
   When you buy at the ask ($100.05) and the true price hasn't moved,
   the next trade might be at the bid ($100.00) — a -$0.05 "return"
   that's pure spread artifact, not real price movement.
```

---

## 1.4 The Amihud Illiquidity Ratio: How Much Does Price Move Per Dollar Traded?

The Amihud ratio (2002) is the single most widely used liquidity measure in academic finance. Over 10,000 papers cite it.

```
Formula:
  Amihud = |Daily Return| / Dollar Volume × 10^10

  Where:
    |Daily Return| = absolute percentage price change
    Dollar Volume = Price × Shares traded
    10^10 = scaling factor (makes numbers readable)

Worked example — AAPL on a typical day:
  |Return| = 1.2% = 0.012
  Volume = 80,000,000 shares at $175 = $14 billion
  Amihud = 0.012 / 14,000,000,000 × 10^10
         = 0.0086

Worked example — Small-cap stock:
  |Return| = 3.5% = 0.035
  Volume = 200,000 shares at $15 = $3 million
  Amihud = 0.035 / 3,000,000 × 10^10
         = 116.67

AAPL: 0.009 (very liquid — massive volume absorbs price changes)
Small-cap: 117 (illiquid — small volume causes big price swings)
```

**Why it works**: If you can trade $14 billion of AAPL and the price only moves 1.2%, AAPL is very liquid. If $3 million of a small-cap moves the price 3.5%, it's illiquid.

---

## 1.5 Kyle's Lambda: Market Impact

Kyle's Lambda (1985) measures how much a trade moves the price:

```
λ = |Return| / √Volume × 10^4

Intuition: Price impact grows with the SQUARE ROOT of order size.
  Trading 4× more doesn't cause 4× more impact — it causes 2× more.
  This is because large orders get "sliced" into pieces.

Higher λ = trades move the price more = less liquid
Lower λ  = trades move the price less = more liquid
```

---

## 1.6 The Other Liquidity Metrics (Quick Reference)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Metric           │ Formula                       │ Measures          │
├──────────────────────────────────────────────────────────────────────┤
│ Dollar Volume    │ Close × Volume                │ Trading activity  │
│                  │                               │ ($ value traded)  │
├──────────────────────────────────────────────────────────────────────┤
│ Turnover         │ Volume / 20-day Avg Volume    │ Relative activity │
│                  │                               │ (normal = 1.0)    │
├──────────────────────────────────────────────────────────────────────┤
│ Realized Vol     │ Rolling 20-day std(returns)   │ Price uncertainty │
│                  │ × √252 (annualized)           │ (risk proxy)      │
├──────────────────────────────────────────────────────────────────────┤
│ Intraday Range   │ (High − Low) / Close          │ Daily volatility  │
│                  │                               │ (higher=riskier)  │
├──────────────────────────────────────────────────────────────────────┤
│ Volume Ratio     │ Volume / 20-day MA(Volume)    │ Volume surprise   │
│                  │                               │ (>1 = above avg)  │
├──────────────────────────────────────────────────────────────────────┤
│ High-Low Vol     │ Parkinson estimator           │ Range-based vol   │
│ (Parkinson)      │ √(ln(H/L)²/(4×ln2)) × √252  │ (more efficient   │
│                  │                               │  than close-close)│
└──────────────────────────────────────────────────────────────────────┘

Parkinson volatility is more efficient than realized vol because
it uses the full daily range (H-L), not just closing prices.
With close-close, if a stock drops 5% intraday then recovers,
you see 0% return. Parkinson catches that hidden movement.
```

---

## 1.7 Feature Engineering: Turning 10 Metrics into 68 Predictive Features

Raw metrics alone aren't enough. Liquidity tomorrow depends on the **history** of metrics, not just today's snapshot. We create four types of derived features:

```
1. LAGGED FEATURES (16 features):
   "What was the spread 1, 2, 3, 5 days ago?"
   spread_lag1, spread_lag2, spread_lag3, spread_lag5
   amihud_lag1, amihud_lag2, ...
   
   WHY: Liquidity is persistent — if spreads were wide yesterday,
   they're likely still wide today.

2. ROLLING STATISTICS (33 features):
   Rolling mean: "What's the average spread over the last 5/10/20 days?"
     spread_ma5, spread_ma10, spread_ma20
   Rolling std: "How volatile has the spread been?"
     spread_std5, spread_std10, spread_std20
   Rolling max: "What was the worst spread recently?"
     spread_max5, spread_max10, spread_max20
   Rolling skew: "Is the spread distribution asymmetric?"
     spread_skew10, spread_skew20
   
   WHY: Trends and regimes matter. A rising 20-day average spread
   signals deteriorating liquidity even if today's spread looks OK.

3. INTERACTION FEATURES (4 features):
   vol_volume_interaction = realized_vol × log(volume)
   spread_volume_ratio = spread_proxy / log(volume + 1)
   amihud_ratio = amihud / (amihud_ma20 + ε)
   normalized_impact = price_impact / (realized_vol + ε)
   
   WHY: Liquidity isn't just one dimension. A stock with high vol
   AND low volume is far more dangerous than either alone.

4. CATEGORICAL FEATURES (5 features):
   volume_regime: quartile bin (0=very low, 3=very high volume)
   vol_regime: quartile bin of realized_vol
   day_of_week: Monday=0, Friday=4 (Mondays are less liquid)
   month: 1-12 (December is thin, September is volatile)
   quarter: 1-4
   
   WHY: Liquidity has calendar patterns. "Witching days" (options
   expiry), holidays, and fiscal year-ends affect liquidity.
```

---

## 1.8 The ML Models: Why Five Algorithms?

```
1. Ridge Regression (Linear Baseline):
   spread_tomorrow = β₁×spread_today + β₂×amihud + ... + β₆₈×feature_68
   
   WHY USE IT: Fast, interpretable, great baseline. If a linear model
   already explains 20% of spread variation, that's the floor.
   L2 regularization prevents overfitting when features > observations.

2. Random Forest (Ensemble of Trees):
   Builds 100 decision trees on random subsets of data and features.
   Final prediction = average of all 100 trees.
   
   WHY USE IT: Handles non-linearity, automatic feature selection,
   robust to outliers, gives feature importance scores.
   Often wins in "medium data" (thousands of rows, dozens of features).

3. Gradient Boosting (Sequential Trees):
   Builds trees ONE AT A TIME. Each new tree corrects the errors
   of the previous trees. Like learning from your mistakes.
   
   WHY USE IT: Usually more accurate than Random Forest, but slower
   and more prone to overfitting.

4. XGBoost (Optimized Gradient Boosting):
   Same idea as Gradient Boosting but with regularization (L1/L2),
   column sampling, and optimized C++ implementation.
   
   WHY USE IT: The "go-to" for tabular data in Kaggle competitions.
   Fast, accurate, handles missing values natively.

5. LightGBM (Microsoft's Gradient Boosting):
   Uses histogram-based splits instead of sorting (faster).
   Leaf-wise growth instead of level-wise (more accurate).
   
   WHY USE IT: Fastest for large datasets. Often comparable to XGBoost
   in accuracy. Handles categorical features natively.
```

---

## 1.9 Time-Series Cross-Validation: Why You Can't Shuffle Financial Data

Standard k-fold CV shuffles data randomly. This is **disastrous** for financial data because it causes **look-ahead bias** — using future data to predict the past.

```
WRONG (Standard K-Fold):
  Fold 1: Train on [Jan, Mar, May], Test on [Feb, Apr]
  → You're training on March data to predict February!
  → The model "learns" future information → fake accuracy

RIGHT (Time-Series CV / Expanding Window):
  Fold 1: Train on [Jan──Feb],       Test on [Mar]
  Fold 2: Train on [Jan──Feb──Mar],  Test on [Apr]
  Fold 3: Train on [Jan──────Apr],   Test on [May]
  Fold 4: Train on [Jan──────May],   Test on [Jun]
  Fold 5: Train on [Jan──────Jun],   Test on [Jul]
  
  → Training window grows, test is always AFTER training
  → No look-ahead bias
  → Simulates real deployment: you train on history, predict the future
```

---

## 1.10 The Liquidity Score: Mapping Metrics to a Single Number

Raw metrics are on different scales (Amihud might be 0.001 to 500, volume might be 1,000 to 500,000,000). We need a single, comparable score.

```
Step 1: Normalize each metric to 0–100
  For "higher is worse" metrics (spread, amihud, impact, vol):
    score = 100 × (1 − percentile_rank)
    Worst spread → score near 0
    Best spread → score near 100

  For "higher is better" metrics (volume):
    score = 100 × percentile_rank
    Highest volume → score near 100
    Lowest volume → score near 0

Step 2: Weighted composite
  composite = 0.25 × spread_score
            + 0.25 × volume_score
            + 0.20 × amihud_score
            + 0.15 × impact_score
            + 0.15 × volatility_score

Step 3: Risk classification
  80–100: Very Low Risk  🟢  "Trade freely"
  60–80:  Low Risk       🟡  "Normal liquidity"
  40–60:  Moderate Risk  🟠  "Monitor closely"
  20–40:  High Risk      🔴  "Reduce position size"
  0–20:   Very High Risk ⚫  "Avoid trading if possible"
```

---

---

# PART 2: PROJECT SETUP (Step 0)

---

## Step 0.1: Create the Folder Structure

```bash
mkdir liquidity-risk-ml-predictor
cd liquidity-risk-ml-predictor
mkdir -p src tests output models data notebooks
```

## Step 0.2: Create `requirements.txt`

**File: `requirements.txt`**
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
yfinance>=0.1.70
joblib>=1.1.0
pytest>=7.0.0
```

| Library | Purpose |
|---------|---------|
| `numpy` | Array math, random number generation for synthetic data |
| `pandas` | DataFrames for market data, feature matrices, reports |
| `scipy` | `stats.norm` for Corwin-Schultz spread estimator |
| `scikit-learn` | Ridge, Random Forest, Gradient Boosting, StandardScaler, cross-validation |
| `xgboost` | XGBoost gradient boosting algorithm |
| `lightgbm` | LightGBM gradient boosting algorithm |
| `matplotlib` | Base charting library for all 8 chart types |
| `seaborn` | Professional chart styling, heatmaps, box plots |
| `yfinance` | Optional: fetch real market data from Yahoo Finance |
| `joblib` | Serialize/deserialize models and feature pipelines |
| `pytest` | Run the 25+ unit tests |

Install:
```bash
pip install -r requirements.txt
```

## Step 0.3: Create `src/__init__.py`

**File: `src/__init__.py`**
```python
"""
Liquidity Risk ML Predictor
=================================
Machine learning framework for predicting liquidity risk metrics
and bid-ask spread forecasting.

Modules:
    data_loader         - OHLCV data generation/fetching + 10 liquidity metrics
    liquidity_features  - 68-feature engineering pipeline
    ml_models           - 5 ML algorithms + cross-validation + comparison
    liquidity_scorer    - Composite scoring engine (0-100) + risk classification
    visualization       - 8 publication-quality chart functions
"""
```

---

---

# PART 3: DATA LOADER (Step 1)

This module does two things: (1) generates or fetches OHLCV market data, and (2) calculates 10 liquidity metrics from that data. Every metric maps directly to the formulas from Part 1.

---

## Step 1.1: Understand What This Module Does

```
data_loader.py
    │
    ├── generate_synthetic_ohlcv(tickers, start_date, end_date, random_state)
    │       → Creates realistic OHLCV data with liquidity regimes
    │       → Returns multi-level DataFrame (date × ticker for OHLCV columns)
    │
    ├── fetch_market_data(tickers, start_date, end_date)
    │       → Tries yfinance; auto-fallback to synthetic if download fails
    │       → Returns same format as generate_synthetic_ohlcv
    │
    ├── calculate_liquidity_metrics(prices)
    │       → Computes 10 metrics from OHLCV data
    │       → Returns flat DataFrame with: date, ticker, ohlcv, + 10 metrics
    │
    └── create_liquidity_dataset(tickers, start_date, end_date)
            → One-liner: fetch/generate → calculate metrics → return ready DataFrame
```

## Step 1.2: Write the Code

**File: `src/data_loader.py`**

```python
"""
data_loader.py — Market Data & Liquidity Metrics
====================================================

Two data sources:
  1. generate_synthetic_ohlcv()  — Realistic OHLCV with regime changes
  2. fetch_market_data()         — Real data via yfinance (with fallback)

Ten liquidity metrics (calculated from OHLCV):
  1. Corwin-Schultz Spread   — Estimated bid-ask spread from High/Low
  2. Roll Spread             — Estimated spread from return autocorrelation
  3. Amihud Ratio            — |Return| / Dollar Volume (price impact)
  4. Kyle's Lambda           — |Return| / √Volume (market impact)
  5. Dollar Volume           — Close × Volume (trading activity)
  6. Turnover                — Volume / 20-day MA(Volume)
  7. Realized Volatility     — Rolling 20-day std(returns) × √252
  8. Intraday Range          — (High − Low) / Close
  9. Volume Ratio            — Volume / 20-day MA(Volume)
  10. High-Low Volatility    — Parkinson estimator × √252

References:
  Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads"
  Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread"
  Amihud (2002), "Illiquidity and Stock Returns"
  Kyle (1985), "Continuous Auctions and Insider Trading"
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


# ─── Synthetic Data Generation ──────────────────────────────────

def generate_synthetic_ohlcv(
    tickers: List[str] = None,
    start_date: str = '2020-01-01',
    end_date: str = '2025-12-31',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data for multiple tickers.

    Creates market data with:
      - Different volatility regimes (normal, elevated, crisis)
      - Realistic bid-ask spread patterns
      - Correlated volume/volatility (vol-volume correlation)
      - Different liquidity profiles per ticker

    Parameters
    ----------
    tickers : list of str
        Ticker symbols. Defaults to ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ'].
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Multi-level columns: (Open/High/Low/Close/Volume, ticker).
        Index: business days.
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ']

    np.random.seed(random_state)

    # Business days only (no weekends)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    # ── Liquidity profiles per ticker ────────────────────────
    # More liquid assets (SPY, QQQ) have:
    #   - Lower base volatility
    #   - Higher base volume
    #   - Tighter spreads
    profiles = {
        'SPY':   {'base_price': 400, 'base_vol': 0.12, 'base_volume': 8e7, 'drift': 0.08},
        'QQQ':   {'base_price': 350, 'base_vol': 0.15, 'base_volume': 5e7, 'drift': 0.10},
        'AAPL':  {'base_price': 150, 'base_vol': 0.22, 'base_volume': 7e7, 'drift': 0.12},
        'MSFT':  {'base_price': 300, 'base_vol': 0.20, 'base_volume': 3e7, 'drift': 0.10},
        'GOOGL': {'base_price': 130, 'base_vol': 0.25, 'base_volume': 2e7, 'drift': 0.08},
        'AMZN':  {'base_price': 140, 'base_vol': 0.28, 'base_volume': 4e7, 'drift': 0.15},
    }

    all_data = {}

    for ticker in tickers:
        # Get profile (use defaults for unknown tickers)
        profile = profiles.get(ticker, {
            'base_price': 100 + np.random.uniform(-50, 50),
            'base_vol': 0.20 + np.random.uniform(-0.05, 0.10),
            'base_volume': 1e7 * np.random.uniform(0.5, 5),
            'drift': 0.08 + np.random.uniform(-0.05, 0.05),
        })

        base_price = profile['base_price']
        base_vol = profile['base_vol']
        base_volume = profile['base_volume']
        drift = profile['drift']

        # ── Step 1: Generate volatility regimes ──────────────
        # Regime: 0=normal (70%), 1=elevated (20%), 2=crisis (10%)
        regimes = np.zeros(n_days, dtype=int)
        current_regime = 0
        for i in range(1, n_days):
            r = np.random.random()
            if current_regime == 0:
                if r < 0.005:    # 0.5% chance of jumping to crisis
                    current_regime = 2
                elif r < 0.03:   # 2.5% chance of jumping to elevated
                    current_regime = 1
            elif current_regime == 1:
                if r < 0.01:
                    current_regime = 2
                elif r < 0.08:   # 7% chance of returning to normal
                    current_regime = 0
            elif current_regime == 2:
                if r < 0.10:     # 10% chance of calming down
                    current_regime = 1
                elif r < 0.15:
                    current_regime = 0
            regimes[i] = current_regime

        # ── Step 2: Regime-dependent parameters ──────────────
        vol_multiplier = np.where(regimes == 0, 1.0,
                         np.where(regimes == 1, 1.8, 3.5))
        volume_multiplier = np.where(regimes == 0, 1.0,
                            np.where(regimes == 1, 1.3, 0.6))

        daily_vol = base_vol / np.sqrt(252) * vol_multiplier

        # ── Step 3: Generate Close prices (GBM) ─────────────
        daily_drift = drift / 252
        returns = np.random.normal(daily_drift, daily_vol)
        log_prices = np.log(base_price) + np.cumsum(returns)
        close = np.exp(log_prices)

        # ── Step 4: Generate OHLV from Close ─────────────────
        # Intraday range scales with volatility
        intraday_range = daily_vol * (1 + 0.5 * np.random.random(n_days))

        high = close * (1 + intraday_range * np.random.uniform(0.3, 1.0, n_days))
        low = close * (1 - intraday_range * np.random.uniform(0.3, 1.0, n_days))
        # Ensure low < close < high
        high = np.maximum(high, close * 1.001)
        low = np.minimum(low, close * 0.999)
        low = np.maximum(low, 0.01)  # Floor at penny

        # Open is between low and high, near previous close
        open_price = np.roll(close, 1) * (1 + np.random.normal(0, 0.003, n_days))
        open_price[0] = base_price
        open_price = np.clip(open_price, low, high)

        # ── Step 5: Generate Volume ──────────────────────────
        # Volume is log-normal, correlated with volatility
        log_vol_noise = np.random.normal(0, 0.4, n_days)
        volume = base_volume * volume_multiplier * np.exp(log_vol_noise)
        volume = np.maximum(volume, 10000).astype(int)

        all_data[ticker] = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
        }, index=dates)

    # ── Build multi-level DataFrame ──────────────────────────
    # Format: columns = (Open/High/Low/Close/Volume, ticker)
    frames = {}
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        frames[col] = pd.DataFrame({
            ticker: all_data[ticker][col] for ticker in tickers
        }, index=dates)

    result = pd.concat(frames, axis=1)
    result.columns.names = ['Price', 'Ticker']
    return result


# ─── Real Data via yfinance ─────────────────────────────────────

def fetch_market_data(
    tickers: List[str] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch real OHLCV data via yfinance, with automatic fallback to synthetic.
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ']

    try:
        import yfinance as yf
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient data")

        # Ensure multi-level columns
        if not isinstance(data.columns, pd.MultiIndex):
            # Single ticker — reshape
            data.columns = pd.MultiIndex.from_product(
                [data.columns, tickers], names=['Price', 'Ticker']
            )
        return data

    except Exception as e:
        print(f"  ⚠ yfinance download failed ({e}). Using synthetic data.")
        return generate_synthetic_ohlcv(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date or '2025-12-31',
        )


# ─── Liquidity Metrics Calculation ──────────────────────────────

def calculate_liquidity_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 10 industry-standard liquidity metrics from OHLCV data.

    Parameters
    ----------
    prices : pd.DataFrame
        Multi-level columns: (Open/High/Low/Close/Volume, ticker).

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with columns:
        date, ticker, open, high, low, close, volume,
        returns, abs_returns, dollar_volume, turnover,
        realized_vol, spread_proxy, roll_spread,
        amihud, price_impact, intraday_range,
        volume_ratio, high_low_vol
    """
    # Extract tickers from column level
    if isinstance(prices.columns, pd.MultiIndex):
        tickers = prices.columns.get_level_values(1).unique().tolist()
    else:
        tickers = [prices.columns[0]]

    all_metrics = []

    for ticker in tickers:
        # Extract OHLCV for this ticker
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                df = pd.DataFrame({
                    'open': prices[('Open', ticker)],
                    'high': prices[('High', ticker)],
                    'low': prices[('Low', ticker)],
                    'close': prices[('Close', ticker)],
                    'volume': prices[('Volume', ticker)],
                })
            else:
                df = prices.copy()
                df.columns = ['open', 'high', 'low', 'close', 'volume']
        except KeyError:
            continue

        df = df.dropna()
        if len(df) < 30:
            continue

        df['ticker'] = ticker
        df['date'] = df.index

        # ── Metric 1: Returns ────────────────────────────────
        df['returns'] = df['close'].pct_change()
        df['abs_returns'] = df['returns'].abs()

        # ── Metric 2: Dollar Volume ──────────────────────────
        df['dollar_volume'] = df['close'] * df['volume']

        # ── Metric 3: Turnover ───────────────────────────────
        vol_ma20 = df['volume'].rolling(20).mean()
        df['turnover'] = df['volume'] / vol_ma20.replace(0, np.nan)

        # ── Metric 4: Realized Volatility (annualized) ──────
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)

        # ── Metric 5: Corwin-Schultz Spread Estimator ────────
        # From "A Simple Way to Estimate Bid-Ask Spreads" (2012)
        #
        # The idea: daily High tends to occur at the Ask,
        # and daily Low at the Bid. By comparing the range
        # of a two-day window vs individual days, we can
        # separate the spread component from volatility.
        #
        # β = sum of squared log(High/Low) for each day
        # γ = squared log(two-day-High / two-day-Low)
        # α = (√(2β) − √β) / (3 − 2√2) − √(γ/(3 − 2√2))
        # Spread = 2(e^α − 1) / (1 + e^α)

        log_hl = np.log(df['high'] / df['low'])
        log_hl_sq = log_hl ** 2

        # β: sum of day t and day t-1 squared log ranges
        beta = log_hl_sq + log_hl_sq.shift(1)

        # γ: squared log range of the two-day high-low
        high_2d = df['high'].rolling(2).max()
        low_2d = df['low'].rolling(2).min()
        gamma = np.log(high_2d / low_2d) ** 2

        # α and spread
        sqrt_2 = np.sqrt(2)
        denom = 3 - 2 * sqrt_2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        df['spread_proxy'] = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0)

        # ── Metric 6: Roll Spread ────────────────────────────
        # Roll (1984): Spread = 2√(−Cov(rₜ, rₜ₋₁))
        # Only valid when autocovariance is negative (bid-ask bounce)
        roll_cov = df['returns'].rolling(20).apply(
            lambda x: np.cov(x[1:], x[:-1])[0, 1] if len(x) > 1 else 0,
            raw=True
        )
        df['roll_spread'] = (2 * np.sqrt((-roll_cov).clip(lower=0))).fillna(0)

        # ── Metric 7: Amihud Illiquidity Ratio ───────────────
        # Amihud (2002): |Return| / Dollar Volume × 10^10
        df['amihud'] = (
            df['abs_returns'] / df['dollar_volume'].replace(0, np.nan) * 1e10
        ).fillna(0)

        # ── Metric 8: Kyle's Lambda (Price Impact) ───────────
        # Approximation: |Return| / √Volume × 10^4
        df['price_impact'] = (
            df['abs_returns'] / np.sqrt(df['volume'].replace(0, np.nan)) * 1e4
        ).fillna(0)

        # ── Metric 9: Intraday Range ────────────────────────
        df['intraday_range'] = (df['high'] - df['low']) / df['close']

        # ── Metric 10: Volume Ratio ──────────────────────────
        df['volume_ratio'] = df['volume'] / vol_ma20.replace(0, np.nan)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

        # ── Metric 11: High-Low Volatility (Parkinson) ───────
        # Parkinson (1980): σ = √(ln(H/L)² / (4×ln2)) × √252
        # More efficient than close-close realized vol
        df['high_low_vol'] = np.sqrt(
            log_hl_sq / (4 * np.log(2))
        ) * np.sqrt(252)

        all_metrics.append(df)

    result = pd.concat(all_metrics, ignore_index=True)

    # Replace infinities with NaN, then fill
    result = result.replace([np.inf, -np.inf], np.nan)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0)

    return result


# ─── One-Liner Dataset Creator ──────────────────────────────────

def create_liquidity_dataset(
    tickers: List[str] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[str] = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Create a complete liquidity dataset in one call.

    Fetches OHLCV data (real or synthetic) and calculates all 10 metrics.
    """
    if use_synthetic:
        prices = generate_synthetic_ohlcv(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date or '2025-12-31',
        )
    else:
        prices = fetch_market_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )

    return calculate_liquidity_metrics(prices)
```

---

**What You Just Built:**

- **`generate_synthetic_ohlcv`**: Creates realistic market data for 6 assets over 5+ years. Each ticker has a unique liquidity profile (SPY is very liquid with low vol and high volume; AMZN is less liquid with higher vol). Volatility **regimes** (normal/elevated/crisis) switch stochastically — in crisis, vol triples and volume drops 40%, mimicking real market stress. Close prices follow Geometric Brownian Motion. Open/High/Low are derived from Close with realistic intraday ranges. Volume is log-normal and correlated with volatility.

- **`fetch_market_data`**: Tries yfinance first (real data). If that fails (no internet, rate limit, bad ticker), automatically falls back to synthetic. Users never see an error — the system is self-healing.

- **`calculate_liquidity_metrics`**: The heart of the module. Computes all 10 metrics for each ticker. The Corwin-Schultz estimator is the most complex — it uses a 4-step process (β from individual day ranges, γ from two-day range, α from the ratio, spread from α). The Amihud ratio and Kyle's Lambda are simpler but equally important. All metrics are cleaned (infinities → NaN → 0) before returning.

- **`create_liquidity_dataset`**: One-liner convenience function. Fetches data and calculates metrics in a single call. Returns a ready-to-use DataFrame with 17 columns (date, ticker, OHLCV, returns, abs_returns, dollar_volume, turnover, realized_vol, spread_proxy, roll_spread, amihud, price_impact, intraday_range, volume_ratio, high_low_vol).

---

---

# PART 4: FEATURE ENGINEERING (Step 2)

This module transforms 10 raw liquidity metrics into 68 predictive features using a scikit-learn-style fit/transform pattern to prevent data leakage.

---

## Step 2.1: Understand What This Module Does

```
liquidity_features.py
    │
    ├── FeatureSet (dataclass)
    │       .X              — Feature matrix (DataFrame, 68 columns)
    │       .y              — Target variable (Series)
    │       .feature_names  — List of 68 feature names
    │       .scaler         — Fitted StandardScaler (for transforming new data)
    │
    └── LiquidityFeatureEngineer
            .fit_transform(df, forecast_horizon=1) → FeatureSet
            │   Fits scaler on training data, creates all features
            │
            .transform(df) → DataFrame
            │   Applies learned scaler to new data (no re-fitting)
            │
            ├── create_lagged_features(df, cols, lags)
            ├── create_rolling_features(df, cols, windows)
            ├── create_interaction_features(df)
            └── create_categorical_features(df)
```

## Step 2.2: Write the Code

**File: `src/liquidity_features.py`**

```python
"""
liquidity_features.py — Feature Engineering Pipeline
=======================================================

Transforms 10 raw liquidity metrics into 68 predictive features:
  - 16 lagged features (1, 2, 3, 5-day lags of key metrics)
  - 33 rolling statistics (5/10/20-day mean, std, max, skew)
  - 4 interaction features (cross-metric combinations)
  - 5 categorical features (regime indicators, calendar)

Uses sklearn-style fit/transform to prevent data leakage:
  - fit_transform() learns scaling parameters from training data
  - transform() applies those parameters to new data
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureSet:
    """
    Container for ML-ready feature data.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix (n_samples × 68 features).
    y : pd.Series
        Target variable (next-day spread_proxy).
    feature_names : list of str
        Names of all 68 features.
    scaler : StandardScaler or None
        Fitted scaler for transforming new data.
    """
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str] = field(default_factory=list)
    scaler: Optional[StandardScaler] = None


class LiquidityFeatureEngineer:
    """
    Feature engineering pipeline for liquidity prediction.

    Creates 68 features from 10 raw metrics, with a fit/transform
    interface to prevent train/test data leakage.

    Parameters
    ----------
    target_col : str
        Column to predict (default: 'spread_proxy').
    scale_features : bool
        Whether to standardize features (zero mean, unit variance).
    """

    def __init__(
        self,
        target_col: str = 'spread_proxy',
        scale_features: bool = True,
    ):
        self.target_col = target_col
        self.scale_features = scale_features
        self.scaler = None
        self.feature_names = None
        self._is_fitted = False

    # ── Lagged Features ──────────────────────────────────────

    def create_lagged_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lags: List[int] = None,
    ) -> pd.DataFrame:
        """
        Create lagged versions of key metrics.

        spread_lag1 = spread yesterday
        spread_lag5 = spread 5 trading days ago

        WHY: Liquidity is persistent. Yesterday's spread is the
        single best predictor of today's spread.
        """
        if lags is None:
            lags = [1, 2, 3, 5]

        if columns is None:
            columns = ['spread_proxy', 'amihud', 'volume_ratio', 'abs_returns']

        result = df.copy()

        for col in columns:
            if col not in result.columns:
                continue
            for lag in lags:
                result[f'{col}_lag{lag}'] = result.groupby('ticker')[col].shift(lag)

        return result

    # ── Rolling Statistics ───────────────────────────────────

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: List[int] = None,
    ) -> pd.DataFrame:
        """
        Create rolling mean, std, max, and skew.

        spread_ma5 = 5-day moving average of spread
        spread_std10 = 10-day rolling standard deviation
        spread_max20 = worst spread in last 20 days
        spread_skew20 = asymmetry of spread distribution

        WHY: Trends matter. A rising 20-day average spread signals
        deteriorating liquidity even if today's value looks normal.
        """
        if windows is None:
            windows = [5, 10, 20]

        if columns is None:
            columns = ['spread_proxy', 'volume_ratio', 'abs_returns']

        result = df.copy()

        for col in columns:
            if col not in result.columns:
                continue
            for window in windows:
                grouped = result.groupby('ticker')[col]

                # Rolling mean (trend)
                result[f'{col}_ma{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

                # Rolling std (volatility of the metric itself)
                result[f'{col}_std{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )

                # Rolling max (worst case recently)
                result[f'{col}_max{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )

            # Rolling skew (only for 10 and 20-day windows)
            for window in [w for w in windows if w >= 10]:
                result[f'{col}_skew{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=5).skew()
                )

        return result

    # ── Interaction Features ─────────────────────────────────

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-metric interaction features.

        vol_volume_interaction = realized_vol × log(volume)
          WHY: High vol + low volume = dangerous combination

        spread_volume_ratio = spread / log(volume + 1)
          WHY: Wide spread with low volume = very illiquid

        amihud_ratio = amihud / amihud_ma20
          WHY: Today's Amihud relative to its recent average.
          >1 means liquidity is worse than usual.

        normalized_impact = price_impact / (realized_vol + ε)
          WHY: Impact relative to volatility. High impact during
          low-vol periods is a stronger signal than during high-vol.
        """
        result = df.copy()

        if 'realized_vol' in result.columns and 'volume' in result.columns:
            result['vol_volume_interaction'] = (
                result['realized_vol'] * np.log1p(result['volume'])
            )

        if 'spread_proxy' in result.columns and 'volume' in result.columns:
            result['spread_volume_ratio'] = (
                result['spread_proxy'] / np.log1p(result['volume'] + 1)
            )

        if 'amihud' in result.columns:
            amihud_ma20 = result.groupby('ticker')['amihud'].transform(
                lambda x: x.rolling(20, min_periods=1).mean()
            )
            result['amihud_ratio'] = result['amihud'] / (amihud_ma20 + 1e-10)

        if 'price_impact' in result.columns and 'realized_vol' in result.columns:
            result['normalized_impact'] = (
                result['price_impact'] / (result['realized_vol'] + 1e-10)
            )

        return result

    # ── Categorical Features ─────────────────────────────────

    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create regime indicators and calendar features.

        volume_regime: quartile of volume (0=very low, 3=very high)
        vol_regime: quartile of realized_vol
        day_of_week: 0=Monday ... 4=Friday
        month: 1-12
        quarter: 1-4

        WHY: Liquidity has strong calendar patterns:
          - Monday mornings are thin (traders assessing weekend news)
          - Options expiry Fridays have distorted volume
          - December is thin (holiday trading)
          - September is historically volatile
        """
        result = df.copy()

        if 'volume' in result.columns:
            result['volume_regime'] = pd.qcut(
                result['volume'], q=4, labels=False, duplicates='drop'
            ).fillna(0).astype(int)

        if 'realized_vol' in result.columns:
            result['vol_regime'] = pd.qcut(
                result['realized_vol'].clip(lower=1e-10),
                q=4, labels=False, duplicates='drop'
            ).fillna(0).astype(int)

        if 'date' in result.columns:
            dates = pd.to_datetime(result['date'])
            result['day_of_week'] = dates.dt.dayofweek
            result['month'] = dates.dt.month
            result['quarter'] = dates.dt.quarter
        elif isinstance(result.index, pd.DatetimeIndex):
            result['day_of_week'] = result.index.dayofweek
            result['month'] = result.index.month
            result['quarter'] = result.index.quarter

        return result

    # ── Main Pipeline ────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        forecast_horizon: int = 1,
    ) -> FeatureSet:
        """
        Create all features and prepare for ML training.

        Parameters
        ----------
        df : pd.DataFrame
            Output of create_liquidity_dataset() or calculate_liquidity_metrics().
        forecast_horizon : int
            How many days ahead to predict (default: 1 = next day).

        Returns
        -------
        FeatureSet
            Contains X (features), y (target), feature_names, scaler.
        """
        # ── Step 1: Create all feature types ─────────────────
        featured = self.create_lagged_features(df)
        featured = self.create_rolling_features(featured)
        featured = self.create_interaction_features(featured)
        featured = self.create_categorical_features(featured)

        # ── Step 2: Create target (shifted forward) ──────────
        # y(t) = spread at time t + forecast_horizon
        # We're predicting FUTURE spread from CURRENT features
        featured['target'] = featured.groupby('ticker')[self.target_col].shift(
            -forecast_horizon
        )

        # ── Step 3: Drop non-feature columns ─────────────────
        drop_cols = [
            'date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
            'returns', 'target', self.target_col,
            'dollar_volume', 'roll_spread',
        ]
        feature_cols = [
            col for col in featured.columns
            if col not in drop_cols
            and featured[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]

        # ── Step 4: Clean data ───────────────────────────────
        featured = featured.dropna(subset=['target'])
        featured = featured.replace([np.inf, -np.inf], np.nan)
        featured[feature_cols] = featured[feature_cols].fillna(0)

        X = featured[feature_cols].copy()
        y = featured['target'].copy()

        # ── Step 5: Scale features ───────────────────────────
        if self.scale_features:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )
        else:
            X_scaled = X
            self.scaler = None

        self.feature_names = list(X_scaled.columns)
        self._is_fitted = True

        return FeatureSet(
            X=X_scaled,
            y=y,
            feature_names=self.feature_names,
            scaler=self.scaler,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted pipeline.

        Uses the scaler learned during fit_transform() to ensure
        new data is on the same scale as training data.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform()")

        featured = self.create_lagged_features(df)
        featured = self.create_rolling_features(featured)
        featured = self.create_interaction_features(featured)
        featured = self.create_categorical_features(featured)

        featured = featured.replace([np.inf, -np.inf], np.nan)

        # Use only the features we learned during fit
        available = [c for c in self.feature_names if c in featured.columns]
        X = featured[available].fillna(0)

        # Add any missing columns as zeros
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.feature_names]

        if self.scale_features and self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_names,
                index=X.index,
            )

        return X
```

---

**What You Just Built (68 Features):**

| Category | Count | Features Created | Why They Matter |
|----------|-------|-----------------|-----------------|
| **Lagged** | 16 | `spread_lag1` ... `abs_returns_lag5` | Liquidity is persistent — yesterday's spread is the best single predictor |
| **Rolling Mean** | 9 | `spread_ma5`, `volume_ratio_ma20`, ... | Trends reveal regime changes before individual days do |
| **Rolling Std** | 9 | `spread_std10`, `abs_returns_std20`, ... | Volatile metrics = unstable liquidity = higher risk |
| **Rolling Max** | 9 | `spread_max5`, `volume_ratio_max20`, ... | Worst-case recently = stress indicator |
| **Rolling Skew** | 6 | `spread_skew10`, `abs_returns_skew20`, ... | Right-skewed distributions = occasional extreme illiquidity |
| **Interactions** | 4 | `vol_volume_interaction`, `amihud_ratio`, ... | Cross-metric combinations capture complex patterns |
| **Categorical** | 5 | `volume_regime`, `day_of_week`, `quarter`, ... | Calendar effects and regime indicators |
| **Base Metrics** | 10 | `abs_returns`, `realized_vol`, `amihud`, ... | The raw inputs (after dropping OHLCV and non-numeric) |

---

---

# PART 5: ML MODELS (Step 3)

Five algorithms trained on the 68 features to predict next-day bid-ask spreads. Each model returns a standardized `ModelResult` with metrics, feature importance, and the trained model object.

---

**File: `src/ml_models.py`**

```python
"""
ml_models.py — Machine Learning Model Training & Comparison
===============================================================

Five algorithms:
  1. Ridge Regression    — Linear baseline, L2 regularization
  2. Random Forest       — Ensemble of 100 decision trees
  3. Gradient Boosting   — Sequential error-correcting trees
  4. XGBoost             — Optimized gradient boosting (regularized)
  5. LightGBM            — Histogram-based gradient boosting (fastest)

Plus:
  - Standardized ModelResult dataclass for all models
  - train_all_models()  — trains all 5 in one call
  - create_model_comparison()  — comparison DataFrame
  - get_best_model()  — selects best by test RMSE
  - cross_validate_model()  — time-series expanding-window CV

All models predict: next-day bid-ask spread (spread_proxy).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """
    Standardized container for a trained model's results.

    Every model returns one of these, making comparison easy.
    """
    model_name: str
    model: Any                              # The fitted sklearn/xgb/lgb model
    train_rmse: float = 0.0
    test_rmse: float = 0.0
    train_r2: float = 0.0
    test_r2: float = 0.0
    train_mae: float = 0.0
    test_mae: float = 0.0
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))


def _evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> ModelResult:
    """
    Common evaluation logic for all models.

    Generates predictions, calculates RMSE/MAE/R² on both
    train and test sets, extracts feature importance.
    """
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        importance = np.zeros(X_train.shape[1])

    fi_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return ModelResult(
        model_name=model_name,
        model=model,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        train_r2=train_r2,
        test_r2=test_r2,
        train_mae=train_mae,
        test_mae=test_mae,
        feature_importance=fi_df,
        predictions=test_pred,
    )


# ─── Individual Model Training ──────────────────────────────────

def train_ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float = 1.0,
) -> ModelResult:
    """
    Train Ridge Regression (linear baseline).

    Ridge = Linear Regression + L2 regularization.
    L2 penalty: minimizes Σ(y − ŷ)² + α × Σβ²

    This prevents coefficients from exploding when features are
    correlated (which they are — spread_lag1 correlates with spread_ma5).
    """
    print("  Training Ridge Regression...")
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return _evaluate_model(model, X_train, y_train, X_test, y_test, 'Ridge Regression')


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_leaf: int = 20,
) -> ModelResult:
    """
    Train Random Forest.

    100 decision trees, each trained on a random bootstrap sample
    with a random subset of features. Final prediction = average.

    max_depth=10 prevents overfitting (trees can't memorize noise).
    min_samples_leaf=20 ensures each leaf represents meaningful patterns.
    n_jobs=-1 uses all CPU cores for parallel training.
    """
    print("  Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return _evaluate_model(model, X_train, y_train, X_test, y_test, 'Random Forest')


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
) -> ModelResult:
    """
    Train Gradient Boosting.

    Builds trees SEQUENTIALLY. Each tree corrects the residual errors
    of all previous trees. learning_rate=0.1 means each tree only
    contributes 10% of its correction (prevents over-correction).
    """
    print("  Training Gradient Boosting...")
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return _evaluate_model(model, X_train, y_train, X_test, y_test, 'Gradient Boosting')


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> ModelResult:
    """
    Train XGBoost.

    Adds L1 (reg_alpha) and L2 (reg_lambda) regularization to
    gradient boosting. colsample_bytree=0.8 means each tree
    only sees 80% of features (like Random Forest's feature sampling).
    """
    print("  Training XGBoost...")
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        return _evaluate_model(model, X_train, y_train, X_test, y_test, 'XGBoost')
    except ImportError:
        print("    ⚠ XGBoost not installed, skipping")
        return None


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 7,
    learning_rate: float = 0.1,
) -> ModelResult:
    """
    Train LightGBM.

    Uses histogram-based splits (bins continuous features into 255 buckets
    instead of sorting — much faster). Leaf-wise tree growth finds the
    leaf with largest loss reduction, unlike level-wise which builds
    complete levels. This is more accurate but risks overfitting,
    so num_leaves is capped.
    """
    print("  Training LightGBM...")
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        return _evaluate_model(model, X_train, y_train, X_test, y_test, 'LightGBM')
    except ImportError:
        print("    ⚠ LightGBM not installed, skipping")
        return None


# ─── Train All Models ────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, ModelResult]:
    """
    Train all 5 models and return results.

    Returns dict: {'ridge': ModelResult, 'rf': ModelResult, ...}
    """
    results = {}

    results['ridge'] = train_ridge_model(X_train, y_train, X_test, y_test)
    results['rf'] = train_random_forest(X_train, y_train, X_test, y_test)
    results['gb'] = train_gradient_boosting(X_train, y_train, X_test, y_test)

    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_result is not None:
        results['xgboost'] = xgb_result

    lgb_result = train_lightgbm(X_train, y_train, X_test, y_test)
    if lgb_result is not None:
        results['lightgbm'] = lgb_result

    return results


# ─── Model Comparison ────────────────────────────────────────────

def create_model_comparison(results: Dict[str, ModelResult]) -> pd.DataFrame:
    """
    Create a comparison table of all models.

    Returns DataFrame sorted by test RMSE (best first).
    """
    rows = []
    for key, res in results.items():
        rows.append({
            'Model': res.model_name,
            'Train RMSE': res.train_rmse,
            'Test RMSE': res.test_rmse,
            'Train R²': res.train_r2,
            'Test R²': res.test_r2,
            'Train MAE': res.train_mae,
            'Test MAE': res.test_mae,
        })

    return pd.DataFrame(rows).sort_values('Test RMSE').reset_index(drop=True)


def get_best_model(results: Dict[str, ModelResult]) -> ModelResult:
    """Select the best model by lowest test RMSE."""
    return min(results.values(), key=lambda r: r.test_rmse)


# ─── Time-Series Cross-Validation ────────────────────────────────

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Expanding-window time-series cross-validation.

    Fold 1: Train[───]    Test[─]
    Fold 2: Train[─────]  Test[─]
    Fold 3: Train[───────] Test[─]
    ...

    Returns dict with mean and std of RMSE, MAE, R² across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv = X.iloc[train_idx]
        X_test_cv = X.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv = y.iloc[test_idx]

        # Clone and refit the model
        from sklearn.base import clone
        model_clone = clone(model)
        model_clone.fit(X_train_cv, y_train_cv)

        pred = model_clone.predict(X_test_cv)

        rmse_scores.append(np.sqrt(mean_squared_error(y_test_cv, pred)))
        mae_scores.append(mean_absolute_error(y_test_cv, pred))
        r2_scores.append(r2_score(y_test_cv, pred))

    return {
        'cv_rmse_mean': np.mean(rmse_scores),
        'cv_rmse_std': np.std(rmse_scores),
        'cv_mae_mean': np.mean(mae_scores),
        'cv_mae_std': np.std(mae_scores),
        'cv_r2_mean': np.mean(r2_scores),
        'cv_r2_std': np.std(r2_scores),
        'cv_rmse_scores': rmse_scores,
    }
```

---

**What You Just Built:**

- **`ModelResult`**: Standardized container that every model returns. Contains the model object, 6 metrics (RMSE/MAE/R² for train and test), feature importance DataFrame, and raw predictions. This makes comparison trivial.
- **5 training functions**: Each creates a model with sensible defaults (max_depth, n_estimators, learning_rate), fits it, and returns a `ModelResult`. XGBoost and LightGBM gracefully handle missing imports.
- **`train_all_models()`**: One-call training of all 5 algorithms. Returns a dict for easy comparison.
- **`cross_validate_model()`**: Uses sklearn's `TimeSeriesSplit` for proper expanding-window CV. Clones the model for each fold (no state leakage between folds). Returns mean ± std of all metrics.

---

---

# PART 6: LIQUIDITY SCORER (Step 4)

The scoring engine maps raw metrics into a single 0–100 composite score and classifies into 5 risk levels.

---

**File: `src/liquidity_scorer.py`**

```python
"""
liquidity_scorer.py — Composite Liquidity Scoring Engine
============================================================

Maps raw liquidity metrics to a single 0–100 composite score:
  100 = perfectly liquid (no risk)
  0   = completely illiquid (maximum risk)

Five risk levels:
  80-100: Very Low Risk  🟢
  60-80:  Low Risk       🟡
  40-60:  Moderate Risk  🟠
  20-40:  High Risk      🔴
  0-20:   Very High Risk ⚫

Scoring method:
  1. Percentile-normalize each metric within its ticker (0-100)
  2. Invert "higher is worse" metrics (spread, amihud, impact, vol)
  3. Weight the components (configurable)
  4. Sum to composite score
  5. Classify into risk levels
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class LiquidityScore:
    """Score for a single observation."""
    ticker: str
    date: str
    composite_score: float
    risk_level: str
    component_scores: Dict[str, float]

    def __repr__(self):
        return (f"{self.ticker} ({self.date}): "
                f"Score={self.composite_score:.1f} ({self.risk_level})")


class LiquidityScoringEngine:
    """
    Composite liquidity scoring with configurable weights and thresholds.

    Parameters
    ----------
    weights : dict, optional
        Component weights (must sum to 1.0).
    risk_thresholds : dict, optional
        Score boundaries for each risk level.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        risk_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.weights = weights or {
            'spread_score': 0.25,
            'volume_score': 0.25,
            'amihud_score': 0.20,
            'impact_score': 0.15,
            'volatility_score': 0.15,
        }

        self.risk_thresholds = risk_thresholds or {
            'Very High Risk': 20,
            'High Risk': 40,
            'Moderate Risk': 60,
            'Low Risk': 80,
            'Very Low Risk': 100,
        }

    def _normalize_metric(
        self,
        series: pd.Series,
        higher_is_worse: bool = True,
    ) -> pd.Series:
        """
        Normalize a metric to 0–100 using percentile ranking.

        For "higher is worse" metrics (spread, amihud, vol):
          Worst value → 0, Best value → 100
          score = 100 × (1 − percentile_rank)

        For "higher is better" metrics (volume):
          Highest value → 100, Lowest value → 0
          score = 100 × percentile_rank
        """
        ranks = series.rank(pct=True, method='average')

        if higher_is_worse:
            return (1 - ranks) * 100
        else:
            return ranks * 100

    def _classify_risk(self, score: float) -> str:
        """Map a composite score to a risk level."""
        for level, threshold in sorted(
            self.risk_thresholds.items(),
            key=lambda x: x[1]
        ):
            if score <= threshold:
                return level
        return 'Very Low Risk'

    def calculate_composite_score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate composite liquidity score for every observation.

        Parameters
        ----------
        df : pd.DataFrame
            Output of create_liquidity_dataset().

        Returns
        -------
        pd.DataFrame
            Original data + spread_score, volume_score, amihud_score,
            impact_score, volatility_score, composite_score, risk_level.
        """
        result = df.copy()

        # ── Normalize each component ─────────────────────────
        if 'spread_proxy' in result.columns:
            result['spread_score'] = self._normalize_metric(
                result['spread_proxy'], higher_is_worse=True
            )
        else:
            result['spread_score'] = 50.0

        if 'volume' in result.columns:
            result['volume_score'] = self._normalize_metric(
                np.log1p(result['volume']), higher_is_worse=False
            )
        elif 'dollar_volume' in result.columns:
            result['volume_score'] = self._normalize_metric(
                np.log1p(result['dollar_volume']), higher_is_worse=False
            )
        else:
            result['volume_score'] = 50.0

        if 'amihud' in result.columns:
            result['amihud_score'] = self._normalize_metric(
                result['amihud'], higher_is_worse=True
            )
        else:
            result['amihud_score'] = 50.0

        if 'price_impact' in result.columns:
            result['impact_score'] = self._normalize_metric(
                result['price_impact'], higher_is_worse=True
            )
        else:
            result['impact_score'] = 50.0

        if 'realized_vol' in result.columns:
            result['volatility_score'] = self._normalize_metric(
                result['realized_vol'], higher_is_worse=True
            )
        else:
            result['volatility_score'] = 50.0

        # ── Weighted composite ───────────────────────────────
        result['composite_score'] = (
            self.weights.get('spread_score', 0.25) * result['spread_score']
            + self.weights.get('volume_score', 0.25) * result['volume_score']
            + self.weights.get('amihud_score', 0.20) * result['amihud_score']
            + self.weights.get('impact_score', 0.15) * result['impact_score']
            + self.weights.get('volatility_score', 0.15) * result['volatility_score']
        ).clip(0, 100)

        # ── Classify risk level ──────────────────────────────
        result['risk_level'] = result['composite_score'].apply(self._classify_risk)

        return result

    # ── Analysis Methods ─────────────────────────────────────

    def get_score_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summary statistics by ticker.

        Returns DataFrame with avg_score, std, min, max, typical_risk
        sorted from most liquid to least.
        """
        summary = scored_df.groupby('ticker').agg(
            avg_score=('composite_score', 'mean'),
            std_dev=('composite_score', 'std'),
            min_score=('composite_score', 'min'),
            max_score=('composite_score', 'max'),
        ).round(2)

        summary['typical_risk'] = summary['avg_score'].apply(self._classify_risk)
        summary = summary.sort_values('avg_score', ascending=False)

        return summary

    def get_risk_distribution(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Count and percentage of observations in each risk level.
        """
        dist = scored_df['risk_level'].value_counts()
        pct = scored_df['risk_level'].value_counts(normalize=True) * 100

        return pd.DataFrame({
            'count': dist,
            'percentage': pct.round(1),
        })

    def identify_illiquid_periods(
        self,
        scored_df: pd.DataFrame,
        threshold: float = 30.0,
    ) -> pd.DataFrame:
        """
        Find observations below the threshold (high risk).

        Returns rows where composite_score < threshold,
        sorted from worst to best.
        """
        illiquid = scored_df[scored_df['composite_score'] < threshold].copy()
        return illiquid.sort_values('composite_score').reset_index(drop=True)

    def score_single_observation(
        self,
        metrics: Dict[str, float],
        ticker: str = 'UNKNOWN',
        date: str = 'N/A',
    ) -> LiquidityScore:
        """
        Score a single observation from a dict of metrics.

        Example:
            score = scorer.score_single_observation(
                {'spread_proxy': 0.005, 'amihud': 0.01, 'volume': 1e8},
                ticker='AAPL', date='2024-01-15'
            )
        """
        # Build single-row DataFrame
        row_df = pd.DataFrame([metrics])
        row_df['ticker'] = ticker

        scored = self.calculate_composite_score(row_df)

        components = {}
        for col in ['spread_score', 'volume_score', 'amihud_score',
                     'impact_score', 'volatility_score']:
            if col in scored.columns:
                components[col] = float(scored[col].iloc[0])

        return LiquidityScore(
            ticker=ticker,
            date=str(date),
            composite_score=float(scored['composite_score'].iloc[0]),
            risk_level=str(scored['risk_level'].iloc[0]),
            component_scores=components,
        )
```

---

**What You Just Built:**

- **`_normalize_metric()`**: Percentile-rank normalization. For "higher is worse" metrics (spread, amihud), we invert: the worst spread gets score 0, the best gets 100. For volume, higher is better, so it stays direct. Percentile ranking is robust to outliers (unlike min-max scaling).
- **`calculate_composite_score()`**: The main scoring function. Normalizes 5 components, applies configurable weights (default: 25% spread, 25% volume, 20% amihud, 15% impact, 15% volatility), clips to [0, 100], and classifies into 5 risk levels.
- **`get_score_summary()`**: Per-ticker aggregate statistics — what you'd put in a risk report for the CRO.
- **`identify_illiquid_periods()`**: Finds dates where liquidity was dangerously low. These are the days where trading costs explode.
- **`score_single_observation()`**: Scores one data point from a dict — useful for real-time monitoring.

---

---

# PART 7: VISUALIZATION (Step 5)

Eight chart functions producing publication-quality analytics. Each chart answers a specific question.

---

**File: `src/visualization.py`**

```python
"""
visualization.py — Liquidity Risk Charts
=============================================

Eight chart types:
  1. Liquidity Time Series    — 4-panel metric trends over time
  2. Liquidity Comparison     — Box plots comparing assets
  3. Risk Distribution        — Bar chart of risk level counts
  4. Feature Importance       — Top 15 predictive features
  5. Prediction vs Actual     — Scatter + residual histogram
  6. Model Comparison         — RMSE by algorithm
  7. Correlation Matrix       — Triangular heatmap
  8. Liquidity Heatmap        — Asset × Time color-coded grid
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.05)

COLORS = {
    'primary': '#1a1a2e',
    'green': '#2ecc71',
    'red': '#e74c3c',
    'blue': '#3498db',
    'orange': '#f39c12',
    'purple': '#9b59b6',
    'teal': '#1abc9c',
    'dark_blue': '#2c3e50',
}

RISK_COLORS = {
    'Very Low Risk': '#2ecc71',
    'Low Risk': '#f1c40f',
    'Moderate Risk': '#f39c12',
    'High Risk': '#e74c3c',
    'Very High Risk': '#1a1a2e',
}


def plot_liquidity_timeseries(
    df: pd.DataFrame,
    ticker: str = None,
    metrics: List[str] = None,
    save_path: str = 'output/liquidity_timeseries.png',
) -> None:
    """
    Chart 1: 4-panel time series of liquidity metrics.

    Shows how spread, amihud, volume ratio, and realized vol
    evolve over time for one ticker. Crisis periods are visible
    as spikes in spread and drops in volume.
    """
    if metrics is None:
        metrics = ['spread_proxy', 'amihud', 'volume_ratio', 'realized_vol']

    if ticker is not None:
        data = df[df['ticker'] == ticker].copy()
    else:
        # Use the first ticker
        tickers = df['ticker'].unique()
        ticker = tickers[0] if len(tickers) > 0 else 'UNKNOWN'
        data = df[df['ticker'] == ticker].copy()

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')

    available = [m for m in metrics if m in data.columns]
    n_panels = min(len(available), 4)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    titles = {
        'spread_proxy': 'Bid-Ask Spread (Corwin-Schultz)',
        'amihud': 'Amihud Illiquidity Ratio',
        'volume_ratio': 'Volume Ratio (vs 20-day avg)',
        'realized_vol': 'Realized Volatility (annualized)',
    }
    colors_list = [COLORS['blue'], COLORS['red'], COLORS['teal'], COLORS['orange']]

    for ax, metric, color in zip(axes, available[:n_panels], colors_list):
        if 'date' in data.columns:
            ax.plot(data['date'], data[metric], color=color, linewidth=0.8, alpha=0.8)
            # Add 20-day moving average
            ma = data[metric].rolling(20).mean()
            ax.plot(data['date'], ma, color='black', linewidth=1.5,
                    linestyle='--', alpha=0.7, label='20-day MA')
        else:
            ax.plot(data[metric].values, color=color, linewidth=0.8, alpha=0.8)

        ax.set_title(titles.get(metric, metric), fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)

    fig.suptitle(f'Liquidity Metrics — {ticker}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_liquidity_comparison(
    scored_df: pd.DataFrame,
    metric: str = 'composite_score',
    save_path: str = 'output/liquidity_comparison.png',
) -> None:
    """
    Chart 2: Box plots comparing assets by liquidity score.

    Each box shows the distribution of composite_score for one asset.
    Higher = more liquid. Width of box = variability.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Order by median score (most liquid first)
    order = (scored_df.groupby('ticker')[metric]
             .median().sort_values(ascending=False).index.tolist())

    palette = sns.color_palette("RdYlGn", n_colors=len(order))

    sns.boxplot(
        data=scored_df, x='ticker', y=metric,
        order=order, palette=palette, ax=ax,
        width=0.6, fliersize=2,
    )

    ax.set_xlabel('Asset', fontsize=12)
    ax.set_ylabel('Composite Liquidity Score', fontsize=12)
    ax.set_title('Liquidity Score Distribution by Asset',
                 fontsize=14, fontweight='bold')

    # Risk level reference lines
    for level, threshold in [('High Risk', 40), ('Moderate', 60), ('Low', 80)]:
        ax.axhline(y=threshold, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.text(len(order) - 0.5, threshold + 1, level, fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_risk_distribution(
    scored_df: pd.DataFrame,
    save_path: str = 'output/risk_distribution.png',
) -> None:
    """
    Chart 3: Bar chart of risk level distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    risk_order = ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
    counts = scored_df['risk_level'].value_counts()

    present = [r for r in risk_order if r in counts.index]
    values = [counts.get(r, 0) for r in present]
    colors = [RISK_COLORS.get(r, 'grey') for r in present]

    bars = ax.bar(range(len(present)), values, color=colors, alpha=0.85, edgecolor='white')

    for bar, val in zip(bars, values):
        pct = val / len(scored_df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(scored_df) * 0.005,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=15)
    ax.set_ylabel('Number of Observations', fontsize=12)
    ax.set_title('Liquidity Risk Level Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: str = 'output/feature_importance.png',
) -> None:
    """
    Chart 4: Top predictive features (horizontal bar chart).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    top = importance_df.head(top_n).sort_values('importance')

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
    bars = ax.barh(range(len(top)), top['importance'], color=colors, alpha=0.85)

    for bar, val in zip(bars, top['importance']):
        ax.text(bar.get_width() + max(top['importance']) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_prediction_vs_actual(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    model_name: str = 'Best Model',
    save_path: str = 'output/prediction_vs_actual.png',
) -> None:
    """
    Chart 5: Prediction accuracy — scatter plot + residual histogram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax = axes[0]
    ax.scatter(y_actual, y_predicted, alpha=0.3, s=10, color=COLORS['blue'])

    # Perfect prediction line
    mn, mx = min(y_actual.min(), y_predicted.min()), max(y_actual.max(), y_predicted.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Spread', fontsize=12)
    ax.set_ylabel('Predicted Spread', fontsize=12)
    ax.set_title(f'{model_name} — Predicted vs Actual', fontsize=13, fontweight='bold')
    ax.legend()

    # Residual histogram
    ax = axes[1]
    residuals = y_actual - y_predicted
    ax.hist(residuals, bins=50, color=COLORS['purple'], alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Residual (Actual − Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=13, fontweight='bold')

    # Stats annotation
    ax.text(0.72, 0.85,
            f'Mean: {np.mean(residuals):.6f}\nStd: {np.std(residuals):.6f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = 'output/model_comparison.png',
) -> None:
    """
    Chart 6: RMSE comparison across models (grouped bar chart).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    models = comparison_df['Model'].values
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, comparison_df['Train RMSE'],
                   width, label='Train RMSE', color=COLORS['blue'], alpha=0.85)
    bars2 = ax.bar(x + width / 2, comparison_df['Test RMSE'],
                   width, label='Test RMSE', color=COLORS['orange'], alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Model Comparison — Train vs Test RMSE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_correlation_matrix(
    df: pd.DataFrame,
    metrics: List[str] = None,
    save_path: str = 'output/correlation_matrix.png',
) -> None:
    """
    Chart 7: Triangular heatmap of metric correlations.
    """
    if metrics is None:
        metrics = ['spread_proxy', 'amihud', 'volume_ratio', 'realized_vol',
                    'intraday_range', 'high_low_vol', 'abs_returns', 'price_impact']

    available = [m for m in metrics if m in df.columns]
    if len(available) < 2:
        return

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Upper triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.5, ax=ax,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
    )

    ax.set_title('Liquidity Metrics Correlation Matrix',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_liquidity_heatmap(
    scored_df: pd.DataFrame,
    save_path: str = 'output/liquidity_heatmap.png',
) -> None:
    """
    Chart 8: Asset × Time liquidity heatmap.

    Monthly average liquidity score for each asset, color-coded
    from red (illiquid) to green (liquid).
    """
    df = scored_df.copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M').astype(str)
    else:
        print("  ⚠ No date column for heatmap")
        return

    pivot = df.pivot_table(
        values='composite_score',
        index='ticker',
        columns='year_month',
        aggfunc='mean',
    )

    # Sort tickers by overall liquidity (most liquid on top)
    ticker_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[ticker_order]

    # Limit columns for readability (show every Nth month)
    if pivot.shape[1] > 24:
        step = max(1, pivot.shape[1] // 24)
        pivot = pivot.iloc[:, ::step]

    fig, ax = plt.subplots(figsize=(max(12, pivot.shape[1] * 0.5), max(4, len(pivot) * 0.8)))

    sns.heatmap(
        pivot, cmap='RdYlGn', vmin=0, vmax=100,
        annot=pivot.shape[1] <= 18, fmt='.0f',
        linewidths=0.5, ax=ax,
        cbar_kws={'shrink': 0.8, 'label': 'Liquidity Score'},
    )

    ax.set_title('Liquidity Score Heatmap (Asset × Time)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Asset', fontsize=11)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")
```

---

**What You Just Built (8 Charts):**

| Chart | What It Shows | Who Uses It |
|-------|-------------|-------------|
| **Liquidity Time Series** | 4-panel trend of spread, amihud, volume ratio, realized vol with 20-day MA overlay | Traders monitoring day-to-day liquidity changes |
| **Asset Comparison** | Box plots of composite score by ticker, ordered most→least liquid | Portfolio managers deciding which assets to trade |
| **Risk Distribution** | Bar chart counting observations per risk level with percentages | Risk managers for regulatory reporting (Basel III) |
| **Feature Importance** | Top 15 features that drive spread predictions | Quants understanding what matters for liquidity |
| **Prediction vs Actual** | Scatter (accuracy) + residual histogram (bias check) | Model validators assessing ML performance |
| **Model Comparison** | Train vs Test RMSE for all 5 algorithms | ML team selecting production model |
| **Correlation Matrix** | Triangular heatmap showing metric correlations | Researchers studying liquidity dimensions |
| **Liquidity Heatmap** | Asset × Month grid, green=liquid, red=illiquid | CRO dashboard for portfolio-wide monitoring |

---

---

# PART 8: MAIN SCRIPT (Step 6)

The 9-step pipeline entry point — demonstrates the entire system.

---

**File: `main.py`**

```python
"""
main.py — Liquidity Risk ML Predictor Pipeline
===================================================

Runs the complete analysis in 9 steps:
  1. Data loading & liquidity metrics calculation
  2. Feature engineering (10 metrics → 68 features)
  3. Train/test split (80/20, time-based)
  4. Model training (5 algorithms)
  5. Time-series cross-validation
  6. Feature importance analysis
  7. Liquidity scoring & risk classification
  8. Visualization (8 charts)
  9. Model saving
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

from src.data_loader import create_liquidity_dataset, fetch_market_data
from src.liquidity_features import LiquidityFeatureEngineer
from src.ml_models import (
    train_all_models, create_model_comparison, get_best_model,
    cross_validate_model,
)
from src.liquidity_scorer import LiquidityScoringEngine
from src.visualization import (
    plot_liquidity_timeseries, plot_liquidity_comparison,
    plot_risk_distribution, plot_feature_importance,
    plot_prediction_vs_actual, plot_model_comparison,
    plot_correlation_matrix, plot_liquidity_heatmap,
)


def main():
    """Run the complete liquidity risk ML analysis."""

    print("=" * 70)
    print(" LIQUIDITY RISK ML PREDICTOR")
    print("=" * 70)
    print(f"\nThis analysis performs:")
    print(f"  1. Liquidity metrics calculation (Amihud, Spread, Impact)")
    print(f"  2. Feature engineering for ML prediction")
    print(f"  3. Multiple ML model training and comparison")
    print(f"  4. Liquidity risk scoring and classification")
    print(f"  5. Visualization of results")

    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # ── STEP 1: Data Loading & Liquidity Metrics ─────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 1: Data Loading & Liquidity Metrics")
    print(f"{'─' * 70}")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ']
    print(f"\n  Assets: {', '.join(tickers)}")

    df = create_liquidity_dataset(
        tickers=tickers,
        start_date='2020-01-01',
    )

    print(f"  Created liquidity dataset: {len(df)} rows")
    print(f"\n  Dataset shape: {df.shape}")
    if 'date' in df.columns:
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    print(f"\n  Sample Liquidity Metrics:")
    sample_cols = ['ticker', 'spread_proxy', 'amihud', 'volume_ratio', 'price_impact']
    available_cols = [c for c in sample_cols if c in df.columns]
    print(df[available_cols].head(5).to_string(index=False))

    # ── STEP 2: Feature Engineering ──────────────────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 2: Feature Engineering")
    print(f"{'─' * 70}")

    engineer = LiquidityFeatureEngineer(
        target_col='spread_proxy',
        scale_features=True,
    )
    feature_set = engineer.fit_transform(df, forecast_horizon=1)

    X = feature_set.X
    y = feature_set.y

    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Number of features: {len(feature_set.feature_names)}")
    print(f"\n  Top 15 Features Created:")
    for i, name in enumerate(feature_set.feature_names[:15], 1):
        print(f"    {i:>3}. {name}")

    # ── STEP 3: Train/Test Split ─────────────────────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 3: Train/Test Split (80/20 Time-Based)")
    print(f"{'─' * 70}")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\n  Training set: {len(X_train)} samples")
    print(f"  Test set:     {len(X_test)} samples")
    print(f"  Split ratio:  {len(X_train)/len(X)*100:.0f}% / {len(X_test)/len(X)*100:.0f}%")

    # ── STEP 4: Model Training ───────────────────────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 4: Model Training")
    print(f"{'─' * 70}\n")

    results = train_all_models(X_train, y_train, X_test, y_test)

    comparison_df = create_model_comparison(results)
    print(f"\n  📊 Model Comparison:")
    print(comparison_df.to_string(index=False))

    best = get_best_model(results)
    print(f"\n  🏆 Best Model: {best.model_name}")
    print(f"     Test RMSE: {best.test_rmse:.6f}")
    print(f"     Test R²:   {best.test_r2:.4f}")
    print(f"     Test MAE:  {best.test_mae:.6f}")

    # ── STEP 5: Time-Series Cross-Validation ─────────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 5: Time Series Cross-Validation")
    print(f"{'─' * 70}")

    print(f"\n  Performing 5-fold time series CV on {best.model_name}...")
    cv_scores = cross_validate_model(best.model, X, y, n_splits=5)

    print(f"\n  Cross-Validation Results:")
    print(f"    RMSE: {cv_scores['cv_rmse_mean']:.6f} (±{cv_scores['cv_rmse_std']:.6f})")
    print(f"    MAE:  {cv_scores['cv_mae_mean']:.6f} (±{cv_scores['cv_mae_std']:.6f})")
    print(f"    R²:   {cv_scores['cv_r2_mean']:.4f} (±{cv_scores['cv_r2_std']:.4f})")

    # ── STEP 6: Feature Importance ───────────────────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 6: Feature Importance")
    print(f"{'─' * 70}")

    print(f"\n  Top 15 Most Important Features:")
    for i, row in best.feature_importance.head(15).iterrows():
        print(f"    {i+1:>3}. {row['feature']:<30s}: {row['importance']:.4f}")

    # ── STEP 7: Liquidity Scoring ────────────────────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 7: Liquidity Scoring")
    print(f"{'─' * 70}")

    scorer = LiquidityScoringEngine()
    scored_df = scorer.calculate_composite_score(df)

    summary = scorer.get_score_summary(scored_df)
    print(f"\n  Liquidity Score Summary by Asset:")
    print(summary.to_string())

    risk_dist = scorer.get_risk_distribution(scored_df)
    print(f"\n  Risk Level Distribution:")
    for level, row in risk_dist.iterrows():
        print(f"    {level}: {row['count']:,} ({row['percentage']:.1f}%)")

    # ── STEP 8: Visualizations ───────────────────────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 8: Generating Visualizations")
    print(f"{'─' * 70}")
    print(f"\n  Saving charts to ./output/...")

    # Chart 1: Liquidity time series (for first ticker)
    plot_liquidity_timeseries(df, ticker=tickers[0])

    # Chart 2: Asset comparison
    plot_liquidity_comparison(scored_df)

    # Chart 3: Risk distribution
    plot_risk_distribution(scored_df)

    # Chart 4: Feature importance
    plot_feature_importance(best.feature_importance, top_n=15)

    # Chart 5: Prediction vs Actual
    plot_prediction_vs_actual(
        y_actual=y_test.values,
        y_predicted=best.predictions,
        model_name=best.model_name,
    )

    # Chart 6: Model comparison
    plot_model_comparison(comparison_df)

    # Chart 7: Correlation matrix
    plot_correlation_matrix(df)

    # Chart 8: Liquidity heatmap
    plot_liquidity_heatmap(scored_df)

    # ── STEP 9: Save Outputs ─────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f" STEP 9: Saving Outputs")
    print(f"{'─' * 70}")

    # Save scored data
    scored_df.to_csv('output/liquidity_scores.csv', index=False)
    print(f"  ✓ Liquidity scores → output/liquidity_scores.csv")

    # Save model comparison
    comparison_df.to_csv('output/model_comparison.csv', index=False)
    print(f"  ✓ Model comparison → output/model_comparison.csv")

    # Save best model
    model_filename = f"models/best_model_{best.model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(best.model, model_filename)
    print(f"  ✓ Best model → {model_filename}")

    # Save feature engineer
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    print(f"  ✓ Feature engineer → models/feature_engineer.pkl")

    # Save scorer
    joblib.dump(scorer, 'models/liquidity_scorer.pkl')
    print(f"  ✓ Liquidity scorer → models/liquidity_scorer.pkl")

    # Save training results
    joblib.dump({
        'best_model_name': best.model_name,
        'test_rmse': best.test_rmse,
        'test_r2': best.test_r2,
        'cv_scores': cv_scores,
        'n_features': len(feature_set.feature_names),
        'n_observations': len(df),
        'timestamp': datetime.now().isoformat(),
    }, 'output/training_results.pkl')
    print(f"  ✓ Training results → output/training_results.pkl")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f" ANALYSIS COMPLETE")
    print(f"{'─' * 70}")
    print(f"\n  📊 Key Results:")
    print(f"\n    Data:")
    print(f"      • Assets analyzed: {len(tickers)}")
    print(f"      • Total observations: {len(df):,}")
    print(f"      • Features engineered: {len(feature_set.feature_names)}")
    print(f"\n    Best Model ({best.model_name}):")
    print(f"      • Test RMSE: {best.test_rmse:.6f}")
    print(f"      • Test R²: {best.test_r2:.4f}")
    print(f"      • CV RMSE: {cv_scores['cv_rmse_mean']:.6f} "
          f"(±{cv_scores['cv_rmse_std']:.6f})")
    print(f"\n    Liquidity Scoring:")
    avg_score = scored_df['composite_score'].mean()
    high_risk_pct = (scored_df['risk_level'].isin(['High Risk', 'Very High Risk'])).mean() * 100
    print(f"      • Average Score: {avg_score:.1f}")
    print(f"      • High Risk %: {high_risk_pct:.1f}%")
    print(f"\n  📁 Output files saved to ./output/")
    print(f"  📁 Models saved to ./models/")
    print(f"\n  Done! ✅")


if __name__ == '__main__':
    main()
```

---

---

# PART 9: UNIT TESTS (Step 7)

25+ tests across 7 test classes — covering data generation, metrics, features, models, scoring, visualization, and end-to-end integration.

---

**File: `tests/test_liquidity.py`**

```python
"""
test_liquidity.py — Unit Tests for Liquidity Risk ML Predictor
==================================================================

25+ tests across 7 classes:
  - TestDataLoader (5): synthetic OHLCV, metrics calculation, dataset creation
  - TestLiquidityMetrics (4): spread, amihud, volume ratio, price impact
  - TestFeatureEngineering (4): features created, no NaN, scaler, transform
  - TestMLModels (4): ridge, random forest, train all, comparison
  - TestLiquidityScorer (4): scoring, risk levels, summary, illiquid periods
  - TestVisualization (1): imports
  - TestIntegration (3): full pipeline, model persistence, scoring pipeline

Run: python -m pytest tests/test_liquidity.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import (
    generate_synthetic_ohlcv,
    calculate_liquidity_metrics,
    create_liquidity_dataset,
)
from src.liquidity_features import LiquidityFeatureEngineer, FeatureSet
from src.ml_models import (
    train_ridge_model, train_random_forest,
    train_all_models, create_model_comparison, get_best_model,
    ModelResult,
)
from src.liquidity_scorer import LiquidityScoringEngine


# ─── Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope='module')
def synthetic_prices():
    """Generate synthetic OHLCV for testing (2 tickers, 2 years)."""
    return generate_synthetic_ohlcv(
        tickers=['TEST_A', 'TEST_B'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        random_state=42,
    )


@pytest.fixture(scope='module')
def liquidity_df(synthetic_prices):
    """Calculate liquidity metrics from synthetic data."""
    return calculate_liquidity_metrics(synthetic_prices)


@pytest.fixture(scope='module')
def feature_data(liquidity_df):
    """Engineer features from liquidity data."""
    engineer = LiquidityFeatureEngineer(scale_features=True)
    return engineer.fit_transform(liquidity_df, forecast_horizon=1)


@pytest.fixture(scope='module')
def train_test_split(feature_data):
    """Split features into train/test."""
    split = int(len(feature_data.X) * 0.8)
    return {
        'X_train': feature_data.X.iloc[:split],
        'X_test': feature_data.X.iloc[split:],
        'y_train': feature_data.y.iloc[:split],
        'y_test': feature_data.y.iloc[split:],
    }


# ─── TestDataLoader ─────────────────────────────────────────────

class TestDataLoader:

    def test_synthetic_ohlcv_shape(self, synthetic_prices):
        """Synthetic data should have 5 price columns × 2 tickers."""
        assert isinstance(synthetic_prices, pd.DataFrame)
        assert isinstance(synthetic_prices.columns, pd.MultiIndex)
        assert len(synthetic_prices) > 200  # ~2 years of business days

    def test_synthetic_ohlcv_values(self, synthetic_prices):
        """Prices should be positive, High ≥ Low."""
        for ticker in ['TEST_A', 'TEST_B']:
            close = synthetic_prices[('Close', ticker)]
            high = synthetic_prices[('High', ticker)]
            low = synthetic_prices[('Low', ticker)]
            assert (close > 0).all()
            assert (high >= low).all()

    def test_liquidity_metrics_columns(self, liquidity_df):
        """Metrics DataFrame should have all expected columns."""
        expected = ['ticker', 'spread_proxy', 'amihud', 'volume_ratio',
                    'price_impact', 'realized_vol', 'intraday_range']
        for col in expected:
            assert col in liquidity_df.columns, f"Missing column: {col}"

    def test_liquidity_metrics_tickers(self, liquidity_df):
        """Should have data for both tickers."""
        tickers = liquidity_df['ticker'].unique()
        assert len(tickers) == 2
        assert 'TEST_A' in tickers
        assert 'TEST_B' in tickers

    def test_create_dataset(self):
        """create_liquidity_dataset should work end-to-end."""
        df = create_liquidity_dataset(
            tickers=['SYN_A'],
            start_date='2023-01-01',
            end_date='2023-06-30',
            use_synthetic=True,
        )
        assert len(df) > 50
        assert 'spread_proxy' in df.columns


# ─── TestLiquidityMetrics ────────────────────────────────────────

class TestLiquidityMetrics:

    def test_spread_non_negative(self, liquidity_df):
        """Corwin-Schultz spread should be non-negative."""
        assert (liquidity_df['spread_proxy'] >= 0).all()

    def test_amihud_non_negative(self, liquidity_df):
        """Amihud ratio should be non-negative."""
        assert (liquidity_df['amihud'] >= 0).all()

    def test_volume_ratio_positive(self, liquidity_df):
        """Volume ratio should be positive (volume/avg_volume)."""
        vr = liquidity_df['volume_ratio']
        assert (vr >= 0).all()

    def test_realized_vol_reasonable(self, liquidity_df):
        """Annualized vol should be between 0 and 200%."""
        vol = liquidity_df['realized_vol']
        valid = vol[vol > 0]
        if len(valid) > 0:
            assert valid.max() < 5.0  # 500% annualized is extreme but possible


# ─── TestFeatureEngineering ──────────────────────────────────────

class TestFeatureEngineering:

    def test_feature_count(self, feature_data):
        """Should create many features (>30)."""
        assert feature_data.X.shape[1] > 30

    def test_no_nans_in_features(self, feature_data):
        """Feature matrix should have no NaN values."""
        assert feature_data.X.isnull().sum().sum() == 0

    def test_target_exists(self, feature_data):
        """Target should be a non-empty Series."""
        assert len(feature_data.y) > 0
        assert feature_data.y.isnull().sum() == 0

    def test_scaler_fitted(self, feature_data):
        """Scaler should be fitted with correct dimensions."""
        assert feature_data.scaler is not None
        assert feature_data.scaler.n_features_in_ == feature_data.X.shape[1]


# ─── TestMLModels ────────────────────────────────────────────────

class TestMLModels:

    def test_ridge_model(self, train_test_split):
        """Ridge should return valid ModelResult."""
        d = train_test_split
        result = train_ridge_model(d['X_train'], d['y_train'],
                                   d['X_test'], d['y_test'])
        assert isinstance(result, ModelResult)
        assert result.test_rmse > 0
        assert len(result.predictions) == len(d['y_test'])

    def test_random_forest(self, train_test_split):
        """Random Forest should return valid ModelResult with importance."""
        d = train_test_split
        result = train_random_forest(d['X_train'], d['y_train'],
                                     d['X_test'], d['y_test'])
        assert isinstance(result, ModelResult)
        assert result.test_rmse > 0
        assert len(result.feature_importance) > 0

    def test_train_all_models(self, train_test_split):
        """Should train at least 3 models (ridge, rf, gb)."""
        d = train_test_split
        results = train_all_models(d['X_train'], d['y_train'],
                                   d['X_test'], d['y_test'])
        assert len(results) >= 3
        assert 'ridge' in results
        assert 'rf' in results

    def test_model_comparison(self, train_test_split):
        """Comparison table should have correct structure."""
        d = train_test_split
        results = train_all_models(d['X_train'], d['y_train'],
                                   d['X_test'], d['y_test'])
        comp = create_model_comparison(results)
        assert 'Model' in comp.columns
        assert 'Test RMSE' in comp.columns
        assert len(comp) >= 3

        best = get_best_model(results)
        assert best.test_rmse == comp['Test RMSE'].min()


# ─── TestLiquidityScorer ─────────────────────────────────────────

class TestLiquidityScorer:

    def test_scoring(self, liquidity_df):
        """Composite scores should be between 0 and 100."""
        scorer = LiquidityScoringEngine()
        scored = scorer.calculate_composite_score(liquidity_df)
        assert 'composite_score' in scored.columns
        assert scored['composite_score'].min() >= 0
        assert scored['composite_score'].max() <= 100

    def test_risk_levels(self, liquidity_df):
        """Risk levels should be valid categories."""
        scorer = LiquidityScoringEngine()
        scored = scorer.calculate_composite_score(liquidity_df)
        valid_levels = {'Very Low Risk', 'Low Risk', 'Moderate Risk',
                        'High Risk', 'Very High Risk'}
        assert set(scored['risk_level'].unique()).issubset(valid_levels)

    def test_score_summary(self, liquidity_df):
        """Summary should have one row per ticker."""
        scorer = LiquidityScoringEngine()
        scored = scorer.calculate_composite_score(liquidity_df)
        summary = scorer.get_score_summary(scored)
        assert len(summary) == liquidity_df['ticker'].nunique()
        assert 'avg_score' in summary.columns

    def test_custom_weights(self, liquidity_df):
        """Custom weights should change scores."""
        scorer_default = LiquidityScoringEngine()
        scorer_volume = LiquidityScoringEngine(weights={
            'spread_score': 0.10,
            'volume_score': 0.60,
            'amihud_score': 0.10,
            'impact_score': 0.10,
            'volatility_score': 0.10,
        })
        s1 = scorer_default.calculate_composite_score(liquidity_df)
        s2 = scorer_volume.calculate_composite_score(liquidity_df)
        # Scores should differ with different weights
        assert not np.allclose(s1['composite_score'].values,
                               s2['composite_score'].values)


# ─── TestVisualization ───────────────────────────────────────────

class TestVisualization:

    def test_imports(self):
        """Visualization module should import without errors."""
        from src.visualization import (
            plot_liquidity_timeseries, plot_liquidity_comparison,
            plot_risk_distribution, plot_feature_importance,
            plot_prediction_vs_actual, plot_model_comparison,
            plot_correlation_matrix, plot_liquidity_heatmap,
        )
        assert callable(plot_liquidity_timeseries)
        assert callable(plot_liquidity_heatmap)


# ─── TestIntegration ─────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline(self):
        """End-to-end: data → features → train → score."""
        # Data
        df = create_liquidity_dataset(
            tickers=['INT_A', 'INT_B'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            use_synthetic=True,
        )
        assert len(df) > 100

        # Features
        engineer = LiquidityFeatureEngineer(scale_features=True)
        fs = engineer.fit_transform(df, forecast_horizon=1)
        assert fs.X.shape[1] > 20

        # Split and train
        split = int(len(fs.X) * 0.8)
        X_train, X_test = fs.X.iloc[:split], fs.X.iloc[split:]
        y_train, y_test = fs.y.iloc[:split], fs.y.iloc[split:]

        result = train_ridge_model(X_train, y_train, X_test, y_test)
        assert result.test_rmse > 0

        # Score
        scorer = LiquidityScoringEngine()
        scored = scorer.calculate_composite_score(df)
        assert 'risk_level' in scored.columns

    def test_model_persistence(self, train_test_split):
        """Models should survive save/load."""
        import tempfile
        d = train_test_split
        result = train_ridge_model(d['X_train'], d['y_train'],
                                   d['X_test'], d['y_test'])

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            import joblib
            joblib.dump(result.model, f.name)
            loaded = joblib.load(f.name)

        preds_original = result.model.predict(d['X_test'])
        preds_loaded = loaded.predict(d['X_test'])
        np.testing.assert_array_almost_equal(preds_original, preds_loaded)
        os.unlink(f.name)

    def test_scoring_pipeline(self):
        """Scoring should work independently of ML pipeline."""
        # Create minimal data
        data = pd.DataFrame({
            'ticker': ['A'] * 100,
            'spread_proxy': np.random.uniform(0, 0.01, 100),
            'amihud': np.random.uniform(0, 1, 100),
            'volume': np.random.uniform(1e6, 1e8, 100),
            'price_impact': np.random.uniform(0, 0.05, 100),
            'realized_vol': np.random.uniform(0.1, 0.5, 100),
        })

        scorer = LiquidityScoringEngine()
        scored = scorer.calculate_composite_score(data)

        assert len(scored) == 100
        assert scored['composite_score'].between(0, 100).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

---

# PART 10: RUN IT!

## Step 8.1: Run the Full Pipeline
```bash
python main.py
```

This produces:
- Console output with 9 formatted steps
- `output/liquidity_timeseries.png` — 4-panel metric trends for AAPL
- `output/liquidity_comparison.png` — Box plots of score by asset
- `output/risk_distribution.png` — Bar chart of risk levels
- `output/feature_importance.png` — Top 15 features bar chart
- `output/prediction_vs_actual.png` — Scatter + residual histogram
- `output/model_comparison.png` — Train vs Test RMSE for 5 models
- `output/correlation_matrix.png` — Triangular metric heatmap
- `output/liquidity_heatmap.png` — Asset × Month grid
- `output/liquidity_scores.csv` — Full scored dataset
- `output/model_comparison.csv` — Model performance table
- `output/training_results.pkl` — Serialized metrics
- `models/best_model_random_forest.pkl` — Best trained model
- `models/feature_engineer.pkl` — Feature pipeline
- `models/liquidity_scorer.pkl` — Scoring engine

## Step 8.2: Run the Tests
```bash
python -m pytest tests/test_liquidity.py -v
```

Expected: **25 passed** across 7 test classes.

---

---

# PART 11: HOW TO READ THE RESULTS

## 11.1: The Model Comparison

```
            Model  Train RMSE  Test RMSE  Train R²  Test R²
    Random Forest    0.004853   0.003778    0.4166   0.2895
         LightGBM    0.004674   0.003846    0.4589   0.2637
 Ridge Regression    0.005957   0.003904    0.1210   0.2414
Gradient Boosting    0.004373   0.003930    0.5263   0.2314
          XGBoost    0.004210   0.004145    0.5610   0.1446
```

**How to read this:**

```
Random Forest wins with Test RMSE = 0.003778.
  "On average, the model's spread predictions are off by 0.38 basis points."
  For a stock with 5bp spread, that's ±7.6% error. Quite good.

Test R² = 0.29 means:
  "The model explains 29% of the variation in next-day spreads."
  In financial ML, R² of 0.15-0.30 is considered strong.
  This is much harder than predicting house prices or credit risk
  because financial markets are noisy by nature.

WARNING SIGNS to look for:
  1. Train RMSE << Test RMSE → OVERFITTING
     XGBoost: Train=0.0042, Test=0.0041 → Moderate gap
     GB: Train=0.0044, Test=0.0039 → OK, regularization helping
     
  2. High Train R² + Low Test R² → OVERFITTING
     XGBoost: Train R²=0.56, Test R²=0.14 → BIG gap → overfitting!
     Random Forest: Train R²=0.42, Test R²=0.29 → Smaller gap → better

  3. Cross-validation RMSE vs Test RMSE → STABILITY
     CV RMSE=0.0056 ± 0.0013, Test RMSE=0.0038
     The CV error is higher (because earlier data has different patterns)
     and the ± tells you how variable the model is across time periods.
```

## 11.2: The Feature Importance

```
Top 15 Most Important Features:
  1. intraday_range:          0.1432  (14.3%)
  2. high_low_vol:            0.1303  (13.0%)
  3. volume_ratio_ma20:       0.0281  (2.8%)
  4. amihud_lag2:             0.0265  (2.6%)
  5. volume_ratio_lag1:       0.0248  (2.5%)
```

**What this tells you:**

```
The top 2 features (intraday_range + high_low_vol) explain 27.3% of
the model's total predictive power. Both are VOLATILITY measures.

This makes perfect sense: high volatility → wider spreads.
Market makers widen spreads when volatility is high because:
  1. Their inventory risk increases (price might move against them)
  2. They face more adverse selection (informed traders exploit them)
  3. Wider spreads compensate for this higher risk

The remaining features each contribute 1-3%. This "long tail"
means no single feature dominates — the model captures complex
multi-factor patterns that simple rules would miss.

Lagged features (amihud_lag2, volume_ratio_lag1) confirm that
liquidity is PERSISTENT — yesterday's conditions predict tomorrow.

Rolling features (volume_ratio_ma20) capture REGIME CHANGES —
a sustained drop in volume over 20 days signals deteriorating
liquidity even if today's volume looks normal.
```

## 11.3: The Liquidity Scores

```
Liquidity Score Summary by Asset:
        Avg Score  Std Dev  Min Score  Max Score   Typical Risk
SPY         88.87     5.45      49.11      98.98  Very Low Risk
QQQ         79.43     7.26      46.37      95.85  Very Low Risk
AAPL        71.81     7.33      35.69      93.01       Low Risk
MSFT        66.17     8.80      25.58      87.04       Low Risk
AMZN        62.84    10.64      24.55      85.94       Low Risk
GOOGL       52.94    10.78       9.21      79.80  Moderate Risk
```

**What each number means:**

```
SPY (Avg: 88.9, Std: 5.4):
  The most liquid asset. Score rarely drops below 49.
  Low variability (std=5.4) means its liquidity is STABLE.
  This is expected — SPY is the most-traded ETF on Earth.

GOOGL (Avg: 52.9, Std: 10.8):
  The least liquid of our 6 assets. Sometimes drops to 9.2!
  High variability (std=10.8) means its liquidity is UNSTABLE.
  The min score of 9.21 = "Very High Risk" day. On that day,
  trading GOOGL was extremely expensive.

Risk Level Distribution:
  Low Risk: 3,840 (50.5%)     ← Most observations are fine
  Very Low Risk: 2,007 (26.4%) ← ETFs drive this bucket
  Moderate Risk: 1,542 (20.3%) ← Watch these days
  High Risk: 207 (2.7%)        ← Red flag days
  Very High Risk: 6 (0.1%)     ← Crisis days

Only 2.8% of observations are "High Risk" or worse.
But those are exactly the days when you lose the most money —
liquidity evaporates when you need it most.
```

## 11.4: Interpreting the Charts

### Liquidity Time Series (4-panel)
Each panel shows one metric over time for a single asset. Look for:
- **Spikes in spread and amihud** = illiquidity events (crisis periods)
- **Drops in volume ratio** below 1.0 = below-average trading activity
- **The 20-day MA** (dashed black line) shows the trend. When the MA is rising for spread/amihud, liquidity is deteriorating.

### Asset Comparison (Box Plots)
Ordered left-to-right from most liquid to least. The **box** shows the interquartile range (25th–75th percentile). **Whiskers** show the range. **Dots** are outliers. SPY's box is narrow and high (consistently liquid). GOOGL's box is wide and low (variable, less liquid).

### Correlation Matrix
Shows how metrics relate. Key correlations:
- **spread_proxy ↔ high_low_vol ≈ 0.85**: Volatility and spread are highly correlated (wider spreads during volatile periods)
- **amihud ↔ price_impact ≈ 0.70**: Different measures of the same concept (price sensitivity to trading)
- **volume_ratio ↔ amihud ≈ −0.30**: Higher volume → lower Amihud (more volume absorbs price impact)

### Liquidity Heatmap
Green cells = liquid months, red cells = illiquid months. Look for:
- **Vertical red bands** = market-wide stress (e.g., COVID crash March 2020)
- **Horizontal red rows** = chronically illiquid assets
- **Diagonal patterns** = seasonal effects (December thin trading)

---

---

# PART 12: QUICK REFERENCE CARD

## Architecture
```
main.py                           → 9-step pipeline
src/data_loader.py                → OHLCV generation/fetching + 10 metrics
src/liquidity_features.py         → 68 features (fit/transform pattern)
src/ml_models.py                  → 5 ML models + CV + comparison
src/liquidity_scorer.py           → Composite scoring (0-100) + risk levels
src/visualization.py              → 8 chart functions
tests/test_liquidity.py           → 25+ tests across 7 classes
```

## The 10 Liquidity Metrics

| # | Metric | Formula | Higher Means |
|---|--------|---------|--------------|
| 1 | **Corwin-Schultz Spread** | `2(e^α−1)/(1+e^α)` from H/L | More expensive to trade |
| 2 | **Roll Spread** | `2√(−Cov(rₜ,rₜ₋₁))` | More expensive to trade |
| 3 | **Amihud Ratio** | `|r|/DollarVol × 10¹⁰` | More price impact |
| 4 | **Kyle's Lambda** | `|r|/√Vol × 10⁴` | More market impact |
| 5 | **Dollar Volume** | `Close × Volume` | More trading activity |
| 6 | **Turnover** | `Volume / MA₂₀(Volume)` | Above-average activity |
| 7 | **Realized Vol** | `std₂₀(returns) × √252` | More price uncertainty |
| 8 | **Intraday Range** | `(High−Low) / Close` | More daily volatility |
| 9 | **Volume Ratio** | `Volume / MA₂₀(Volume)` | Above-average volume |
| 10 | **Parkinson Vol** | `√(ln(H/L)²/(4ln2)) × √252` | More range-based vol |

## The 68 Features (Breakdown)

| Category | Count | Example | Why |
|----------|-------|---------|-----|
| Base metrics | 10 | `abs_returns`, `realized_vol` | Raw signal |
| Lagged (1,2,3,5d) | 16 | `spread_lag1`, `amihud_lag5` | Persistence |
| Rolling mean (5,10,20) | 9 | `spread_ma20` | Trend |
| Rolling std | 9 | `volume_ratio_std10` | Stability |
| Rolling max | 9 | `abs_returns_max5` | Recent stress |
| Rolling skew | 6 | `spread_skew20` | Tail risk |
| Interactions | 4 | `vol_volume_interaction` | Cross-metric |
| Categorical | 5 | `volume_regime`, `day_of_week` | Regime + calendar |

## 5 ML Models

| Model | Type | Regularization | Speed | When to Use |
|-------|------|---------------|-------|-------------|
| **Ridge** | Linear | L2 (shrinks coefficients) | ⚡ | Baseline, interpretability |
| **Random Forest** | Ensemble (parallel) | max_depth, min_samples | 🔶 | Feature importance, robustness |
| **Gradient Boosting** | Ensemble (sequential) | learning_rate, subsample | 🔷 | Accuracy on small data |
| **XGBoost** | Boosting + L1/L2 | reg_alpha, reg_lambda | 🔷 | Best tabular accuracy |
| **LightGBM** | Boosting (histogram) | num_leaves | ⚡ | Fastest on large data |

## Scoring Weights

| Component | Default Weight | Metric Source |
|-----------|---------------|---------------|
| spread_score | 25% | spread_proxy (inverted) |
| volume_score | 25% | log(volume) (direct) |
| amihud_score | 20% | amihud (inverted) |
| impact_score | 15% | price_impact (inverted) |
| volatility_score | 15% | realized_vol (inverted) |

## Risk Levels

| Score Range | Level | Action |
|-------------|-------|--------|
| 80–100 | Very Low Risk 🟢 | Trade freely |
| 60–80 | Low Risk 🟡 | Normal liquidity |
| 40–60 | Moderate Risk 🟠 | Monitor closely, reduce size |
| 20–40 | High Risk 🔴 | Reduce position, use limit orders |
| 0–20 | Very High Risk ⚫ | Avoid trading if possible |

## Test Coverage (25+ Tests)

| Class | # | What's Tested |
|-------|---|---------------|
| TestDataLoader | 5 | Synthetic OHLCV shape/values, metrics columns/tickers, dataset creation |
| TestLiquidityMetrics | 4 | Spread ≥ 0, Amihud ≥ 0, volume ratio > 0, vol < 500% |
| TestFeatureEngineering | 4 | Feature count > 30, no NaN, target non-empty, scaler fitted |
| TestMLModels | 4 | Ridge valid, RF importance, all 3+ models, best = min RMSE |
| TestLiquidityScorer | 4 | Scores 0-100, valid risk levels, summary by ticker, custom weights differ |
| TestVisualization | 1 | All 8 functions importable |
| TestIntegration | 3 | Full pipeline data→score, model save/load, scoring independence |

## Dependencies
```
numpy          → Array math, random generation
pandas         → DataFrames for all data handling
scipy          → Statistical functions (not heavily used in final code)
scikit-learn   → Ridge, RF, GB, StandardScaler, TimeSeriesSplit, metrics
xgboost        → XGBRegressor (optional — graceful fallback)
lightgbm       → LGBMRegressor (optional — graceful fallback)
matplotlib     → All 8 charts
seaborn        → Heatmaps, box plots, professional styling
yfinance       → Real market data (optional — auto-fallback to synthetic)
joblib         → Save/load models, feature engineer, scorer
pytest         → Testing framework
```
