"""
Data Loader Module
Fetches and generates market data with liquidity metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_market_data(
    tickers: List[str],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for given tickers.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date (default: today)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if HAS_YFINANCE:
        try:
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False
            )
            
            if len(tickers) == 1:
                data.columns = pd.MultiIndex.from_product(
                    [data.columns, tickers]
                )
            
            print(f"Fetched {len(data)} days of data for {len(tickers)} assets")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Generating synthetic data...")
            return generate_synthetic_ohlcv(tickers, start_date, end_date)
    else:
        print("yfinance not installed. Generating synthetic data...")
        return generate_synthetic_ohlcv(tickers, start_date, end_date)


def generate_synthetic_ohlcv(
    tickers: List[str],
    start_date: str,
    end_date: str,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with realistic liquidity characteristics.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date, end_date : str
        Date range
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Synthetic OHLCV data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # Asset-specific parameters
    asset_params = {
        'AAPL': {'price': 150, 'vol': 0.25, 'avg_volume': 80_000_000},
        'MSFT': {'price': 300, 'vol': 0.22, 'avg_volume': 25_000_000},
        'GOOGL': {'price': 140, 'vol': 0.28, 'avg_volume': 20_000_000},
        'AMZN': {'price': 170, 'vol': 0.30, 'avg_volume': 40_000_000},
        'SPY': {'price': 450, 'vol': 0.18, 'avg_volume': 100_000_000},
        'QQQ': {'price': 380, 'vol': 0.22, 'avg_volume': 50_000_000},
        'IWM': {'price': 200, 'vol': 0.24, 'avg_volume': 30_000_000},
        'GLD': {'price': 180, 'vol': 0.15, 'avg_volume': 10_000_000},
        'TLT': {'price': 100, 'vol': 0.16, 'avg_volume': 15_000_000},
    }
    
    default_params = {'price': 100, 'vol': 0.25, 'avg_volume': 5_000_000}
    
    data_dict = {}
    
    for ticker in tickers:
        params = asset_params.get(ticker, default_params)
        
        # Generate price path using GBM
        dt = 1/252
        mu = 0.08
        sigma = params['vol']
        
        log_returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt,
            sigma * np.sqrt(dt),
            n_days
        )
        log_returns[0] = 0
        
        close_prices = params['price'] * np.exp(np.cumsum(log_returns))
        
        # Generate OHLC from close
        daily_range = np.abs(np.random.normal(0, sigma * np.sqrt(dt), n_days))
        high_prices = close_prices * (1 + daily_range * 0.5)
        low_prices = close_prices * (1 - daily_range * 0.5)
        
        # Open is previous close with overnight gap
        open_prices = np.roll(close_prices, 1) * (1 + np.random.normal(0, 0.002, n_days))
        open_prices[0] = params['price']
        
        # Ensure OHLC consistency
        high_prices = np.maximum.reduce([open_prices, close_prices, high_prices])
        low_prices = np.minimum.reduce([open_prices, close_prices, low_prices])
        
        # Generate volume with regime changes
        base_volume = params['avg_volume']
        volume_multiplier = np.exp(np.random.normal(0, 0.5, n_days))
        
        # Add volume spikes on high volatility days
        vol_shock = np.abs(log_returns) > 2 * sigma * np.sqrt(dt)
        volume_multiplier[vol_shock] *= 2
        
        volumes = base_volume * volume_multiplier
        
        data_dict[('Open', ticker)] = open_prices
        data_dict[('High', ticker)] = high_prices
        data_dict[('Low', ticker)] = low_prices
        data_dict[('Close', ticker)] = close_prices
        data_dict[('Volume', ticker)] = volumes.astype(int)
    
    df = pd.DataFrame(data_dict, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    
    print(f"Generated {len(df)} days of synthetic data for {len(tickers)} assets")
    return df


def calculate_liquidity_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various liquidity metrics from OHLCV data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data with MultiIndex columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with liquidity metrics
    """
    tickers = data['Close'].columns.tolist()
    metrics_list = []
    
    for ticker in tickers:
        ohlcv = pd.DataFrame({
            'open': data['Open'][ticker],
            'high': data['High'][ticker],
            'low': data['Low'][ticker],
            'close': data['Close'][ticker],
            'volume': data['Volume'][ticker]
        })
        
        metrics = pd.DataFrame(index=ohlcv.index)
        metrics['ticker'] = ticker
        
        # Returns
        metrics['returns'] = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
        metrics['abs_returns'] = np.abs(metrics['returns'])
        
        # Volatility measures
        metrics['realized_vol'] = metrics['returns'].rolling(20).std() * np.sqrt(252)
        metrics['high_low_vol'] = np.log(ohlcv['high'] / ohlcv['low'])
        
        # Volume metrics
        metrics['volume'] = ohlcv['volume']
        metrics['volume_ma20'] = ohlcv['volume'].rolling(20).mean()
        metrics['volume_ratio'] = ohlcv['volume'] / metrics['volume_ma20']
        
        # Dollar volume
        metrics['dollar_volume'] = ohlcv['close'] * ohlcv['volume']
        
        # Amihud illiquidity ratio
        metrics['amihud'] = (
            metrics['abs_returns'] / metrics['dollar_volume'] * 1e10
        )
        metrics['amihud_ma20'] = metrics['amihud'].rolling(20).mean()
        
        # Bid-Ask Spread Proxy (Corwin-Schultz)
        high = ohlcv['high']
        low = ohlcv['low']
        
        beta = np.log(high / low) ** 2 + np.log(high.shift(1) / low.shift(1)) ** 2
        gamma = np.log(high.rolling(2).max() / low.rolling(2).min()) ** 2
        
        alpha_term = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
        alpha_term = alpha_term - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        
        metrics['spread_proxy'] = 2 * (np.exp(alpha_term) - 1) / (1 + np.exp(alpha_term))
        metrics['spread_proxy'] = metrics['spread_proxy'].clip(lower=0, upper=0.1)
        
        # Roll's spread estimator
        cov_returns = metrics['returns'].rolling(20).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan,
            raw=True
        )
        metrics['roll_spread'] = 2 * np.sqrt(np.abs(-cov_returns))
        metrics['roll_spread'] = metrics['roll_spread'].clip(upper=0.1)
        
        # Price impact (Kyle's lambda proxy)
        metrics['price_impact'] = (
            metrics['abs_returns'] / np.sqrt(ohlcv['volume']) * 1e4
        )
        
        # Trading activity
        metrics['turnover'] = ohlcv['volume'] / ohlcv['volume'].rolling(252).mean()
        
        # Intraday range as % of price
        metrics['intraday_range'] = (ohlcv['high'] - ohlcv['low']) / ohlcv['close']
        
        metrics_list.append(metrics)
    
    result = pd.concat(metrics_list, ignore_index=False)
    result = result.reset_index().rename(columns={'index': 'date'})
    
    return result.dropna()


def create_liquidity_dataset(
    tickers: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a complete liquidity dataset ready for ML.
    
    Parameters:
    -----------
    tickers : List[str], optional
        Tickers to include (default: diversified set)
    start_date, end_date : str
        Date range
    
    Returns:
    --------
    pd.DataFrame
        Complete liquidity dataset
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ']
    
    # Fetch data
    ohlcv = fetch_market_data(tickers, start_date, end_date)
    
    # Calculate metrics
    liquidity_data = calculate_liquidity_metrics(ohlcv)
    
    print(f"Created liquidity dataset: {len(liquidity_data)} rows")
    
    return liquidity_data


def get_default_dataset() -> Dict:
    """
    Get default dataset configuration.
    
    Returns:
    --------
    Dict with dataset parameters
    """
    return {
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ'],
        'target': 'spread_proxy',
        'features': [
            'realized_vol', 'volume_ratio', 'amihud_ma20', 
            'price_impact', 'intraday_range', 'turnover',
            'volume', 'dollar_volume', 'abs_returns'
        ]
    }


if __name__ == "__main__":
    print("Testing Data Loader...")
    
    # Create dataset
    df = create_liquidity_dataset(
        tickers=['SPY', 'AAPL', 'MSFT'],
        start_date="2022-01-01"
    )
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample data:")
    print(df.head())
    
    print(f"\nLiquidity statistics:")
    print(df.groupby('ticker')['spread_proxy'].describe())
