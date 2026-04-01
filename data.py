import yfinance as yf
import pandas as pd
import numpy as np


def get_stock_data(ticker, lookback_days):
    """
    Downloads historical stock data with robust error handling.

    Parameters:
    - ticker (str): Stock symbol, e.g., "AAPL"
    - lookback_days (int): Number of past trading days to retrieve

    Returns:
    - current_price (float): Latest closing price
    - prices (pd.Series): Historical closing prices
    - log_returns (np.ndarray): Daily log returns
    """
    # Use calendar days with buffer to ensure enough trading days
    calendar_days = int(lookback_days * 1.6) + 10
    period = f"{calendar_days}d"

    data = yf.download(ticker, period=period, progress=False)

    if data.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. "
            "Please check the symbol and try again."
        )

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        raise ValueError(f"No 'Close' column found in data for '{ticker}'.")

    prices = data["Close"].dropna()

    if len(prices) < 10:
        raise ValueError(
            f"Insufficient data for '{ticker}': only {len(prices)} data points. "
            "Need at least 10 trading days."
        )

    # Trim to requested lookback (trading days)
    prices = prices.tail(lookback_days)

    current_price = float(prices.iloc[-1])
    log_returns = np.log(prices / prices.shift(1)).dropna().to_numpy()

    return current_price, prices, log_returns
