"""
data_fetcher.py
---------------
Handles all historical market data retrieval and persistence for the
Portfolio Risk Analyzer.  Uses yfinance to pull adjusted closing prices,
validates and cleans the resulting DataFrame, saves a raw CSV snapshot,
and prints a human-readable summary.

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import os
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

# Suppress the noisy FutureWarnings that yfinance can emit
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    save_path: str = "data/raw_prices.csv",
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers and return a
    clean DataFrame ready for downstream analysis.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance ticker symbols, e.g. ["AAPL", "MSFT", "SPY"].
    start_date : str
        Inclusive start date in "YYYY-MM-DD" format.
    end_date : str
        Exclusive end date in "YYYY-MM-DD" format (yfinance convention).
    save_path : str, optional
        Relative or absolute path where the raw CSV will be written.
        Default: "data/raw_prices.csv".

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices indexed by date.
        Columns are ticker symbols.  All rows with any NaN are dropped.

    Raises
    ------
    ValueError
        If no data is returned for any ticker, or if the cleaned DataFrame
        is empty after dropping NaN rows.
    """

    print("=" * 60)
    print("  PORTFOLIO RISK ANALYZER — Data Fetcher")
    print("=" * 60)
    print(f"  Tickers    : {', '.join(tickers)}")
    print(f"  Start date : {start_date}")
    print(f"  End date   : {end_date}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # 1. Download data via yfinance
    # ------------------------------------------------------------------
    print("  Fetching data from Yahoo Finance …")

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,   # gives us split/dividend-adjusted prices
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            "yfinance returned an empty DataFrame. "
            "Check your tickers and date range."
        )

    # ------------------------------------------------------------------
    # 2. Extract the "Close" column (= adjusted close when auto_adjust=True)
    # ------------------------------------------------------------------
    # yfinance returns a MultiIndex when multiple tickers are requested.
    # With auto_adjust=True the column name is "Close" (not "Adj Close").
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single-ticker edge-case: promote to a DataFrame with named col
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    # Ensure column order matches the input list (yfinance may reorder)
    prices = prices[[t for t in tickers if t in prices.columns]]

    # ------------------------------------------------------------------
    # 3. Clean: drop any rows that have at least one NaN
    # ------------------------------------------------------------------
    n_raw = len(prices)
    missing_before = prices.isna().sum()

    prices.dropna(inplace=True)

    n_clean = len(prices)
    n_dropped = n_raw - n_clean

    if prices.empty:
        raise ValueError(
            "DataFrame is empty after dropping NaN rows. "
            "Try extending the date range or checking tickers."
        )

    # ------------------------------------------------------------------
    # 4. Warn about tickers that were missing data before cleaning
    # ------------------------------------------------------------------
    print()
    if missing_before.sum() > 0:
        print("  ⚠  Missing-value warnings (before cleaning):")
        for ticker, count in missing_before[missing_before > 0].items():
            print(f"     {ticker:>6s} — {count} missing row(s)")
        print(f"     Rows dropped during cleaning : {n_dropped}")
    else:
        print("  ✓  No missing values detected in raw download.")

    # ------------------------------------------------------------------
    # 5. Save raw prices to CSV
    # ------------------------------------------------------------------
    _ensure_dir(save_path)
    prices.to_csv(save_path)
    print(f"\n  ✓  Saved raw prices → {os.path.abspath(save_path)}")

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    _print_summary(prices)

    print("\n" + "=" * 60)
    return prices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(filepath: str) -> None:
    """Create all parent directories for *filepath* if they do not exist."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _print_summary(prices: pd.DataFrame) -> None:
    """
    Print a formatted summary table: one row per ticker showing
    trading days fetched, first price, last price, and total return.

    Parameters
    ----------
    prices : pd.DataFrame
        Clean DataFrame of adjusted closing prices (output of fetch_stock_data).
    """
    trading_days = len(prices)
    date_min = prices.index.min().strftime("%Y-%m-%d")
    date_max = prices.index.max().strftime("%Y-%m-%d")

    print(f"\n  Trading days : {trading_days}  ({date_min} → {date_max})")
    print()

    # Header
    col_w = [8, 14, 12, 12, 14]
    header = (
        f"  {'Ticker':<{col_w[0]}}"
        f"{'Trading Days':>{col_w[1]}}"
        f"{'First Price':>{col_w[2]}}"
        f"{'Last Price':>{col_w[3]}}"
        f"{'Total Return':>{col_w[4]}}"
    )
    print(header)
    print("  " + "-" * (sum(col_w) + 2))

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        days   = len(series)
        first  = series.iloc[0]
        last   = series.iloc[-1]
        ret    = (last / first - 1) * 100

        row = (
            f"  {ticker:<{col_w[0]}}"
            f"{days:>{col_w[1]}}"
            f"{first:>{col_w[2]}.2f}"
            f"{last:>{col_w[3]}.2f}"
            f"{ret:>{col_w[4] - 1}.2f}%"
        )
        print(row)

    # Null check confirmation
    null_counts = prices.isna().sum()
    print()
    if null_counts.sum() == 0:
        print("  ✓  Null values in final DataFrame : 0  (clean)")
    else:
        print("  ✗  Null values remaining per ticker:")
        print(null_counts.to_string())
