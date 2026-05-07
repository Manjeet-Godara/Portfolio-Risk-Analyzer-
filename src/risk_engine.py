"""
risk_engine.py
--------------
Returns and volatility engine for the Portfolio Risk Analyzer.

Computes daily log returns, annualised volatility, rolling volatility,
correlation matrices, and distributional statistics (skewness, kurtosis)
from a DataFrame of adjusted closing prices.

All intermediate results are persisted to the data/ directory so that
downstream steps (VaR, optimisation, reporting) can load them directly.

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
"""Standard annualisation factor used across the industry."""


# ---------------------------------------------------------------------------
# 1. Returns
# ---------------------------------------------------------------------------

def compute_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns and simple returns from adjusted closing prices.

    Log return  : r_t = ln(P_t / P_{t-1})
    Simple return: r_t = (P_t / P_{t-1}) - 1

    The first row is dropped because it has no prior price to compute
    a return against.  Only log returns are returned (and saved as the
    primary series); simple returns are saved alongside them for reference.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame of adjusted closing prices indexed by date.
        Columns are ticker symbols.

    Returns
    -------
    pd.DataFrame
        Daily log returns, indexed by date, one column per ticker.
        Saved to data/daily_returns.csv.
    """
    # ── Log returns ─────────────────────────────────────────────────────────
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

    # ── Simple returns (stored for reference / cross-checks) ────────────────
    simple_returns = prices_df.pct_change().dropna()

    # ── Persist ─────────────────────────────────────────────────────────────
    _ensure_dir("data/daily_returns.csv")

    # Save log returns as the primary file
    log_returns.to_csv("data/daily_returns.csv")

    # Save simple returns alongside, with a "_simple" suffix on columns
    simple_col_map = {t: f"{t}_simple" for t in simple_returns.columns}
    combined = log_returns.join(simple_returns.rename(columns=simple_col_map))
    combined.to_csv("data/daily_returns_combined.csv")

    print(f"  ✓  Log returns saved      → {os.path.abspath('data/daily_returns.csv')}")
    print(f"  ✓  Combined returns saved → {os.path.abspath('data/daily_returns_combined.csv')}")
    print(f"     Shape : {log_returns.shape[0]} trading days × {log_returns.shape[1]} tickers")

    return log_returns


# ---------------------------------------------------------------------------
# 2. Volatility
# ---------------------------------------------------------------------------

def compute_volatility(
    returns_df: pd.DataFrame,
    window_short: int = 30,
    window_long: int = 60,
) -> dict[str, float]:
    """
    Compute annualised volatility (full-period), and rolling volatility
    at two lookback windows.

    Annualised vol = daily_std × √252
    Rolling vol    = rolling_std(window) × √252  (computed for each date)

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns (output of compute_returns).
    window_short : int, optional
        Short rolling window in trading days. Default: 30.
    window_long : int, optional
        Long rolling window in trading days. Default: 60.

    Returns
    -------
    dict[str, float]
        Mapping of ticker → annualised volatility (as a decimal, e.g. 0.284
        means 28.4%).  Saved alongside rolling vol to data/rolling_volatility.csv.
    """
    ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR)

    # ── Full-period annualised vol ───────────────────────────────────────────
    vol_summary: dict[str, float] = {
        ticker: returns_df[ticker].std() * ann_factor
        for ticker in returns_df.columns
    }

    # ── Rolling volatility ───────────────────────────────────────────────────
    rolling_short = (
        returns_df.rolling(window=window_short).std() * ann_factor
    )
    rolling_short.columns = [f"{t}_{window_short}d" for t in returns_df.columns]

    rolling_long = (
        returns_df.rolling(window=window_long).std() * ann_factor
    )
    rolling_long.columns = [f"{t}_{window_long}d" for t in returns_df.columns]

    rolling_vol = rolling_short.join(rolling_long).dropna()

    # ── Persist ─────────────────────────────────────────────────────────────
    _ensure_dir("data/rolling_volatility.csv")
    rolling_vol.to_csv("data/rolling_volatility.csv")

    print(f"  ✓  Rolling volatility saved → {os.path.abspath('data/rolling_volatility.csv')}")
    print(f"     Windows : {window_short}-day and {window_long}-day")

    return vol_summary


# ---------------------------------------------------------------------------
# 3. Correlation matrix
# ---------------------------------------------------------------------------

def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix of daily log returns.

    Correlation measures how co-integrated two assets' daily moves are.
    Values close to +1 mean assets move together; close to −1 means they
    move inversely.  A diversified portfolio wants low average pairwise
    correlations.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns (output of compute_returns).

    Returns
    -------
    pd.DataFrame
        Square correlation matrix (tickers × tickers).
        Saved to data/correlation_matrix.csv.
    """
    corr_matrix = returns_df.corr(method="pearson")

    # ── Persist ─────────────────────────────────────────────────────────────
    _ensure_dir("data/correlation_matrix.csv")
    corr_matrix.to_csv("data/correlation_matrix.csv")

    print(f"  ✓  Correlation matrix saved → {os.path.abspath('data/correlation_matrix.csv')}")

    # ── Pretty-print to console ──────────────────────────────────────────────
    print()
    print("  Pearson Correlation Matrix (log returns):")
    print()
    _print_corr_matrix(corr_matrix)

    return corr_matrix


# ---------------------------------------------------------------------------
# 4. Rolling correlation between two tickers
# ---------------------------------------------------------------------------

def compute_rolling_correlation(
    returns_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 60,
) -> pd.Series:
    """
    Compute the rolling Pearson correlation between two tickers.

    Useful for spotting regime changes — e.g. two assets that are normally
    uncorrelated suddenly becoming highly correlated during a market stress
    event (correlation breakdown is a well-known risk in portfolio models).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns (output of compute_returns).
    ticker_a : str
        First ticker symbol, must be a column in returns_df.
    ticker_b : str
        Second ticker symbol, must be a column in returns_df.
    window : int, optional
        Rolling window in trading days. Default: 60.

    Returns
    -------
    pd.Series
        Rolling correlation series, indexed by date.
        NaN for the first (window − 1) dates.

    Raises
    ------
    KeyError
        If either ticker is not present in returns_df.
    """
    for t in (ticker_a, ticker_b):
        if t not in returns_df.columns:
            raise KeyError(
                f"Ticker '{t}' not found in returns DataFrame. "
                f"Available: {list(returns_df.columns)}"
            )

    rolling_corr = (
        returns_df[ticker_a]
        .rolling(window=window)
        .corr(returns_df[ticker_b])
    )
    rolling_corr.name = f"{ticker_a}_{ticker_b}_{window}d_corr"
    return rolling_corr


# ---------------------------------------------------------------------------
# 5. Summary statistics table
# ---------------------------------------------------------------------------

def print_stats_summary(
    returns_df: pd.DataFrame,
    vol_summary: dict[str, float],
) -> None:
    """
    Print a formatted table of key descriptive statistics for each ticker.

    Columns printed
    ---------------
    - Mean daily log return (%)
    - Annualised volatility  (%)
    - Best single day        (%)
    - Worst single day       (%)
    - Skewness  (Fisher definition, via scipy.stats.skew)
    - Excess kurtosis (scipy.stats.kurtosis, Fisher=True → normal = 0)

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns (output of compute_returns).
    vol_summary : dict[str, float]
        Annualised volatility per ticker (output of compute_volatility).
    """
    # Column widths for alignment
    c = [8, 10, 16, 10, 11, 8, 10]

    header = (
        f"  {'Ticker':<{c[0]}}"
        f"{'Ann. Vol':>{c[1]}}"
        f"{'Mean Daily Ret':>{c[2]}}"
        f"{'Best Day':>{c[3]}}"
        f"{'Worst Day':>{c[4]}}"
        f"{'Skew':>{c[5]}}"
        f"{'Kurtosis':>{c[6]}}"
    )
    divider = "  " + "-" * (sum(c) + 2)

    print()
    print(header)
    print(divider)

    for ticker in returns_df.columns:
        series = returns_df[ticker].dropna()

        mean_ret  = series.mean()
        ann_vol   = vol_summary[ticker]
        best_day  = series.max()
        worst_day = series.min()
        skewness  = stats.skew(series)
        kurt      = stats.kurtosis(series, fisher=True)  # excess kurtosis

        row = (
            f"  {ticker:<{c[0]}}"
            f"{ann_vol * 100:>{c[1]}.1f}%"
            f"{mean_ret * 100:>{c[2] - 1}.3f}%"
            f"{best_day * 100:>{c[3] - 1}.2f}%"
            f"{worst_day * 100:>{c[4] - 1}.2f}%"
            f"{skewness:>{c[5]}.2f}"
            f"{kurt:>{c[6]}.2f}"
        )
        print(row)

    print(divider)
    print()
    print("  Notes:")
    print("  · Ann. Vol   = daily std × √252")
    print("  · Skew < 0   = left tail heavier (more large negative days)")
    print("  · Kurtosis   = excess kurtosis; > 0 = fat tails vs. normal (leptokurtic)")
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(filepath: str) -> None:
    """Create all parent directories for *filepath* if they do not exist."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _print_corr_matrix(corr: pd.DataFrame) -> None:
    """
    Print a correlation matrix to the console with right-aligned columns
    and 3 decimal places.

    Parameters
    ----------
    corr : pd.DataFrame
        Square correlation DataFrame.
    """
    tickers = list(corr.columns)
    col_w = 9  # width per cell

    # Header row
    header = " " * 8 + "".join(f"{t:>{col_w}}" for t in tickers)
    print("  " + header)
    print("  " + "-" * len(header))

    for row_ticker in tickers:
        row_str = f"  {row_ticker:<8}"
        for col_ticker in tickers:
            val = corr.loc[row_ticker, col_ticker]
            row_str += f"{val:>{col_w}.3f}"
        print(row_str)

    print()
