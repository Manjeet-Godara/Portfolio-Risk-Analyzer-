"""
portfolio.py
------------
Portfolio construction and performance measurement layer for the
Portfolio Risk Analyzer.

Takes individual-asset log returns and combines them into a weighted
portfolio time series, then computes the standard suite of performance
and risk metrics used in quantitative portfolio management.

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# 1. Build weighted portfolio
# ---------------------------------------------------------------------------

def build_portfolio(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    portfolio_value: float = 100_000,
) -> pd.DataFrame:
    """
    Construct a weighted portfolio time series from individual asset log returns.

    The portfolio daily return on day t is the dot product of the weight
    vector and each asset's return vector:
        r_p,t = sum_i( w_i * r_i,t )

    This is a *static* (buy-and-hold) weighting scheme — weights are set
    once at inception and drift with market moves.  Rebalancing is left
    for a future step.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns, indexed by date (output of risk_engine.compute_returns).
    weights : dict[str, float]
        Mapping of ticker → portfolio weight.  Must sum to 1.0 (±0.001
        tolerance for floating-point rounding).
        Example: {"AAPL": 0.25, "MSFT": 0.25, "JPM": 0.20, "GS": 0.15, "SPY": 0.15}
    portfolio_value : float, optional
        Starting notional value in USD.  Default: 100,000.

    Returns
    -------
    pd.DataFrame
        Time series with four columns:
          - daily_return     : weighted portfolio log return (decimal)
          - daily_pnl        : dollar P&L for that day
          - cumulative_value : running portfolio value starting from portfolio_value
          - cumulative_return: cumulative return from inception (decimal)
        Saved to data/portfolio_timeseries.csv.

    Raises
    ------
    ValueError
        If weights do not sum to 1.0, or if a ticker in weights is missing
        from returns_df.
    """

    # ── Validate tickers ────────────────────────────────────────────────────
    missing = [t for t in weights if t not in returns_df.columns]
    if missing:
        raise ValueError(
            f"Tickers in weights not found in returns DataFrame: {missing}\n"
            f"Available tickers: {list(returns_df.columns)}"
        )

    # ── Validate weights sum to 1.0 ─────────────────────────────────────────
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        raise ValueError(
            f"Portfolio weights must sum to 1.0.  "
            f"Current sum: {total_weight:.6f}  "
            f"Difference: {total_weight - 1.0:+.6f}\n"
            f"Weights provided: {weights}"
        )

    # ── Align returns to the ordered weight vector ───────────────────────────
    tickers = list(weights.keys())
    w_array = np.array([weights[t] for t in tickers], dtype=float)
    r_matrix = returns_df[tickers].values          # shape: (T, N)

    # ── Weighted daily portfolio return ──────────────────────────────────────
    # Matrix multiply: (T, N) · (N,) → (T,)
    daily_returns = r_matrix @ w_array

    # ── Dollar P&L  (using simple-return approximation for small r) ──────────
    # For daily log returns r ≈ simple return, so P&L ≈ portfolio_value * r
    # More precisely we track the running portfolio value and take differences
    # to get P&L, which is done below.

    # ── Cumulative portfolio value ───────────────────────────────────────────
    # Convert log returns to simple returns for compounding:
    #   cumulative_value_t = V_0 * prod(1 + simple_r_i)
    #                      = V_0 * exp(sum(log_r_i))    ← exact equivalence
    simple_returns    = np.exp(daily_returns) - 1
    cumulative_growth = np.cumprod(1 + simple_returns)
    cumulative_value  = portfolio_value * cumulative_growth

    # Daily dollar P&L: today's value minus yesterday's value
    prev_values = np.concatenate([[portfolio_value], cumulative_value[:-1]])
    daily_pnl   = cumulative_value - prev_values

    # Cumulative return from inception
    cumulative_return = cumulative_value / portfolio_value - 1

    # ── Assemble output DataFrame ────────────────────────────────────────────
    portfolio_df = pd.DataFrame(
        {
            "daily_return":      daily_returns,
            "daily_pnl":         daily_pnl,
            "cumulative_value":  cumulative_value,
            "cumulative_return": cumulative_return,
        },
        index=returns_df.index,
    )

    # ── Persist ─────────────────────────────────────────────────────────────
    _ensure_dir("data/portfolio_timeseries.csv")
    portfolio_df.to_csv("data/portfolio_timeseries.csv")

    return portfolio_df


# ---------------------------------------------------------------------------
# 2. Equal-weight portfolio (convenience wrapper)
# ---------------------------------------------------------------------------

def build_equal_weight_portfolio(
    returns_df: pd.DataFrame,
    portfolio_value: float = 100_000,
) -> pd.DataFrame:
    """
    Build a 1/N equal-weight portfolio across all tickers in returns_df.

    Each ticker receives weight = 1 / number_of_tickers.
    Internally delegates to build_portfolio(), so all validation and
    persistence logic is shared.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns (output of risk_engine.compute_returns).
    portfolio_value : float, optional
        Starting notional value in USD.  Default: 100,000.

    Returns
    -------
    pd.DataFrame
        Same structure as build_portfolio() output.
    """
    n = len(returns_df.columns)
    equal_weights = {ticker: round(1.0 / n, 10) for ticker in returns_df.columns}

    # Correct any floating-point drift so weights sum exactly to 1.0
    tickers = list(equal_weights.keys())
    total   = sum(equal_weights.values())
    equal_weights[tickers[-1]] += 1.0 - total   # absorb rounding residual

    return build_portfolio(returns_df, equal_weights, portfolio_value)


# ---------------------------------------------------------------------------
# 3. Portfolio-level performance metrics
# ---------------------------------------------------------------------------

def compute_portfolio_metrics(
    portfolio_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Compute a standard suite of portfolio performance and risk metrics.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Time series output of build_portfolio() or build_equal_weight_portfolio().
        Must contain columns: daily_return, cumulative_value.
    risk_free_rate : float, optional
        Annualised risk-free rate as a decimal (e.g. 0.05 = 5%).
        Used in Sharpe and Calmar ratio calculations.  Default: 0.05.

    Returns
    -------
    dict
        Keys and value types:
          total_return        (float)  — total % return over full period
          annualised_return   (float)  — CAGR over full period
          annualised_vol      (float)  — annualised std of daily log returns
          sharpe_ratio        (float)  — (ann_return - rfr) / ann_vol
          max_drawdown        (float)  — largest peak-to-trough decline (negative)
          calmar_ratio        (float)  — ann_return / abs(max_drawdown)
          best_month          (float)  — best calendar-month return
          worst_month         (float)  — worst calendar-month return
          pct_positive_days   (float)  — fraction of days with positive return
          n_trading_days      (int)    — number of trading days in the series
    """
    daily_ret  = portfolio_df["daily_return"]
    cum_values = portfolio_df["cumulative_value"]
    n_days     = len(daily_ret)

    # ── Total return ────────────────────────────────────────────────────────
    total_return = cum_values.iloc[-1] / cum_values.iloc[0] - 1

    # ── Annualised return (CAGR) ─────────────────────────────────────────────
    n_years          = n_days / TRADING_DAYS_PER_YEAR
    annualised_return = (1 + total_return) ** (1 / n_years) - 1

    # ── Annualised volatility ────────────────────────────────────────────────
    annualised_vol = daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # ── Sharpe Ratio ────────────────────────────────────────────────────────
    # Excess return per unit of annualised vol
    sharpe_ratio = (
        (annualised_return - risk_free_rate) / annualised_vol
        if annualised_vol != 0 else np.nan
    )

    # ── Maximum Drawdown ────────────────────────────────────────────────────
    # At each point, drawdown = (current value - running peak) / running peak
    running_peak  = cum_values.cummax()
    drawdown_series = (cum_values - running_peak) / running_peak
    max_drawdown    = drawdown_series.min()   # most negative value

    # ── Calmar Ratio ────────────────────────────────────────────────────────
    # Reward-to-risk using max drawdown as the risk denominator
    calmar_ratio = (
        annualised_return / abs(max_drawdown)
        if max_drawdown != 0 else np.nan
    )

    # ── Monthly returns ──────────────────────────────────────────────────────
    # Resample daily log returns to monthly by summing (log returns are additive)
    monthly_returns = daily_ret.resample("ME").sum()
    best_month  = monthly_returns.max()
    worst_month = monthly_returns.min()

    # ── % of days with positive return ──────────────────────────────────────
    pct_positive_days = (daily_ret > 0).mean()

    return {
        "total_return":       total_return,
        "annualised_return":  annualised_return,
        "annualised_vol":     annualised_vol,
        "sharpe_ratio":       sharpe_ratio,
        "max_drawdown":       max_drawdown,
        "calmar_ratio":       calmar_ratio,
        "best_month":         best_month,
        "worst_month":        worst_month,
        "pct_positive_days":  pct_positive_days,
        "n_trading_days":     n_days,
    }


# ---------------------------------------------------------------------------
# 4. Compare two portfolios side by side
# ---------------------------------------------------------------------------

def compare_portfolios(
    returns_df: pd.DataFrame,
    portfolio_value: float = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two portfolios with different weight schemes and print a
    side-by-side performance comparison.

    Portfolio A — Equal weight (1/N):
        Each of the 5 tickers receives 20%.

    Portfolio B — Tech-overweight custom:
        {"AAPL": 0.35, "MSFT": 0.30, "JPM": 0.15, "GS": 0.10, "SPY": 0.10}
        Tilts toward large-cap tech vs. the equal-weight benchmark.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns (output of risk_engine.compute_returns).
    portfolio_value : float, optional
        Starting notional value in USD for both portfolios.  Default: 100,000.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (portfolio_a_df, portfolio_b_df) — time series DataFrames for each.
    """

    # ── Build portfolios ─────────────────────────────────────────────────────
    print("  Building Portfolio A — Equal Weight (20% each) …")
    portfolio_a_df = build_equal_weight_portfolio(returns_df, portfolio_value)

    custom_weights = {
        "AAPL": 0.35,
        "MSFT": 0.30,
        "JPM":  0.15,
        "GS":   0.10,
        "SPY":  0.10,
    }
    print("  Building Portfolio B — Tech-Overweight Custom …")
    portfolio_b_df = build_portfolio(returns_df, custom_weights, portfolio_value)

    # ── Compute metrics for both ─────────────────────────────────────────────
    metrics_a = compute_portfolio_metrics(portfolio_a_df)
    metrics_b = compute_portfolio_metrics(portfolio_b_df)

    # ── Print individual summaries ───────────────────────────────────────────
    print()
    print_portfolio_summary(metrics_a, portfolio_name="Portfolio A — Equal Weight")
    print_portfolio_summary(metrics_b, portfolio_name="Portfolio B — Tech Overweight")

    # ── Print side-by-side comparison table ─────────────────────────────────
    _print_comparison_table(metrics_a, metrics_b)

    # ── Save both time series to CSV ─────────────────────────────────────────
    _ensure_dir("data/portfolio_a_equal.csv")
    portfolio_a_df.to_csv("data/portfolio_a_equal.csv")
    portfolio_b_df.to_csv("data/portfolio_b_custom.csv")

    print(f"  Saved → {os.path.abspath('data/portfolio_a_equal.csv')}")
    print(f"  Saved → {os.path.abspath('data/portfolio_b_custom.csv')}")

    return portfolio_a_df, portfolio_b_df


# ---------------------------------------------------------------------------
# 5. Print formatted summary block
# ---------------------------------------------------------------------------

def print_portfolio_summary(
    metrics: dict,
    portfolio_name: str = "Portfolio",
) -> None:
    """
    Print a clean, formatted performance summary block for one portfolio.

    Parameters
    ----------
    metrics : dict
        Output of compute_portfolio_metrics().
    portfolio_name : str, optional
        Display name shown in the header.  Default: "Portfolio".
    """
    bar_len = max(32, len(portfolio_name) + 4)
    bar     = "=" * bar_len

    # Sign-aware formatters
    def pct(v: float, decimals: int = 1) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v * 100:.{decimals}f}%"

    def ratio(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "N/A"

    print(bar)
    print(f"  {portfolio_name}")
    print(bar)
    print(f"  {'Total Return':<22}: {pct(metrics['total_return'])}")
    print(f"  {'Annualised Return':<22}: {pct(metrics['annualised_return'])}")
    print(f"  {'Annualised Vol':<22}: {pct(metrics['annualised_vol'])}")
    print(f"  {'Sharpe Ratio':<22}: {ratio(metrics['sharpe_ratio'])}")
    print(f"  {'Max Drawdown':<22}: {pct(metrics['max_drawdown'])}")
    print(f"  {'Calmar Ratio':<22}: {ratio(metrics['calmar_ratio'])}")
    print(f"  {'Best Month':<22}: {pct(metrics['best_month'])}")
    print(f"  {'Worst Month':<22}: {pct(metrics['worst_month'])}")
    print(f"  {'Positive Days':<22}: {metrics['pct_positive_days'] * 100:.1f}%")
    print(f"  {'Trading Days':<22}: {metrics['n_trading_days']}")
    print(bar)
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_comparison_table(
    metrics_a: dict,
    metrics_b: dict,
    name_a: str = "Equal Weight",
    name_b: str = "Tech OW",
) -> None:
    """
    Print a side-by-side comparison table for two metrics dicts.

    Parameters
    ----------
    metrics_a, metrics_b : dict
        Outputs of compute_portfolio_metrics().
    name_a, name_b : str
        Short display labels for the column headers.
    """

    def pct(v: float, decimals: int = 1) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v * 100:.{decimals}f}%"

    def ratio(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "N/A"

    col_label = 24
    col_val   = 14

    header  = (
        f"\n  {'Metric':<{col_label}}"
        f"{name_a:>{col_val}}"
        f"{name_b:>{col_val}}"
    )
    divider = "  " + "-" * (col_label + col_val * 2)

    rows = [
        ("Total Return",       pct(metrics_a["total_return"]),       pct(metrics_b["total_return"])),
        ("Annualised Return",  pct(metrics_a["annualised_return"]),   pct(metrics_b["annualised_return"])),
        ("Annualised Vol",     pct(metrics_a["annualised_vol"]),      pct(metrics_b["annualised_vol"])),
        ("Sharpe Ratio",       ratio(metrics_a["sharpe_ratio"]),      ratio(metrics_b["sharpe_ratio"])),
        ("Max Drawdown",       pct(metrics_a["max_drawdown"]),        pct(metrics_b["max_drawdown"])),
        ("Calmar Ratio",       ratio(metrics_a["calmar_ratio"]),      ratio(metrics_b["calmar_ratio"])),
        ("Best Month",         pct(metrics_a["best_month"]),          pct(metrics_b["best_month"])),
        ("Worst Month",        pct(metrics_a["worst_month"]),         pct(metrics_b["worst_month"])),
        ("Positive Days",      f"{metrics_a['pct_positive_days']*100:.1f}%",
                               f"{metrics_b['pct_positive_days']*100:.1f}%"),
    ]

    print("  " + "=" * (col_label + col_val * 2))
    print("  SIDE-BY-SIDE COMPARISON")
    print("  " + "=" * (col_label + col_val * 2))
    print(header)
    print(divider)
    for label, val_a, val_b in rows:
        print(
            f"  {label:<{col_label}}"
            f"{val_a:>{col_val}}"
            f"{val_b:>{col_val}}"
        )
    print(divider)
    print()


def _ensure_dir(filepath: str) -> None:
    """Create all parent directories for *filepath* if they do not exist."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
