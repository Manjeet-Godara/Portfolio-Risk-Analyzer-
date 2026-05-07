"""
main.py
-------
Entry point for the Portfolio Risk Analyzer.

Steps completed
---------------
  Step 1 : Project setup & data fetching      (src/data_fetcher.py)
  Step 2 : Returns & volatility engine        (src/risk_engine.py)
  Step 3 : Portfolio construction & metrics   (src/portfolio.py)

Run:
    python main.py

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import sys
import os
from datetime import date

# ---------------------------------------------------------------------------
# Ensure src/ is importable regardless of working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_fetcher import fetch_stock_data
from risk_engine import (
    compute_returns,
    compute_volatility,
    compute_correlation_matrix,
    compute_rolling_correlation,
    print_stats_summary,
)
from portfolio import (
    compare_portfolios,
    compute_portfolio_metrics,
    print_portfolio_summary,
)

# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------

TICKERS: list[str] = ["AAPL", "MSFT", "JPM", "GS", "SPY"]
START_DATE: str = "2020-01-01"
END_DATE:   str = date.today().isoformat()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Print a clearly visible section header to stdout."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> dict:
    """
    Orchestrate all completed steps of the Portfolio Risk Analyzer.

    Returns a dict of all computed artefacts so this module can be
    imported and called interactively (e.g. from a Jupyter notebook).
    """

    # ── STEP 1: Fetch prices ─────────────────────────────────────────────────
    section("STEP 1 — DATA FETCHING")

    prices = fetch_stock_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        save_path="data/raw_prices.csv",
    )

    print(f"\n  Shape  : {prices.shape[0]} rows x {prices.shape[1]} columns")
    print(f"  Dates  : {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"  Nulls  : {prices.isna().sum().sum()}")

    # ── STEP 2: Returns & volatility ─────────────────────────────────────────
    section("STEP 2 — RETURNS & VOLATILITY ANALYSIS")

    print("\n  [1/4]  Computing daily log returns ...")
    log_returns = compute_returns(prices)

    print("\n  [2/4]  Computing volatility ...")
    vol_summary = compute_volatility(log_returns, window_short=30, window_long=60)

    print("\n  [3/4]  Computing correlation matrix ...")
    corr_matrix = compute_correlation_matrix(log_returns)

    print("\n  [4/4]  Computing rolling 60-day correlation: JPM vs GS ...")
    jpm_gs_rolling_corr = compute_rolling_correlation(
        log_returns, ticker_a="JPM", ticker_b="GS", window=60
    )
    print(f"  Latest JPM-GS 60-day correlation : "
          f"{jpm_gs_rolling_corr.dropna().iloc[-1]:.3f}")

    print("\n── Descriptive Statistics ───────────────────────────────────────")
    print_stats_summary(log_returns, vol_summary)

    # ── STEP 3: Portfolio construction ───────────────────────────────────────
    section("STEP 3 — PORTFOLIO CONSTRUCTION")

    portfolio_a_df, portfolio_b_df = compare_portfolios(
        log_returns, portfolio_value=100_000
    )

    # Compute metrics for use in later steps (VaR, reporting)
    metrics_a = compute_portfolio_metrics(portfolio_a_df)
    metrics_b = compute_portfolio_metrics(portfolio_b_df)

    # ── Summary of all saved files ───────────────────────────────────────────
    section("OUTPUTS WRITTEN")
    outputs = [
        "data/raw_prices.csv",
        "data/daily_returns.csv",
        "data/daily_returns_combined.csv",
        "data/rolling_volatility.csv",
        "data/correlation_matrix.csv",
        "data/portfolio_a_equal.csv",
        "data/portfolio_b_custom.csv",
    ]
    for path in outputs:
        status = "OK     " if os.path.exists(path) else "MISSING"
        print(f"  [{status}]  {path}")

    print("\n  Step 3 complete. Ready for Step 4 (VaR / CVaR).\n")

    return {
        # Step 1
        "prices":               prices,
        # Step 2
        "log_returns":          log_returns,
        "vol_summary":          vol_summary,
        "corr_matrix":          corr_matrix,
        "jpm_gs_rolling_corr":  jpm_gs_rolling_corr,
        # Step 3
        "portfolio_a_df":       portfolio_a_df,
        "portfolio_b_df":       portfolio_b_df,
        "metrics_a":            metrics_a,
        "metrics_b":            metrics_b,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
