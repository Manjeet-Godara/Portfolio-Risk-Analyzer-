"""
main.py
-------
Entry point for the Portfolio Risk Analyzer.

Steps completed
---------------
  Step 1 : Data fetching                     (src/data_fetcher.py)
  Step 2 : Returns & volatility engine       (src/risk_engine.py)
  Step 3 : Portfolio construction & metrics  (src/portfolio.py)
  Step 4 : VaR / CVaR engine                 (src/var_engine.py)
  Step 5 : Visualisations                    (src/visualizer.py)

Run:
    python main.py

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import sys
import os
from datetime import date

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
)
from var_engine import print_var_report
from visualizer import generate_all_charts

# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------

TICKERS: list[str] = ["AAPL", "MSFT", "JPM", "GS", "SPY"]
START_DATE: str = "2020-01-01"
END_DATE:   str = date.today().isoformat()

WEIGHTS_A: dict[str, float] = {t: 0.20 for t in TICKERS}
WEIGHTS_B: dict[str, float] = {
    "AAPL": 0.35, "MSFT": 0.30, "JPM": 0.15, "GS": 0.10, "SPY": 0.10
}


def section(title: str) -> None:
    """Print a clearly visible section header."""
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> dict:
    """
    Orchestrate all five completed steps of the Portfolio Risk Analyzer.
    Returns a dict of every computed artefact for interactive / notebook use.
    """

    # ── STEP 1: Fetch prices ─────────────────────────────────────────────────
    section("STEP 1 — DATA FETCHING")
    prices = fetch_stock_data(
        tickers=TICKERS, start_date=START_DATE,
        end_date=END_DATE, save_path="data/raw_prices.csv",
    )
    print(f"\n  Shape : {prices.shape[0]} rows x {prices.shape[1]} columns")
    print(f"  Dates : {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"  Nulls : {prices.isna().sum().sum()}")

    # ── STEP 2: Returns & volatility ─────────────────────────────────────────
    section("STEP 2 — RETURNS & VOLATILITY ANALYSIS")
    print("\n  [1/4]  Computing daily log returns ...")
    log_returns = compute_returns(prices)

    print("\n  [2/4]  Computing volatility ...")
    vol_summary = compute_volatility(log_returns, window_short=30, window_long=60)

    print("\n  [3/4]  Computing correlation matrix ...")
    corr_matrix = compute_correlation_matrix(log_returns)

    print("\n  [4/4]  Rolling 60-day correlation: JPM vs GS ...")
    jpm_gs_rolling_corr = compute_rolling_correlation(
        log_returns, ticker_a="JPM", ticker_b="GS", window=60
    )
    print(f"  Latest JPM-GS 60-day corr : {jpm_gs_rolling_corr.dropna().iloc[-1]:.3f}")

    print("\n── Descriptive Statistics ───────────────────────────────────────")
    print_stats_summary(log_returns, vol_summary)

    # ── STEP 3: Portfolio construction ───────────────────────────────────────
    section("STEP 3 — PORTFOLIO CONSTRUCTION")
    portfolio_a_df, portfolio_b_df = compare_portfolios(
        log_returns, portfolio_value=100_000
    )
    metrics_a = compute_portfolio_metrics(portfolio_a_df)
    metrics_b = compute_portfolio_metrics(portfolio_b_df)

    # ── STEP 4: VaR analysis ──────────────────────────────────────────────────
    section("STEP 4 — VALUE AT RISK ANALYSIS")

    print("\n  Running VaR report — Portfolio A (Equal Weight) ...")
    var_results_a = print_var_report(
        portfolio_returns=portfolio_a_df["daily_return"],
        returns_df=log_returns,
        weights=WEIGHTS_A,
        portfolio_value=100_000,
        portfolio_name="Portfolio A — Equal Weight",
    )

    print("\n  Running VaR report — Portfolio B (Tech Overweight) ...")
    var_results_b = print_var_report(
        portfolio_returns=portfolio_b_df["daily_return"],
        returns_df=log_returns,
        weights=WEIGHTS_B,
        portfolio_value=100_000,
        portfolio_name="Portfolio B — Tech Overweight",
    )

    rolling_var_series  = var_results_a["rolling_var"]
    backtest_results_a  = var_results_a["backtest"]
    backtest_results_b  = var_results_b["backtest"]

    # ── STEP 5: Visualisations ────────────────────────────────────────────────
    section("STEP 5 — GENERATING VISUALISATIONS")

    saved_paths = generate_all_charts(
        portfolio_a_df=portfolio_a_df,
        portfolio_b_df=portfolio_b_df,
        returns_df=log_returns,
        correlation_matrix=corr_matrix,
        rolling_var_a=var_results_a["rolling_var"],
        rolling_var_b=var_results_b["rolling_var"],
        component_var_a=var_results_a["component_var"],
        component_var_b=var_results_b["component_var"],
        var_metrics_a=var_results_a,
        var_metrics_b=var_results_b,
        portfolio_value=100_000,
    )

    print(f"\n  All charts saved to outputs/ — {len(saved_paths)} files generated")

    # ── Final file inventory ──────────────────────────────────────────────────
    section("ALL OUTPUTS WRITTEN")
    all_outputs = [
        "data/raw_prices.csv",
        "data/daily_returns.csv",
        "data/daily_returns_combined.csv",
        "data/rolling_volatility.csv",
        "data/correlation_matrix.csv",
        "data/portfolio_a_equal.csv",
        "data/portfolio_b_custom.csv",
        "data/rolling_var.csv",
        *[os.path.relpath(p) for p in saved_paths],
    ]
    for path in all_outputs:
        status = "OK     " if os.path.exists(path) else "MISSING"
        print(f"  [{status}]  {path}")

    print("\n  Project complete through Step 5. Ready for Step 6 (PDF Report).\n")

    return {
        "prices":               prices,
        "log_returns":          log_returns,
        "vol_summary":          vol_summary,
        "corr_matrix":          corr_matrix,
        "jpm_gs_rolling_corr":  jpm_gs_rolling_corr,
        "portfolio_a_df":       portfolio_a_df,
        "portfolio_b_df":       portfolio_b_df,
        "metrics_a":            metrics_a,
        "metrics_b":            metrics_b,
        "var_results_a":        var_results_a,
        "var_results_b":        var_results_b,
        "rolling_var_series":   rolling_var_series,
        "backtest_results_a":   backtest_results_a,
        "backtest_results_b":   backtest_results_b,
        "saved_chart_paths":    saved_paths,
    }


if __name__ == "__main__":
    main()
