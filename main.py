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
  Step 6 : PDF risk report                   (src/report_generator.py)

Run:
    python main.py

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import sys
import os
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_fetcher    import fetch_stock_data
from risk_engine     import (compute_returns, compute_volatility,
                              compute_correlation_matrix,
                              compute_rolling_correlation, print_stats_summary)
from portfolio       import compare_portfolios, compute_portfolio_metrics
from var_engine      import print_var_report
from visualizer      import generate_all_charts
from report_generator import generate_pdf_report

# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------

TICKERS: list[str] = ["AAPL", "MSFT", "JPM", "GS", "SPY"]
START_DATE: str    = "2020-01-01"
END_DATE: str      = date.today().isoformat()

WEIGHTS_A: dict[str, float] = {t: 0.20 for t in TICKERS}
WEIGHTS_B: dict[str, float] = {
    "AAPL": 0.35, "MSFT": 0.30, "JPM": 0.15, "GS": 0.10, "SPY": 0.10
}


def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    """Run all six steps and return every computed artefact."""

    # ── STEP 1 ───────────────────────────────────────────────────────────────
    section("STEP 1 — DATA FETCHING")
    prices = fetch_stock_data(
        tickers=TICKERS, start_date=START_DATE,
        end_date=END_DATE, save_path="data/raw_prices.csv",
    )
    print(f"\n  Shape : {prices.shape[0]} rows x {prices.shape[1]} columns")
    print(f"  Dates : {prices.index.min().date()} -> {prices.index.max().date()}")

    # ── STEP 2 ───────────────────────────────────────────────────────────────
    section("STEP 2 — RETURNS & VOLATILITY")
    log_returns = compute_returns(prices)
    vol_summary = compute_volatility(log_returns, window_short=30, window_long=60)
    corr_matrix = compute_correlation_matrix(log_returns)
    jpm_gs_corr = compute_rolling_correlation(log_returns, "JPM", "GS", window=60)
    print_stats_summary(log_returns, vol_summary)

    # ── STEP 3 ───────────────────────────────────────────────────────────────
    section("STEP 3 — PORTFOLIO CONSTRUCTION")
    portfolio_a_df, portfolio_b_df = compare_portfolios(log_returns, 100_000)
    metrics_a = compute_portfolio_metrics(portfolio_a_df)
    metrics_b = compute_portfolio_metrics(portfolio_b_df)

    # ── STEP 4 ───────────────────────────────────────────────────────────────
    section("STEP 4 — VALUE AT RISK ANALYSIS")
    var_results_a = print_var_report(
        portfolio_returns=portfolio_a_df["daily_return"],
        returns_df=log_returns, weights=WEIGHTS_A,
        portfolio_value=100_000,
        portfolio_name="Portfolio A — Equal Weight",
    )
    var_results_b = print_var_report(
        portfolio_returns=portfolio_b_df["daily_return"],
        returns_df=log_returns, weights=WEIGHTS_B,
        portfolio_value=100_000,
        portfolio_name="Portfolio B — Tech Overweight",
    )
    backtest_results_a = var_results_a["backtest"]
    backtest_results_b = var_results_b["backtest"]
    rolling_var_series = var_results_a["rolling_var"]

    # ── STEP 5 ───────────────────────────────────────────────────────────────
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

    # ── STEP 6 ───────────────────────────────────────────────────────────────
    section("STEP 6 — GENERATING RISK REPORT")
    report_path = generate_pdf_report(
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        var_metrics_a=var_results_a,
        var_metrics_b=var_results_b,
        backtest_a=backtest_results_a,
        backtest_b=backtest_results_b,
        returns_df=log_returns,
        correlation_matrix=corr_matrix,
        component_var_a=var_results_a["component_var"],
        component_var_b=var_results_b["component_var"],
        weights_a=WEIGHTS_A,
        weights_b=WEIGHTS_B,
        output_path="outputs/risk_report.pdf",
        chart_dir="outputs",
    )
    print(f"\n  Full analysis complete. Report saved to {report_path}")

    # ── Final inventory ───────────────────────────────────────────────────────
    section("ALL OUTPUTS")
    all_files = [
        "data/raw_prices.csv", "data/daily_returns.csv",
        "data/daily_returns_combined.csv", "data/rolling_volatility.csv",
        "data/correlation_matrix.csv", "data/portfolio_a_equal.csv",
        "data/portfolio_b_custom.csv", "data/rolling_var.csv",
        *[os.path.relpath(p) for p in saved_paths],
        "outputs/risk_report.pdf",
    ]
    for f in all_files:
        status = "OK     " if os.path.exists(f) else "MISSING"
        print(f"  [{status}]  {f}")

    print("\n  Project complete — all 6 steps finished.\n")

    return {
        "prices": prices, "log_returns": log_returns,
        "vol_summary": vol_summary, "corr_matrix": corr_matrix,
        "portfolio_a_df": portfolio_a_df, "portfolio_b_df": portfolio_b_df,
        "metrics_a": metrics_a, "metrics_b": metrics_b,
        "var_results_a": var_results_a, "var_results_b": var_results_b,
        "rolling_var_series": rolling_var_series,
        "backtest_results_a": backtest_results_a,
        "backtest_results_b": backtest_results_b,
        "saved_chart_paths": saved_paths,
        "report_path": report_path,
    }


if __name__ == "__main__":
    main()
