"""
main.py
-------
Entry point for the Portfolio Risk Analyzer.

Step 1: Project Setup & Data Fetching
    - Defines the portfolio universe and date range.
    - Fetches adjusted closing prices via data_fetcher.fetch_stock_data().
    - Runs basic diagnostics on the returned DataFrame.

Run:
    python main.py

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import sys
import os
from datetime import date

# ---------------------------------------------------------------------------
# Make sure Python can find modules inside src/ regardless of where the
# script is launched from.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_fetcher import fetch_stock_data  # noqa: E402  (after sys.path insert)


# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------

TICKERS: list[str] = ["AAPL", "MSFT", "JPM", "GS", "SPY"]
"""
Tickers chosen to represent:
  AAPL, MSFT — large-cap tech (growth)
  JPM, GS    — major US banks (financials / risk exposure)
  SPY        — S&P 500 ETF (benchmark)
"""

START_DATE: str = "2020-01-01"
END_DATE:   str = date.today().isoformat()   # always uses today dynamically


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrates Step 1 of the Portfolio Risk Analyzer:
      1. Fetch and clean historical price data.
      2. Display head / tail rows for a quick sanity-check.
      3. Print DataFrame-level diagnostics (shape, date range, null counts).
    """

    # ── 1. Fetch data ───────────────────────────────────────────────────────
    prices = fetch_stock_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        save_path="data/raw_prices.csv",
    )

    # ── 2. Head / tail preview ───────────────────────────────────────────────
    print("\n── First 5 rows ─────────────────────────────────────────────────")
    print(prices.head().to_string())

    print("\n── Last 5 rows ──────────────────────────────────────────────────")
    print(prices.tail().to_string())

    # ── 3. DataFrame diagnostics ─────────────────────────────────────────────
    print("\n── DataFrame Diagnostics ────────────────────────────────────────")

    shape = prices.shape
    print(f"  Shape          : {shape[0]} rows × {shape[1]} columns")
    print(f"  Min date       : {prices.index.min().date()}")
    print(f"  Max date       : {prices.index.max().date()}")

    null_counts = prices.isna().sum()
    print("\n  Null counts per ticker:")
    for ticker, count in null_counts.items():
        status = "✓" if count == 0 else "✗"
        print(f"    {status} {ticker:>6s} : {count}")

    total_nulls = null_counts.sum()
    print(f"\n  Total nulls across all columns : {total_nulls}")

    if total_nulls == 0:
        print("\n  ✓  Data is clean and ready for Step 2 analysis.\n")
    else:
        print("\n  ✗  Warning: unexpected nulls remain — review data_fetcher.\n")


# ---------------------------------------------------------------------------
# Guard: only run when executed directly (not when imported)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
