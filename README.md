# Portfolio Risk Analyzer

A Python-based quantitative risk management tool for analyzing equity portfolio risk.
Built as a resume project targeting quant/risk roles.

## Project Structure

```
portfolio-risk-analyzer/
├── data/               # Raw and processed price data (CSV)
├── outputs/            # Generated charts and PDF reports
├── src/
│   └── data_fetcher.py # Step 1: data retrieval & cleaning
├── main.py             # Entry point
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Step 1
python main.py
```

## Steps (Roadmap)

| Step | Description                          | Status      |
|------|--------------------------------------|-------------|
| 1    | Project setup & data fetching        | ✅ Complete |
| 2    | Returns, volatility & correlation    | 🔜 Upcoming |
| 3    | Value at Risk (VaR) & CVaR           | 🔜 Upcoming |
| 4    | Portfolio optimisation               | 🔜 Upcoming |
| 5    | PDF risk report generation           | 🔜 Upcoming |

## Portfolio Universe

| Ticker | Description                  |
|--------|------------------------------|
| AAPL   | Apple Inc. (Tech)            |
| MSFT   | Microsoft Corp. (Tech)       |
| JPM    | JPMorgan Chase (Financials)  |
| GS     | Goldman Sachs (Financials)   |
| SPY    | S&P 500 ETF (Benchmark)      |
