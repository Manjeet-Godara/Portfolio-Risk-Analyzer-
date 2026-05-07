"""
var_engine.py
-------------
Value at Risk (VaR) engine for the Portfolio Risk Analyzer.

Implements five standard risk measurement approaches used in practice
at investment banks and asset managers:

  1. Historical Simulation VaR   — empirical, no distributional assumption
  2. Parametric (Normal) VaR     — variance-covariance, Gaussian assumption
  3. Conditional VaR / CVaR      — Expected Shortfall, tail-risk metric
  4. Rolling VaR                 — time-series of VaR to show regime changes
  5. VaR Backtest                — Kupiec proportion-of-failures test
  6. Component VaR               — per-asset risk contribution decomposition

All dollar figures assume the portfolio_value is fully invested (no cash).
All VaR figures are expressed as *losses* — i.e. negative numbers.

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
_CONF_LEVELS: tuple[float, ...] = (0.95, 0.99)   # reported by default


# ---------------------------------------------------------------------------
# 1. Historical Simulation VaR
# ---------------------------------------------------------------------------

def historical_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 100_000,
) -> dict:
    """
    Compute 1-day VaR using Historical Simulation.

    Sort all observed daily returns and take the empirical percentile
    corresponding to (1 - confidence_level).  No distributional assumption
    is made — the full empirical return distribution drives the estimate.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log returns of the portfolio (single Series).
    confidence_level : float
        Confidence level, e.g. 0.95 for 95% VaR.
    portfolio_value : float
        Current notional portfolio value in USD.

    Returns
    -------
    dict
        {
          "method"            : "Historical Simulation",
          "confidence_level"  : 0.95,
          "var_pct"           : -0.0124,   # as decimal
          "var_dollar"        : -1240.0,
          "n_observations"    : 1258,
        }
    """
    # The VaR percentile sits at the (1 - confidence_level) quantile
    # e.g. for 95% VaR → 5th percentile of the return distribution
    var_pct = float(np.percentile(portfolio_returns, (1 - confidence_level) * 100))
    var_dollar = var_pct * portfolio_value

    return {
        "method":           "Historical Simulation",
        "confidence_level": confidence_level,
        "var_pct":          var_pct,
        "var_dollar":       var_dollar,
        "n_observations":   len(portfolio_returns),
    }


# ---------------------------------------------------------------------------
# 2. Parametric (Variance-Covariance) VaR
# ---------------------------------------------------------------------------

def parametric_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 100_000,
) -> dict:
    """
    Compute 1-day VaR using the Parametric (Variance-Covariance) method.

    Assumes returns are normally distributed.  Uses the mean and standard
    deviation of observed returns to fit a Gaussian, then applies
    scipy.stats.norm.ppf() to find the loss at the given confidence level.

    Note: This method systematically underestimates tail risk when the
    actual return distribution is leptokurtic (fat-tailed), which is
    almost always the case for equity returns.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log returns of the portfolio.
    confidence_level : float
        Confidence level, e.g. 0.95 for 95% VaR.
    portfolio_value : float
        Current notional portfolio value in USD.

    Returns
    -------
    dict
        Same structure as historical_var(), with method = "Parametric (Normal)".
    """
    mu    = portfolio_returns.mean()
    sigma = portfolio_returns.std()

    # z-score at the (1 - confidence_level) tail
    # ppf(0.05) ≈ -1.645 for 95%; ppf(0.01) ≈ -2.326 for 99%
    z_score = stats.norm.ppf(1 - confidence_level)

    # VaR = mu + z * sigma  (z is negative, so VaR is a loss)
    var_pct    = mu + z_score * sigma
    var_dollar = var_pct * portfolio_value

    return {
        "method":           "Parametric (Normal)",
        "confidence_level": confidence_level,
        "var_pct":          var_pct,
        "var_dollar":       var_dollar,
        "mean_daily_return": mu,
        "daily_volatility":  sigma,
        "z_score":           z_score,
        "n_observations":    len(portfolio_returns),
    }


# ---------------------------------------------------------------------------
# 3. Conditional VaR (Expected Shortfall / CVaR)
# ---------------------------------------------------------------------------

def compute_cvar(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 100_000,
) -> dict:
    """
    Compute Conditional VaR (CVaR), also known as Expected Shortfall (ES).

    CVaR is the *expected* (average) loss on days when the loss exceeds
    the VaR threshold.  It fully characterises the tail rather than just
    marking its boundary, making it a coherent risk measure in the
    mathematical sense (VaR is not coherent for non-normal distributions).

    CVaR became the primary regulatory metric under Basel III / FRTB,
    replacing VaR for internal model capital calculations.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log returns of the portfolio.
    confidence_level : float
        Confidence level, e.g. 0.95.  CVaR averages all returns that are
        *worse* than the corresponding VaR.
    portfolio_value : float
        Current notional portfolio value in USD.

    Returns
    -------
    dict
        {
          "confidence_level" : 0.95,
          "var_pct"          : -0.0124,
          "var_dollar"       : -1240.0,
          "cvar_pct"         : -0.0189,
          "cvar_dollar"      : -1890.0,
          "n_tail_obs"       : 63,    # number of days in the tail
        }
    """
    var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    # Isolate returns that fall below (are worse than) the VaR cutoff
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]

    cvar_pct    = float(tail_returns.mean())
    cvar_dollar = cvar_pct * portfolio_value

    return {
        "confidence_level": confidence_level,
        "var_pct":          float(var_threshold),
        "var_dollar":       float(var_threshold) * portfolio_value,
        "cvar_pct":         cvar_pct,
        "cvar_dollar":      cvar_dollar,
        "n_tail_obs":       len(tail_returns),
        "n_observations":   len(portfolio_returns),
    }


# ---------------------------------------------------------------------------
# 4. Rolling Historical VaR
# ---------------------------------------------------------------------------

def rolling_var(
    portfolio_returns: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    portfolio_value: float = 100_000,
) -> pd.Series:
    """
    Compute a rolling 1-day Historical VaR as a time series.

    At each date t, VaR is estimated using the past *window* trading days.
    This produces a time series that captures regime changes:
      - Spikes during COVID crash (Feb-Mar 2020)
      - Elevated throughout 2022 rate-hike cycle
      - Lower during low-volatility bull markets

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log returns of the portfolio.
    window : int, optional
        Lookback window in trading days.  Default: 252 (1 year).
    confidence_level : float, optional
        Confidence level.  Default: 0.95.
    portfolio_value : float, optional
        Current notional portfolio value in USD.

    Returns
    -------
    pd.Series
        Rolling VaR in dollar terms (negative values).
        Saved to data/rolling_var.csv with columns:
          date | rolling_var_pct | rolling_var_dollar
    """
    percentile_cutoff = (1 - confidence_level) * 100

    # Rolling quantile at the VaR percentile
    rolling_var_pct = portfolio_returns.rolling(window=window).quantile(
        percentile_cutoff / 100
    )

    rolling_var_dollar = rolling_var_pct * portfolio_value

    # ── Persist ─────────────────────────────────────────────────────────────
    rolling_df = pd.DataFrame(
        {
            "rolling_var_pct":    rolling_var_pct,
            "rolling_var_dollar": rolling_var_dollar,
        },
        index=portfolio_returns.index,
    ).dropna()

    _ensure_dir("data/rolling_var.csv")
    rolling_df.to_csv("data/rolling_var.csv")
    print(f"  Saved rolling VaR → {os.path.abspath('data/rolling_var.csv')}  "
          f"({len(rolling_df)} rows after {window}-day warm-up)")

    return rolling_var_dollar


# ---------------------------------------------------------------------------
# 5. VaR Backtest (Kupiec proportion-of-failures)
# ---------------------------------------------------------------------------

def var_backtest(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 100_000,
) -> dict:
    """
    Backtest the Historical Simulation VaR model using the
    proportion-of-failures (Kupiec) approach.

    At a 95% confidence level we *expect* actual losses to exceed the VaR
    estimate on approximately 5% of trading days.  If the observed breach
    rate is significantly higher, the model is underestimating tail risk.

    Basel Traffic Light framework (for 250-day backtest):
      Green  (0–4 breaches)  : model acceptable
      Yellow (5–9 breaches)  : model under scrutiny
      Red    (10+ breaches)  : model rejected; capital add-on applied

    This implementation uses the full return history rather than a fixed
    250-day window, and flags the model as:
      - "OK"                   if breach rate is within [4%, 6%]
      - "UNDERESTIMATES RISK"  if breach rate > 6%
      - "OVERESTIMATES RISK"   if breach rate < 4%

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log returns of the portfolio.
    confidence_level : float
        Confidence level used to compute the VaR threshold.
    portfolio_value : float
        Current notional portfolio value in USD.

    Returns
    -------
    dict
        {
          "n_days"             : 1258,
          "expected_breaches"  : 62.9,
          "actual_breaches"    : 67,
          "breach_rate_pct"    : 5.32,
          "expected_rate_pct"  : 5.0,
          "var_pct"            : -0.0124,
          "var_dollar"         : -1240.0,
          "model_status"       : "OK",
          "confidence_level"   : 0.95,
        }
    """
    var_result   = historical_var(portfolio_returns, confidence_level, portfolio_value)
    var_threshold = var_result["var_pct"]

    n_days            = len(portfolio_returns)
    expected_rate     = 1 - confidence_level              # e.g. 0.05
    expected_breaches = n_days * expected_rate

    # A breach occurs when the realised return is worse than the VaR cutoff
    actual_breaches = int((portfolio_returns < var_threshold).sum())
    breach_rate_pct = (actual_breaches / n_days) * 100

    # Determine model status with ±1% tolerance band around expected rate
    expected_rate_pct = expected_rate * 100
    if breach_rate_pct > expected_rate_pct + 1.0:
        model_status = "UNDERESTIMATES RISK"
    elif breach_rate_pct < expected_rate_pct - 1.0:
        model_status = "OVERESTIMATES RISK"
    else:
        model_status = "OK"

    return {
        "n_days":              n_days,
        "expected_breaches":   round(expected_breaches, 1),
        "actual_breaches":     actual_breaches,
        "breach_rate_pct":     round(breach_rate_pct, 2),
        "expected_rate_pct":   expected_rate_pct,
        "var_pct":             var_threshold,
        "var_dollar":          var_threshold * portfolio_value,
        "model_status":        model_status,
        "confidence_level":    confidence_level,
    }


# ---------------------------------------------------------------------------
# 6. Component VaR
# ---------------------------------------------------------------------------

def compute_component_var(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    confidence_level: float = 0.95,
    portfolio_value: float = 100_000,
) -> pd.DataFrame:
    """
    Decompose total portfolio VaR into per-asset Component VaR contributions.

    Component VaR uses the covariance between each asset and the portfolio
    to attribute the total portfolio variance — and therefore VaR — back to
    individual positions.  This is the standard attribution methodology
    used in risk systems at investment banks.

    Methodology (linear, Gaussian approximation):
        Portfolio variance  = w^T * Σ * w
        Marginal VaR_i      = z * (Σ * w)_i / portfolio_vol  [per unit of w_i]
        Component VaR_i     = w_i * Marginal VaR_i * portfolio_value

    This approach gives component VaRs that *sum exactly* to total
    parametric portfolio VaR — a property called "VaR additivity."

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns, one column per ticker.
    weights : dict[str, float]
        Portfolio weight per ticker.  Must sum to 1.0.
    confidence_level : float
        Confidence level.  Default: 0.95.
    portfolio_value : float
        Current notional portfolio value in USD.

    Returns
    -------
    pd.DataFrame
        Columns:
          ticker | weight | individual_var_pct | individual_var_dollar |
          component_var_dollar | pct_contribution
        Rows ordered by absolute component VaR (largest contributor first).
    """
    tickers = list(weights.keys())
    w       = np.array([weights[t] for t in tickers], dtype=float)

    # ── Covariance matrix of returns ────────────────────────────────────────
    cov_matrix = returns_df[tickers].cov().values   # shape (N, N)

    # ── Portfolio volatility (daily) ─────────────────────────────────────────
    port_variance = float(w @ cov_matrix @ w)
    port_vol      = np.sqrt(port_variance)

    # ── Z-score at the confidence level ──────────────────────────────────────
    z_score = abs(stats.norm.ppf(1 - confidence_level))

    # ── Individual (stand-alone) VaR per ticker ──────────────────────────────
    individual_vols    = returns_df[tickers].std().values   # shape (N,)
    individual_var_pct = -z_score * individual_vols         # negative (loss)

    # ── Component VaR via marginal contribution ──────────────────────────────
    # Covariance of each asset with the portfolio = (Σ * w)
    cov_with_port = cov_matrix @ w              # shape (N,)

    # Marginal VaR (per unit weight): z * cov(asset, port) / port_vol
    marginal_var_pct = -z_score * cov_with_port / port_vol  # shape (N,)

    # Component VaR (dollar): weight_i * marginal_var_i * portfolio_value
    component_var_dollar = w * marginal_var_pct * portfolio_value   # shape (N,)

    # Total portfolio parametric VaR (for normalising contribution %)
    total_portfolio_var_dollar = -z_score * port_vol * portfolio_value

    pct_contribution = (component_var_dollar / total_portfolio_var_dollar) * 100

    # ── Assemble output DataFrame ────────────────────────────────────────────
    component_df = pd.DataFrame(
        {
            "ticker":                 tickers,
            "weight":                 w,
            "individual_var_pct":     individual_var_pct,
            "individual_var_dollar":  individual_var_pct * portfolio_value,
            "component_var_dollar":   component_var_dollar,
            "pct_contribution":       pct_contribution,
        }
    ).sort_values("component_var_dollar", ascending=True)  # most negative first

    component_df = component_df.reset_index(drop=True)
    return component_df


# ---------------------------------------------------------------------------
# 7. Master VaR report
# ---------------------------------------------------------------------------

def print_var_report(
    portfolio_returns: pd.Series,
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    portfolio_value: float = 100_000,
    portfolio_name: str = "Portfolio",
) -> dict:
    """
    Generate and print a complete VaR report for a single portfolio.

    Calls all five VaR functions in sequence and formats their output
    into the standard risk report layout used in this project.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log returns of the portfolio (weighted sum of assets).
    returns_df : pd.DataFrame
        Individual asset daily log returns (for component VaR calculation).
    weights : dict[str, float]
        Portfolio weight per ticker.
    portfolio_value : float, optional
        Current notional portfolio value in USD.  Default: 100,000.
    portfolio_name : str, optional
        Display name used in the report header.

    Returns
    -------
    dict
        {
          "hist_95"        : dict   — historical VaR at 95%
          "hist_99"        : dict   — historical VaR at 99%
          "param_95"       : dict   — parametric VaR at 95%
          "param_99"       : dict   — parametric VaR at 99%
          "cvar_95"        : dict   — CVaR at 95%
          "cvar_99"        : dict   — CVaR at 99%
          "rolling_var"    : Series — rolling VaR dollar time series
          "backtest"       : dict   — backtest results
          "component_var"  : DataFrame — component VaR breakdown
        }
    """

    bar_wide = "=" * 50
    bar_thin = "-" * 50

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{bar_wide}")
    print(f"  VALUE AT RISK REPORT — {portfolio_name}")
    print(bar_wide)

    # ── 1 & 2. Historical and Parametric VaR at 95% and 99% ─────────────────
    hist_95  = historical_var(portfolio_returns, 0.95, portfolio_value)
    hist_99  = historical_var(portfolio_returns, 0.99, portfolio_value)
    param_95 = parametric_var(portfolio_returns, 0.95, portfolio_value)
    param_99 = parametric_var(portfolio_returns, 0.99, portfolio_value)

    col = 22
    print(f"\n  {'Method':<{col}} {'95% VaR':>10}   {'99% VaR':>10}")
    print(f"  {'-'*col} {'----------':>10}   {'----------':>10}")
    print(
        f"  {'Historical Sim':<{col}} "
        f"{_fmt_var(hist_95['var_dollar'], hist_95['var_pct']):>22}   "
        f"{_fmt_var(hist_99['var_dollar'], hist_99['var_pct']):>22}"
    )
    print(
        f"  {'Parametric (Normal)':<{col}} "
        f"{_fmt_var(param_95['var_dollar'], param_95['var_pct']):>22}   "
        f"{_fmt_var(param_99['var_dollar'], param_99['var_pct']):>22}"
    )

    # Explicit confirmation lines used in Step 5 PDF
    print(f"\n  {'Confidence':>14}  {'Method':>22}  {'VaR $':>10}  {'VaR %':>8}")
    print(f"  {'-'*58}")
    for result, label in [
        (hist_95,  "Historical Sim    95%"),
        (hist_99,  "Historical Sim    99%"),
        (param_95, "Parametric        95%"),
        (param_99, "Parametric        99%"),
    ]:
        print(
            f"  {label:>36}  "
            f"${result['var_dollar']:>9,.0f}  "
            f"{result['var_pct']*100:>7.2f}%"
        )

    # ── 3. CVaR ──────────────────────────────────────────────────────────────
    cvar_95 = compute_cvar(portfolio_returns, 0.95, portfolio_value)
    cvar_99 = compute_cvar(portfolio_returns, 0.99, portfolio_value)

    print(f"\n{bar_thin}")
    print("  CONDITIONAL VaR  (Expected Shortfall)")
    print(bar_thin)
    print(
        f"  95% CVaR : ${cvar_95['cvar_dollar']:>9,.0f}  "
        f"({cvar_95['cvar_pct']*100:.2f}%)  "
        f"[avg of {cvar_95['n_tail_obs']} tail days]"
    )
    print(
        f"  99% CVaR : ${cvar_99['cvar_dollar']:>9,.0f}  "
        f"({cvar_99['cvar_pct']*100:.2f}%)  "
        f"[avg of {cvar_99['n_tail_obs']} tail days]"
    )

    # ── 4. Rolling VaR ───────────────────────────────────────────────────────
    print(f"\n{bar_thin}")
    print("  ROLLING VaR  (252-day window)")
    print(bar_thin)
    rolling_var_series = rolling_var(
        portfolio_returns, window=252, confidence_level=0.95,
        portfolio_value=portfolio_value
    )
    valid = rolling_var_series.dropna()
    print(f"  Min (most dangerous period) : ${valid.min():>9,.0f}")
    print(f"  Max (least dangerous period): ${valid.max():>9,.0f}")
    print(f"  Latest rolling 95% VaR      : ${valid.iloc[-1]:>9,.0f}")

    # ── 5. Backtest ───────────────────────────────────────────────────────────
    bt = var_backtest(portfolio_returns, 0.95, portfolio_value)
    verdict = "✓  Model OK" if bt["model_status"] == "OK" else f"✗  {bt['model_status']}"

    print(f"\n{bar_thin}")
    print("  VaR BACKTEST  (95% confidence)")
    print(bar_thin)
    print(f"  Trading days analysed  : {bt['n_days']}")
    print(
        f"  Expected breaches (5%) : {bt['expected_breaches']:.1f}   "
        f"Actual : {bt['actual_breaches']}"
    )
    print(f"  Breach rate            : {bt['breach_rate_pct']:.2f}%  "
          f"(expected {bt['expected_rate_pct']:.1f}%)")
    print(f"  Model Status           : {verdict}")

    # ── 6. Component VaR ─────────────────────────────────────────────────────
    comp_df = compute_component_var(returns_df, weights, 0.95, portfolio_value)

    print(f"\n{bar_thin}")
    print("  COMPONENT VaR BREAKDOWN  (95%)")
    print(bar_thin)
    _print_component_table(comp_df)

    print(f"\n{bar_wide}\n")

    return {
        "hist_95":       hist_95,
        "hist_99":       hist_99,
        "param_95":      param_95,
        "param_99":      param_99,
        "cvar_95":       cvar_95,
        "cvar_99":       cvar_99,
        "rolling_var":   rolling_var_series,
        "backtest":      bt,
        "component_var": comp_df,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_var(dollar: float, pct: float) -> str:
    """Format a VaR value as '$-1,234 (-1.23%)' for table display."""
    return f"${dollar:>9,.0f} ({pct*100:.2f}%)"


def _print_component_table(comp_df: pd.DataFrame) -> None:
    """
    Print a clean component VaR table to the console.

    Parameters
    ----------
    comp_df : pd.DataFrame
        Output of compute_component_var().
    """
    c = [8, 8, 18, 18, 14]
    header = (
        f"  {'Ticker':<{c[0]}}"
        f"{'Weight':>{c[1]}}"
        f"{'Indiv. VaR $':>{c[2]}}"
        f"{'Component VaR $':>{c[3]}}"
        f"{'% Contribution':>{c[4]}}"
    )
    print(header)
    print("  " + "-" * sum(c))

    total_comp = 0.0
    for _, row in comp_df.iterrows():
        total_comp += row["component_var_dollar"]
        print(
            f"  {row['ticker']:<{c[0]}}"
            f"{row['weight']*100:>{c[1]}.0f}%"
            f"${row['individual_var_dollar']:>{c[2]-1},.0f}"
            f"${row['component_var_dollar']:>{c[3]-1},.0f}"
            f"{row['pct_contribution']:>{c[4]-1}.1f}%"
        )

    print("  " + "-" * sum(c))
    print(
        f"  {'TOTAL':<{c[0]+c[1]}}"
        f"{'':>{c[2]}}"
        f"${total_comp:>{c[3]-1},.0f}"
        f"{'100.0':>{c[4]-1}}%"
    )


def _ensure_dir(filepath: str) -> None:
    """Create all parent directories for *filepath* if they do not exist."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
