"""
visualizer.py
-------------
Publication-quality chart generation for the Portfolio Risk Analyzer.

Produces 8 professional PNG charts saved to outputs/:
  1. cumulative_returns.png     — portfolio value growth over time
  2. return_distribution_A.png  — return histogram + VaR overlays (Port A)
  3. return_distribution_B.png  — return histogram + VaR overlays (Port B)
  4. correlation_heatmap.png    — Pearson correlation matrix heatmap
  5. rolling_volatility.png     — per-ticker 30-day rolling vol with regime shading
  6. rolling_var_A.png          — rolling VaR vs daily P&L (Port A)
  7. rolling_var_B.png          — rolling VaR vs daily P&L (Port B)
  8. component_var_A.png        — component VaR bar chart (Port A)
  9. component_var_B.png        — component VaR bar chart (Port B)
 10. drawdown.png               — underwater drawdown curves, both portfolios

Design conventions
------------------
  Portfolio A : #2196F3  (blue)
  Portfolio B : #FF5722  (orange)
  VaR 95%     : #F44336  (red)
  VaR 99%     : #B71C1C  (dark red)
  CVaR        : #9C27B0  (purple)
  Normal fit  : #9E9E9E  (mid-gray)

All functions save at dpi=150 and never call plt.show().

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must precede pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global style & palette
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

# Tighten up matplotlib defaults for a cleaner look
plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#CCCCCC",
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "grid.color":          "#EEEEEE",
    "grid.linewidth":      0.6,
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.titlesize":      13,
    "axes.titleweight":    "bold",
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#CCCCCC",
    "figure.dpi":          100,
})

# Brand palette
C_A      = "#2196F3"    # Portfolio A  (blue)
C_B      = "#FF5722"    # Portfolio B  (orange)
C_VAR95  = "#F44336"    # VaR 95%      (red)
C_VAR99  = "#B71C1C"    # VaR 99%      (dark red)
C_CVAR   = "#9C27B0"    # CVaR         (purple)
C_NORMAL = "#9E9E9E"    # Normal fit   (gray)
C_BREAK  = "#4CAF50"    # Breakeven    (green)

# Market regime date ranges
COVID_START  = "2020-02-20"
COVID_END    = "2020-04-30"
HIKE_START   = "2022-01-01"
HIKE_END     = "2022-12-31"

OUTPUT_DIR = "outputs"


# ---------------------------------------------------------------------------
# 1. Cumulative returns
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    portfolio_a_df: pd.DataFrame,
    portfolio_b_df: pd.DataFrame,
) -> str:
    """
    Plot cumulative portfolio value over time for both portfolios.

    Features
    --------
    - Dual-line chart starting at $100,000
    - Horizontal breakeven line at $100,000
    - COVID crash and 2022 rate-hike periods shaded
    - Maximum drawdown period annotated for each portfolio

    Parameters
    ----------
    portfolio_a_df : pd.DataFrame
        Time series from portfolio.build_portfolio() for Portfolio A.
        Must contain column: cumulative_value.
    portfolio_b_df : pd.DataFrame
        Same for Portfolio B.

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    dates_a = portfolio_a_df.index
    dates_b = portfolio_b_df.index

    ax.plot(dates_a, portfolio_a_df["cumulative_value"], color=C_A,
            linewidth=2.0, label="Portfolio A — Equal Weight", zorder=4)
    ax.plot(dates_b, portfolio_b_df["cumulative_value"], color=C_B,
            linewidth=2.0, label="Portfolio B — Tech Overweight", zorder=4)

    # Breakeven line
    ax.axhline(100_000, color=C_BREAK, linewidth=1.0, linestyle="--",
               alpha=0.7, label="Breakeven ($100k)", zorder=3)

    # Market regime shading
    _shade_regime(ax, COVID_START, COVID_END, "#FFF9C4", "COVID\nCrash", dates_a)
    _shade_regime(ax, HIKE_START,  HIKE_END,  "#FFE0CC", "2022 Rate\nHike Cycle", dates_a)

    # Max drawdown annotation for each portfolio
    for df, color, label in [
        (portfolio_a_df, C_A, "A"),
        (portfolio_b_df, C_B, "B"),
    ]:
        peak      = df["cumulative_value"].cummax()
        dd_series = (df["cumulative_value"] - peak) / peak
        dd_idx    = dd_series.idxmin()
        dd_val    = dd_series.min()
        ax.annotate(
            f"Max DD\n{dd_val*100:.1f}%",
            xy=(dd_idx, df.loc[dd_idx, "cumulative_value"]),
            xytext=(0, -38),
            textcoords="offset points",
            fontsize=8,
            color=color,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Cumulative Portfolio Value (2020 – Present)", pad=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    path = _save(fig, "cumulative_returns.png")
    return path


# ---------------------------------------------------------------------------
# 2. Return distribution with VaR overlays
# ---------------------------------------------------------------------------

def plot_return_distribution(
    portfolio_returns: pd.Series,
    var_95: float,
    var_99: float,
    cvar_95: float,
    portfolio_name: str = "Portfolio",
    suffix: str = "A",
) -> str:
    """
    Histogram of daily log returns with fitted normal curve and VaR lines.

    Features
    --------
    - 50-bin histogram of observed returns
    - Fitted Gaussian overlay to visualise fat tails
    - Vertical lines for VaR 95%, VaR 99%, CVaR 95%
    - Tail region beyond VaR 95% shaded in light red
    - Stats text box: VaR 95% | VaR 99% | CVaR

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily log returns of the portfolio.
    var_95, var_99, cvar_95 : float
        VaR / CVaR values as decimals (e.g. -0.0124).
    portfolio_name : str
        Display name for chart title and legend.
    suffix : str
        "A" or "B" — used in the output filename.

    Returns
    -------
    str
        Absolute path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    returns_arr = portfolio_returns.dropna().values

    # Histogram
    n, bins, patches = ax.hist(
        returns_arr, bins=50, density=True,
        color=C_A if suffix == "A" else C_B,
        alpha=0.55, edgecolor="white", linewidth=0.4, zorder=3,
    )

    # Shade tail beyond VaR 95%
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < var_95:
            patch.set_facecolor("#FFCDD2")
            patch.set_alpha(0.85)

    # Fitted normal overlay
    mu, sigma = returns_arr.mean(), returns_arr.std()
    x_range = np.linspace(returns_arr.min(), returns_arr.max(), 400)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
            color=C_NORMAL, linewidth=1.8, linestyle="-",
            label="Fitted Normal", zorder=5)

    # VaR / CVaR vertical lines
    ax.axvline(var_95,  color=C_VAR95,  linewidth=1.8, linestyle="--",
               label=f"VaR 95%  {var_95*100:.2f}%",  zorder=6)
    ax.axvline(var_99,  color=C_VAR99,  linewidth=1.8, linestyle="--",
               label=f"VaR 99%  {var_99*100:.2f}%",  zorder=6)
    ax.axvline(cvar_95, color=C_CVAR,   linewidth=1.8, linestyle="--",
               label=f"CVaR 95% {cvar_95*100:.2f}%", zorder=6)

    # Stats text box
    textstr = (
        f"VaR 95%  : {var_95*100:.2f}%\n"
        f"VaR 99%  : {var_99*100:.2f}%\n"
        f"CVaR 95% : {cvar_95*100:.2f}%\n"
        f"Skewness : {stats.skew(returns_arr):.2f}\n"
        f"Kurtosis : {stats.kurtosis(returns_arr):.2f}"
    )
    ax.text(
        0.02, 0.97, textstr,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9),
    )

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title(f"Daily Return Distribution — {portfolio_name}", pad=14)
    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = _save(fig, f"return_distribution_{suffix}.png")
    return path


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> str:
    """
    Seaborn annotated heatmap of the asset return correlation matrix.

    Features
    --------
    - Diverging coolwarm palette: red = strong positive, blue = negative
    - Values annotated with 2 decimal places
    - Tick labels rotated 45 degrees for readability

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Square Pearson correlation matrix (output of risk_engine).

    Returns
    -------
    str
        Absolute path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))

    mask = np.zeros_like(correlation_matrix, dtype=bool)
    # Do NOT mask — show full matrix so all pairwise values are visible

    sns.heatmap(
        correlation_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        linecolor="#EEEEEE",
        annot_kws={"size": 10, "weight": "bold"},
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )

    ax.set_title("Asset Return Correlation Matrix (2020 – Present)", pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    path = _save(fig, "correlation_heatmap.png")
    return path


# ---------------------------------------------------------------------------
# 4. Rolling volatility
# ---------------------------------------------------------------------------

def plot_rolling_volatility(returns_df: pd.DataFrame) -> str:
    """
    Multi-line chart of 30-day rolling annualised volatility per ticker.

    Features
    --------
    - One line per ticker using the tab10 palette for distinctiveness
    - COVID crash and 2022 rate-hike periods shaded
    - Horizontal threshold line at 20% annualised vol
    - Annualisation factor: daily std × √252

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns, one column per ticker.

    Returns
    -------
    str
        Absolute path to saved PNG.
    """
    WINDOW     = 30
    ANN_FACTOR = np.sqrt(252)

    rolling_vol = returns_df.rolling(WINDOW).std() * ANN_FACTOR

    tab10 = plt.cm.get_cmap("tab10")
    colors = [tab10(i) for i in range(len(returns_df.columns))]

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, ticker in enumerate(returns_df.columns):
        ax.plot(rolling_vol.index, rolling_vol[ticker] * 100,
                color=colors[i], linewidth=1.6, label=ticker, zorder=4)

    # Regime shading
    _shade_regime(ax, COVID_START, COVID_END, "#FFFDE7",
                  "COVID\nCrash", returns_df.index)
    _shade_regime(ax, HIKE_START,  HIKE_END,  "#FFF3E0",
                  "2022 Rate\nHike Cycle", returns_df.index)

    # Elevated-vol threshold
    ax.axhline(20, color="#F44336", linewidth=1.2, linestyle="--",
               alpha=0.7, label="20% vol threshold", zorder=3)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_title(f"30-Day Rolling Annualised Volatility by Ticker", pad=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised Volatility (%)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()

    path = _save(fig, "rolling_volatility.png")
    return path


# ---------------------------------------------------------------------------
# 5. Rolling VaR vs daily P&L
# ---------------------------------------------------------------------------

def plot_rolling_var(
    rolling_var_series: pd.Series,
    portfolio_returns: pd.Series,
    portfolio_value: float = 100_000,
    portfolio_name: str = "Portfolio",
    suffix: str = "A",
) -> str:
    """
    Dual-axis chart: rolling VaR (line) vs actual daily dollar P&L (bars).

    Features
    --------
    - Left axis  : rolling 252-day 95% VaR in USD (red line)
    - Right axis : daily P&L bars (green = gain, red = loss)
    - Red × markers where actual loss exceeds VaR (breach days)

    Parameters
    ----------
    rolling_var_series : pd.Series
        Rolling dollar VaR from var_engine.rolling_var().
    portfolio_returns : pd.Series
        Daily log returns of the portfolio.
    portfolio_value : float
        Current notional portfolio value for converting returns to dollars.
    portfolio_name : str
        Display name for chart title.
    suffix : str
        "A" or "B" — used in output filename.

    Returns
    -------
    str
        Absolute path to saved PNG.
    """
    # Align on common dates
    pnl_dollar = (np.exp(portfolio_returns) - 1) * portfolio_value
    combined   = pd.DataFrame({
        "var":  rolling_var_series,
        "pnl":  pnl_dollar,
    }).dropna()

    breach_mask = combined["pnl"] < combined["var"]

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    # ── Right axis: daily P&L bars ───────────────────────────────────────────
    colors = np.where(combined["pnl"] >= 0, "#81C784", "#E57373")
    ax2.bar(combined.index, combined["pnl"], color=colors,
            alpha=0.55, width=1.0, zorder=2, label="Daily P&L")

    # Breach markers
    breaches = combined[breach_mask]
    if not breaches.empty:
        ax2.scatter(breaches.index, breaches["pnl"],
                    marker="x", color=C_VAR95, s=45, linewidths=1.6,
                    zorder=5, label=f"VaR Breach ({len(breaches)} days)")

    # ── Left axis: rolling VaR line ──────────────────────────────────────────
    ax1.plot(combined.index, combined["var"], color=C_VAR95,
             linewidth=1.8, zorder=4, label="95% Rolling VaR")
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Format both axes as currency
    fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    ax1.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_formatter(fmt)

    ax1.set_title(f"Rolling VaR vs Actual Daily P&L — {portfolio_name}", pad=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("95% Rolling VaR (USD)", color=C_VAR95)
    ax2.set_ylabel("Daily P&L (USD)")
    ax1.tick_params(axis="y", labelcolor=C_VAR95)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    fig.tight_layout()
    path = _save(fig, f"rolling_var_{suffix}.png")
    return path


# ---------------------------------------------------------------------------
# 6. Component VaR bar chart
# ---------------------------------------------------------------------------

def plot_component_var(
    component_var_df: pd.DataFrame,
    portfolio_name: str = "Portfolio",
    suffix: str = "A",
) -> str:
    """
    Horizontal bar chart of each ticker's dollar contribution to portfolio VaR.

    Features
    --------
    - Bars coloured on a red gradient: darkest = largest contributor
    - Each bar annotated with % contribution
    - Sorted from largest to smallest contributor

    Parameters
    ----------
    component_var_df : pd.DataFrame
        Output of var_engine.compute_component_var().
        Required columns: ticker, component_var_dollar, pct_contribution.
    portfolio_name : str
        Display name for chart title.
    suffix : str
        "A" or "B" — used in output filename.

    Returns
    -------
    str
        Absolute path to saved PNG.
    """
    # Sort by absolute contribution (largest first) — component VaRs are negative
    df = component_var_df.sort_values("component_var_dollar", ascending=True).copy()

    # Map contribution magnitude to red intensity
    pct_vals   = df["pct_contribution"].abs().values
    norm_vals  = pct_vals / pct_vals.max() if pct_vals.max() > 0 else pct_vals
    bar_colors = [plt.cm.Reds(0.4 + 0.5 * v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        df["ticker"],
        df["component_var_dollar"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.55,
    )

    # Annotate each bar with % contribution
    for bar, pct in zip(bars, df["pct_contribution"]):
        ax.text(
            bar.get_width() - abs(bar.get_width()) * 0.04,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="right",
            fontsize=9, fontweight="bold", color="white",
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.axvline(0, color="#AAAAAA", linewidth=0.8)
    ax.set_title(f"Component VaR Breakdown — {portfolio_name}", pad=14)
    ax.set_xlabel("Dollar Contribution to Portfolio VaR (USD)")
    ax.set_ylabel("Ticker")
    fig.tight_layout()

    path = _save(fig, f"component_var_{suffix}.png")
    return path


# ---------------------------------------------------------------------------
# 7. Underwater drawdown chart
# ---------------------------------------------------------------------------

def plot_drawdown(
    portfolio_a_df: pd.DataFrame,
    portfolio_b_df: pd.DataFrame,
) -> str:
    """
    Underwater (drawdown) chart for both portfolios.

    At each date t:  drawdown_t = (value_t − peak_t) / peak_t

    Features
    --------
    - Area fill under each drawdown curve (Portfolio A blue, B orange)
    - Maximum drawdown point annotated with label and arrow for each portfolio
    - Zero line for reference

    Parameters
    ----------
    portfolio_a_df, portfolio_b_df : pd.DataFrame
        Portfolio time series with column: cumulative_value.

    Returns
    -------
    str
        Absolute path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    for df, color, label in [
        (portfolio_a_df, C_A, "Portfolio A — Equal Weight"),
        (portfolio_b_df, C_B, "Portfolio B — Tech Overweight"),
    ]:
        peak = df["cumulative_value"].cummax()
        dd   = (df["cumulative_value"] - peak) / peak * 100  # in %

        ax.plot(dd.index, dd.values, color=color, linewidth=1.6,
                label=label, zorder=4)
        ax.fill_between(dd.index, dd.values, 0,
                        color=color, alpha=0.18, zorder=2)

        # Annotate maximum drawdown
        dd_idx = dd.idxmin()
        dd_val = dd.min()
        ax.annotate(
            f"Max DD\n{dd_val:.1f}%",
            xy=(dd_idx, dd_val),
            xytext=(0, -34),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

    ax.axhline(0, color="#AAAAAA", linewidth=0.8, zorder=3)

    # Shade market stress regimes
    _shade_regime(ax, COVID_START, COVID_END, "#FFF9C4",
                  "COVID\nCrash", portfolio_a_df.index)
    _shade_regime(ax, HIKE_START,  HIKE_END,  "#FFE0CC",
                  "2022 Rate\nHike Cycle", portfolio_a_df.index)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_title("Portfolio Drawdown — Underwater Chart", pad=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.invert_yaxis()   # convention: drawdown chart reads top = 0, down = worse
    fig.tight_layout()

    path = _save(fig, "drawdown.png")
    return path


# ---------------------------------------------------------------------------
# 8. Master function
# ---------------------------------------------------------------------------

def generate_all_charts(
    portfolio_a_df: pd.DataFrame,
    portfolio_b_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    rolling_var_a: pd.Series,
    rolling_var_b: pd.Series,
    component_var_a: pd.DataFrame,
    component_var_b: pd.DataFrame,
    var_metrics_a: dict,
    var_metrics_b: dict,
    portfolio_value: float = 100_000,
) -> list[str]:
    """
    Generate and save all 10 portfolio risk visualisation charts.

    Calls each individual chart function in sequence, prints a confirmation
    line after each save, and returns the full list of output paths for use
    in the PDF report generator (Step 6).

    Parameters
    ----------
    portfolio_a_df : pd.DataFrame
        Portfolio A time series (equal weight).
    portfolio_b_df : pd.DataFrame
        Portfolio B time series (tech overweight).
    returns_df : pd.DataFrame
        Individual asset daily log returns.
    correlation_matrix : pd.DataFrame
        Pearson correlation matrix of log returns.
    rolling_var_a : pd.Series
        Rolling dollar VaR time series for Portfolio A.
    rolling_var_b : pd.Series
        Rolling dollar VaR time series for Portfolio B.
    component_var_a : pd.DataFrame
        Component VaR breakdown for Portfolio A.
    component_var_b : pd.DataFrame
        Component VaR breakdown for Portfolio B.
    var_metrics_a : dict
        Full VaR report dict for Portfolio A (from var_engine.print_var_report).
    var_metrics_b : dict
        Same for Portfolio B.
    portfolio_value : float, optional
        Notional portfolio value used for dollar P&L conversion.

    Returns
    -------
    list[str]
        Absolute paths of all saved PNG files, in generation order.
    """
    _ensure_output_dir()
    saved_paths: list[str] = []

    # Extract scalar VaR values needed by plot_return_distribution
    def _get_var_vals(vm: dict) -> tuple[float, float, float]:
        """Return (var_95_pct, var_99_pct, cvar_95_pct) from a var_metrics dict."""
        return (
            vm["hist_95"]["var_pct"],
            vm["hist_99"]["var_pct"],
            vm["cvar_95"]["cvar_pct"],
        )

    charts = [
        # (function_call, label)
        (
            lambda: plot_cumulative_returns(portfolio_a_df, portfolio_b_df),
            "cumulative_returns.png",
        ),
        (
            lambda: plot_return_distribution(
                portfolio_a_df["daily_return"],
                *_get_var_vals(var_metrics_a),
                portfolio_name="Portfolio A — Equal Weight",
                suffix="A",
            ),
            "return_distribution_A.png",
        ),
        (
            lambda: plot_return_distribution(
                portfolio_b_df["daily_return"],
                *_get_var_vals(var_metrics_b),
                portfolio_name="Portfolio B — Tech Overweight",
                suffix="B",
            ),
            "return_distribution_B.png",
        ),
        (
            lambda: plot_correlation_heatmap(correlation_matrix),
            "correlation_heatmap.png",
        ),
        (
            lambda: plot_rolling_volatility(returns_df),
            "rolling_volatility.png",
        ),
        (
            lambda: plot_rolling_var(
                rolling_var_a,
                portfolio_a_df["daily_return"],
                portfolio_value=portfolio_value,
                portfolio_name="Portfolio A — Equal Weight",
                suffix="A",
            ),
            "rolling_var_A.png",
        ),
        (
            lambda: plot_rolling_var(
                rolling_var_b,
                portfolio_b_df["daily_return"],
                portfolio_value=portfolio_value,
                portfolio_name="Portfolio B — Tech Overweight",
                suffix="B",
            ),
            "rolling_var_B.png",
        ),
        (
            lambda: plot_component_var(
                component_var_a,
                portfolio_name="Portfolio A — Equal Weight",
                suffix="A",
            ),
            "component_var_A.png",
        ),
        (
            lambda: plot_component_var(
                component_var_b,
                portfolio_name="Portfolio B — Tech Overweight",
                suffix="B",
            ),
            "component_var_B.png",
        ),
        (
            lambda: plot_drawdown(portfolio_a_df, portfolio_b_df),
            "drawdown.png",
        ),
    ]

    for fn, filename in charts:
        path = fn()
        saved_paths.append(path)
        print(f"  \u2713  Saved {path}")

    return saved_paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> str:
    """
    Save a matplotlib figure to the outputs/ directory at dpi=150.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    filename : str
        Filename (not full path).  Will be placed in outputs/.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return os.path.abspath(path)


def _ensure_output_dir() -> None:
    """Create the outputs/ directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _shade_regime(
    ax: plt.Axes,
    start_str: str,
    end_str: str,
    color: str,
    label: str,
    index: pd.Index,
) -> None:
    """
    Shade a date range on a chart axis with a label.

    Parameters
    ----------
    ax : plt.Axes
        Axes object to draw on.
    start_str, end_str : str
        ISO date strings "YYYY-MM-DD" for the regime boundaries.
    color : str
        Fill colour (hex or named).
    label : str
        Text label placed at the top of the shaded region.
    index : pd.Index
        The chart's date index — used to check if the range overlaps.
    """
    try:
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str)

        # Only shade if dates fall within the chart's range
        if start > index.max() or end < index.min():
            return

        start = max(start, index.min())
        end   = min(end,   index.max())

        ax.axvspan(start, end, color=color, alpha=0.60, zorder=1)
        ax.text(
            start + (end - start) / 2,
            ax.get_ylim()[1],
            label,
            fontsize=7.5,
            ha="center",
            va="top",
            color="#555555",
            style="italic",
        )
    except Exception:
        pass   # silently skip if date formatting fails
