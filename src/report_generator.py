"""
report_generator.py
-------------------
Professional PDF risk report generator for the Portfolio Risk Analyzer.

Produces a 7-page institutional-quality PDF saved to outputs/risk_report.pdf,
structured identically to a real trading-desk risk summary:

  Page 1 — Cover Page
  Page 2 — Executive Summary
  Page 3 — Portfolio Performance Analysis
  Page 4 — Value at Risk Analysis
  Page 5 — Risk Decomposition
  Page 6 — Volatility Analysis
  Page 7 — Methodology & Disclaimers

Uses reportlab (Platypus high-level layout engine) with Helvetica throughout.

Colour palette
--------------
  Navy  (26,  42,  74)  — headers, table headers
  Gray  (50,  50,  50)  — body text
  Green (34, 139,  34)  — positive numbers
  Red  (178,  34,  34)  — negative numbers
  LightGray (245,245,245) — alternating table rows
  Rule  (200,200,200)   — horizontal dividers

Author : <your name>
Project: Portfolio Risk Analyzer  (Quant / Risk Management)
"""

import os
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ---------------------------------------------------------------------------
# Colour constants  (r, g, b  as 0-1 floats for reportlab)
# ---------------------------------------------------------------------------

def _rgb(r: int, g: int, b: int) -> colors.Color:
    return colors.Color(r / 255, g / 255, b / 255)


NAVY        = _rgb(26,  42,  74)
DARK_GRAY   = _rgb(50,  50,  50)
MID_GRAY    = _rgb(120, 120, 120)
LIGHT_GRAY  = _rgb(245, 245, 245)
RULE_GRAY   = _rgb(200, 200, 200)
GREEN       = _rgb(34,  139,  34)
RED         = _rgb(178,  34,  34)
WHITE       = colors.white
ACCENT_BLUE = _rgb(33, 150, 243)

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4          # 595 × 842 pts
MARGIN_L = 2.0 * cm
MARGIN_R = 2.0 * cm
MARGIN_T = 2.5 * cm
MARGIN_B = 2.0 * cm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R   # ≈ 491 pts

TOTAL_PAGES = 7

# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------

_BASE = getSampleStyleSheet()

def _style(name: str, **kwargs) -> ParagraphStyle:
    """Create a named ParagraphStyle with sensible defaults."""
    defaults = dict(
        fontName="Helvetica",
        fontSize=10,
        textColor=DARK_GRAY,
        leading=14,
        spaceAfter=4,
    )
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)


S_COVER_TITLE = _style("CoverTitle",
    fontName="Helvetica-Bold", fontSize=26,
    textColor=NAVY, alignment=TA_CENTER, leading=32, spaceAfter=10)

S_COVER_SUB = _style("CoverSub",
    fontName="Helvetica", fontSize=13,
    textColor=MID_GRAY, alignment=TA_CENTER, leading=18, spaceAfter=6)

S_COVER_META = _style("CoverMeta",
    fontName="Helvetica", fontSize=10,
    textColor=DARK_GRAY, alignment=TA_CENTER, leading=15)

S_SECTION = _style("Section",
    fontName="Helvetica-Bold", fontSize=14,
    textColor=NAVY, leading=20, spaceBefore=10, spaceAfter=6)

S_SUBSECTION = _style("SubSection",
    fontName="Helvetica-Bold", fontSize=11,
    textColor=NAVY, leading=16, spaceBefore=6, spaceAfter=4)

S_BODY = _style("Body",
    fontSize=10, leading=15, spaceAfter=5, alignment=TA_JUSTIFY)

S_BODY_SMALL = _style("BodySmall",
    fontSize=8.5, leading=13, textColor=MID_GRAY)

S_CAPTION = _style("Caption",
    fontName="Helvetica-BoldOblique", fontSize=8,
    textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=6)

S_TH = _style("TH",
    fontName="Helvetica-Bold", fontSize=8.5,
    textColor=WHITE, alignment=TA_CENTER, leading=11)

S_TD = _style("TD",
    fontSize=8.5, textColor=DARK_GRAY,
    alignment=TA_CENTER, leading=11)

S_FOOTER = _style("Footer",
    fontSize=8, textColor=MID_GRAY, alignment=TA_CENTER)

S_BULLET = _style("Bullet",
    fontSize=9.5, leading=15, leftIndent=12, spaceAfter=3)


# ---------------------------------------------------------------------------
# Helpers — formatting
# ---------------------------------------------------------------------------

def _pct(v: float, decimals: int = 1, sign: bool = True) -> str:
    """Format a decimal as a percentage string."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    prefix = "+" if sign and v >= 0 else ""
    return f"{prefix}{v * 100:.{decimals}f}%"


def _dollar(v: float) -> str:
    """Format as a dollar amount with sign."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.0f}"


def _ratio(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}"


def _colored_para(text: str, style: ParagraphStyle,
                  positive_green: bool = False, value: Optional[float] = None) -> Paragraph:
    """
    Return a coloured Paragraph for numeric cells.
    Positive values → green, negative → red, zero/neutral → dark gray.
    """
    if positive_green and value is not None:
        color_hex = "#228B22" if value >= 0 else "#B22222"
        return Paragraph(f'<font color="{color_hex}">{text}</font>', style)
    return Paragraph(text, style)


def _rule() -> HRFlowable:
    """Thin horizontal rule for section separation."""
    return HRFlowable(width="100%", thickness=0.5,
                      color=RULE_GRAY, spaceAfter=8, spaceBefore=4)


def _section_header(title: str) -> list:
    """Return [rule, section paragraph, rule] for a section header block."""
    return [
        _rule(),
        Paragraph(title, S_SECTION),
        _rule(),
    ]


def _embed_image(path: str,
                 width: float = CONTENT_W,
                 max_height: float = 16 * cm,
                 caption: Optional[str] = None) -> list:
    """
    Embed a PNG chart while safely constraining both width and height.
    Prevents ReportLab LayoutError for oversized images.
    """
    flowables = []

    if os.path.exists(path):

        # Create image without forcing dimensions first
        img = Image(path)

        # Original dimensions
        img_w = img.imageWidth
        img_h = img.imageHeight

        # Compute scale ratio
        width_ratio = width / img_w
        height_ratio = max_height / img_h

        ratio = min(width_ratio, height_ratio)

        # Apply scaled dimensions
        img.drawWidth = img_w * ratio
        img.drawHeight = img_h * ratio

        img.hAlign = "CENTER"

        flowables.append(img)

        if caption:
            flowables.append(Paragraph(caption, S_CAPTION))

    else:
        flowables.append(
            Paragraph(
                f'<i>[Chart not found: {os.path.basename(path)}]</i>',
                S_BODY_SMALL
            )
        )

    return flowables


def _two_images(path_a: str,
                path_b: str,
                caption_a: str = "",
                caption_b: str = "") -> list:
    """
    Place two images side by side with safe scaling.
    Prevents oversized ReportLab table/image LayoutErrors.
    """

    half_w = (CONTENT_W - 12) / 2
    max_h = 9 * cm

    def _scaled_image(path: str):

        if not os.path.exists(path):
            return Paragraph(
                f"[Chart missing: {os.path.basename(path)}]",
                S_BODY_SMALL
            )

        img = Image(path)

        # Original dimensions
        iw = img.imageWidth
        ih = img.imageHeight

        # Scale safely
        ratio = min(
            half_w / iw,
            max_h / ih
        )

        img.drawWidth = iw * ratio
        img.drawHeight = ih * ratio
        img.hAlign = "CENTER"

        return img

    cell_a = [
        _scaled_image(path_a),
        Paragraph(caption_a, S_CAPTION)
    ]

    cell_b = [
        _scaled_image(path_b),
        Paragraph(caption_b, S_CAPTION)
    ]

    tbl = Table(
        [[cell_a, cell_b]],
        colWidths=[half_w + 6, half_w + 6],
    )

    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))

    return [tbl]


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _build_table(
    headers: list[str],
    rows: list[list],
    col_widths: Optional[list[float]] = None,
    row_colors: bool = True,
) -> Table:
    """
    Build a styled reportlab Table from header and row data.

    Parameters
    ----------
    headers : list[str]
        Column header labels.
    rows : list[list]
        Each inner list is one data row; cells can be str or Paragraph.
    col_widths : list[float], optional
        Column widths in points.  If None, distributed evenly.
    row_colors : bool
        Alternate row background between white and light-gray.

    Returns
    -------
    Table
        Styled reportlab Table flowable.
    """
    if col_widths is None:
        col_widths = [CONTENT_W / len(headers)] * len(headers)

    # Header row: wrap each label in a styled Paragraph
    header_cells = [Paragraph(h, S_TH) for h in headers]
    all_rows = [header_cells] + rows

    tbl = Table(all_rows, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        # Header background
        ("BACKGROUND",   (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUND", (0, 0), (-1, 0), NAVY),
        # Grid
        ("GRID",         (0, 0), (-1, -1), 0.3, RULE_GRAY),
        ("LINEBELOW",    (0, 0), (-1, 0),  0.8, RULE_GRAY),
        # Padding
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        # Alignment
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]

    if row_colors:
        for i in range(1, len(all_rows)):
            bg = WHITE if i % 2 == 1 else LIGHT_GRAY
            style_cmds.append(("ROWBACKGROUND", (0, i), (-1, i), bg))

    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ---------------------------------------------------------------------------
# Page templates (header / footer callbacks)
# ---------------------------------------------------------------------------

class _RiskReport(BaseDocTemplate):
    """
    Custom BaseDocTemplate that renders a page number footer on every
    page except the cover, and a thin navy top bar on content pages.
    """

    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        self._build_templates()

    def _build_templates(self):
        # Cover page: full-bleed frame, no footer
        cover_frame = Frame(
            MARGIN_L, MARGIN_B,
            CONTENT_W, PAGE_H - MARGIN_T - MARGIN_B,
            id="cover",
        )
        cover_tpl = PageTemplate(
            id="Cover",
            frames=[cover_frame],
            onPage=self._draw_cover_bg,
        )

        # Content pages: same frame + footer
        content_frame = Frame(
            MARGIN_L, MARGIN_B + 0.8 * cm,
            CONTENT_W, PAGE_H - MARGIN_T - MARGIN_B - 0.8 * cm,
            id="content",
        )
        content_tpl = PageTemplate(
            id="Content",
            frames=[content_frame],
            onPage=self._draw_content_chrome,
        )

        self.addPageTemplates([cover_tpl, content_tpl])

    @staticmethod
    def _draw_cover_bg(canvas, doc):
        """Draw navy top band and footer bar on the cover page."""
        canvas.saveState()
        # Top navy band
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 3.5 * cm, PAGE_W, 3.5 * cm, fill=1, stroke=0)
        # Bottom footer band
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_W, 1.4 * cm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(
            PAGE_W / 2, 0.55 * cm,
            "CONFIDENTIAL — FOR INTERNAL USE ONLY"
        )
        canvas.restoreState()

    @staticmethod
    def _draw_content_chrome(canvas, doc):
        """Draw page number footer and top rule on content pages."""
        canvas.saveState()
        # Thin navy rule at top
        canvas.setStrokeColor(NAVY)
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN_L, PAGE_H - 1.2 * cm,
                    PAGE_W - MARGIN_R, PAGE_H - 1.2 * cm)
        # Header: report name left, date right
        canvas.setFillColor(MID_GRAY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN_L, PAGE_H - 0.9 * cm,
                          "PORTFOLIO RISK ANALYSIS REPORT")
        canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 0.9 * cm,
                               f"Generated: {date.today().strftime('%d %B %Y')}")
        # Footer rule
        canvas.setStrokeColor(RULE_GRAY)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_L, MARGIN_B + 0.5 * cm,
                    PAGE_W - MARGIN_R, MARGIN_B + 0.5 * cm)
        # Page number
        canvas.setFillColor(MID_GRAY)
        canvas.setFont("Helvetica", 8)
        page_num = doc.page
        canvas.drawCentredString(
            PAGE_W / 2, MARGIN_B + 0.2 * cm,
            f"Page {page_num} of {TOTAL_PAGES}"
        )
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Page 1 — Cover
# ---------------------------------------------------------------------------

def _page_cover(weights_a: dict, weights_b: dict) -> list:
    """
    Build flowables for the cover page.

    Parameters
    ----------
    weights_a, weights_b : dict
        Portfolio weight dicts for label display.

    Returns
    -------
    list
        List of reportlab flowables ending with a PageBreak.
    """
    today_str    = date.today().strftime("%d %B %Y")
    tickers_str  = ", ".join(weights_a.keys())
    period_str   = f"January 2020 – {date.today().strftime('%B %Y')}"

    weights_a_str = "  |  ".join(
        f"{t}: {w*100:.0f}%" for t, w in weights_a.items()
    )
    weights_b_str = "  |  ".join(
        f"{t}: {w*100:.0f}%" for t, w in weights_b.items()
    )

    s_white_center = _style("WhiteCenter",
        fontName="Helvetica", fontSize=11,
        textColor=WHITE, alignment=TA_CENTER, leading=16)
    s_white_bold = _style("WhiteBold",
        fontName="Helvetica-Bold", fontSize=20,
        textColor=WHITE, alignment=TA_CENTER, leading=26)

    return [
        # Spacer to push below the navy band drawn by the canvas callback
        Spacer(1, 3.8 * cm),
        Paragraph("PORTFOLIO RISK ANALYSIS REPORT", S_COVER_TITLE),
        Spacer(1, 0.3 * cm),
        Paragraph("Value at Risk &amp; Risk Metrics Summary", S_COVER_SUB),
        Spacer(1, 0.8 * cm),
        _rule(),
        Spacer(1, 0.5 * cm),
        Paragraph(f"Generated: {today_str}", S_COVER_META),
        Paragraph(f"Analysis Period: {period_str}", S_COVER_META),
        Paragraph(f"Asset Universe: {tickers_str}", S_COVER_META),
        Paragraph(f"Base Portfolio Value: $100,000", S_COVER_META),
        Spacer(1, 0.8 * cm),
        _rule(),
        Spacer(1, 0.6 * cm),
        Paragraph("<b>Portfolio A — Equal Weight</b>", S_COVER_META),
        Paragraph(weights_a_str, S_COVER_META),
        Spacer(1, 0.35 * cm),
        Paragraph("<b>Portfolio B — Tech Overweight</b>", S_COVER_META),
        Paragraph(weights_b_str, S_COVER_META),
        Spacer(1, 1.2 * cm),
        _rule(),
        Spacer(1, 0.6 * cm),
        Paragraph(
            "This report presents a comprehensive risk analysis of a multi-asset "
            "equity portfolio using historical market data sourced via Yahoo Finance. "
            "Risk metrics include Value at Risk (VaR), Conditional VaR (Expected "
            "Shortfall), volatility analysis, and portfolio performance statistics "
            "computed in accordance with industry-standard methodologies.",
            S_BODY,
        ),
        NextPageTemplate("Content"),
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# Page 2 — Executive Summary
# ---------------------------------------------------------------------------

def _page_exec_summary(
    metrics_a: dict,
    metrics_b: dict,
    var_metrics_a: dict,
    var_metrics_b: dict,
) -> list:
    """
    Build the Executive Summary page with a key-metrics grid and
    dynamically generated risk narrative.

    Parameters
    ----------
    metrics_a, metrics_b : dict
        Output of portfolio.compute_portfolio_metrics().
    var_metrics_a, var_metrics_b : dict
        Output of var_engine.print_var_report().

    Returns
    -------
    list
        Flowables ending with PageBreak.
    """
    flowables = _section_header("EXECUTIVE SUMMARY")

    # ── Key metrics grid ────────────────────────────────────────────────────
    flowables.append(Paragraph("Key Risk & Performance Metrics", S_SUBSECTION))
    flowables.append(Spacer(1, 4))

    def _metric_row(label: str,
                    val_a: str, val_b: str,
                    raw_a: float = 0.0, raw_b: float = 0.0) -> list:
        color_a = GREEN if raw_a >= 0 else RED
        color_b = GREEN if raw_b >= 0 else RED
        return [
            Paragraph(label, S_TD),
            Paragraph(f'<font color="#{_to_hex(color_a)}">{val_a}</font>', S_TD),
            Paragraph(f'<font color="#{_to_hex(color_b)}">{val_b}</font>', S_TD),
        ]

    ma, mb = metrics_a, metrics_b
    va, vb = var_metrics_a, var_metrics_b

    grid_rows = [
        _metric_row("Total Return",
            _pct(ma["total_return"]),      _pct(mb["total_return"]),
            ma["total_return"],             mb["total_return"]),
        _metric_row("Annualised Return",
            _pct(ma["annualised_return"]),  _pct(mb["annualised_return"]),
            ma["annualised_return"],        mb["annualised_return"]),
        _metric_row("Annualised Volatility",
            _pct(ma["annualised_vol"], sign=False),
            _pct(mb["annualised_vol"], sign=False), 1.0, 1.0),
        _metric_row("Sharpe Ratio",
            _ratio(ma["sharpe_ratio"]),    _ratio(mb["sharpe_ratio"]),
            ma["sharpe_ratio"],             mb["sharpe_ratio"]),
        _metric_row("Max Drawdown",
            _pct(ma["max_drawdown"]),       _pct(mb["max_drawdown"]),
            ma["max_drawdown"],             mb["max_drawdown"]),
        _metric_row("95% Hist. VaR (1-day)",
            _dollar(va["hist_95"]["var_dollar"]),
            _dollar(vb["hist_95"]["var_dollar"]),
            va["hist_95"]["var_dollar"],    vb["hist_95"]["var_dollar"]),
    ]

    col_w = [CONTENT_W * 0.40, CONTENT_W * 0.30, CONTENT_W * 0.30]
    grid = _build_table(
        ["Metric", "Portfolio A — Equal Wt", "Portfolio B — Tech OW"],
        grid_rows, col_widths=col_w,
    )
    flowables.append(grid)
    flowables.append(Spacer(1, 0.6 * cm))

    # ── Auto-generated narrative ─────────────────────────────────────────────
    flowables.append(Paragraph("Risk Narrative", S_SUBSECTION))
    flowables.append(_narrative_exec(ma, mb, va))
    flowables.append(Spacer(1, 0.4 * cm))

    # ── Divider ──────────────────────────────────────────────────────────────
    flowables.append(_rule())
    flowables.append(Paragraph(
        "<i>Detailed analysis on the following pages covers performance attribution, "
        "VaR methodology comparisons, risk decomposition, and volatility regimes.</i>",
        S_BODY_SMALL,
    ))
    flowables.append(PageBreak())
    return flowables


def _narrative_exec(ma: dict, mb: dict, va: dict) -> Paragraph:
    """Generate a dynamic 4-sentence executive narrative from computed metrics."""
    ret_a   = ma["total_return"] * 100
    vol_a   = ma["annualised_vol"] * 100
    sr_a    = ma["sharpe_ratio"]
    dd_a    = ma["max_drawdown"] * 100
    var_a_d = va["hist_95"]["var_dollar"]

    best_port  = "Portfolio A" if ma["sharpe_ratio"] > mb["sharpe_ratio"] else "Portfolio B"
    best_sr    = max(ma["sharpe_ratio"], mb["sharpe_ratio"])
    dd_comment = (
        "the COVID-19 market dislocation in early 2020"
        if abs(dd_a) > 15 else "elevated market volatility during the analysis period"
    )

    text = (
        f"Portfolio A generated a cumulative total return of {_pct(ma['total_return'])} "
        f"over the analysis period, against Portfolio B's {_pct(mb['total_return'])}, "
        f"with annualised volatility of {vol_a:.1f}% and {mb['annualised_vol']*100:.1f}% respectively. "
        f"{best_port} achieved the superior risk-adjusted return with a Sharpe Ratio of {best_sr:.2f}, "
        f"indicating {('strong' if best_sr > 1 else 'moderate')} compensation per unit of risk taken. "
        f"The maximum drawdown of {dd_a:.1f}% for Portfolio A reflects the portfolio's sensitivity to "
        f"{dd_comment}. "
        f"At 95% confidence, Portfolio A is not expected to lose more than "
        f"${abs(var_a_d):,.0f} in a single trading day under normal market conditions."
    )
    return Paragraph(text, S_BODY)


# ---------------------------------------------------------------------------
# Page 3 — Portfolio Performance
# ---------------------------------------------------------------------------

def _page_performance(
    metrics_a: dict,
    metrics_b: dict,
    chart_dir: str = "outputs",
) -> list:
    """
    Build the Portfolio Performance Analysis page.

    Parameters
    ----------
    metrics_a, metrics_b : dict
        Output of portfolio.compute_portfolio_metrics().
    chart_dir : str
        Directory where PNG charts are saved.

    Returns
    -------
    list
        Flowables ending with PageBreak.
    """
    flowables = _section_header("PORTFOLIO PERFORMANCE ANALYSIS")

    def _prow(label: str, val_a, val_b, raw_a=0.0, raw_b=0.0) -> list:
        ca = "#228B22" if raw_a >= 0 else "#B22222"
        cb = "#228B22" if raw_b >= 0 else "#B22222"
        return [
            Paragraph(label, S_TD),
            Paragraph(f'<font color="{ca}">{val_a}</font>', S_TD),
            Paragraph(f'<font color="{cb}">{val_b}</font>', S_TD),
        ]

    ma, mb = metrics_a, metrics_b
    perf_rows = [
        _prow("Total Return",
              _pct(ma["total_return"]),       _pct(mb["total_return"]),
              ma["total_return"],              mb["total_return"]),
        _prow("Annualised Return",
              _pct(ma["annualised_return"]),   _pct(mb["annualised_return"]),
              ma["annualised_return"],          mb["annualised_return"]),
        _prow("Annualised Volatility",
              _pct(ma["annualised_vol"], sign=False),
              _pct(mb["annualised_vol"], sign=False), 1, 1),
        _prow("Sharpe Ratio",
              _ratio(ma["sharpe_ratio"]),      _ratio(mb["sharpe_ratio"]),
              ma["sharpe_ratio"],               mb["sharpe_ratio"]),
        _prow("Max Drawdown",
              _pct(ma["max_drawdown"]),        _pct(mb["max_drawdown"]),
              ma["max_drawdown"],               mb["max_drawdown"]),
        _prow("Calmar Ratio",
              _ratio(ma["calmar_ratio"]),       _ratio(mb["calmar_ratio"]),
              ma["calmar_ratio"],               mb["calmar_ratio"]),
        _prow("Best Day",
              _pct(ma["best_month"]),           _pct(mb["best_month"]),
              ma["best_month"],                 mb["best_month"]),
        _prow("Worst Month",
              _pct(ma["worst_month"]),          _pct(mb["worst_month"]),
              ma["worst_month"],                mb["worst_month"]),
        _prow("% Positive Days",
              f"{ma['pct_positive_days']*100:.1f}%",
              f"{mb['pct_positive_days']*100:.1f}%",
              1, 1),
        _prow("Trading Days",
              str(ma["n_trading_days"]),        str(mb["n_trading_days"]),
              1, 1),
    ]

    col_w = [CONTENT_W * 0.42, CONTENT_W * 0.29, CONTENT_W * 0.29]
    flowables.append(_build_table(
        ["Metric", "Portfolio A — Equal Wt", "Portfolio B — Tech OW"],
        perf_rows, col_widths=col_w,
    ))
    flowables.append(Spacer(1, 0.5 * cm))

    # Charts
    flowables += _embed_image(
        os.path.join(chart_dir, "cumulative_returns.png"),
        caption="Figure 1 — Cumulative portfolio value (Jan 2020 – Present)",
    )
    flowables.append(Spacer(1, 0.3 * cm))
    flowables += _embed_image(
        os.path.join(chart_dir, "drawdown.png"),
        caption="Figure 2 — Underwater drawdown chart",
    )
    flowables.append(PageBreak())
    return flowables


# ---------------------------------------------------------------------------
# Page 4 — VaR Analysis
# ---------------------------------------------------------------------------

def _page_var(
    var_metrics_a: dict,
    var_metrics_b: dict,
    backtest_a: dict,
    backtest_b: dict,
    chart_dir: str = "outputs",
) -> list:
    """
    Build the VaR Analysis page.

    Parameters
    ----------
    var_metrics_a, var_metrics_b : dict
        Output of var_engine.print_var_report().
    backtest_a, backtest_b : dict
        Backtest result dicts (contained within var_metrics).
    chart_dir : str
        Directory where PNG charts are saved.

    Returns
    -------
    list
        Flowables ending with PageBreak.
    """
    flowables = _section_header("VALUE AT RISK ANALYSIS")
    va, vb = var_metrics_a, var_metrics_b

    # ── VaR comparison table ─────────────────────────────────────────────────
    flowables.append(Paragraph("1-Day VaR by Method and Confidence Level", S_SUBSECTION))

    def _var_cell(val: float) -> Paragraph:
        col = "#B22222"   # all VaR values are losses → always red
        return Paragraph(f'<font color="{col}">{_dollar(val)}</font>', S_TD)

    var_rows = [
        [Paragraph("Historical Simulation", S_TD),
         _var_cell(va["hist_95"]["var_dollar"]),
         _var_cell(va["hist_99"]["var_dollar"]),
         _var_cell(vb["hist_95"]["var_dollar"]),
         _var_cell(vb["hist_99"]["var_dollar"])],
        [Paragraph("Parametric (Normal)", S_TD),
         _var_cell(va["param_95"]["var_dollar"]),
         _var_cell(va["param_99"]["var_dollar"]),
         _var_cell(vb["param_95"]["var_dollar"]),
         _var_cell(vb["param_99"]["var_dollar"])],
        [Paragraph("CVaR / Exp. Shortfall", S_TD),
         _var_cell(va["cvar_95"]["cvar_dollar"]),
         _var_cell(va["cvar_99"]["cvar_dollar"]),
         _var_cell(vb["cvar_95"]["cvar_dollar"]),
         _var_cell(vb["cvar_99"]["cvar_dollar"])],
    ]

    col_w = [CONTENT_W * 0.26, CONTENT_W * 0.185,
             CONTENT_W * 0.185, CONTENT_W * 0.185, CONTENT_W * 0.185]
    flowables.append(_build_table(
        ["Method", "Port A 95%", "Port A 99%", "Port B 95%", "Port B 99%"],
        var_rows, col_widths=col_w,
    ))
    flowables.append(Spacer(1, 0.5 * cm))

    # ── Backtest results ─────────────────────────────────────────────────────
    flowables.append(Paragraph("VaR Model Backtest Results (95% Confidence)", S_SUBSECTION))

    def _bt_status(bt: dict) -> Paragraph:
        ok   = bt["model_status"] == "OK"
        col  = "#228B22" if ok else "#B22222"
        txt  = "PASS" if ok else "FAIL"
        return Paragraph(f'<font color="{col}"><b>{txt}</b></font>', S_TD)

    bt_rows = [
        [Paragraph("Portfolio A — Equal Wt", S_TD),
         Paragraph(str(backtest_a["n_days"]), S_TD),
         Paragraph(f"{backtest_a['expected_breaches']:.1f}", S_TD),
         Paragraph(str(backtest_a["actual_breaches"]), S_TD),
         Paragraph(f"{backtest_a['breach_rate_pct']:.2f}%", S_TD),
         _bt_status(backtest_a)],
        [Paragraph("Portfolio B — Tech OW", S_TD),
         Paragraph(str(backtest_b["n_days"]), S_TD),
         Paragraph(f"{backtest_b['expected_breaches']:.1f}", S_TD),
         Paragraph(str(backtest_b["actual_breaches"]), S_TD),
         Paragraph(f"{backtest_b['breach_rate_pct']:.2f}%", S_TD),
         _bt_status(backtest_b)],
    ]

    col_w2 = [CONTENT_W * 0.28, CONTENT_W * 0.12,
              CONTENT_W * 0.14, CONTENT_W * 0.12,
              CONTENT_W * 0.14, CONTENT_W * 0.20]
    flowables.append(_build_table(
        ["Portfolio", "Days", "Expected", "Actual", "Breach Rate", "Status"],
        bt_rows, col_widths=col_w2,
    ))
    flowables.append(Spacer(1, 0.4 * cm))

    # ── VaR narrative ────────────────────────────────────────────────────────
    flowables.append(_narrative_var(va, vb, backtest_a, backtest_b))
    flowables.append(Spacer(1, 0.4 * cm))

    # ── Return distribution charts ───────────────────────────────────────────
    flowables += _two_images(
        os.path.join(chart_dir, "return_distribution_A.png"),
        os.path.join(chart_dir, "return_distribution_B.png"),
        caption_a="Figure 3 — Return distribution, Portfolio A",
        caption_b="Figure 4 — Return distribution, Portfolio B",
    )
    flowables.append(PageBreak())
    return flowables


def _narrative_var(va: dict, vb: dict, bt_a: dict, bt_b: dict) -> Paragraph:
    """Generate a dynamic 3-sentence VaR interpretation."""
    diff = abs(va["hist_95"]["var_dollar"]) - abs(vb["hist_95"]["var_dollar"])
    higher_port = "Portfolio B" if diff < 0 else "Portfolio A"
    gap_str     = f"${abs(diff):,.0f}"

    bt_ok_a = bt_a["model_status"] == "OK"
    bt_ok_b = bt_b["model_status"] == "OK"
    model_str = (
        "Both models pass the Kupiec backtest"
        if bt_ok_a and bt_ok_b else
        "At least one model fails the Kupiec backtest and requires recalibration"
    )

    hist_vs_param = abs(va["hist_95"]["var_dollar"]) - abs(va["param_95"]["var_dollar"])
    tail_comment = (
        f"Historical Simulation produces a ${abs(hist_vs_param):,.0f} larger loss "
        f"estimate than the Parametric method for Portfolio A, confirming the presence "
        f"of fat tails in the empirical return distribution."
        if hist_vs_param > 0 else
        f"Parametric VaR is conservative relative to Historical Simulation for Portfolio A, "
        f"suggesting returns are closer to normally distributed than typical equity portfolios."
    )

    text = (
        f"{higher_port} carries higher 1-day 95% VaR, with a {gap_str} difference "
        f"reflecting its greater concentration in volatile technology names. "
        f"{tail_comment} "
        f"{model_str} at {bt_a['expected_rate_pct']:.0f}% expected breach rate, "
        f"with Portfolio A recording {bt_a['actual_breaches']} actual breaches "
        f"({bt_a['breach_rate_pct']:.2f}%) against an expected "
        f"{bt_a['expected_breaches']:.1f}."
    )
    return Paragraph(text, S_BODY)


# ---------------------------------------------------------------------------
# Page 5 — Risk Decomposition
# ---------------------------------------------------------------------------

def _page_decomposition(
    component_var_a: pd.DataFrame,
    component_var_b: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    chart_dir: str = "outputs",
) -> list:
    """
    Build the Risk Decomposition page.

    Parameters
    ----------
    component_var_a, component_var_b : pd.DataFrame
        Output of var_engine.compute_component_var().
    correlation_matrix : pd.DataFrame
        Pearson correlation matrix from risk_engine.
    chart_dir : str
        Directory where PNG charts are saved.

    Returns
    -------
    list
        Flowables ending with PageBreak.
    """
    flowables = _section_header("RISK DECOMPOSITION")

    # ── Component VaR tables side by side ────────────────────────────────────
    flowables.append(Paragraph("Component VaR Contribution", S_SUBSECTION))

    def _comp_table(comp_df: pd.DataFrame, label: str) -> Table:
        rows = []
        for _, row in comp_df.iterrows():
            pct_val = row["pct_contribution"]
            col_hex = "#B22222"
            rows.append([
                Paragraph(row["ticker"], S_TD),
                Paragraph(f"{row['weight']*100:.0f}%", S_TD),
                Paragraph(f'<font color="{col_hex}">{_dollar(row["component_var_dollar"])}</font>', S_TD),
                Paragraph(f"{pct_val:.1f}%", S_TD),
            ])
        cw = [(CONTENT_W / 2 - 4) * x for x in [0.18, 0.15, 0.37, 0.30]]
        return _build_table(
            [label, "Weight", "Comp VaR $", "% Contrib"],
            rows, col_widths=cw,
        )

    side_by_side = Table(
        [[_comp_table(component_var_a, "Port A Ticker"),
          _comp_table(component_var_b, "Port B Ticker")]],
        colWidths=[(CONTENT_W / 2) - 4, (CONTENT_W / 2) + 4],
    )
    side_by_side.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    flowables.append(side_by_side)
    flowables.append(Spacer(1, 0.5 * cm))

    # ── Correlation matrix table ──────────────────────────────────────────────
    flowables.append(Paragraph("Pearson Correlation Matrix", S_SUBSECTION))
    tickers = list(correlation_matrix.columns)
    corr_header = [""] + tickers
    corr_rows = []
    for row_t in tickers:
        cells = [Paragraph(f"<b>{row_t}</b>", S_TD)]
        for col_t in tickers:
            val = correlation_matrix.loc[row_t, col_t]
            if val > 0.7:
                col = "#B22222"
            elif val < 0.3:
                col = "#228B22"
            else:
                col = "#323232"
            cells.append(
                Paragraph(f'<font color="{col}">{val:.2f}</font>', S_TD)
            )
        corr_rows.append(cells)

    n_cols = len(corr_header)
    cw = [CONTENT_W * 0.14] + [CONTENT_W * (0.86 / (n_cols - 1))] * (n_cols - 1)
    flowables.append(_build_table(corr_header, corr_rows, col_widths=cw))
    flowables.append(Spacer(1, 0.3 * cm))

    # ── Diversification commentary ───────────────────────────────────────────
    flowables.append(_narrative_correlation(correlation_matrix))
    flowables.append(Spacer(1, 0.3 * cm))

    # ── Charts ───────────────────────────────────────────────────────────────
    flowables += _embed_image(
        os.path.join(chart_dir, "correlation_heatmap.png"),
        width=CONTENT_W * 0.80,
        caption="Figure 5 — Correlation heatmap (coolwarm diverging palette)",
    )
    flowables.append(Spacer(1, 0.2 * cm))
    flowables += _two_images(
        os.path.join(chart_dir, "component_var_A.png"),
        os.path.join(chart_dir, "component_var_B.png"),
        caption_a="Figure 6 — Component VaR, Portfolio A",
        caption_b="Figure 7 — Component VaR, Portfolio B",
    )
    flowables.append(PageBreak())
    return flowables


def _narrative_correlation(corr: pd.DataFrame) -> Paragraph:
    """Generate a 2-sentence diversification commentary from the correlation matrix."""
    # Find max / min off-diagonal pairs using a mask instead of fill_diagonal
    # (avoids the read-only array issue that can occur with certain DataFrame internals)
    arr      = corr.values.copy().astype(float)   # writable copy
    mask     = np.eye(len(arr), dtype=bool)

    arr_no_diag_max        = arr.copy(); arr_no_diag_max[mask] = -999
    max_val  = arr_no_diag_max.max()
    max_idx  = np.unravel_index(arr_no_diag_max.argmax(), arr.shape)
    max_pair = (corr.columns[max_idx[0]], corr.columns[max_idx[1]])

    arr_no_diag_min        = arr.copy(); arr_no_diag_min[mask] = 999
    min_val  = arr_no_diag_min.min()
    min_idx  = np.unravel_index(arr_no_diag_min.argmin(), arr.shape)
    min_pair = (corr.columns[min_idx[0]], corr.columns[min_idx[1]])
    text = (
        f"The highest pairwise correlation in the portfolio is between "
        f"{max_pair[0]} and {max_pair[1]} at {max_val:.2f}, suggesting limited "
        f"independent return contribution between these two holdings and concentrated "
        f"sector exposure. "
        f"The lowest pairwise correlation is between {min_pair[0]} and {min_pair[1]} "
        f"at {min_val:.2f}, indicating the best available diversification benefit "
        f"within the current universe; expanding the portfolio to include lower-correlation "
        f"asset classes would further reduce portfolio variance."
    )
    return Paragraph(text, S_BODY)


# ---------------------------------------------------------------------------
# Page 6 — Volatility Analysis
# ---------------------------------------------------------------------------

def _page_volatility(
    returns_df: pd.DataFrame,
    chart_dir: str = "outputs",
) -> list:
    """
    Build the Volatility Analysis page.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Individual asset daily log returns.
    chart_dir : str
        Directory where PNG charts are saved.

    Returns
    -------
    list
        Flowables ending with PageBreak.
    """
    flowables = _section_header("VOLATILITY ANALYSIS")

    # ── Vol summary table ────────────────────────────────────────────────────
    flowables.append(Paragraph("Annualised Volatility Summary by Asset", S_SUBSECTION))

    ann_vol = returns_df.std() * np.sqrt(252)
    roll_30 = returns_df.rolling(30).std().iloc[-1] * np.sqrt(252)
    roll_60 = returns_df.rolling(60).std().iloc[-1] * np.sqrt(252)

    vol_rows = []
    for ticker in returns_df.columns:
        av   = ann_vol[ticker]
        r30  = roll_30[ticker]
        r60  = roll_60[ticker]
        trend = ("↑" if r30 > av else "↓") + " vs full-period"
        vol_rows.append([
            Paragraph(ticker,               S_TD),
            Paragraph(_pct(av, sign=False), S_TD),
            Paragraph(_pct(r30, sign=False),S_TD),
            Paragraph(_pct(r60, sign=False),S_TD),
            Paragraph(trend,                S_TD),
        ])

    col_w = [CONTENT_W * 0.15, CONTENT_W * 0.21,
             CONTENT_W * 0.21, CONTENT_W * 0.21, CONTENT_W * 0.22]
    flowables.append(_build_table(
        ["Ticker", "Ann. Vol (Full)", "30-Day Roll. Vol", "60-Day Roll. Vol", "Trend"],
        vol_rows, col_widths=col_w,
    ))
    flowables.append(Spacer(1, 0.3 * cm))
    flowables.append(_narrative_vol(returns_df, ann_vol, roll_30))
    flowables.append(Spacer(1, 0.3 * cm))

    # ── Charts ───────────────────────────────────────────────────────────────
    flowables += _embed_image(
        os.path.join(chart_dir, "rolling_volatility.png"),
        caption="Figure 8 — 30-day rolling annualised volatility by ticker",
    )
    flowables.append(Spacer(1, 0.2 * cm))
    flowables += _two_images(
        os.path.join(chart_dir, "rolling_var_A.png"),
        os.path.join(chart_dir, "rolling_var_B.png"),
        caption_a="Figure 9 — Rolling VaR vs P&L, Portfolio A",
        caption_b="Figure 10 — Rolling VaR vs P&L, Portfolio B",
    )
    flowables.append(PageBreak())
    return flowables


def _narrative_vol(returns_df: pd.DataFrame,
                   ann_vol: pd.Series,
                   roll_30: pd.Series) -> Paragraph:
    """Generate a 2-sentence volatility commentary."""
    most_vol   = ann_vol.idxmax()
    least_vol  = ann_vol.idxmin()
    trending_up = [t for t in returns_df.columns if roll_30[t] > ann_vol[t]]

    trend_str = (
        f"{', '.join(trending_up)} show rising near-term volatility relative to "
        f"their full-period averages, suggesting elevated risk conditions."
        if trending_up else
        "Near-term volatility is below full-period averages across all tickers, "
        "suggesting a relatively calm current market regime."
    )

    text = (
        f"{most_vol} is the most volatile asset with annualised volatility of "
        f"{ann_vol[most_vol]*100:.1f}%, while {least_vol} exhibits the lowest "
        f"realised volatility at {ann_vol[least_vol]*100:.1f}%, consistent with its "
        f"role as a diversified benchmark ETF. "
        f"{trend_str}"
    )
    return Paragraph(text, S_BODY)


# ---------------------------------------------------------------------------
# Page 7 — Methodology & Disclaimers
# ---------------------------------------------------------------------------

def _page_methodology() -> list:
    """
    Build the Methodology and Disclaimers page.

    Returns
    -------
    list
        Flowables (no PageBreak — last page).
    """
    flowables = _section_header("METHODOLOGY")

    bullets = [
        ("<b>Data Source:</b>  Historical adjusted closing prices sourced from Yahoo "
         "Finance via the yfinance Python library.  Prices are split- and "
         "dividend-adjusted to ensure return continuity."),
        ("<b>Return Calculation:</b>  Daily log returns computed as "
         "r<sub>t</sub> = ln(P<sub>t</sub> / P<sub>t-1</sub>), which are "
         "time-additive and approximately normally distributed for small moves."),
        ("<b>VaR Methodology:</b>  Historical Simulation uses the empirical 5th "
         "percentile of the past 252-day return distribution.  Parametric VaR "
         "assumes a Normal distribution fitted to observed mean and standard "
         "deviation.  CVaR is the average of all losses exceeding the VaR threshold."),
        ("<b>Annualisation:</b>  Daily volatility is annualised by multiplying by "
         "sqrt(252) under the assumption of i.i.d. daily returns and 252 "
         "trading days per year."),
        ("<b>Risk-Free Rate:</b>  5.0% per annum, approximating the US 10-year "
         "Treasury yield over the analysis period.  Used in Sharpe and Calmar "
         "ratio calculations."),
        ("<b>Component VaR:</b>  Computed using the Gaussian covariance "
         "decomposition w<sub>i</sub> × (Sigma × w)<sub>i</sub> / sigma<sub>p</sub>, "
         "ensuring components sum exactly to total parametric VaR."),
        ("<b>Backtest:</b>  Kupiec proportion-of-failures test.  Model flagged "
         "as passing if observed breach rate falls within ±1% of the expected "
         "exceedance rate (5% at 95% confidence)."),
    ]

    for b in bullets:
        flowables.append(Paragraph(f"&bull;  {b}", S_BULLET))
    flowables.append(Spacer(1, 0.6 * cm))

    flowables += _section_header("DISCLAIMERS")
    disclaimer = (
        "This report is generated for educational and analytical purposes only and "
        "does not constitute investment advice. Past performance is not indicative of "
        "future results. VaR and CVaR estimates are based on historical data and may "
        "not capture tail risks in unprecedented or structurally different market "
        "conditions. The models presented assume stationarity of return distributions, "
        "which may not hold during market stress events. No representation is made as "
        "to the accuracy or completeness of information herein. Positions described are "
        "hypothetical and no actual trading has occurred. This report is for internal "
        "analytical and educational use only."
    )
    flowables.append(Paragraph(disclaimer, S_BODY))
    flowables.append(Spacer(1, 0.8 * cm))
    flowables.append(_rule())
    flowables.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%d %B %Y at %H:%M')}  |  "
        "Portfolio Risk Analyzer v1.0  |  Python / reportlab",
        S_BODY_SMALL,
    ))
    return flowables


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_hex(color: colors.Color) -> str:
    """Convert a reportlab Color to a 6-digit hex string."""
    return (f"{int(color.red*255):02X}"
            f"{int(color.green*255):02X}"
            f"{int(color.blue*255):02X}")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def generate_pdf_report(
    metrics_a: dict,
    metrics_b: dict,
    var_metrics_a: dict,
    var_metrics_b: dict,
    backtest_a: dict,
    backtest_b: dict,
    returns_df: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    component_var_a: pd.DataFrame,
    component_var_b: pd.DataFrame,
    weights_a: dict,
    weights_b: dict,
    output_path: str = "outputs/risk_report.pdf",
    chart_dir: str = "outputs",
) -> str:
    """
    Generate the complete 7-page PDF risk report.

    Calls each page-building function in sequence, assembles the flowables
    into a single document, and saves to output_path.

    Parameters
    ----------
    metrics_a, metrics_b : dict
        Output of portfolio.compute_portfolio_metrics().
    var_metrics_a, var_metrics_b : dict
        Output of var_engine.print_var_report().
    backtest_a, backtest_b : dict
        Backtest result dicts (nested within var_metrics).
    returns_df : pd.DataFrame
        Individual asset daily log returns.
    correlation_matrix : pd.DataFrame
        Pearson correlation matrix.
    component_var_a, component_var_b : pd.DataFrame
        Component VaR DataFrames.
    weights_a, weights_b : dict
        Portfolio weight dicts.
    output_path : str, optional
        Output file path.  Default: "outputs/risk_report.pdf".
    chart_dir : str, optional
        Directory containing chart PNG files.  Default: "outputs".

    Returns
    -------
    str
        Absolute path of the saved PDF.
    """
    _ensure_dir(output_path)

    doc = _RiskReport(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN_L,
        rightMargin=MARGIN_R,
        topMargin=MARGIN_T,
        bottomMargin=MARGIN_B,
        title="Portfolio Risk Analysis Report",
        author="Portfolio Risk Analyzer",
        subject="VaR and Risk Metrics",
    )

    pages = [
        ("Cover Page",             _page_cover(weights_a, weights_b)),
        ("Executive Summary",      _page_exec_summary(metrics_a, metrics_b,
                                       var_metrics_a, var_metrics_b)),
        ("Portfolio Performance",  _page_performance(metrics_a, metrics_b, chart_dir)),
        ("VaR Analysis",           _page_var(var_metrics_a, var_metrics_b,
                                       backtest_a, backtest_b, chart_dir)),
        ("Risk Decomposition",     _page_decomposition(
                                       component_var_a, component_var_b,
                                       correlation_matrix, chart_dir)),
        ("Volatility Analysis",    _page_volatility(returns_df, chart_dir)),
        ("Methodology",            _page_methodology()),
    ]

    story: list = []
    for i, (page_name, flowables) in enumerate(pages, 1):
        print(f"  Generating page {i}/{TOTAL_PAGES}: {page_name} ...")
        story.extend(flowables)

    doc.build(story)

    abs_path = os.path.abspath(output_path)
    print(f"\n  \u2713  Risk report saved to {abs_path}")
    return abs_path
