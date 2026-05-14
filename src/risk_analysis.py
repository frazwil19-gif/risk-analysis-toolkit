"""
Risk Analysis Toolkit
======================
Portfolio-level risk metrics: VaR, CVaR, drawdown, Sharpe, Sortino,
correlation analysis and stress-scenario testing.

Dataset: synthetic demo data — not real positions or market data.
Usage:   python src/risk_analysis.py
Outputs: reports/charts/*.png  |  reports/*.csv
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
REPORTS  = ROOT / "reports"
CHARTS   = REPORTS / "charts"

for _dir in (CHARTS,):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Portfolio configuration ───────────────────────────────────────────────────
TOTAL_NOTIONAL = 2_000_000  # £2M synthetic starting capital

# Signed weights: positive = long, negative = short
WEIGHTS: dict[str, float] = {
    "EURUSD":    +0.190,
    "GBPUSD":    -0.110,   # short position
    "USDJPY":    +0.090,
    "XAUUSD":    +0.175,
    "XAGUSD":    +0.060,
    "SP500":     +0.240,
    "FTSE100":   +0.080,
    "WTI_Crude": +0.055,
}

# Pre-defined stress scenarios: instrument → shocked period return
STRESS_SCENARIOS: dict[str, dict[str, float]] = {
    "COVID-19 Parallel (Mar 2020)": {
        "SP500": -0.35, "FTSE100": -0.32, "XAUUSD": +0.12,
        "XAGUSD": -0.15, "WTI_Crude": -0.40,
        "EURUSD": -0.03, "GBPUSD": -0.08, "USDJPY": +0.08,
    },
    "Rate Shock Parallel (2022)": {
        "SP500": -0.20, "FTSE100": -0.15, "XAUUSD": -0.08,
        "XAGUSD": -0.18, "WTI_Crude": +0.40,
        "EURUSD": -0.08, "GBPUSD": -0.12, "USDJPY": +0.10,
    },
    "USD Strength Shock": {
        "SP500": -0.04, "FTSE100": -0.05, "XAUUSD": -0.06,
        "XAGUSD": -0.09, "WTI_Crude": -0.05,
        "EURUSD": -0.05, "GBPUSD": -0.06, "USDJPY": +0.07,
    },
    "Flash Crash / Risk-Off Spike": {
        "SP500": -0.07, "FTSE100": -0.06, "XAUUSD": +0.05,
        "XAGUSD": +0.04, "WTI_Crude": -0.08,
        "EURUSD": -0.02, "GBPUSD": -0.03, "USDJPY": +0.04,
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_positions(path: Path) -> pd.DataFrame:
    """Load and validate portfolio positions file."""
    if not path.exists():
        log.error(f"Positions file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"instrument", "direction", "notional_gbp", "weight_pct"}
    missing = required - set(df.columns)
    if missing:
        log.error(f"Missing columns: {missing}")
        sys.exit(1)
    log.info(f"Loaded {len(df)} positions | Total notional: £{df['notional_gbp'].sum():,.0f}")
    return df


def load_returns(path: Path) -> pd.DataFrame:
    """Load daily instrument return series."""
    if not path.exists():
        log.error(f"Returns file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info(
        f"Loaded {len(df)} days of returns "
        f"({df['date'].min().strftime('%d %b %Y')} – {df['date'].max().strftime('%d %b %Y')})"
    )
    return df


# ── Portfolio return calculation ──────────────────────────────────────────────

def compute_portfolio_pnl(returns: pd.DataFrame) -> pd.Series:
    """Compute daily portfolio P&L (£) using signed notional weights."""
    instruments = list(WEIGHTS.keys())
    missing = set(instruments) - set(returns.columns)
    if missing:
        log.error(f"Missing return columns: {missing}")
        sys.exit(1)
    daily_pnl = sum(
        returns[inst] * weight * TOTAL_NOTIONAL
        for inst, weight in WEIGHTS.items()
    )
    return daily_pnl.rename("portfolio_pnl")


# ── Risk metrics ──────────────────────────────────────────────────────────────

def historical_var(daily_pnl: pd.Series, confidence: float) -> float:
    """1-day historical-simulation VaR at given confidence level."""
    return float(np.percentile(daily_pnl, (1 - confidence) * 100))


def conditional_var(daily_pnl: pd.Series, confidence: float) -> float:
    """Conditional VaR (Expected Shortfall) beyond the VaR threshold."""
    threshold = historical_var(daily_pnl, confidence)
    tail = daily_pnl[daily_pnl <= threshold]
    return float(tail.mean()) if len(tail) > 0 else threshold


def compute_risk_metrics(daily_pnl: pd.Series) -> dict:
    """Compute full suite of portfolio risk KPIs."""
    equity   = daily_pnl.cumsum() + TOTAL_NOTIONAL
    drawdown = equity - equity.cummax()

    ann_vol = daily_pnl.std() * np.sqrt(252) / TOTAL_NOTIONAL
    ann_ret = daily_pnl.mean() * 252 / TOTAL_NOTIONAL
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside_std = daily_pnl[daily_pnl < 0].std() * np.sqrt(252) / TOTAL_NOTIONAL
    sortino = ann_ret / downside_std if downside_std > 0 else 0.0

    var_95  = historical_var(daily_pnl, 0.95)
    var_99  = historical_var(daily_pnl, 0.99)
    cvar_95 = conditional_var(daily_pnl, 0.95)

    return {
        "Total Notional Exposure":       f"£{TOTAL_NOTIONAL:,.0f}",
        "Observation Days":              str(len(daily_pnl)),
        "1-Day VaR (95% confidence)": f"£{var_95:,.0f}",
        "1-Day VaR (99% confidence)": f"£{var_99:,.0f}",
        "CVaR / Expected Shortfall (95%)": f"£{cvar_95:,.0f}",
        "Annualised Volatility":         f"{ann_vol:.2%}",
        "Max 1-Day Loss":                f"£{daily_pnl.min():,.0f}",
        "Max 1-Day Gain":                f"£{daily_pnl.max():,.0f}",
        "Max Drawdown (£)":              f"£{drawdown.min():,.0f}",
        "Max Drawdown (%)": f"{drawdown.min() / TOTAL_NOTIONAL:.2%}",
        "Estimated Annualised Sharpe":   f"{sharpe:.2f}",
        "Estimated Sortino Ratio":       f"{sortino:.2f}",
        "Positive Return Days":          f"{(daily_pnl > 0).sum()} / {len(daily_pnl)}",
    }


def compute_stress_impacts() -> pd.DataFrame:
    """Apply pre-defined stress scenarios and compute portfolio P&L impact."""
    rows = []
    for scenario, shocks in STRESS_SCENARIOS.items():
        pnl_raw = sum(
            shocks.get(inst, 0.0) * WEIGHTS[inst] * TOTAL_NOTIONAL
            for inst in WEIGHTS
        )
        pct = pnl_raw / TOTAL_NOTIONAL
        rows.append({
            "Scenario":         scenario,
            "pnl_raw":          pnl_raw,
            "Portfolio_PnL":    f"£{pnl_raw:,.0f}",
            "Portfolio_Return": f"{pct:.2%}",
            "Severity": (
                "Critical" if pct < -0.15 else
                "Severe"   if pct < -0.08 else
                "Moderate" if pct < -0.03 else
                "Low"
            ),
        })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_pnl_distribution(daily_pnl: pd.Series):
    """Daily P&L histogram with 95% VaR marker."""
    var_95 = historical_var(daily_pnl, 0.95)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(daily_pnl.values, bins=18, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(var_95, color="#d62728", linewidth=1.8, linestyle="--",
               label=f"95% VaR: £{var_95:,.0f}")
    ax.axvline(float(daily_pnl.mean()), color="#2ca02c", linewidth=1.5, linestyle="--",
               label=f"Mean: £{daily_pnl.mean():,.0f}")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.set_title("Daily Portfolio P&L Distribution (Synthetic Demo Data)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Daily P&L (£)", labelpad=8)
    ax.set_ylabel("Frequency", labelpad=8)
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    fig.savefig(CHARTS / "pnl_distribution.png", dpi=150)
    plt.close()
    log.info("Saved pnl_distribution.png")


def chart_drawdown(daily_pnl: pd.Series):
    """Drawdown from rolling peak equity."""
    equity   = daily_pnl.cumsum() + TOTAL_NOTIONAL
    drawdown = equity - equity.cummax()
    x        = list(range(len(drawdown)))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(x, drawdown.values, 0, color="#d62728", alpha=0.35)
    ax.plot(x, drawdown.values, color="#d62728", linewidth=1.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.set_title("Portfolio Drawdown from Rolling Peak (Synthetic Demo Data)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Trading Day", labelpad=8)
    ax.set_ylabel("Drawdown (£)", labelpad=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "drawdown.png", dpi=150)
    plt.close()
    log.info("Saved drawdown.png")


def chart_correlation(returns: pd.DataFrame):
    """Pearson correlation heatmap across all instruments."""
    instruments = list(WEIGHTS.keys())
    corr = returns[instruments].corr().values

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_xticks(range(len(instruments)))
    ax.set_yticks(range(len(instruments)))
    ax.set_xticklabels(instruments, rotation=45, ha="right")
    ax.set_yticklabels(instruments)
    for i in range(len(instruments)):
        for j in range(len(instruments)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr[i, j]) > 0.6 else "black")
    ax.set_title("Instrument Return Correlation Matrix",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(CHARTS / "correlation_matrix.png", dpi=150)
    plt.close()
    log.info("Saved correlation_matrix.png")


def chart_stress_test(stress_df: pd.DataFrame):
    """Horizontal bar chart of stress scenario portfolio P&L impacts."""
    pnl_vals = stress_df["pnl_raw"].tolist()
    colors   = [
        "#d62728" if v < -100_000 else
        "#ff7f0e" if v < -30_000 else
        "#2ca02c"
        for v in pnl_vals
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(stress_df["Scenario"], pnl_vals,
                   color=colors, height=0.45, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.set_title("Stress Test Scenarios — Estimated Portfolio P&L Impact",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Estimated P&L Impact (£)", labelpad=8)
    for bar, val, ret in zip(bars, pnl_vals, stress_df["Portfolio_Return"]):
        offset = -4000 if val < 0 else 4000
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"£{val:,.0f}  ({ret})",
                va="center", ha="right" if val < 0 else "left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(CHARTS / "stress_scenarios.png", dpi=150)
    plt.close()
    log.info("Saved stress_scenarios.png")


def chart_exposure_breakdown(positions: pd.DataFrame):
    """Donut chart of gross notional exposure by instrument."""
    colors_list = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#17becf",
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        positions["notional_gbp"],
        labels=positions["instrument"],
        autopct="%1.1f%%",
        colors=colors_list[:len(positions)],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(8.5)
    ax.set_title(
        "Gross Notional Exposure by Instrument\n"
        "£2,000,000 Total | Synthetic Demo Data",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "exposure_breakdown.png", dpi=150)
    plt.close()
    log.info("Saved exposure_breakdown.png")


# ── Exports ───────────────────────────────────────────────────────────────────

def export_reports(metrics: dict, stress_df: pd.DataFrame):
    pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_csv(
        REPORTS / "risk_metrics.csv", index=False
    )
    log.info("Saved risk_metrics.csv")

    export_cols = ["Scenario", "Portfolio_PnL", "Portfolio_Return", "Severity"]
    stress_df[export_cols].to_csv(REPORTS / "stress_scenarios.csv", index=False)
    log.info("Saved stress_scenarios.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("── Risk Analysis Toolkit ────────────────────────")
    positions = load_positions(DATA_RAW / "portfolio_positions.csv")
    returns   = load_returns(DATA_RAW / "instrument_returns.csv")
    daily_pnl = compute_portfolio_pnl(returns)
    metrics   = compute_risk_metrics(daily_pnl)
    stress_df = compute_stress_impacts()

    print("\n" + "=" * 60)
    print("  PORTFOLIO RISK SUMMARY — SYNTHETIC DEMO DATA")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"  {metric:<38} {value}")
    print("=" * 60)

    print("\n  STRESS TEST RESULTS:")
    print("-" * 60)
    for _, row in stress_df.iterrows():
        print(f"  {row['Scenario']:<42} {row['Portfolio_PnL']}  ({row['Portfolio_Return']})  [{row['Severity']}]")
    print("=" * 60 + "\n")

    log.info("Generating charts…")
    chart_pnl_distribution(daily_pnl)
    chart_drawdown(daily_pnl)
    chart_correlation(returns)
    chart_stress_test(stress_df)
    chart_exposure_breakdown(positions)

    log.info("Exporting reports…")
    export_reports(metrics, stress_df)

    log.info("Done. All outputs saved to reports/")


if __name__ == "__main__":
    main()
