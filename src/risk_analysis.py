  """
Risk Analysis Toolkit — Professional Edition
==============================================
Generates a full suite of risk analytics charts and a markdown risk report
for a synthetic 8-instrument, £2M portfolio.

All data is synthetically generated (numpy.random.seed=42) for demonstration
purposes. This is not real position or market data.

Usage:
    python src/risk_analysis.py

Outputs:
    reports/charts/*.svg   (8 SVG charts)
    reports/risk_report.md (markdown risk report)
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Configuration ────────────────────────────────────────────────────────────
OUT_DIR  = 'reports/charts'
REPORT   = 'reports/risk_report.md'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('reports', exist_ok=True)

INSTRUMENTS = ['SP500', 'EURUSD', 'GBPUSD', 'GOLD', 'BUND', 'CRUDE', 'NASDAQ', 'USDJPY']
NOTIONALS   = [450_000, 280_000, 180_000, 320_000, 250_000, 150_000, 220_000, 150_000]
DIRECTIONS  = [1, 1, -1, 1, 1, 1, 1, 1]          # -1 = Short
VOLS        = [0.010, 0.006, 0.006, 0.009, 0.004, 0.024, 0.011, 0.007]
DRIFTS      = [0.0003] * 8
DAYS        = 30

rng = np.random.default_rng(42)

# ── Generate synthetic returns ────────────────────────────────────────────────
raw_returns = np.array([
    rng.normal(drift, vol, DAYS)
    for drift, vol in zip(DRIFTS, VOLS)
])

# Apply direction (short positions negate return)
signed_returns = raw_returns * np.array(DIRECTIONS)[:, None]

# P&L matrix: instrument × day
pnl_matrix = signed_returns * np.array(NOTIONALS)[:, None]

# Portfolio daily P&L
portfolio_pnl = pnl_matrix.sum(axis=0)

# Cumulative P&L and drawdown
cumulative = np.cumsum(portfolio_pnl)
drawdown   = cumulative - np.maximum.accumulate(cumulative)

# Risk metrics
var95        = np.percentile(portfolio_pnl, 5)
var99        = np.percentile(portfolio_pnl, 1)
cvar95       = portfolio_pnl[portfolio_pnl <= var95].mean()
daily_vol    = portfolio_pnl.std()
daily_vol_pct = daily_vol / sum(NOTIONALS)
ann_vol      = daily_vol_pct * np.sqrt(252)
max_dd       = drawdown.min()
max_dd_pct   = max_dd / sum(NOTIONALS)
max_loss     = portfolio_pnl.min()
max_gain     = portfolio_pnl.max()
positive_days= (portfolio_pnl > 0).sum()
sharpe_est   = (portfolio_pnl.mean() * 252) / (daily_vol * np.sqrt(252)) if daily_vol > 0 else 0

days = np.arange(1, DAYS + 1)
weights = np.array(NOTIONALS) / sum(NOTIONALS)
pnl_std = pnl_matrix.std(axis=1)

# ── Stress scenarios ──────────────────────────────────────────────────────────
stress_shocks = {
    'COVID-19 Crash': {
        'SP500': -0.32, 'EURUSD': -0.03, 'GBPUSD': -0.04,
        'GOLD': +0.12, 'BUND': +0.04, 'CRUDE': -0.40,
        'NASDAQ': -0.35, 'USDJPY': -0.06
    },
    'Rate Shock': {
        'SP500': -0.18, 'EURUSD': +0.02, 'GBPUSD': +0.01,
        'GOLD': -0.08, 'BUND': -0.07, 'CRUDE': +0.40,
        'NASDAQ': -0.20, 'USDJPY': +0.05
    },
    'USD Strength': {
        'SP500': -0.04, 'EURUSD': -0.06, 'GBPUSD': -0.05,
        'GOLD': -0.03, 'BUND': +0.01, 'CRUDE': -0.05,
        'NASDAQ': -0.03, 'USDJPY': +0.07
    },
    'Flash Crash': {
        'SP500': -0.07, 'EURUSD': -0.01, 'GBPUSD': -0.01,
        'GOLD': +0.02, 'BUND': +0.02, 'CRUDE': -0.04,
        'NASDAQ': -0.06, 'USDJPY': 0.00
    },
}

stress_pnl = {}
for scenario, shocks in stress_shocks.items():
    total = sum(
        shocks[inst] * DIRECTIONS[i] * NOTIONALS[i]
        for i, inst in enumerate(INSTRUMENTS)
    )
    stress_pnl[scenario] = total


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — VaR Distribution
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.5))
n, bins, patches_ = ax.hist(
    portfolio_pnl, bins=15, color='#546e7a',
    edgecolor='white', linewidth=0.8, alpha=0.85
)
for patch, left in zip(patches_, bins[:-1]):
    if left < var95:
        patch.set_facecolor('#c62828')
        patch.set_alpha(0.75)
ax.axvline(var95,  color='#c62828', linestyle='--', linewidth=1.6,
           label=f'95% VaR  £{var95:,.0f}')
ax.axvline(cvar95, color='#e65100', linestyle='--', linewidth=1.6,
           label=f'CVaR (ES) £{cvar95:,.0f}')
ax.axvline(portfolio_pnl.mean(), color='#2e7d32', linestyle='--', linewidth=1.4,
           label=f'Mean  £{portfolio_pnl.mean():,.0f}')
ax.set_xlabel('Daily P&L (£)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Daily P&L Distribution — 95% VaR & CVaR (Synthetic Demo)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/var_distribution.svg', format='svg')
plt.close(fig)
print('  [1/8] var_distribution.svg')


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Drawdown
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')
ax.set_facecolor('#fafafa')
ax.fill_between(days, drawdown, 0, color='#c62828', alpha=0.35)
ax.plot(days, drawdown, color='#c62828', linewidth=1.5)
ax.axhline(0, color='#546e7a', linewidth=0.8, linestyle='--', alpha=0.6)
worst_day = int(np.argmin(drawdown))
ax.annotate(
    f'Worst: £{drawdown[worst_day]:,.0f}',
    xy=(days[worst_day], drawdown[worst_day]),
    xytext=(days[worst_day] + 4, drawdown[worst_day] * 0.65),
    arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.4),
    fontsize=9, fontweight='bold', color='#c62828'
)
ax.set_xlabel('Trading Day', fontsize=10)
ax.set_ylabel('Drawdown (£)', fontsize=10)
ax.set_title('Portfolio Drawdown — 30-Day Window (Synthetic Demo)',
             fontweight='bold', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'£{y:,.0f}'))
ax.grid(axis='both', color='#cccccc', linewidth=0.6, alpha=0.35)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/drawdown_annotated.svg', format='svg')
plt.close(fig)
print('  [2/8] drawdown_annotated.svg')


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
corr = np.corrcoef(pnl_matrix)
fig, ax = plt.subplots(figsize=(5.4, 4.4))
n_inst = len(INSTRUMENTS)

def corr_colour(v: float) -> str:
    if   v >  0.5: return '#1a6e2f'
    elif v >  0.2: return '#a8d5a2'
    elif v > -0.2: return '#fffde7'
    elif v > -0.5: return '#ffab91'
    else:          return '#b71c1c'

for i in range(n_inst):
    for j in range(n_inst):
        rect = mpatches.FancyBboxPatch(
            (j, n_inst - 1 - i), 1, 1,
            boxstyle='square,pad=0', linewidth=0.8,
            edgecolor='white', facecolor=corr_colour(corr[i, j])
        )
        ax.add_patch(rect)
        tc = 'white' if abs(corr[i, j]) > 0.5 else '#222222'
        ax.text(j + 0.5, n_inst - 0.5 - i, f'{corr[i, j]:.2f}',
                ha='center', va='center', fontsize=7.5, color=tc)

ax.set_xlim(0, n_inst)
ax.set_ylim(0, n_inst)
ax.set_xticks([x + 0.5 for x in range(n_inst)])
ax.set_xticklabels(INSTRUMENTS, fontsize=8, rotation=45, ha='right')
ax.set_yticks([y + 0.5 for y in range(n_inst)])
ax.set_yticklabels(INSTRUMENTS[::-1], fontsize=8)
ax.set_title('Instrument Return Correlations (Synthetic Demo)',
             fontweight='bold', fontsize=10)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/correlation_heatmap.svg', format='svg')
plt.close(fig)
print('  [3/8] correlation_heatmap.svg')


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Stress Test
# ═══════════════════════════════════════════════════════════════════════════════
scenario_labels = list(stress_pnl.keys())
scenario_values = list(stress_pnl.values())
stress_colors   = ['#b71c1c', '#c62828', '#e53935', '#ef5350']

fig, ax = plt.subplots(figsize=(8, 3.8), facecolor='white')
ax.set_facecolor('#fafafa')
bars = ax.barh(
    scenario_labels,
    [abs(v) for v in scenario_values],
    color=stress_colors, edgecolor='white', linewidth=0.8, height=0.55
)
for bar, val in zip(bars, scenario_values):
    ax.text(
        bar.get_width() + 3_000,
        bar.get_y() + bar.get_height() / 2,
        f'£{val:,.0f}', va='center', fontsize=9, color='#212121'
    )
ax.set_xlabel('Estimated Loss (£)', fontsize=10)
ax.set_title(
    'Stress Test Scenarios — Estimated Portfolio P&L Impact (Synthetic)',
    fontweight='bold', fontsize=11
)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{-x:,.0f}'))
ax.invert_yaxis()
ax.grid(axis='x', color='#cccccc', linewidth=0.6, alpha=0.4)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/stress_test.svg', format='svg')
plt.close(fig)
print('  [4/8] stress_test.svg')


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Exposure Breakdown
# ═══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11.8, 5),
    gridspec_kw={'width_ratios': [1.35, 1]},
    facecolor='white'
)
fig.suptitle(
    'Portfolio Exposure & Composition — £2,000,000 (Synthetic Demo)',
    fontweight='bold', fontsize=12
)

bar_colors = ['#1565c0' if d == 1 else '#c62828' for d in DIRECTIONS]
ax1.set_facecolor('#fafafa')
bars1 = ax1.barh(
    INSTRUMENTS, NOTIONALS,
    color=bar_colors, edgecolor='white', linewidth=0.8, height=0.55
)
for b, n_, d in zip(bars1, NOTIONALS, DIRECTIONS):
    lbl = f'£{n_:,}  [{"Long" if d == 1 else "Short"}]'
    ax1.text(
        b.get_width() + 4_000,
        b.get_y() + b.get_height() / 2,
        lbl, va='center', fontsize=8, color='#212121'
    )
ax1.set_xlabel('Notional Exposure (£)', fontsize=10)
ax1.set_title('Notional by Instrument', fontweight='bold', fontsize=12)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x/1000:.0f}k'))
ax1.grid(axis='x', color='#cccccc', linewidth=0.6, alpha=0.35)
leg_handles = [
    mpatches.Patch(color='#1565c0', label='Long'),
    mpatches.Patch(color='#c62828', label='Short'),
]
ax1.legend(handles=leg_handles, fontsize=8, loc='lower right')
ax1.text(
    0.01, -0.09, 'Total Portfolio: £2,000,000',
    transform=ax1.transAxes, fontsize=9, color='#546e7a', style='italic'
)

PIE_COLORS = [
    '#1565c0', '#2e7d32', '#c62828', '#e65100',
    '#6a1b9a', '#546e7a', '#00838f', '#558b2f'
]
ax2.pie(
    weights, labels=INSTRUMENTS, autopct='%1.1f%%',
    colors=PIE_COLORS, startangle=90,
    textprops={'fontsize': 8},
    wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
)
ax2.set_title('Portfolio Weight (%)', fontweight='bold', fontsize=12)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/exposure_breakdown.svg', format='svg')
plt.close(fig)
print('  [5/8] exposure_breakdown.svg')


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Risk Contribution
# ═══════════════════════════════════════════════════════════════════════════════
sorted_idx = np.argsort(pnl_std)[::-1]
max_std    = pnl_std.max()

def risk_colour(v: float) -> str:
    t = v / max_std
    r = int(198 * t + 21  * (1 - t))
    g = int(40  * t + 101 * (1 - t))
    b = int(40  * t + 58  * (1 - t))
    return f'#{r:02x}{g:02x}{b:02x}'

fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
ax.set_facecolor('#fafafa')
sorted_instr = [INSTRUMENTS[i] for i in sorted_idx]
sorted_std   = [pnl_std[i]     for i in sorted_idx]
bars = ax.bar(
    sorted_instr, sorted_std,
    color=[risk_colour(v) for v in sorted_std],
    edgecolor='white', linewidth=0.8
)
for bar, val in zip(bars, sorted_std):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 12,
        f'£{val:,.0f}',
        ha='center', va='bottom', fontsize=8.5, color='#212121'
    )
ax.set_ylabel('Daily P&L Std Dev (£)', fontsize=10)
ax.set_title('Volatility Contribution by Instrument (Synthetic Demo)',
             fontweight='bold', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'£{y:,.0f}'))
ax.grid(axis='y', color='#cccccc', linewidth=0.6, alpha=0.4)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/risk_contribution.svg', format='svg')
plt.close(fig)
print('  [6/8] risk_contribution.svg')


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Return Stats
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(14, 6), facecolor='white')
fig.suptitle('Return Distributions by Instrument (Synthetic Demo)',
             fontweight='bold', fontsize=12)
INST_COLORS = [
    '#1565c0', '#2e7d32', '#c62828', '#e65100',
    '#6a1b9a', '#546e7a', '#00838f', '#558b2f'
]
for ax_, inst, col, vol, raw_ret in zip(
        axes.flat, INSTRUMENTS, INST_COLORS, VOLS, raw_returns
):
    ax_.set_facecolor('#fafafa')
    ax_.hist(raw_ret, bins=10, color=col, edgecolor='white', linewidth=0.6, alpha=0.85)
    ax_.set_title(inst, fontsize=9, fontweight='bold')
    ax_.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    ann_vol_i = vol * np.sqrt(252) * 100
    ax_.text(
        0.97, 0.92, f'σ≈{ann_vol_i:.1f}%',
        transform=ax_.transAxes, fontsize=7.5, ha='right', va='top',
        color='#212121', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
    )
    ax_.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/return_stats.svg', format='svg')
plt.close(fig)
print('  [7/8] return_stats.svg')


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8 — KPI Risk Panel
# ═══════════════════════════════════════════════════════════════════════════════
KPIS = [
    ('Portfolio Size',        '£2,000,000',                        '#1565c0'),
    ('95% VaR (daily)',       f'£{var95:,.0f}',                    '#c62828'),
    ('CVaR (95%)',            f'£{cvar95:,.0f}',                   '#e65100'),
    ('Max Drawdown',          f'£{max_dd:,.0f}',                   '#c62828'),
    ('Portfolio Vol (daily)', f'{daily_vol_pct * 100:.2f}%',       '#6a1b9a'),
    ('Instruments',           '8',                                 '#2e7d32'),
]

fig, axes = plt.subplots(2, 3, figsize=(10.9, 3.9))
fig.suptitle('Risk Dashboard — Portfolio Overview (Synthetic Demo)',
             fontweight='bold', fontsize=13)
for ax_, (label, value, color) in zip(axes.flat, KPIS):
    ax_.set_facecolor('white')
    for spine in ax_.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)
    header = mpatches.FancyBboxPatch(
        (0, 0.78), 1, 0.22,
        boxstyle='square,pad=0',
        transform=ax_.transAxes,
        facecolor=color, alpha=0.85, clip_on=False, zorder=2
    )
    ax_.add_patch(header)
    ax_.text(
        0.5, 0.895, label,
        transform=ax_.transAxes, ha='center', va='center',
        fontsize=8.5, fontweight='bold', color='#546e7a', zorder=3
    )
    ax_.text(
        0.5, 0.42, value,
        transform=ax_.transAxes, ha='center', va='center',
        fontsize=13, fontweight='bold', color=color
    )
    ax_.set_xticks([])
    ax_.set_yticks([])
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/kpi_risk_panel.svg', format='svg')
plt.close(fig)
print('  [8/8] kpi_risk_panel.svg')


# ═══════════════════════════════════════════════════════════════════════════════
# RISK REPORT
# ═══════════════════════════════════════════════════════════════════════════════
report_lines = [
    '# Risk Report — Synthetic Portfolio',
    '> **Data note**: All positions and returns are synthetically generated (numpy seed=42).',
    '> This report is for demonstration purposes only — not real positions or market data.',
    '',
    '---',
    '',
    '## Executive Summary',
    '',
    f'This report covers a synthetic £2,000,000 multi-asset portfolio comprising 8 instruments',
    f'across equities (SP500, NASDAQ), FX (EURUSD, GBPUSD short, USDJPY), commodities',
    f'(GOLD, CRUDE), and fixed income (BUND). Analysis is based on {DAYS} days of synthetic',
    f'daily returns generated with `numpy.random.seed(42)`.',
    '',
    f'The portfolio carries a 1-day 95% VaR of **£{var95:,.0f}** and a CVaR of **£{cvar95:,.0f}**,',
    f'representing {abs(var95)/sum(NOTIONALS)*100:.2f}% and {abs(cvar95)/sum(NOTIONALS)*100:.2f}% of total notional respectively.',
    f'The largest single-day stress loss scenario (COVID-19 Crash parallel) is estimated at',
    f'**£{abs(stress_pnl["COVID-19 Crash"]):,.0f} ({stress_pnl["COVID-19 Crash"]/sum(NOTIONALS)*100:.1f}% of portfolio)**.',
    '',
    '---',
    '',
    '## Portfolio Overview',
    '',
    '| Instrument | Asset Class | Direction | Notional (£) | Weight |',
    '|---|---|---|---|---|',
]
assет_classes = {
    'SP500': 'Equities', 'EURUSD': 'FX', 'GBPUSD': 'FX',
    'GOLD': 'Commodities', 'BUND': 'Fixed Income', 'CRUDE': 'Commodities',
    'NASDAQ': 'Equities', 'USDJPY': 'FX'
}
for inst, notional, direction in zip(INSTRUMENTS, NOTIONALS, DIRECTIONS):
    dir_str = 'Long' if direction == 1 else '**Short**'
    wt = notional / sum(NOTIONALS) * 100
    report_lines.append(
        f'| {inst} | {assет_classes[inst]} | {dir_str} | £{notional:,} | {wt:.1f}% |'
    )
report_lines += [
    f'| **TOTAL** | | | **£{sum(NOTIONALS):,}** | **100.0%** |',
    '',
    '---',
    '',
    '## Key Risk Metrics',
    '',
    '| Metric | Value |',
    '|---|---|',
    f'| Total Notional Exposure | £{sum(NOTIONALS):,} |',
    f'| Observation Days | {DAYS} |',
    f'| 1-Day VaR (95% confidence) | £{var95:,.0f} |',
    f'| 1-Day VaR (99% confidence) | £{var99:,.0f} |',
    f'| CVaR / Expected Shortfall (95%) | £{cvar95:,.0f} |',
    f'| Daily Portfolio Volatility (£) | £{daily_vol:,.0f} |',
    f'| Daily Portfolio Volatility (%) | {daily_vol_pct*100:.3f}% |',
    f'| Annualised Volatility | {ann_vol*100:.2f}% |',
    f'| Max Drawdown (30-day window) | £{max_dd:,.0f} ({max_dd_pct*100:.2f}%) |',
    f'| Max Single-Day Loss | £{max_loss:,.0f} |',
    f'| Max Single-Day Gain | +£{max_gain:,.0f} |',
    f'| Estimated Annualised Sharpe | {sharpe_est:.2f} |',
    f'| Positive Return Days | {positive_days} / {DAYS} |',
    '',
    '> Note: Exact values are computed from synthetic data at runtime by `src/risk_analysis.py`.',
    '> The numbers above are representative outputs from seed=42.',
    '',
    '---',
    '',
    '## Stress Test Results',
    '',
    '| Scenario | Estimated P&L Impact | % of Portfolio |',
    '|---|---|---|',
]
for scenario, pnl_val in stress_pnl.items():
    report_lines.append(
        f'| {scenario} | £{pnl_val:,.0f} | {pnl_val/sum(NOTIONALS)*100:.2f}% |'
    )
report_lines += [
    '',
    '> Stress scenarios are illustrative point-in-time shocks calibrated to historical analogues.',
    '> They are **not** probability-weighted estimates.',
    '',
    '---',
    '',
    '*Generated by risk-analysis-toolkit | Synthetic demo data | Not investment advice*',
]

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f'  Risk report → {REPORT}')

print('\n✓ All outputs generated successfully.')
