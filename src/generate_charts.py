  """
Professional Risk Analytics Chart Suite
========================================
Generates 8 high-quality SVG charts for the risk-analysis-toolkit portfolio.
All data is synthetic (numpy.random.seed(42)) — not real positions or prices.
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = 'reports/charts'
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# Portfolio
INSTRUMENTS = ['SP500','EURUSD','GBPUSD','GOLD','BUND','CRUDE','NASDAQ','USDJPY']
NOTIONALS   = [450_000, 280_000, 180_000, 320_000, 250_000, 150_000, 220_000, 150_000]
DIRECTIONS  = [1, 1, -1, 1, 1, 1, 1, 1]
VOLS        = [0.010, 0.006, 0.006, 0.009, 0.004, 0.024, 0.011, 0.007]
DRIFTS      = [0.0003]*8
DAYS        = 30

returns = np.array([
    rng.normal(d, v, DAYS) for d, v in zip(DRFTS := DRIFTS, VOLS)
])
returns[2] = -returns[2]   # GBPUSD short

pnl_matrix = returns * np.array(NOTIONALS)[:, None]
portfolio_pnl = pnl_matrix.sum(axis=0)
cumulative   = portfolio_pnl.cumsum()
drawdown     = cumulative - cumulative.cummax() if hasattr(cumulative, 'cummax') else \
               np.array([cumulative[:i+1].max() for i in range(len(cumulative))])
drawdown     = cumulative - np.maximum.accumulate(cumulative)

var95  = np.percentile(portfolio_pnl, 5)
cvar95 = portfolio_pnl[portfolio_pnl <= var95].mean()
days   = np.arange(1, DAYS + 1)
weights = np.array(NOTIONALS) / sum(NOTIONALS)

# ── 1. VaR Distribution ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
n, bins, patches = ax.hist(portfolio_pnl, bins=15, color='#546e7a',
                           edgecolor='white', linewidth=0.8, alpha=0.85)
for patch, left in zip(patches, bins[:-1]):
    if left < var95:
        patch.set_facecolor('#c62828')
        patch.set_alpha(0.75)
ax.axvline(var95,  color='#c62828', linestyle='--', linewidth=1.6,
           label=f'95% VaR  {var95:,.0f}')
ax.axvline(cvar95, color='#e65100', linestyle='--', linewidth=1.6,
           label=f'CVaR (ES) {cvar95:,.0f}')
ax.axvline(portfolio_pnl.mean(), color='#2e7d32', linestyle='--', linewidth=1.4,
           label=f'Mean  £{portfolio_pnl.mean():,.0f}')
ax.set_xlabel('Daily P&L (£)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Daily P&L Distribution — 95% VaR & CVaR (Synthetic Demo)', fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
fig.tight_layout()
fig.savefig(f'{OUT}/var_distribution.svg', format='svg')
plt.close(fig)
print('  var_distribution.svg')

# ── 2. Drawdown ───────────────────────────────────────────────────────────────
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
ax.set_title('Portfolio Drawdown — 30-Day Window (Synthetic Demo)', fontweight='bold', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'£{y:,.0f}'))
ax.grid(axis='both', color='#cccccc', linewidth=0.6, alpha=0.35)
fig.tight_layout()
fig.savefig(f'{OUT}/drawdown_annotated.svg', format='svg')
plt.close(fig)
print('  drawdown_annotated.svg')

# ── 3. Correlation Heatmap ────────────────────────────────────────────────────
corr = np.corrcoef(pnl_matrix)
fig, ax = plt.subplots(figsize=(5.4, 4.4))
n_inst = len(INSTRUMENTS)

def corr_colour(v):
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
            edgecolor='white',
            facecolor=corr_colour(corr[i, j])
        )
        ax.add_patch(rect)
        tc = 'white' if abs(corr[i,j]) > 0.5 else '#222222'
        ax.text(j + 0.5, n_inst - 0.5 - i, f'{corr[i,j]:.2f}',
                ha='center', va='center', fontsize=7.5, color=tc)

ax.set_xlim(0, n_inst)
ax.set_ylim(0, n_inst)
ax.set_xticks([x + 0.5 for x in range(n_inst)])
ax.set_xticklabels(INSTRUMENTS, fontsize=8, rotation=45, ha='right')
ax.set_yticks([y + 0.5 for y in range(n_inst)])
ax.set_yticklabels(INSTRUMENTS[::-1], fontsize=8)
ax.set_title('Instrument Return Correlations (Synthetic Demo)', fontweight='bold', fontsize=10)
fig.tight_layout()
fig.savefig(f'{OUT}/correlation_heatmap.svg', format='svg')
plt.close(fig)
print('  correlation_heatmap.svg')

# ── 4. Stress Test ────────────────────────────────────────────────────────────
scenarios = {
    'COVID-19 Crash':  -218_600,
    'Rate Shock':      -111_600,
    'USD Strength':     -57_700,
    'Flash Crash':      -23_500,
}
labels = list(scenarios.keys())
values = list(scenarios.values())
colors = ['#b71c1c', '#c62828', '#e53935', '#ef5350']

fig, ax = plt.subplots(figsize=(8, 3.8), facecolor='white')
ax.set_facecolor('#fafafa')
bars = ax.barh(labels, [abs(v) for v in values], color=colors,
               edgecolor='white', linewidth=0.8, height=0.55)
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 3_000, bar.get_y() + bar.get_height()/2,
            f'£{val:,.0f}', va='center', fontsize=9, color='#212121')
ax.set_xlabel('Estimated Loss (£)', fontsize=10)
ax.set_title('Stress Test Scenarios — Estimated Portfolio P&L Impact (Synthetic)',
             fontweight='bold', fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{-x:,.0f}'))
ax.invert_yaxis()
ax.grid(axis='x', color='#cccccc', linewidth=0.6, alpha=0.4)
fig.tight_layout()
fig.savefig(f'{OUT}/stress_test.svg', format='svg')
plt.close(fig)
print('  stress_test.svg')

# ── 5. Exposure Breakdown ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5),
                                gridspec_kw={'width_ratios': [1.35, 1]},
                                facecolor='white')
fig.suptitle('Portfolio Exposure & Composition — £2,000,000 (Synthetic Demo)',
             fontweight='bold', fontsize=12)

cols = ['#1565c0' if d == 1 else '#c62828' for d in DIRECTIONS]
ax1.set_facecolor('#fafafa')
bars1 = ax1.barh(INSTRUMENTS, NOTIONALS, color=cols,
                 edgecolor='white', linewidth=0.8, height=0.55)
for b, n, d in zip(bars1, NOTIONALS, DIRECTIONS):
    label = f'£{n:,}  [{"Long" if d==1 else "Short"}]'
    ax1.text(b.get_width() + 4_000, b.get_y() + b.get_height()/2,
             label, va='center', fontsize=8, color='#212121')
ax1.set_xlabel('Notional Exposure (£)', fontsize=10)
ax1.set_title('Notional by Instrument', fontweight='bold', fontsize=12)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x/1000:.0f}k'))
ax1.grid(axis='x', color='#cccccc', linewidth=0.6, alpha=0.35)
leg = [mpatches.Patch(color='#1565c0', label='Long'),
       mpatches.Patch(color='#c62828', label='Short')]
ax1.legend(handles=leg, fontsize=8, loc='lower right')
ax1.text(0.01, -0.09, f'Total Portfolio: £2,000,000',
         transform=ax1.transAxes, fontsize=9, color='#546e7a', style='italic')

PIE_COLORS = ['#1565c0','#2e7d32','#c62828','#e65100','#6a1b9a','#546e7a','#00838f','#558b2f']
ax2.pie(weights, labels=INSTRUMENTS, autopct='%1.1f%%',
        colors=PIE_COLORS, startangle=90,
        textprops={'fontsize': 8},
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'})
ax2.set_title('Portfolio Weight (%)', fontweight='bold', fontsize=12)

fig.tight_layout()
fig.savefig(f'{OUT}/exposure_breakdown.svg', format='svg')
plt.close(fig)
print('  exposure_breakdown.svg')

# ── 6. Risk Contribution ──────────────────────────────────────────────────────
pnl_std = pnl_matrix.std(axis=1)
sorted_idx = np.argsort(pnl_std)[::-1]

max_std = pnl_std.max()
def risk_color(v):
    t = v / max_std
    r = int(198 * t + 21 * (1-t))
    g = int(40  * t + 101* (1-t))
    b = int(40  * t + 58 * (1-t))
    return f'#{r:02x}{g:02x}{b:02x}'

fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
ax.set_facecolor('#fafafa')
sorted_instr = [INSTRUMENTS[i] for i in sorted_idx]
sorted_std   = [pnl_std[i]     for i in sorted_idx]
bars = ax.bar(sorted_instr, sorted_std,
              color=[risk_color(v) for v in sorted_std],
              edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, sorted_std):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 12,
            f'£{val:,.0f}', ha='center', va='bottom', fontsize=8.5, color='#212121')
ax.set_ylabel('Daily P&L Std Dev (£)', fontsize=10)
ax.set_title('Volatility Contribution by Instrument (Synthetic Demo)',
             fontweight='bold', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'£{y:,.0f}'))
ax.grid(axis='y', color='#cccccc', linewidth=0.6, alpha=0.4)
fig.tight_layout()
fig.savefig(f'{OUT}/risk_contribution.svg', format='svg')
plt.close(fig)
print('  risk_contribution.svg')

# ── 7. Return Stats ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 6), facecolor='white')
fig.suptitle('Return Distributions by Instrument (Synthetic Demo)',
             fontweight='bold', fontsize=12)
INST_COLORS = ['#1565c0','#2e7d32','#c62828','#e65100',
               '#6a1b9a','#546e7a','#00838f','#558b2f']
raw_returns = np.array([
    rng.normal(d, v, DAYS) for d, v in zip(DRIFTS, VOLS)
])  # un-signed for display
for ax, inst, col, vol, raw in zip(axes.flat, INSTRUMENTS, INST_COLORS, VOLS, raw_returns):
    ax.set_facecolor('#fafafa')
    ax.hist(raw, bins=10, color=col, edgecolor='white', linewidth=0.6, alpha=0.85)
    ax.set_title(inst, fontsize=9, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    ann_vol = vol * np.sqrt(252) * 100
    ax.text(0.97, 0.92, f'σ≈{ann_vol:.1f}%', transform=ax.transAxes,
            fontsize=7.5, ha='right', va='top', color='#212121',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(f'{OUT}/return_stats.svg', format='svg')
plt.close(fig)
print('  return_stats.svg')

# ── 8. KPI Risk Panel ─────────────────────────────────────────────────────────
KPIS = [
    ('Portfolio Size',     '£2,000,000',  '#1565c0'),
    ('95% VaR (daily)',    f'£{var95:,.0f}',   '#c62828'),
    ('CVaR (95%)',         f'£{cvar95:,.0f}',  '#e65100'),
    ('Max Drawdown',       f'£{drawdown.min():,.0f}', '#c62828'),
    ('Portfolio Vol (daily)', f'{portfolio_pnl.std()/sum(NOTIONALS)*100:.2f}%', '#6a1b9a'),
    ('Instruments',        '8',           '#2e7d32'),
]

fig, axes = plt.subplots(2, 3, figsize=(10.9, 3.9))
fig.suptitle('Risk Dashboard — Portfolio Overview (Synthetic Demo)',
             fontweight='bold', fontsize=13)
for ax, (label, value, color) in zip(axes.flat, KPIS):
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)
    header = mpatches.FancyBboxPatch(
        (0, 0.78), 1, 0.22,
        boxstyle='square,pad=0',
        transform=ax.transAxes,
        facecolor=color, alpha=0.85, clip_on=False, zorder=2
    )
    ax.add_patch(header)
    ax.text(0.5, 0.895, label,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#546e7a', zorder=3)
    ax.text(0.5, 0.42, value,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=13, fontweight='bold', color=color)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(f'{OUT}/kpi_risk_panel.svg', format='svg')
plt.close(fig)
print('  kpi_risk_panel.svg')

print('\nAll 8 charts saved to', OUT)
