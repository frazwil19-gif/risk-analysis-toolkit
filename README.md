# Risk Analysis Toolkit

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Type](https://img.shields.io/badge/Type-Risk_Analytics-orange)
![Status](https://img.shields.io/badge/Status-Research_Project-green)

A quantitative risk-analysis and stress-testing project demonstrating portfolio risk measurement, scenario analysis, and reporting workflows using Python.

> This repository uses synthetic portfolio and return data for research and portfolio demonstration purposes only.

---

## Project Objective

This project demonstrates a structured Python workflow for generating:

- downside-risk analysis
- drawdown analysis
- stress testing
- correlation analysis
- exposure monitoring
- volatility contribution analysis

The emphasis is on transparent methodology and clearly labelled assumptions.

---

## Portfolio Overview

**Portfolio size:** £2,000,000  
**Instruments:** 8 synthetic positions across equities, FX, commodities, and fixed income.

| Instrument | Asset Class | Direction | Notional |
|---|---|---|---|
| SP500 | Equities | Long | £450,000 |
| EURUSD | FX | Long | £280,000 |
| GBPUSD | FX | Short | £180,000 |
| GOLD | Commodities | Long | £320,000 |
| BUND | Fixed Income | Long | £250,000 |
| CRUDE | Commodities | Long | £150,000 |
| NASDAQ | Equities | Long | £220,000 |
| USDJPY | FX | Long | £150,000 |

---

## Visual Outputs

### VaR Distribution
![VaR Distribution](reports/charts/var_distribution.svg)

### Drawdown Analysis
![Drawdown](reports/charts/drawdown_annotated.svg)

### Correlation Heatmap
![Correlations](reports/charts/correlation_heatmap.svg)

### Stress Test Scenarios
![Stress Test](reports/charts/stress_test.svg)

### Exposure Breakdown
![Exposure](reports/charts/exposure_breakdown.svg)

### Volatility Contribution
![Risk Contribution](reports/charts/risk_contribution.svg)

### Instrument Return Distributions
![Return Stats](reports/charts/return_stats.svg)

### Risk KPI Panel
![KPI Panel](reports/charts/kpi_risk_panel.svg)

---

## Methodology

### Historical Simulation VaR

```text
VaR(95%) = percentile(portfolio_pnl, 5)
```

### Conditional VaR

```text
CVaR(95%) = mean(pnl[pnl <= VaR])
```

### Portfolio P&L

```text
portfolio_pnl = Σ (signed_return × notional)
```

---

## Repository Structure

```text
risk-analysis-toolkit/
├── src/
│   └── risk_analysis.py
├── data/
│   └── raw/
├── reports/
│   ├── charts/
│   └── risk_report.md
└── README.md
```

---

## Usage

```bash
pip install matplotlib numpy pandas
python src/risk_analysis.py
```

---

## Limitations

- All returns are synthetic.
- Real markets exhibit fat tails and regime changes.
- 30 trading days is insufficient for institutional-grade inference.
- No funding costs or slippage are modelled.

---

## Relevance

This repository demonstrates:

- financial risk analysis
- quantitative reasoning
- portfolio analytics
- Python-based reporting workflows
- communication of assumptions and limitations

---

## Disclaimer

This repository is for research and portfolio purposes only. It is not financial advice and does not represent live investment performance.
