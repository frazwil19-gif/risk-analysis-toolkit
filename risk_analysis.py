import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load sample trade data

df = pd.read_csv("sample_trades.csv")

# Basic metrics

total_trades = len(df)
wins = df[df["pnl"] > 0]
losses = df[df["pnl"] < 0]

win_rate = len(wins) / total_trades * 100
avg_win = wins["pnl"].mean()
avg_loss = losses["pnl"].mean()
expectancy = df["pnl"].mean()

gross_profit = wins["pnl"].sum()
gross_loss = abs(losses["pnl"].sum())
profit_factor = gross_profit / gross_loss

# Equity curve

df["equity_curve"] = df["pnl"].cumsum()

# Drawdown

rolling_max = df["equity_curve"].cummax()
drawdown = df["equity_curve"] - rolling_max
max_drawdown = drawdown.min()

# Streak calculation

streaks = np.sign(df["pnl"])
longest_win_streak = 0
longest_loss_streak = 0
current_win = 0
current_loss = 0

for s in streaks:
    if s > 0:
        current_win += 1
        current_loss = 0
    elif s < 0:
        current_loss += 1
        current_win = 0

    longest_win_streak = max(longest_win_streak, current_win)
    longest_loss_streak = max(longest_loss_streak, current_loss)

# Print summary

print("===== RISK ANALYSIS SUMMARY =====")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Win: {avg_win:.2f}")
print(f"Average Loss: {avg_loss:.2f}")
print(f"Expectancy: {expectancy:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2f}")
print(f"Longest Win Streak: {longest_win_streak}")
print(f"Longest Loss Streak: {longest_loss_streak}")

# Save summary metrics

summary = pd.DataFrame({
    "Metric": [
        "Total Trades",
        "Win Rate",
        "Average Win",
        "Average Loss",
        "Expectancy",
        "Profit Factor",
        "Maximum Drawdown",
        "Longest Win Streak",
        "Longest Loss Streak"
    ],
    "Value": [
        total_trades,
        round(win_rate, 2),
        round(avg_win, 2),
        round(avg_loss, 2),
        round(expectancy, 2),
        round(profit_factor, 2),
        round(max_drawdown, 2),
        longest_win_streak,
        longest_loss_streak
    ]
})

summary.to_csv("outputs/summary_metrics.csv", index=False)

# Equity curve chart

plt.figure(figsize=(10, 5))
plt.plot(df["equity_curve"])
plt.title("Equity Curve")
plt.xlabel("Trade Number")
plt.ylabel("Cumulative PnL")
plt.grid(True)
plt.savefig("outputs/equity_curve.png")

# Drawdown chart

plt.figure(figsize=(10, 5))
plt.plot(drawdown)
plt.title("Drawdown")
plt.xlabel("Trade Number")
plt.ylabel("Drawdown")
plt.grid(True)
plt.savefig("outputs/drawdown_chart.png")

# Symbol-level performance

symbol_summary = df.groupby("symbol")["pnl"].sum().reset_index()
symbol_summary.to_csv("outputs/symbol_performance.csv", index=False)

# Strategy-level performance

strategy_summary = df.groupby("strategy")["pnl"].sum().reset_index()
strategy_summary.to_csv("outputs/strategy_performance.csv", index=False)

print("Outputs generated successfully.")
