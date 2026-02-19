import pandas as pd
import numpy as np


def run_backtest(
    df,
    holding_days_list=(1, 3, 5, 10),
    risk_per_trade=0.01,
    cost_per_trade=0.001
):
    df = df.copy()
    close = df["Close"].squeeze()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    zscore = (close - ma20) / ma20
    trend_distance = abs(close - ma50) / ma50

    df["zscore"] = zscore
    df["is_lateral"] = trend_distance < 0.01

    df = df.dropna()
    close = close.loc[df.index]  # sincronizzazione sicura

    results = {}

    for holding_days in holding_days_list:

        equity = 1.0
        equity_curve = []
        trades = []

        for i in range(50, len(df) - holding_days):

            if not bool(df["is_lateral"].iloc[i]):
                equity_curve.append(equity)
                continue

            entry_price = float(close.iloc[i])
            exit_price = float(close.iloc[i + holding_days])
            z = float(df["zscore"].iloc[i])

            if z > 0.01:
                ret = (entry_price - exit_price) / entry_price
            elif z < -0.01:
                ret = (exit_price - entry_price) / entry_price
            else:
                equity_curve.append(equity)
                continue

            ret -= cost_per_trade
            equity *= (1 + risk_per_trade * ret)

            equity_curve.append(equity)
            trades.append(ret)

        equity_series = pd.Series(equity_curve)

        if len(equity_series) > 0:
            total_return = equity_series.iloc[-1] - 1
            max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
        else:
            total_return = 0
            max_drawdown = 0

        win_rate = float(np.mean([t > 0 for t in trades])) if trades else 0

        results[str(holding_days)] = {
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "num_trades": int(len(trades)),
            "avg_holding_days": float(holding_days),
        }

    return results


def walkforward_mean_reversion(
    df,
    start_year=2012,
    end_year=2024,
    holding_days=3,
    risk_per_trade=0.01,
    cost_per_trade=0.001
):
    df = df.copy()
    close = df["Close"].squeeze()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    zscore = (close - ma20) / ma20
    trend_distance = abs(close - ma50) / ma50

    df["zscore"] = zscore
    df["is_lateral"] = trend_distance < 0.01
    df["year"] = df.index.year

    df = df.dropna()
    close = close.loc[df.index]  # sincronizzazione robusta

    yearly_results = []
    equity = 1.0

    for year in range(start_year, end_year + 1):

        mask = df["year"] == year
        if mask.sum() == 0:
            continue

        year_df = df[mask]
        year_close = close[mask]

        year_start_equity = equity
        trades = []

        for i in range(len(year_df) - holding_days):

            if not bool(year_df["is_lateral"].iloc[i]):
                continue

            entry_price = float(year_close.iloc[i])
            exit_price = float(year_close.iloc[i + holding_days])
            z = float(year_df["zscore"].iloc[i])

            if z > 0.01:
                ret = (entry_price - exit_price) / entry_price
            elif z < -0.01:
                ret = (exit_price - entry_price) / entry_price
            else:
                continue

            ret -= cost_per_trade
            equity *= (1 + risk_per_trade * ret)
            trades.append(ret)

        year_return = equity - year_start_equity

        yearly_results.append({
            "year": year,
            "return": float(year_return),
            "num_trades": int(len(trades)),
            "win_rate": float(sum(t > 0 for t in trades) / len(trades)) if trades else 0
        })

    return yearly_results, equity
