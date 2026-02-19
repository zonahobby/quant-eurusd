import pandas as pd
import numpy as np


def run_backtest(
    df,
    risk_per_trade=0.01,
    cost_per_trade=0.001
):
    df = df.copy()
    close = df["Close"].squeeze()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    vol10 = close.pct_change().rolling(10).std()

    zscore = (close - ma20) / ma20
    trend_distance = abs(close - ma50) / ma50

    df["zscore"] = zscore
    df["is_lateral"] = trend_distance < 0.01
    df["low_vol"] = vol10 < vol10.median()

    df = df.dropna()
    close = close.loc[df.index]

    equity = 1.0
    equity_curve = []
    trades = []

    i = 50
    while i < len(df) - 1:

        if not bool(df["is_lateral"].iloc[i]) or not bool(df["low_vol"].iloc[i]):
            equity_curve.append(equity)
            i += 1
            continue

        z = float(df["zscore"].iloc[i])

        # ---- ENTRY ----
        if z < -0.01:
            direction = 1   # BUY
        elif z > 0.01:
            direction = -1  # SELL
        else:
            equity_curve.append(equity)
            i += 1
            continue

        entry_price = float(close.iloc[i])

        # ---- EXIT: ritorno verso media ----
        j = i + 1
        while j < len(df):

            z_exit = float(df["zscore"].iloc[j])

            if abs(z_exit) < 0.002:
                exit_price = float(close.iloc[j])
                break

            j += 1
        else:
            # se non rientra mai, esci ultimo giorno
            exit_price = float(close.iloc[-1])
            j = len(df) - 1

        ret = direction * (exit_price - entry_price) / entry_price
        ret -= cost_per_trade

        equity *= (1 + risk_per_trade * ret)

        equity_curve.append(equity)
        trades.append(ret)

        i = j + 1

    equity_series = pd.Series(equity_curve)

    if len(equity_series) > 0:
        total_return = equity_series.iloc[-1] - 1
        max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
    else:
        total_return = 0
        max_drawdown = 0

    win_rate = float(np.mean([t > 0 for t in trades])) if trades else 0

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "num_trades": int(len(trades))
    }
