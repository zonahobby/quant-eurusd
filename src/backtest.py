import pandas as pd
import numpy as np


def run_backtest(
    df,
    risk_per_trade=0.01,
    cost_per_trade=0.0005  # spread realistico H1
):
    df = df.copy()
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    # === Indicatori ===
    df["high20"] = high.rolling(20).max()
    df["low20"] = low.rolling(20).min()
    df["range20"] = df["high20"] - df["low20"]
    df["range_mean"] = df["range20"].rolling(50).mean()

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    df["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    df = df.dropna()

    equity = 1.0
    trades = []

    i = 50
    while i < len(df) - 1:

        row = df.iloc[i]

        # condizione compressione
        if row["range20"] > row["range_mean"]:
            i += 1
            continue

        entry_price = close.iloc[i]

        # LONG breakout
        if close.iloc[i] > row["high20"]:
            direction = 1
        # SHORT breakout
        elif close.iloc[i] < row["low20"]:
            direction = -1
        else:
            i += 1
            continue

        atr = row["atr"]
        stop_distance = atr
        target_distance = 2 * atr

        j = i + 1
        trade_closed = False

        while j < len(df):

            price = close.iloc[j]

            if direction == 1:
                if price <= entry_price - stop_distance:
                    ret = -stop_distance / entry_price
                    trade_closed = True
                elif price >= entry_price + target_distance:
                    ret = target_distance / entry_price
                    trade_closed = True
            else:
                if price >= entry_price + stop_distance:
                    ret = -stop_distance / entry_price
                    trade_closed = True
                elif price <= entry_price - target_distance:
                    ret = target_distance / entry_price
                    trade_closed = True

            if trade_closed:
                break

            j += 1

        if not trade_closed:
            ret = 0

        ret -= cost_per_trade
        equity *= (1 + risk_per_trade * ret)
        trades.append(ret)

        i = j + 1

    if len(trades) == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0
        }

    equity_curve = np.cumprod([1 + risk_per_trade * t for t in trades])
    equity_series = pd.Series(equity_curve)

    total_return = equity_series.iloc[-1] - 1
    max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
    win_rate = np.mean([t > 0 for t in trades])

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "num_trades": int(len(trades))
    }
