import pandas as pd
import numpy as np


def run_backtest(
    df,
    risk_per_trade=0.01,
    cost_per_trade=0.0005
):
    df = df.copy()

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    # === Breakout levels (CORRETTO) ===
    high20 = high.shift(1).rolling(20).max()
    low20 = low.shift(1).rolling(20).min()

    range20 = high20 - low20
    range_mean = range20.rolling(50).mean()

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    df = pd.DataFrame({
        "close": close,
        "high20": high20,
        "low20": low20,
        "range20": range20,
        "range_mean": range_mean,
        "atr": atr
    }).dropna()

    equity = 1.0
    trades = []

    i = 50
    while i < len(df) - 1:

        # Compressione
        if df["range20"].iloc[i] > df["range_mean"].iloc[i]:
            i += 1
            continue

        entry_price = df["close"].iloc[i]

        # Breakout
        if df["close"].iloc[i] > df["high20"].iloc[i]:
            direction = 1
        elif df["close"].iloc[i] < df["low20"].iloc[i]:
            direction = -1
        else:
            i += 1
            continue

        stop_distance = df["atr"].iloc[i]
        target_distance = 2 * stop_distance

        j = i + 1
        trade_closed = False

        while j < len(df):

            price = df["close"].iloc[j]

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
