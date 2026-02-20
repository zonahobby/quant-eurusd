import pandas as pd
import numpy as np


def run_backtest(
    df,
    risk_per_trade=0.01,
    cost_per_trade=0.0005
):

    df = df.copy().dropna()

    # forza colonne a scalari puri
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    df = pd.DataFrame({
        "close": close,
        "high": high,
        "low": low
    })

    df["hour"] = df.index.hour
    df["date"] = df.index.date

    equity = 1.0
    trades = []

    grouped = df.groupby("date")

    for date, day_data in grouped:

        if len(day_data) < 10:
            continue

        # Asian session 00–07
        asian = day_data[(day_data["hour"] >= 0) & (day_data["hour"] < 8)]

        if len(asian) < 5:
            continue

        high_asia = float(asian["high"].max())
        low_asia = float(asian["low"].min())
        asian_range = high_asia - low_asia

        if asian_range == 0:
            continue

        # London session 08+
        london = day_data[day_data["hour"] >= 8]

        if len(london) == 0:
            continue

        entry_price = None
        direction = None
        entry_index = None

        # breakout detection
        for i in range(len(london)):
            price = float(london["close"].iloc[i])

            if price > high_asia:
                entry_price = price
                direction = 1
                entry_index = i
                break

            elif price < low_asia:
                entry_price = price
                direction = -1
                entry_index = i
                break

        if entry_price is None:
            continue

        stop_distance = asian_range
        target_distance = 1.5 * asian_range

        trade_closed = False

        for j in range(entry_index, len(london)):

            price = float(london["close"].iloc[j])

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

        if not trade_closed:
            ret = 0

        ret -= cost_per_trade
        equity *= (1 + risk_per_trade * ret)
        trades.append(ret)

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
