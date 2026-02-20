import pandas as pd
import numpy as np


def run_backtest(
    df,
    risk_per_trade=0.01,
    cost_per_trade=0.0005
):

    df = df.copy().dropna()

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    # === Indicatori ===
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    high20 = high.shift(1).rolling(20).max()
    low20 = low.shift(1).rolling(20).min()

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    df = pd.DataFrame({
        "close": close,
        "ma50": ma50,
        "ma200": ma200,
        "high20": high20,
        "low20": low20,
        "atr": atr
    }).dropna()

    equity = 1.0
    trades = []

    position = 0
    entry_price = 0
    stop_level = 0

    for i in range(200, len(df)):

        price = df["close"].iloc[i]

        # === Trend direction ===
        if df["ma50"].iloc[i] > df["ma200"].iloc[i]:
            trend = 1
        elif df["ma50"].iloc[i] < df["ma200"].iloc[i]:
            trend = -1
        else:
            trend = 0

        # === Entry logic ===
        if position == 0:

            if trend == 1 and price > df["high20"].iloc[i]:
                position = 1
                entry_price = price
                stop_level = price - 2 * df["atr"].iloc[i]

            elif trend == -1 and price < df["low20"].iloc[i]:
                position = -1
                entry_price = price
                stop_level = price + 2 * df["atr"].iloc[i]

        # === Manage position ===
        else:

            # Stop hit
            if position == 1 and price <= stop_level:
                ret = (price - entry_price) / entry_price
                ret -= cost_per_trade
                equity *= (1 + risk_per_trade * ret)
                trades.append(ret)
                position = 0

            elif position == -1 and price >= stop_level:
                ret = (entry_price - price) / entry_price
                ret -= cost_per_trade
                equity *= (1 + risk_per_trade * ret)
                trades.append(ret)
                position = 0

            # Trend flip exit
            elif (position == 1 and trend == -1) or (position == -1 and trend == 1):
                if position == 1:
                    ret = (price - entry_price) / entry_price
                else:
                    ret = (entry_price - price) / entry_price

                ret -= cost_per_trade
                equity *= (1 + risk_per_trade * ret)
                trades.append(ret)
                position = 0

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
