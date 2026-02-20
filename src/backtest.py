import pandas as pd
import numpy as np


def compute_rsi(series, period=3):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def run_backtest(
    df,
    risk_per_trade=0.01,
    cost_per_trade=0.0005
):

    df = df.copy().dropna()

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    rsi3 = compute_rsi(close, 3)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    df = pd.DataFrame({
        "close": close,
        "ma20": ma20,
        "ma50": ma50,
        "rsi3": rsi3,
        "atr": atr
    }).dropna()

    equity = 1.0
    trades = []

    position = 0
    entry_price = 0
    stop_level = 0
    entry_index = 0

    for i in range(50, len(df)):

        price = df["close"].iloc[i]
        weekday = df.index[i].weekday()

        # ===== ENTRY =====
        if position == 0:

            if df["ma20"].iloc[i] > df["ma50"].iloc[i] and df["rsi3"].iloc[i] < 15:
                position = 1
                entry_price = price
                stop_level = price - 1.5 * df["atr"].iloc[i]
                entry_index = i

            elif df["ma20"].iloc[i] < df["ma50"].iloc[i] and df["rsi3"].iloc[i] > 85:
                position = -1
                entry_price = price
                stop_level = price + 1.5 * df["atr"].iloc[i]
                entry_index = i

        # ===== MANAGE =====
        else:

            holding_days = i - entry_index
            exit_trade = False

            # Stop
            if position == 1 and price <= stop_level:
                ret = (price - entry_price) / entry_price
                exit_trade = True

            elif position == -1 and price >= stop_level:
                ret = (entry_price - price) / entry_price
                exit_trade = True

            # RSI profit exit (più larga)
            elif (position == 1 and df["rsi3"].iloc[i] > 60) or \
                 (position == -1 and df["rsi3"].iloc[i] < 40):
                if position == 1:
                    ret = (price - entry_price) / entry_price
                else:
                    ret = (entry_price - price) / entry_price
                exit_trade = True

            # Max 4 giorni
            elif holding_days >= 4:
                if position == 1:
                    ret = (price - entry_price) / entry_price
                else:
                    ret = (entry_price - price) / entry_price
                exit_trade = True

            # Venerdì chiudi
            elif weekday == 4:
                if position == 1:
                    ret = (price - entry_price) / entry_price
                else:
                    ret = (entry_price - price) / entry_price
                exit_trade = True

            if exit_trade:
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
