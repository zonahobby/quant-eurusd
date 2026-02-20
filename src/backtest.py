import pandas as pd
import numpy as np


def run_backtest(
    df,
    risk_per_trade=0.01,
    cost_per_trade=0.0007  # spread più realistico
):

    df = df.copy().dropna()

    open_ = df["Open"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()

    # Calcolo range giornaliero
    range_daily = high - low
    body = abs(close - open_)

    avg_range5 = range_daily.rolling(5).mean()

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "range": range_daily,
        "body": body,
        "avg_range5": avg_range5
    }).dropna()

    equity = 1.0
    trades = []

    position = 0
    entry_price = 0
    stop_level = 0
    target_level = 0
    entry_index = 0

    for i in range(6, len(df)):

        price = df["close"].iloc[i]
        weekday = df.index[i].weekday()

        # === ENTRY ===
        if position == 0:

            compression = df["avg_range5"].iloc[i] < df["avg_range5"].rolling(20).mean().iloc[i]
            explosion = df["range"].iloc[i] > 1.8 * df["avg_range5"].iloc[i]
            strong_body = df["body"].iloc[i] > 0.6 * df["range"].iloc[i]

            if compression and explosion and strong_body:

                R = df["range"].iloc[i]

                if df["close"].iloc[i] > df["open"].iloc[i]:
                    position = 1
                    entry_price = price
                    stop_level = price - R
                    target_level = price + 2 * R
                else:
                    position = -1
                    entry_price = price
                    stop_level = price + R
                    target_level = price - 2 * R

                entry_index = i

        # === GESTIONE ===
        else:

            holding_days = i - entry_index
            exit_trade = False

            # Stop
            if position == 1 and price <= stop_level:
                ret = -1
                exit_trade = True

            elif position == -1 and price >= stop_level:
                ret = -1
                exit_trade = True

            # Target
            elif position == 1 and price >= target_level:
                ret = 2
                exit_trade = True

            elif position == -1 and price <= target_level:
                ret = 2
                exit_trade = True

            # Max 3 giorni
            elif holding_days >= 3:
                if position == 1:
                    ret = (price - entry_price) / (entry_price - stop_level)
                else:
                    ret = (entry_price - price) / (stop_level - entry_price)
                exit_trade = True

            # Venerdì chiudi
            elif weekday == 4:
                if position == 1:
                    ret = (price - entry_price) / (entry_price - stop_level)
                else:
                    ret = (entry_price - price) / (stop_level - entry_price)
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
