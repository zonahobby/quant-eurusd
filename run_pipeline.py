def run_backtest(df, holding_days_list=(1, 3, 5, 10), risk_per_trade=0.01):
    df = df.copy()

    # --- Close scalare robusto ---
    close = df["Close"].squeeze()

    # --- Feature mean reversion ---
    ma20 = close.rolling(20).mean()
    zscore = (close - ma20) / ma20

    df["zscore"] = zscore
    df = df.dropna()

    results = {}

    for holding_days in holding_days_list:

        equity = 1.0
        equity_curve = []
        trades = []

        for i in range(20, len(df) - holding_days):

            entry_price = float(close.iloc[i])
            exit_price = float(close.iloc[i + holding_days])
            z = float(df["zscore"].iloc[i])

            # --- Segnale contrarian ---
            if z > 0.01:
                ret = (entry_price - exit_price) / entry_price
                direction = "SELL"

            elif z < -0.01:
                ret = (exit_price - entry_price) / entry_price
                direction = "BUY"

            else:
                equity_curve.append(equity)
                continue

            # --- Risk management 1% ---
            equity *= (1 + risk_per_trade * ret)
            equity_curve.append(equity)

            trades.append(ret)

        # --- Metriche ---
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
