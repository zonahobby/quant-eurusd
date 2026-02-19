import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def compute_features(df):
    df = df.copy()

    # forza Close a essere una Series scalare (fix yfinance/pandas)
    close = df["Close"].squeeze()

    # ritorni e volatilità
    df["ret1"] = close.pct_change(1)
    df["ret5"] = close.pct_change(5)
    df["vol10"] = close.pct_change().rolling(10).std()

    # medie mobili
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # trend e momentum
    df["trend"] = df["ma5"] - df["ma20"]
    df["mom10"] = close.pct_change(10)

    # regime di trend robusto
    df["trend_regime"] = (abs(close - df["ma50"]) / df["ma50"]) > 0.01

    return df.dropna()


def train_model(X, y):
    m1 = LogisticRegression(max_iter=1000)
    m2 = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss")

    m1.fit(X, y.values.ravel())
    m2.fit(X, y.values.ravel())

    return m1, m2


def predict_prob(models, X):
    m1, m2 = models
    p1 = m1.predict_proba(X)[0, 1]
    p2 = m2.predict_proba(X)[0, 1]
    return (p1 + p2) / 2


def run_backtest(df, holding_days_list=(1, 3, 5, 10), risk_per_trade=0.01):
    df = df.copy()

    close = df["Close"].squeeze()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    zscore = (close - ma20) / ma20
    trend_distance = abs(close - ma50) / ma50

    df["zscore"] = zscore
    df["is_lateral"] = trend_distance < 0.01

    df = df.dropna()

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


