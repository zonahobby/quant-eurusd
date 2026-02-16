import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def compute_features(df):
    df = df.copy()
    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df["vol10"] = df["Close"].pct_change().rolling(10).std()
    return df.dropna()


def train_model(X, y):
    m1 = LogisticRegression(max_iter=1000)
    m2 = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss")

    m1.fit(X, y)
    m2.fit(X, y)

    return m1, m2


def predict_prob(models, X):
    m1, m2 = models
    p1 = m1.predict_proba(X)[0, 1]
    p2 = m2.predict_proba(X)[0, 1]
    return (p1 + p2) / 2


def run_backtest(df):
    df = compute_features(df)

    equity = 1.0
    equity_curve = []
    trades = []

    for i in range(200, len(df) - 1):
        train = df.iloc[:i]
        test_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        X_train = train[["ret1", "ret5", "vol10"]]
        y_train = (train["Close"].shift(-1) > train["Close"]).astype(int)[:-1]
        X_train = X_train[:-1]

        models = train_model(X_train, y_train)

        X_today = test_row[["ret1", "ret5", "vol10"]].values.reshape(1, -1)
        prob_up = predict_prob(models, X_today)

        entry_price = float(test_row["Close"])
        exit_price = float(next_row["Close"])

        if prob_up > 0.55:
            ret = (exit_price - entry_price) / entry_price
            direction = "BUY"
        elif prob_up < 0.45:
            ret = (entry_price - exit_price) / entry_price
            direction = "SELL"
        else:
            equity_curve.append(equity)
            continue

        equity *= (1 + ret)
        equity_curve.append(equity)

        trades.append({
            "direction": direction,
            "entry": entry_price,
            "exit": exit_price,
            "return": ret,
            "holding_days": 1,
        })

    equity_series = pd.Series(equity_curve)

    total_return = equity_series.iloc[-1] - 1 if len(equity_series) else 0
    max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() if len(equity_series) else 0
    win_rate = np.mean([t["return"] > 0 for t in trades]) if trades else 0
    avg_holding = np.mean([t["holding_days"] for t in trades]) if trades else 0

    metrics = {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "num_trades": len(trades),
        "avg_holding_days": float(avg_holding),
    }

    return equity_series, trades, metrics
