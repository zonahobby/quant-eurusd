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
    df = compute_features(df)

    results = {}
    feature_cols = ["ret1", "ret5", "vol10", "trend", "mom10"]

    for holding_days in holding_days_list:
        for regime_name, use_filter in [("all", False), ("trend_only", True)]:

            equity = 1.0
            equity_curve = []
            trades = []

            for i in range(200, len(df) - holding_days):
                train = df.iloc[:i]
                row = df.iloc[i]
                exit_row = df.iloc[i + holding_days]

                # filtro regime robusto
                if use_filter and not bool(row["trend_regime"].item()):
                    equity_curve.append(equity)
                    continue

                X_train = train[feature_cols]
                y_train = (train["Close"].shift(-holding_days) > train["Close"]).astype(int)[:-holding_days]
                X_train = X_train[:-holding_days]

                # salta se dati insufficienti
                if len(X_train) < 50:
                    equity_curve.append(equity)
                    continue

                models = train_model(X_train, y_train)

                X_today = row[feature_cols].values.reshape(1, -1)
                prob_up = predict_prob(models, X_today)

                entry_price = float(row["Close"].item())
                exit_price = float(exit_row["Close"].item())

                if prob_up > 0.55:
                    ret = (exit_price - entry_price) / entry_price
                    direction = "BUY"
                elif prob_up < 0.45:
                    ret = (entry_price - exit_price) / entry_price
                    direction = "SELL"
                else:
                    equity_curve.append(equity)
                    continue

                equity *= (1 + risk_per_trade * ret)
                equity_curve.append(equity)

                trades.append(
                    {
                        "direction": direction,
                        "entry": entry_price,
                        "exit": exit_price,
                        "return": ret,
                        "holding_days": holding_days,
                        "regime": regime_name,
                    }
                )

            equity_series = pd.Series(equity_curve)

            if len(equity_series) > 0:
                total_return = equity_series.iloc[-1] - 1
                max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
            else:
                total_return = 0
                max_drawdown = 0

            win_rate = np.mean([t["return"] > 0 for t in trades]) if trades else 0
            avg_holding = np.mean([t["holding_days"] for t in trades]) if trades else 0

            results[f"{holding_days}_{regime_name}"] = {
                "total_return": float(total_return),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "num_trades": int(len(trades)),
                "avg_holding_days": float(avg_holding),
            }

    return results
