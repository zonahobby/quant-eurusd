import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def compute_features(df):
    df = df.copy()

    # ritorni e volatilità base
    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df["vol10"] = df["Close"].pct_change().rolling(10).std()

    # trend
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["trend"] = df["ma5"] - df["ma20"]

    # momentum
    df["mom10"] = df["Close"].pct_change(10)

    # regime trend: distanza > 1% dalla MA50
    close = df["Close"].squeeze()
    ma50 = df["ma50"].squeeze()
    df["trend_regime"] = (abs(close - ma50) / ma50) > 0.01

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
            trades = 0

            for i in range(200, len(df) - holding_days):
                row = df.iloc[i]

                # filtro regime
                if use_filter and not bool(row["trend_regime"]):
                    continue

                trades += 1
                equity *= 1.0001  # crescita fittizia minima per test struttura

            # SALVATAGGIO SICURO (sempre eseguito)
            results[f"{holding_days}_{regime_name}"] = {
                "total_return": float(equity - 1),
                "num_trades": int(trades),
            }

    return results
