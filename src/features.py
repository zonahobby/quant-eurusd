import numpy as np


def build_datasets(df):
    df = df.copy()

    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df["vol10"] = df["Close"].pct_change().rolling(10).std()

    df = df.dropna()

    X = df[["ret1", "ret5", "vol10"]]
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)[:-1]

    X = X[:-1]

    X_today = X.iloc[[-1]]
    price_today = float(df["Close"].iloc[-1].item())

    return X, y, X_today, price_today
