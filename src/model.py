from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_and_predict(X, y, X_today):
    m1 = LogisticRegression(max_iter=1000)
    m2 = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss")

    m1.fit(X, y)
    m2.fit(X, y)

    p1 = m1.predict_proba(X_today)[0, 1]
    p2 = m2.predict_proba(X_today)[0, 1]

    return (p1 + p2) / 2
