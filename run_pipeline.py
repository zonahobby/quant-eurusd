from src.download_data import download_all
from src.features import build_datasets
from src.model import train_and_predict
from src.signal import build_signal
from src.plot import create_plot
from src.backtest import run_backtest
import json


def main():
    daily, hourly = download_all()

    # ===== SEGNALE ATTUALE =====
    X_train, y_train, X_today, price_today = build_datasets(daily)
    prob_up = train_and_predict(X_train, y_train, X_today)
    signal = build_signal(prob_up, price_today)

    create_plot(daily, signal["forecast"], signal["price"])

    # ===== BACKTEST MULTI-HOLDING =====
    results = run_backtest(daily)

    # salva solo le metriche riassuntive
    metrics_summary = {str(k): v["metrics"] for k, v in results.items()}

    with open("output/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("Signal:", signal)
    print("Backtest metrics:", metrics_summary)


if __name__ == "__main__":
    main()
