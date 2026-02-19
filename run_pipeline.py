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

    # ===== BACKTEST =====
    results = run_backtest(daily)

    # salva metriche complete
    with open("output/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("BACKTEST RESULTS:", results)
    print("Signal:", signal)


if __name__ == "__main__":
    main()
