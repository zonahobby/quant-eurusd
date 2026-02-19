from src.download_data import download_all
from src.features import build_datasets
from src.model import train_and_predict
from src.signal import build_signal
from src.plot import create_plot
from src.backtest import run_backtest
from src.backtest import walkforward_mean_reversion
import json


def main():
    daily, hourly = download_all()

    # ===== SEGNALE =====
    X_train, y_train, X_today, price_today = build_datasets(daily)
    prob_up = train_and_predict(X_train, y_train, X_today)
    signal = build_signal(prob_up, price_today)

    create_plot(daily, signal["forecast"], signal["price"])

    # ===== BACKTEST =====
    results = run_backtest(daily)

    with open("output/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("BACKTEST RESULTS:", results)
    print("Signal:", signal)   # ← RESTA QUI

    # ===== WALK-FORWARD =====
    wf_results, wf_equity = walkforward_mean_reversion(daily)

    with open("output/walkforward.json", "w") as f:
        json.dump(wf_results, f, indent=2)

    print("WALKFORWARD:", wf_results)
    print("WF FINAL EQUITY:", wf_equity)

if __name__ == "__main__":
    main()
