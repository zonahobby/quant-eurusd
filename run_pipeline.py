import yfinance as yf
import json
import time
from src.backtest import run_backtest


PAIRS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
]


def download_pair(symbol):
    df = yf.download(
        symbol,
        period="15y",     # storico lungo
        interval="1d",    # DAILY
        progress=False
    )
    df = df.dropna()
    return df


def main():

    all_results = {}
    combined_equity = 1.0

    for pair in PAIRS:

        print(f"\nDownloading {pair} ...")

        start_time = time.time()
        df = download_pair(pair)
        elapsed = round(time.time() - start_time, 2)

        rows = len(df)

        print(f"{pair} → rows: {rows} | download time: {elapsed}s")

        pair_result = {
            "rows_downloaded": rows,
            "download_time_seconds": elapsed
        }

        if rows < 500:
            pair_result["status"] = "not_enough_data"
            all_results[pair] = pair_result
            continue

        result = run_backtest(df)

        pair_result["status"] = "backtest_done"
        pair_result["backtest"] = result

        all_results[pair] = pair_result

        # Combiniamo equity in modo semplice
        combined_equity *= (1 + result["total_return"])

    all_results["combined_equity_multiplier"] = combined_equity

    with open("output/multi_asset_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== MACRO TREND DAILY RESULTS ===")
    print(all_results)


if __name__ == "__main__":
    main()
