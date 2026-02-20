from src.backtest import run_backtest
import yfinance as yf
import json
import time


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
        period="730d",      # ~2 anni
        interval="1h",      # 1 ora
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

        num_rows = len(df)

        print(f"{pair} → rows: {num_rows} | download time: {elapsed}s")

        pair_result = {
            "rows_downloaded": num_rows,
            "download_time_seconds": elapsed
        }

        # Se troppo pochi dati, saltiamo ma lo registriamo
        if num_rows < 200:
            pair_result["status"] = "not_enough_data"
            all_results[pair] = pair_result
            continue

        # Esegui backtest
        result = run_backtest(df)

        pair_result["status"] = "backtest_done"
        pair_result["backtest"] = result

        all_results[pair] = pair_result

        combined_equity *= (1 + result["total_return"])

    all_results["combined_equity_multiplier"] = combined_equity

    with open("output/multi_asset_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== H1 MULTI ASSET RESULTS ===")
    print(all_results)


if __name__ == "__main__":
    main()
