from src.backtest import run_backtest
import yfinance as yf
import json


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
        period="5y",
        interval="4h",
        progress=False
    )
    df = df.dropna()
    return df


def main():

    all_results = {}
    combined_equity = 1.0

    for pair in PAIRS:

        print(f"\nRunning H4 backtest for {pair}")

        df = download_pair(pair)

        if len(df) < 200:
            print("Not enough data.")
            continue

        result = run_backtest(df)

        all_results[pair] = result

        combined_equity *= (1 + result["total_return"])

    with open("output/multi_asset_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== H4 MULTI ASSET RESULTS ===")
    print(all_results)
    print("Combined equity multiplier:", combined_equity)


if __name__ == "__main__":
    main()
