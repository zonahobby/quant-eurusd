import yfinance as yf


def download_all():
    pair = "EURUSD=X"

    daily = yf.download(pair, period="10y", interval="1d")
    hourly = yf.download(pair, period="60d", interval="1h")

    return daily, hourly
