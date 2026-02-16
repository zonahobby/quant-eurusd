import matplotlib.pyplot as plt


def create_plot(df, forecast, price):
    recent = df.tail(30)

    plt.figure()
    plt.plot(recent.index, recent["Close"])
    plt.scatter(recent.index[-1], price)
    plt.scatter(recent.index[-1], forecast)

    plt.title("EUR/USD Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.savefig("output/chart.png")
    plt.close()
