def build_signal(prob_up, price):
    if prob_up > 0.55:
        side = "BUY"
    elif prob_up < 0.45:
        side = "SELL"
    else:
        side = "HOLD"

    stop = price * 0.993
    target = price * 1.012

    forecast = price * (1 + (prob_up - 0.5) * 0.02)

    return {
        "prob_up": float(prob_up),
        "signal": side,
        "price": float(price),
        "stop": float(stop),
        "target": float(target),
        "forecast": float(forecast),
    }
