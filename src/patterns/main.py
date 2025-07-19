import src.patterns.candlestick as candlestick
from src.model.predict import download_symbol


if __name__ == "__main__":
    df = download_symbol("AMZN")

    print(candlestick.inverted_hammer(df))
