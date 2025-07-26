import src.patterns.candlestick as candlestick
from src.model.predict import download_symbol
from src.pages.stats import find_pattern


if __name__ == "__main__":
    df = download_symbol("MSFT")

    # print(candlestick.inverted_hammer(df))
    # print(candlestick.hammer(df, target="result"))
    # patterns = [x for x in dir(candlestick) if "__" not in x]
    # for pattern in patterns:
    #     print(find_pattern(df, pattern))
    print(candlestick.bullish_hanging_man(df))
