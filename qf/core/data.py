
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def get_ticker_data():
    
    # S&P500 dataframe: list of tickers
    sp_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp_df['Symbol'] = sp_df['Symbol'].str.replace('.', '-')
    bm_ticker = '^GSPC'
    return sp_df

def get_yahoo_query_data(tickers):
    """
    Fetch detailed stock data for given tickers from Yahoo Finance.
    Args:
        tickers (str or list): Stock ticker symbols.
    Returns:
        dict: Dictionary containing detailed stock data for each ticker."""
  

    from yahooquery import Ticker

    ticker_data = {}

    if isinstance(tickers, str):
        tickers = [tickers]

    for ticker in tickers[:5]:
        try:
            data = Ticker(ticker).all_modules[ticker]
            ticker_data[ticker] = {}

            ticker_data[ticker]["symbol"] = ticker
            ticker_data[ticker]["long_name"] = data["quoteType"]["longName"]
            ticker_data[ticker]["summary"] = data["assetProfile"]["longBusinessSummary"]


            default_key_statistics = data["defaultKeyStatistics"]

            if default_key_statistics:
                ticker_data[ticker]["key_statistics"] = {
                    "market_cap": default_key_statistics.get("marketCap"),
                    "enterprise_value": default_key_statistics.get("enterpriseValue"),
                    "trailing_pe": default_key_statistics.get("trailingPE"),
                    "forward_pe": default_key_statistics.get("forwardPE"),
                    "peg_ratio": default_key_statistics.get("pegRatio"),
                    "price_to_book": default_key_statistics.get("priceToBook"),
                    "dividend_yield": default_key_statistics.get("dividendYield"),
                    "dividend_rate": default_key_statistics.get("dividendRate"),
                    "fifty_two_week_high": default_key_statistics.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": default_key_statistics.get("fiftyTwoWeekLow"),
                    "fifty_two_week_change": default_key_statistics.get("fiftyTwoWeekChange"),
                    "beta": default_key_statistics.get("beta"),
                }
            else:
                ticker_data[ticker]["key_statistics"] = {}
            
            summary_detail = data["summaryDetail"]

            if summary_detail:
                ticker_data[ticker]["summary_detail"] = {
                "previous_close": summary_detail.get("previousClose"),
                "open": summary_detail.get("open"),
                "day_high": summary_detail.get("dayHigh"),
                "day_low": summary_detail.get("dayLow"),
                "fifty_two_week_high": summary_detail.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": summary_detail.get("fiftyTwoWeekLow"),
                "fifty_two_week_change": summary_detail.get("fiftyTwoWeekChange"),
                "volume": summary_detail.get("volume"),
                "average_volume": summary_detail.get("averageVolume"),
                "market_cap": summary_detail.get("marketCap"),
                "beta": summary_detail.get("beta"),
                "dividend_yield": summary_detail.get("dividendYield"),
                "dividend_rate": summary_detail.get("dividendRate"),
                "ex_dividend_date": summary_detail.get("exDividendDate"),}
            else:
                ticker_data[ticker]["summary_detail"] = {}

            esg_scores = data["esgScores"]
            if esg_scores:
                ticker_data[ticker]["esg_scores"] = {
                    "environmental": esg_scores.get("environmentalScore"),
                    "social": esg_scores.get("socialScore"),
                    "governance": esg_scores.get("governanceScore"),
                    "total": esg_scores.get("totalEsg"),
                    "rating": esg_scores.get("rating"),
                    "percentile": esg_scores.get("percentile"),
                    "rank": esg_scores.get("rank"),
                    "last_updated": esg_scores.get("lastUpdated"),
                    "category": esg_scores.get("category"),
                    "category_scores": esg_scores.get("categoryScores"),
                    "category_percentiles": esg_scores.get("categoryPercentiles"),
                    "category_ranks": esg_scores.get("categoryRanks"),  
                }
            else:
                ticker_data[ticker]["esg_scores"]  = {}

            financial_data = data["financialData"]
            if financial_data:
                ticker_data[ticker]["financial_data"] = {
                    "current_price": financial_data.get("currentPrice"),
                    "market_cap": financial_data.get("marketCap"),
                    "trailing_pe": financial_data.get("trailingPE"),
                    "forward_pe": financial_data.get("forwardPE"),
                    "peg_ratio": financial_data.get("pegRatio"),
                    "price_to_book": financial_data.get("priceToBook"),
                    "enterprise_value": financial_data.get("enterpriseValue"),
                    "enterprise_to_ebitda": financial_data.get("enterpriseToEbitda"),
                    "dividend_yield": financial_data.get("dividendYield"),
                    "dividend_rate": financial_data.get("dividendRate"),
                    "five_year_avg_dividend_yield": financial_data.get("fiveYearAvgDividendYield"),
                    "fifty_two_week_high": financial_data.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": financial_data.get("fiftyTwoWeekLow"),
                    "fifty_two_week_change": financial_data.get("fiftyTwoWeekChange"),
                    "beta": financial_data.get("beta"),
                    "return_on_equity": financial_data.get("returnOnEquity"),
                    "return_on_assets": financial_data.get("returnOnAssets"),
                    "return_on_invested_capital": financial_data.get("returnOnInvestedCapital"),
                    "debt_to_equity": financial_data.get("debtToEquity"),
                    "current_ratio": financial_data.get("currentRatio"),
                    "quick_ratio": financial_data.get("quickRatio"),
                    "operating_margin": financial_data.get("operatingMargin"),
                    "profit_margin": financial_data.get("profitMargin"),
                    "gross_profit": financial_data.get("grossProfit"),
                    "revenue": financial_data.get("totalRevenue"),
                    "ebitda": financial_data.get("ebitda"),
                    "net_income": financial_data.get("netIncome"),
                    "free_cash_flow": financial_data.get("freeCashflow"),
                    "cash_and_cash_equivalents": financial_data.get("cashAndCashEquivalents"),
                    "total_assets": financial_data.get("totalAssets"),
                    "total_liabilities": financial_data.get("totalLiabilities"),
                    "total_equity": financial_data.get("totalEquity"),      
                }
            else:
                ticker_data[ticker]["financial_data"]  = {}

        except Exception:
                print(f"Not able to collect data for {ticker}")

    return ticker_data


def get_yahoo_query_historical_data(tickers, period='10y', identifier='adjclose'):
    """
    Fetch historical stock data for given tickers from Yahoo Finance.
    Args:
        tickers (str or list): Stock ticker symbols.
        period (str): Period for which to fetch data (e.g., '1d', '1mo', '1y', '5y', '10y').
        identifier (str): Identifier for the data to fetch (e.g., 'adjclose', 'close').
    Returns:
        pd.DataFrame: DataFrame containing historical stock data."""

    from yahooquery import Ticker

    stock_data = Ticker(tickers, asynchronous=True).history(period=period)[identifier]
    stock_data = stock_data.unstack(level=0)

    stock_data.index = pd.to_datetime(stock_data.index, format='mixed', utc=True)
    stock_data.index = stock_data.index.tz_localize(None)

    if isinstance(tickers_list, str):
        tickers_list = [tickers_list]

    if isinstance(stock_data, pd.DataFrame):
        stock_data = stock_data[tickers_list]
    elif isinstance(stock_data, pd.Series):
        stock_data = pd.DataFrame(stock_data)

    return stock_data


def get_yahoo_data(tickers_list, start_date, end_date, identifier_list=None):
    """
    Fetch historical data for a given ticker from Yahoo Finance.
    
    Args:
        tickers_list (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """

    if identifier_list is None:
        identifier_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Variables instantiation
    window = 252
    show_batch = True
    df_abs = pd.DataFrame()
    batch_size = 20
    loop_size = int(len(tickers_list) // batch_size) + 2

    for t in range(1,loop_size): # Batch download
        m = (t - 1) * batch_size
        n = t * batch_size
        batch_list = tickers_list[m:n]
        print(batch_list,m,n)
        batch_download = yf.download(tickers= batch_list,start= start_date, end = end_date,
                        interval = "1d",group_by = 'column',auto_adjust = True,
                              prepost = True, proxy = None)[identifier_list]
                              
        df_abs = df_abs.join(batch_download, how='outer')
        if show_batch:
            print(f"Batch {t} downloaded: {batch_list}")
            print(batch_download.head())

    return df_abs


def main():

    sp_tickers_df = get_ticker_data()

    tickers = sp_tickers_df['Symbol'].tolist()

    full_data_dict = get_yahoo_query_data(tickers)

    financial_data_df = pd.DataFrame.from_dict(
        {ticker: data['financial_data'] for ticker, data in full_data_dict.items()},
        orient='index'
    )
    financial_data_df.index.name = 'ticker'
    financial_data_df.reset_index(inplace=True)
    print(financial_data_df.head())

    esg_scores_df = pd.DataFrame.from_dict(
        {ticker: data['esg_scores'] for ticker, data in full_data_dict.items()},
        orient='index'
    )
    esg_scores_df.index.name = 'ticker'
    esg_scores_df.reset_index(inplace=True)
    print(esg_scores_df.head())

    summary_detail_df = pd.DataFrame.from_dict(
        {ticker: data['summary_detail'] for ticker, data in full_data_dict.items()},
        orient='index'
    )
    summary_detail_df.index.name = 'ticker'
    summary_detail_df.reset_index(inplace=True)
    print(summary_detail_df.head())

    features_df = financial_data_df[['return_on_equity', 'return_on_assets', 'return_on_invested_capital']]

    # 'return_on_equity', 'return_on_assets', 'return_on_invested_capital'


if __name__ == "__main__":
    main()
    # Example usage:
    # tickers = ['AAPL', 'MSFT', 'GOOGL']
    # data = get_yahoo_query_data(tickers)
    # print(data)

    # history_data = get_yahoo_query_historical_data(tickers, period='1y')
    # print(history_data.head())
    
    # historical_data = get_yahoo_data(tickers, '2020-01-01', '2023-01-01')
    # print(historical_data.head())
        

    