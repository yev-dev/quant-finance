
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def get_instruments_data(columns=None):
    
    # S&P500 dataframe: list of tickers

    if columns is None:
        columns = ['symbol', 'security_name', 'sector', 'sub_industry', 'date_added']
    sp_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    sp_df.rename(columns={
        'Symbol': 'symbol',
        'Security': 'security_name',
        'GICS Sector': 'sector',
        'GICS Sub-Industry': 'sub_industry',
        'Date added': 'date_added'
    }, inplace=True)

    sp_df['symbol'] = sp_df['symbol'].str.replace('.', '-')
    
    return sp_df[columns]

def _extract_key_statistics(default_key_statistics):
    if not default_key_statistics:
        return {}
    return {
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

def _extract_summary_detail(summary_detail):
    if not summary_detail:
        return {}
    return {
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
        "ex_dividend_date": summary_detail.get("exDividendDate"),
    }

def _extract_esg_scores(esg_scores):
    if not esg_scores:
        return {}
    return {
        "environmental": esg_scores.get("environmentalScore"),
        "social": esg_scores.get("socialScore"),
        "governance": esg_scores.get("governanceScore"),
        "highest_controversy": esg_scores.get("highestControversy"),
        "total": esg_scores.get("totalEsg"),
        "peer_count": esg_scores.get("peerCount"),
        "peer_group": esg_scores.get("peerGroup"),
        "rating_year": esg_scores.get("ratingYear"),
        "rating_month": esg_scores.get("ratingMonth"),
        "adult": esg_scores.get("adult"),
        "animal_testing": esg_scores.get("animalTesting"),
        "gambling": esg_scores.get("gambling"),
        "small_arm": esg_scores.get("smallArms"),
        "furLeather": esg_scores.get("furLeather"),
        "coal": esg_scores.get("coal"),
        "controversial_weapons": esg_scores.get("controversialWeapons"),
        "tobacco": esg_scores.get("tobacco"),
    }

def _extract_financial_data(financial_data):
    if not financial_data:
        return {}
    return {
        "current_price": financial_data.get("currentPrice"),
        "return_on_equity": financial_data.get("returnOnEquity"),
        "return_on_assets": financial_data.get("returnOnAssets"),
        "debt_to_equity": financial_data.get("debtToEquity"),
        "current_ratio": financial_data.get("currentRatio"),
        "quick_ratio": financial_data.get("quickRatio"),
        "revenue": financial_data.get("totalRevenue"),
        "ebitda": financial_data.get("ebitda"),
        "free_cash_flow": financial_data.get("freeCashflow"),
    }

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

    for ticker in tickers:
        try:
            data = Ticker(ticker).all_modules[ticker]
            ticker_data[ticker] = {
                "symbol": ticker,
                "long_name": data["quoteType"]["longName"],
                "summary": data["assetProfile"]["longBusinessSummary"],
                "summary_detail": _extract_summary_detail(data.get("summaryDetail")),
                "esg_scores": _extract_esg_scores(data.get("esgScores")),
                "financial_data": _extract_financial_data(data.get("financialData")),
            }
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

    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(stock_data, pd.DataFrame):
        stock_data = stock_data[tickers]
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
    yahoo_data_df = pd.DataFrame()
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
                              
        yahoo_data_df = yahoo_data_df.join(batch_download, how='outer')
        # Always print batch info since show_batch was always True
        print(f"Batch {t} downloaded: {batch_list}")
        print(batch_download.head())

    return yahoo_data_df


def get_yahoo_query_full_data(tickers = None, features_list=None):


    if tickers is None:
        # Get S&P 500 tickers if no tickers are provided   
        tickers = get_instruments_data() 


    if isinstance(tickers, pd.DataFrame):
        tickers = tickers['symbol'].tolist()


    full_data_dict = get_yahoo_query_data(tickers)


    financial_data_df = pd.DataFrame.from_dict(
        {ticker: data['financial_data'] for ticker, data in full_data_dict.items()},
        orient='index'
    )

    financial_data_df.index.name = 'symbol'
    financial_data_df.reset_index(inplace=True)

    summary_detail_df = pd.DataFrame.from_dict(
        {ticker: data['summary_detail'] for ticker, data in full_data_dict.items()},
        orient='index'
    )
    summary_detail_df.index.name = 'symbol'
    summary_detail_df.reset_index(inplace=True) 

    esg_scores_df = pd.DataFrame.from_dict(
        {ticker: data['esg_scores'] for ticker, data in full_data_dict.items()},
        orient='index'
    )
    esg_scores_df.index.name = 'symbol'
    esg_scores_df.reset_index(inplace=True)


    return_data_df = pd.DataFrame.from_dict(
        {ticker: {'symbol': data.get('symbol'), 'long_name': data.get('long_name'), 'summary': data.get('summary')}
         for ticker, data in full_data_dict.items()},
        orient='index'
    )
    return_data_df = summary_detail_df.merge(return_data_df, on='symbol', how='left', validate='one_to_one')
    return_data_df = financial_data_df.merge(return_data_df, on='symbol', how='left', validate='one_to_one')    
    return_data_df = return_data_df.merge(esg_scores_df, on='symbol', how='left', validate='one_to_one')

    if features_list:
        return return_data_df[features_list]

    return return_data_df

if __name__ == "__main__":
    instruments_df = get_instruments_data()
    instruments_df.to_csv('/Users/yevgeniy/Development/projects/quant/quant-finance/qf/clustering/data/instruments.csv', index=False)
    tickers = instruments_df['symbol'].tolist()
    fundamentals_df = get_yahoo_query_full_data(tickers=tickers)
    saving_path = r'/Users/yevgeniy/Development/projects/quant/quant-finance/qf/clustering/data/fundamentals_data.csv'
    fundamentals_df.to_csv(saving_path, index=False)
    print(f"Fundamentals data saved to {saving_path}")

    