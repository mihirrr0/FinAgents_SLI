import os
import pandas as pd
import requests
import yfinance as yf
from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    BenGrahamMetrics,
    BenGrahamMetricsResponse,
    ValuationLineItem,
    ValuationMetrics,
    ValuationMetricsResponse,
)
from typing import List
import numpy as np
from datetime import datetime, timedelta
# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    """
    Fetch historical stock data from Yahoo Finance using yfinance and return it as a list of Price objects.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL" for Apple).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        List[Price]: A list of Price objects conforming to the Price Pydantic model.
    """
    if not ticker.endswith('.NS'):
        ticker = f"{ticker}.NS"
    
    # Create a Ticker object
    stock = yf.Ticker(ticker)
    
    # Fetch historical data
    # interval='1d' ensures daily data;
    hist_data = stock.history(start=start_date, end=end_date, interval='1d')
    
    # Convert DataFrame to list of Price objects
    prices = []
    for index, row in hist_data.iterrows():
        price = Price(
            open=float(row['Open']),
            close=float(row['Close']),
            high=float(row['High']),
            low=float(row['Low']),
            volume=int(row['Volume']),
            time=index.strftime('%Y-%m-%d')  # Format timestamp as string
        )
        prices.append(price)
    

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def clean_numeric(value):
    if isinstance(value, str):
        if '%' in value:
            return float(value.replace('%', '').replace('"', '')) / 100
        return float(value.replace(',', '').replace('"', ''))
    return float(value) if value is not None else None

# Fetch historical price from Yahoo Finance for a specific date
def get_closest_price(ticker: str, date_str: str) -> float:
    """Fetch the closest stock price for the given ticker and date, ensuring .NS suffix."""
    if not ticker.endswith('.NS'):
        ticker = f"{ticker}.NS"  # Add .NS for Yahoo Finance if not present
    stock = yf.Ticker(ticker)
    date = pd.to_datetime(date_str).tz_localize('UTC')
    start_date = date - timedelta(days=5)
    end_date = date + timedelta(days=5)
    hist = stock.history(start=start_date, end=end_date)
    if not hist.empty:
        idx = np.abs((hist.index - date).days).argmin()
        return hist['Close'].iloc[idx]
    print(f"Warning: No price data found for {ticker} around {date_str}")
    return None

# Calculate financial metrics from CSV data
def calculate_financial_metrics(bs_df, cf_df, pnl_df, ticker):
    for df in [bs_df, cf_df, pnl_df]:
        for col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    
    dates = sorted(set(bs_df.columns) & set(cf_df.columns) & set(pnl_df.columns))
    stock = yf.Ticker(ticker)
    shares_outstanding = stock.info.get('sharesOutstanding', 0)
    crore_to_rupee = 10000000
    
    metrics_list = []
    
    for i, date in enumerate(dates):
        equity = bs_df.loc['Equity Capital'][date] * crore_to_rupee
        reserves = bs_df.loc['Reserves'][date] * crore_to_rupee
        borrowings = bs_df.loc['Borrowings\xa0+'][date] * crore_to_rupee
        other_liab = bs_df.loc['Other Liabilities\xa0+'][date] * crore_to_rupee
        total_assets = bs_df.loc['Total Assets'][date] * crore_to_rupee
        
        sales = pnl_df.loc['Sales\xa0+'][date] * crore_to_rupee
        op_profit = pnl_df.loc['Operating Profit'][date] * crore_to_rupee
        net_profit = pnl_df.loc['Net Profit\xa0+'][date] * crore_to_rupee
        eps = pnl_df.loc['EPS in Rs'][date]
        
        cash_op = cf_df.loc['Cash from Operating Activity\xa0+'][date] * crore_to_rupee
        cash_inv = cf_df.loc['Cash from Investing Activity\xa0+'][date] * crore_to_rupee
        
        book_value = equity + reserves
        price = get_closest_price(ticker, date)
        if price is not None:
            market_cap = price * shares_outstanding
        else:
            market_cap = net_profit * 20
            price = market_cap / shares_outstanding if shares_outstanding != 0 else None
        
        pe_ratio = price / eps if eps != 0 else None
        pb_ratio = market_cap / book_value if book_value != 0 else None
        ps_ratio = market_cap / sales if sales != 0 else None
        
        op_margin = op_profit / sales if sales != 0 else None
        net_margin = net_profit / sales if sales != 0 else None
        roe = net_profit / book_value if book_value != 0 else None
        
        rev_growth = None
        earn_growth = None
        bv_growth = None
        if i > 0:
            prev_date = dates[i-1]
            prev_sales = pnl_df.loc['Sales\xa0+'][prev_date] * crore_to_rupee
            prev_profit = pnl_df.loc['Net Profit\xa0+'][prev_date] * crore_to_rupee
            prev_bv = (bs_df.loc['Equity Capital'][prev_date] + bs_df.loc['Reserves'][prev_date]) * crore_to_rupee
            prev_price = get_closest_price(ticker, prev_date)
            if prev_price is not None:
                market_cap_prev = prev_price * shares_outstanding
            else:
                market_cap_prev = prev_profit * 20
                prev_price = market_cap_prev / shares_outstanding if shares_outstanding != 0 else None
            
            rev_growth = (sales - prev_sales) / prev_sales if prev_sales != 0 else None
            earn_growth = (net_profit - prev_profit) / prev_profit if prev_profit != 0 else None
            bv_growth = (book_value - prev_bv) / prev_bv if prev_bv != 0 else None
        
        current_assets = bs_df.loc['Other Assets\xa0+'][date] * crore_to_rupee
        current_liab = other_liab
        current_ratio = current_assets / current_liab if current_liab != 0 else None
        
        de_ratio = borrowings / book_value if book_value != 0 else None
        
        fcf = cash_op + cash_inv
        fcf_per_share = fcf / shares_outstanding if shares_outstanding != 0 else None
        
        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=date,
            price_to_earnings_ratio=round(pe_ratio, 2) if pe_ratio else None,
            price_to_book_ratio=round(pb_ratio, 2) if pb_ratio else None,
            price_to_sales_ratio=round(ps_ratio, 2) if ps_ratio else None,
            operating_margin=round(op_margin, 2) if op_margin else None,
            net_margin=round(net_margin, 2) if net_margin else None,
            return_on_equity=round(roe, 2) if roe else None,
            revenue_growth=round(rev_growth, 2) if rev_growth else None,
            earnings_growth=round(earn_growth, 2) if earn_growth else None,
            book_value_growth=round(bv_growth, 2) if bv_growth else None,
            current_ratio=round(current_ratio, 2) if current_ratio else None,
            debt_to_equity=round(de_ratio, 2) if de_ratio else None,
            free_cash_flow_per_share=round(fcf_per_share, 2) if fcf_per_share else None,
            earnings_per_share=round(eps, 2) if eps else None
        )
        metrics_list.append(metrics)
    
    return FinancialMetricsResponse(financial_metrics=metrics_list)

def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> List[FinancialMetrics]:
    """Fetch financial metrics for Indian stocks using local CSV data and Yahoo Finance."""
    csv_ticker = ticker.split('.NS')[0]
    yf_ticker = f"{csv_ticker}.NS"

    try:
        bs_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_BS.csv', index_col=0)
        cf_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_CF.csv', index_col=0)
        pnl_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_PNL.csv', index_col=0)
    except FileNotFoundError as e:
        print(f"Error: CSV files for {csv_ticker} not found - {e}")
        return []

    metrics_response = calculate_financial_metrics(bs_df, cf_df, pnl_df, yf_ticker)
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    end_date_dt = pd.to_datetime(end_date)
    filtered_metrics = [m for m in financial_metrics if pd.to_datetime(m.report_period) <= end_date_dt]
    filtered_metrics.sort(key=lambda x: x.report_period, reverse=True)

    return filtered_metrics[:min(limit, len(filtered_metrics))]

def ben_graham_metrics(
    ticker: str,
    end_date: str,
    period: str = "annual",
    limit: int = 10,
) -> List[BenGrahamMetrics]:
    """Fetch Ben Graham-specific metrics for Indian stocks using local CSV data and Yahoo Finance."""
    csv_ticker = ticker.split('.NS')[0]  # Strip .NS for CSV lookup
    yf_ticker = f"{csv_ticker}.NS"  # Use .NS for Yahoo Finance and output

    try:
        bs_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_BS.csv', index_col=0)
        cf_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_CF.csv', index_col=0)
        pnl_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_PNL.csv', index_col=0)
    except FileNotFoundError as e:
        print(f"Error: CSV files for {csv_ticker} not found - {e}")
        return []

    for df in [bs_df, cf_df, pnl_df]:
        for col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    dates = sorted(set(bs_df.columns) & set(cf_df.columns) & set(pnl_df.columns))
    stock = yf.Ticker(yf_ticker)
    shares_outstanding = stock.info.get('sharesOutstanding', 0)  # Raw shares in units

    if shares_outstanding == 0:
        print(f"Warning: No shares outstanding data for {yf_ticker}")
        return []

    metrics_list = []
    crore_to_rupee = 10000000  # 1 Cr = 10^7 INR

    for date in dates:
        equity = bs_df.loc['Equity Capital'][date] * crore_to_rupee  # Convert to ₹
        reserves = bs_df.loc['Reserves'][date] * crore_to_rupee
        total_assets = bs_df.loc['Total Assets'][date] * crore_to_rupee
        total_liabilities = (bs_df.loc['Borrowings\xa0+'][date] + bs_df.loc['Other Liabilities\xa0+'][date]) * crore_to_rupee
        current_assets = bs_df.loc['Other Assets\xa0+'][date] * crore_to_rupee
        current_liabilities = bs_df.loc['Other Liabilities\xa0+'][date] * crore_to_rupee

        revenue = pnl_df.loc['Sales\xa0+'][date] * crore_to_rupee
        net_income = pnl_df.loc['Net Profit\xa0+'][date] * crore_to_rupee
        eps = pnl_df.loc['EPS in Rs'][date]  # Already in ₹ per share

        cash_fin = cf_df.loc['Cash from Financing Activity\xa0+'][date] * crore_to_rupee
        dividends_per_share = abs(cash_fin) / shares_outstanding if cash_fin < 0 else None  # ₹ per share

        book_value = equity + reserves  # In ₹
        book_value_per_share = book_value / shares_outstanding if shares_outstanding != 0 else None  # ₹ per share

        if book_value_per_share is not None and book_value_per_share < 0:
            print(f"Warning: Negative book value per share for {yf_ticker} on {date}: {book_value_per_share}")

        metrics = BenGrahamMetrics(
            ticker=yf_ticker,
            report_period=date,
            earnings_per_share=eps,
            revenue=revenue,
            net_income=net_income,
            book_value_per_share=book_value_per_share,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            current_assets=current_assets,
            current_liabilities=current_liabilities,
            dividends_per_share=dividends_per_share,
            outstanding_shares=shares_outstanding
        )
        metrics_list.append(metrics)

    end_date_dt = pd.to_datetime(end_date)
    filtered_metrics = [m for m in metrics_list if pd.to_datetime(m.report_period) <= end_date_dt]
    filtered_metrics.sort(key=lambda x: x.report_period, reverse=True)

    return filtered_metrics[:min(limit, len(filtered_metrics))]


def valuation_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 2,  # Need current and previous period for working capital change
) -> List[ValuationMetrics]:
    """Fetch valuation-specific metrics for the given ticker using local CSV data and Yahoo Finance."""
    # Use ticker without .NS for CSV, add .NS for Yahoo Finance and output
    csv_ticker = ticker.split('.NS')[0]
    yf_ticker = f"{csv_ticker}.NS"

    try:
        bs_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_BS.csv', index_col=0)
        cf_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_CF.csv', index_col=0)
        pnl_df = pd.read_csv(f'src/tools/financial_data/{csv_ticker}_PNL.csv', index_col=0)
    except FileNotFoundError as e:
        print(f"Error: CSV files for {csv_ticker} not found - {e}")
        return []

    # Clean numeric data
    for df in [bs_df, cf_df, pnl_df]:
        for col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # Get common dates across all statements
    dates = sorted(set(bs_df.columns) & set(cf_df.columns) & set(pnl_df.columns))
    if not dates:
        print(f"Error: No overlapping dates found for {csv_ticker}")
        return []

    # Filter dates up to end_date
    end_date_dt = pd.to_datetime(end_date)
    filtered_dates = [d for d in dates if pd.to_datetime(d) <= end_date_dt]
    filtered_dates.sort(reverse=True)
    if len(filtered_dates) < limit:
        print(f"Warning: Insufficient periods for {csv_ticker}; found {len(filtered_dates)}, need {limit}")
        return []

    # Fetch earnings growth (approximate from net income growth over available periods)
    net_incomes = [pnl_df.loc['Net Profit\xa0+'][d] for d in filtered_dates[:limit] if 'Net Profit\xa0+' in pnl_df.index]
    earnings_growth = None
    if len(net_incomes) >= 2 and net_incomes[1] != 0:
        earnings_growth = (net_incomes[0] - net_incomes[1]) / abs(net_incomes[1])

    # Collect line items for the latest two periods
    line_items = []
    for date in filtered_dates[:limit]:
        net_income = pnl_df.loc['Net Profit\xa0+'][date] if 'Net Profit\xa0+' in pnl_df.index else None
        depreciation = cf_df.loc['Depreciation'][date] if 'Depreciation' in cf_df.index else None
        capex = abs(cf_df.loc['Cash from Investing Activity\xa0+'][date]) if 'Cash from Investing Activity\xa0+' in cf_df.index else None
        free_cash_flow = cf_df.loc['Cash from Operating Activity\xa0+'][date] - capex if 'Cash from Operating Activity\xa0+' in cf_df.index and capex else None
        current_assets = bs_df.loc['Other Assets\xa0+'][date] if 'Other Assets\xa0+' in bs_df.index else None
        current_liabilities = bs_df.loc['Other Liabilities\xa0+'][date] if 'Other Liabilities\xa0+' in bs_df.index else None
        working_capital = current_assets - current_liabilities if current_assets is not None and current_liabilities is not None else None

        line_item = ValuationLineItem(
            ticker=yf_ticker,
            report_period=date,
            free_cash_flow=free_cash_flow,
            net_income=net_income,
            depreciation_and_amortization=depreciation,
            capital_expenditure=capex,
            working_capital=working_capital,
        )
        line_items.append(line_item)

    # Create ValuationMetrics object
    metrics = ValuationMetrics(
        ticker=yf_ticker,
        report_period="ttm" if period == "ttm" else filtered_dates[0],
        earnings_growth=earnings_growth,
        line_items=line_items,
    )

    return [metrics]  # Return as a list for consistency with other functions


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break
            
        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news



def get_market_cap(ticker: str, end_date: str) -> float:
    """Fetch market cap for the given ticker and date in rupees."""
    if not ticker.endswith('.NS'):
        ticker = f"{ticker}.NS"
    stock = yf.Ticker(ticker)
    price = get_closest_price(ticker, end_date)  # INR per share
    shares_outstanding = stock.info.get('sharesOutstanding', 0)  # Units (not millions)
    market_cap = price * shares_outstanding if price and shares_outstanding else None
    
    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
