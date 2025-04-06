from pydantic import BaseModel
from typing import Optional

class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str


class PriceResponse(BaseModel):
    ticker: str
    prices: list[Price]



class FinancialMetrics(BaseModel):
    ticker: str
    report_period: str
    price_to_earnings_ratio: float | None
    price_to_book_ratio: float | None
    price_to_sales_ratio: float | None
    operating_margin: float | None
    net_margin: float | None
    return_on_equity: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    book_value_growth: float | None
    current_ratio: float | None
    debt_to_equity: float | None
    free_cash_flow_per_share: float | None
    earnings_per_share: float | None


class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]

class BenGrahamMetrics(BaseModel):
    ticker: str
    report_period: str
    earnings_per_share: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    book_value_per_share: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    dividends_per_share: Optional[float] = None
    outstanding_shares: Optional[float] = None

class BenGrahamMetricsResponse(BaseModel):
    metrics: list[BenGrahamMetrics]

class LineItem(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str

    # Allow additional fields dynamically
    model_config = {"extra": "allow"}


class LineItemResponse(BaseModel):
    search_results: list[LineItem]


class InsiderTrade(BaseModel):
    ticker: str
    issuer: str | None
    name: str | None
    title: str | None
    is_board_director: bool | None
    transaction_date: str | None
    transaction_shares: float | None
    transaction_price_per_share: float | None
    transaction_value: float | None
    shares_owned_before_transaction: float | None
    shares_owned_after_transaction: float | None
    security_title: str | None
    filing_date: str


class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]


class CompanyNews(BaseModel):
    ticker: str
    title: str
    author: str
    source: str
    date: str
    url: str
    sentiment: str | None = None


class CompanyNewsResponse(BaseModel):
    news: list[CompanyNews]


class Position(BaseModel):
    cash: float = 0.0
    shares: int = 0
    ticker: str


class Portfolio(BaseModel):
    positions: dict[str, Position]  # ticker -> Position mapping
    total_cash: float = 0.0


class AnalystSignal(BaseModel):
    signal: str | None = None
    confidence: float | None = None
    reasoning: dict | str | None = None
    max_position_size: float | None = None  # For risk management signals


class TickerAnalysis(BaseModel):
    ticker: str
    analyst_signals: dict[str, AnalystSignal]  # agent_name -> signal mapping


class AgentStateData(BaseModel):
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]  # ticker -> analysis mapping


class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = {"extra": "allow"}
