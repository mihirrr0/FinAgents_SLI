from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json
from tools.api import valuation_metrics, get_market_cap
from data.models import ValuationMetrics
from typing import List

def valuation_agent(state: AgentState):
    """Performs detailed valuation analysis using multiple methodologies for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    valuation_analysis = {}

    for ticker in tickers:
        progress.update_status("valuation_agent", ticker, "Fetching valuation metrics")
        financial_metrics = valuation_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=2,
        )

        if not financial_metrics:
            progress.update_status("valuation_agent", ticker, "Failed: No valuation metrics found")
            continue

        metrics = financial_metrics[0]
        if len(metrics.line_items) < 2:
            progress.update_status("valuation_agent", ticker, "Failed: Insufficient financial line items")
            continue

        current_line_item = metrics.line_items[0]
        previous_line_item = metrics.line_items[1]

        # Debug: Log input metrics (in crores)


        # Convert crores to rupees for consistency with market cap
        crore_to_rupee = 10000000  # 1 Cr = 10^7 INR
        fcf_rupees = (current_line_item.free_cash_flow or 0) * crore_to_rupee
        net_income_rupees = (current_line_item.net_income or 0) * crore_to_rupee
        depreciation_rupees = (current_line_item.depreciation_and_amortization or 0) * crore_to_rupee
        capex_rupees = (current_line_item.capital_expenditure or 0) * crore_to_rupee
        wc_current_rupees = (current_line_item.working_capital or 0) * crore_to_rupee
        wc_previous_rupees = (previous_line_item.working_capital or 0) * crore_to_rupee

        progress.update_status("valuation_agent", ticker, "Calculating owner earnings")
        working_capital_change = wc_current_rupees - wc_previous_rupees

        owner_earnings_value = calculate_owner_earnings_value(
            net_income=net_income_rupees,
            depreciation=depreciation_rupees,
            capex=capex_rupees,
            working_capital_change=working_capital_change,
            growth_rate=metrics.earnings_growth or 0.05,
            required_return=0.15,
            margin_of_safety=0.25,
        )

        progress.update_status("valuation_agent", ticker, "Calculating DCF value")
        dcf_value = calculate_intrinsic_value(
            free_cash_flow=fcf_rupees,
            growth_rate=metrics.earnings_growth or 0.05,
            discount_rate=0.10,
            terminal_growth_rate=0.03,
            num_years=5,
        )

        progress.update_status("valuation_agent", ticker, "Comparing to market value")
        market_cap = get_market_cap(ticker=ticker, end_date=end_date)

        if not market_cap:
            progress.update_status("valuation_agent", ticker, "Failed: Could not determine market cap")
            continue

        dcf_gap = (dcf_value - market_cap) / market_cap
        owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
        valuation_gap = (dcf_gap + owner_earnings_gap) / 2

        if valuation_gap > 0.15:
            signal = "bullish"
        elif valuation_gap < -0.15:
            signal = "bearish"
        else:
            signal = "neutral"

        reasoning = {
            "dcf_analysis": {
                "signal": "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral",
                "details": f"Intrinsic Value: {dcf_value:,.2f}, Market Cap: {market_cap:,.2f}, Gap: {dcf_gap:.1%}",
            },
            "owner_earnings_analysis": {
                "signal": "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral",
                "details": f"Owner Earnings Value: {owner_earnings_value:,.2f}, Market Cap: {market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}",
            },
        }

        confidence = min(round(abs(valuation_gap), 2) * 100, 100)
        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("valuation_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(valuation_analysis),
        name="valuation_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")

    state["data"]["analyst_signals"]["valuation_agent"] = valuation_analysis

    return {
        "messages": [message],
        "data": state["data"],
    }

def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """
    Calculates the intrinsic value using Buffett's Owner Earnings method in rupees.
    """
    if not all([isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]]):
        return 0

    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0

    future_values = []
    for year in range(1, num_years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        discounted_value = future_value / (1 + required_return) ** year
        future_values.append(discounted_value)

    terminal_growth = min(growth_rate, 0.03)
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / (1 + required_return) ** num_years

    intrinsic_value = sum(future_values) + terminal_value_discounted
    value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

    return value_with_safety_margin

def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.03,
    num_years: int = 5,
) -> float:
    """
    Computes the discounted cash flow (DCF) in rupees.
    """
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)

    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    dcf_value = sum(present_values) + terminal_present_value
    return dcf_value