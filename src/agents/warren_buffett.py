from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from tools.api import buffett_metrics
from utils.llm import call_llm
from utils.progress import progress
from data.models import BuffettFinancialMetrics
from typing import List

class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def warren_buffett_agent(state: AgentState):
    """Analyzes stocks using Buffett's principles and LLM reasoning."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status("warren_buffett_agent", ticker, "Fetching Buffett metrics")
        metrics = buffett_metrics(ticker, end_date, period="annual", limit=5)

        if not metrics:
            progress.update_status("warren_buffett_agent", ticker, "Failed: No metrics found")
            analysis_data[ticker] = {
                "signal": "neutral",
                "score": 0,
                "max_score": 10,
                "fundamental_analysis": {"score": 0, "details": "No metrics available"},
                "consistency_analysis": {"score": 0, "details": "No metrics available"},
                "intrinsic_value_analysis": {"intrinsic_value": None, "details": ["No metrics available"]},
                "market_cap": None,
                "margin_of_safety": None,
            }
            buffett_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": "No financial metrics available for analysis.",
            }
            continue

        progress.update_status("warren_buffett_agent", ticker, "Analyzing fundamentals")
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(metrics)

        total_score = fundamental_analysis["score"] + consistency_analysis["score"]
        max_possible_score = 10

        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis.get("intrinsic_value")
        market_cap = metrics[0].market_cap if metrics and metrics[0].market_cap else None
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap
            if margin_of_safety > 0.3:
                total_score += 2
                max_possible_score += 2

        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status("warren_buffett_agent", ticker, "Generating Buffett analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status("warren_buffett_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "Warren Buffett Agent")

    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis

    return {"messages": [message], "data": state["data"]}

def analyze_fundamentals(metrics: List[BuffettFinancialMetrics]) -> dict[str, any]:
    """Analyze company fundamentals based on Buffett's criteria."""
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data", "metrics": {}}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    if latest_metrics.return_on_equity is not None and latest_metrics.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity is not None:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    if latest_metrics.debt_to_equity is not None and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif latest_metrics.debt_to_equity is not None:
        reasoning.append(f"High debt to equity ratio of {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    if latest_metrics.operating_margin is not None and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin is not None:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    if latest_metrics.current_ratio is not None and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif latest_metrics.current_ratio is not None:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {
        "score": score,
        "details": "; ".join(reasoning),
        "metrics": latest_metrics.model_dump() if latest_metrics else {},
    }

def analyze_consistency(metrics: List[BuffettFinancialMetrics]) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    if len(metrics) < 4:
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    earnings_values = [item.net_income for item in metrics if item.net_income is not None]
    if len(earnings_values) >= 4:
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))
        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        if len(earnings_values) >= 2:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1]) if earnings_values[-1] != 0 else 0
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {"score": score, "details": "; ".join(reasoning)}

def calculate_owner_earnings(metrics: List[BuffettFinancialMetrics]) -> dict[str, any]:
    """Calculate owner earnings."""
    if not metrics or len(metrics) < 1:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"], "components": {}}

    latest = metrics[0]
    net_income = latest.net_income
    depreciation = latest.depreciation_and_amortization
    capex = latest.capital_expenditure

    if not all(x is not None for x in [net_income, depreciation, capex]):
        return {
            "owner_earnings": None,
            "details": ["Missing components for owner earnings calculation"],
            "components": {
                "net_income": net_income,
                "depreciation": depreciation,
                "maintenance_capex": None,
            },
        }

    maintenance_capex = capex * 0.75
    owner_earnings = net_income + depreciation - maintenance_capex

    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex,
        },
        "details": ["Owner earnings calculated successfully"],
    }

def calculate_intrinsic_value(metrics: List[BuffettFinancialMetrics]) -> dict[str, any]:
    """Calculate intrinsic value using DCF with owner earnings."""
    if not metrics:
        return {
            "intrinsic_value": None,
            "details": ["Insufficient data for valuation"],
            "owner_earnings": None,
            "assumptions": {},
        }

    earnings_data = calculate_owner_earnings(metrics)
    owner_earnings = earnings_data.get("owner_earnings")
    if not owner_earnings:
        return {
            "intrinsic_value": None,
            "details": earnings_data.get("details", ["No owner earnings available"]),
            "owner_earnings": None,
            "assumptions": {},
        }

    shares_outstanding = metrics[0].outstanding_shares if metrics and metrics[0].outstanding_shares else None
    if not shares_outstanding:
        return {
            "intrinsic_value": None,
            "details": ["Missing shares outstanding data"],
            "owner_earnings": owner_earnings,
            "assumptions": {},
        }

    growth_rate = 0.05
    discount_rate = 0.09
    terminal_multiple = 12
    projection_years = 10

    future_value = 0
    for year in range(1, projection_years + 1):
        future_earnings = owner_earnings * (1 + growth_rate) ** year
        present_value = future_earnings / (1 + discount_rate) ** year
        future_value += present_value

    terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / (1 + discount_rate) ** projection_years
    intrinsic_value = future_value + terminal_value

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years,
        },
        "details": ["Intrinsic value calculated using DCF model with owner earnings"],
    }

def generate_buffett_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with Buffett's principles."""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Warren Buffett AI agent. Decide on investment signals based on Warren Buffett’s principles:

                Circle of Competence: Only invest in businesses you understand
                Margin of Safety: Buy well below intrinsic value
                Economic Moat: Prefer companies with lasting advantages
                Quality Management: Look for conservative, shareholder-oriented teams
                Financial Strength: Low debt, strong returns on equity
                Long-term Perspective: Invest in businesses, not just stocks

                Rules:
                - Buy only if margin of safety > 30%
                - Focus on owner earnings and intrinsic value
                - Prefer consistent earnings growth
                - Avoid high debt or poor management
                - Hold good businesses long term
                - Sell when fundamentals deteriorate or the valuation is too high
                """,
            ),
            (
                "human",
                """Based on the following data, create the investment signal as Warren Buffett would.

                Analysis Data for {ticker}:
                {analysis_data}

                Return the trading signal in the following JSON format:
                {{
                  "signal": "bullish/bearish/neutral",
                  "confidence": float (0-100),
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=WarrenBuffettSignal,
        agent_name="warren_buffett_agent",
        default_factory=create_default_warren_buffett_signal,
    )