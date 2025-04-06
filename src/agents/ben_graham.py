from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import math
import yfinance as yf
from tools.api import ben_graham_metrics, get_closest_price
from typing import List
from data.models import BenGrahamMetrics

class BenGrahamSignal(BaseModel):
    signal: Literal["neutral", "bullish", "bearish"]
    confidence: float
    reasoning: str

def ben_graham_agent(state: AgentState):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, "Fetching Ben Graham metrics")
        financial_metrics = ben_graham_metrics(
            ticker=ticker,
            end_date=end_date,
            period="annual",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status("ben_graham_agent", ticker, "Failed: No metrics found")
            continue

        yf_ticker = financial_metrics[0].ticker  # Already includes .NS

        progress.update_status("ben_graham_agent", ticker, "Getting market cap")
        price = get_closest_price(ticker, end_date)  # ₹ per share
        shares_outstanding = financial_metrics[0].outstanding_shares  # Raw shares from yfinance
        market_cap = price * shares_outstanding if price and shares_outstanding else None  # ₹

        if not market_cap and financial_metrics:
            latest_metrics = financial_metrics[0]
            market_cap = latest_metrics.net_income * 20  # ₹ (net_income already in ₹)

        if not market_cap:
            progress.update_status("ben_graham_agent", ticker, "Failed: Could not determine market cap")
            continue

        progress.update_status("ben_graham_agent", ticker, "Analyzing earnings stability")
        earnings_analysis = analyze_earnings_stability(financial_metrics)

        progress.update_status("ben_graham_agent", ticker, "Analyzing financial strength")
        strength_analysis = analyze_financial_strength(financial_metrics)

        progress.update_status("ben_graham_agent", ticker, "Analyzing Graham valuation")
        valuation_analysis = analyze_valuation_graham(financial_metrics, market_cap)

        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 16

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
            "earnings_analysis": earnings_analysis,
            "strength_analysis": strength_analysis,
            "valuation_analysis": valuation_analysis
        }

        progress.update_status("ben_graham_agent", ticker, "Generating Graham-style analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        graham_analysis[ticker] = {
            "signal": graham_output.signal,
            "confidence": graham_output.confidence,
            "reasoning": graham_output.reasoning
        }

        progress.update_status("ben_graham_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis

    return {"messages": [message], "data": state["data"]}

def analyze_earnings_stability(financial_metrics: List[BenGrahamMetrics]) -> dict:
    score = 0
    details = []
    eps_vals = [m.earnings_per_share for m in financial_metrics if m.earnings_per_share is not None]

    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
    else:
        positive_eps_years = sum(1 for e in eps_vals if e > 0)
        total_eps_years = len(eps_vals)
        if positive_eps_years == total_eps_years:
            score += 3
            details.append("EPS was positive in all periods.")
        elif positive_eps_years >= (total_eps_years * 0.8):
            score += 2
            details.append("EPS was positive in most periods.")
        else:
            details.append("EPS was negative in some periods.")
        if eps_vals[-1] > eps_vals[0]:
            score += 1
            details.append("EPS grew over time.")
        else:
            details.append("EPS did not grow over time.")
    return {"score": score, "details": "; ".join(details)}

def analyze_financial_strength(financial_metrics: List[BenGrahamMetrics]) -> dict:
    score = 0
    details = []
    latest = financial_metrics[0]

    if latest.current_liabilities is not None and latest.current_liabilities > 0:
        current_ratio = latest.current_assets / latest.current_liabilities  # ₹ / ₹ = unitless
        details.append(f"Current ratio = {current_ratio:.2f}")
        if current_ratio >= 2.0:
            score += 2
            details.append("Current ratio >= 2.0.")
        elif current_ratio >= 1.5:
            score += 1
            details.append("Current ratio >= 1.5.")
    else:
        details.append("Current liabilities missing or zero.")

    if latest.total_assets is not None and latest.total_assets > 0:
        debt_ratio = latest.total_liabilities / latest.total_assets  # ₹ / ₹ = unitless
        details.append(f"Debt ratio = {debt_ratio:.2f}")
        if debt_ratio < 0.5:
            score += 2
            details.append("Debt ratio < 0.5.")
        elif debt_ratio < 0.8:
            score += 1
            details.append("Debt ratio < 0.8.")
    else:
        details.append("Total assets missing or zero.")

    div_periods = [m.dividends_per_share for m in financial_metrics if m.dividends_per_share is not None]
    if div_periods:
        div_paid_years = sum(1 for d in div_periods if d > 0)
        if div_paid_years >= (len(div_periods) // 2 + 1):
            score += 1
            details.append("Dividends paid in majority of years.")
        else:
            details.append("Dividends paid in some years.")
    else:
        details.append("No dividend data available.")

    return {"score": score, "details": "; ".join(details)}

def analyze_valuation_graham(financial_metrics: List[BenGrahamMetrics], market_cap: float) -> dict:
    if not financial_metrics or not market_cap or market_cap <= 0:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    score = 0
    details = []
    latest = financial_metrics[0]

    # NCAV Analysis (all in ₹)
    if latest.current_assets is not None and latest.total_liabilities is not None:
        ncav = latest.current_assets - latest.total_liabilities
        if latest.outstanding_shares > 0:
            ncav_ps = ncav / latest.outstanding_shares  # ₹ per share
            price_ps = market_cap / latest.outstanding_shares  # ₹ per share
            details.append(f"NCAV = ₹{ncav:,.2f}, NCAV/Share = ₹{ncav_ps:.2f}, Price/Share = ₹{price_ps:.2f}")
            if ncav > market_cap:
                score += 4
                details.append("NCAV > Market Cap (strong buy signal).")
            elif ncav_ps >= (price_ps * 0.67):
                score += 2
                details.append("NCAV >= 2/3 of Price (moderate value).")
            else:
                details.append("NCAV below 2/3 of Price.")
        else:
            details.append("No shares outstanding for NCAV/Share calculation.")
    else:
        details.append("Missing current_assets or total_liabilities for NCAV.")

    # Graham Number Analysis
    if latest.earnings_per_share is not None and latest.book_value_per_share is not None:
        if latest.earnings_per_share > 0 and latest.book_value_per_share > 0:
            graham_number = math.sqrt(22.5 * latest.earnings_per_share * latest.book_value_per_share)  # ₹ per share
            current_price = market_cap / latest.outstanding_shares if latest.outstanding_shares > 0 else 0  # ₹ per share
            if current_price > 0:
                margin_of_safety = (graham_number - current_price) / current_price
                details.append(f"Graham Number = ₹{graham_number:.2f}, Current Price = ₹{current_price:.2f}, Margin of Safety = {margin_of_safety:.2%}")
                if margin_of_safety > 0.5:
                    score += 3
                    details.append("Margin of Safety > 50% (undervalued).")
                elif margin_of_safety > 0.2:
                    score += 1
                    details.append("Margin of Safety > 20% (some value).")
                else:
                    details.append("Margin of Safety <= 20% (limited value).")
            else:
                details.append("Current price invalid for margin of safety.")
        else:
            details.append("EPS or Book Value/Share not positive for Graham Number.")
    else:
        details.append("Missing EPS or Book Value/Share for Graham Number.")

    return {"score": score, "details": "; ".join(details)}

def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BenGrahamSignal:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Benjamin Graham AI agent, making investment decisions using his principles:
            1. Insist on a margin of safety by buying below intrinsic value (e.g., using Graham Number, net-net).
            2. Emphasize the company's financial strength (low leverage, ample current assets).
            3. Prefer stable earnings over multiple years.
            4. Consider dividend record for extra safety.
            5. Avoid speculative or high-growth assumptions; focus on proven metrics.
                        
            Return a rational recommendation: bullish, bearish, or neutral, with a confidence level (0-100) and concise reasoning.
            """
        ),
        (
            "human",
            """Based on the following analysis, create a Graham-style investment signal:

            Analysis Data for {ticker}:
            {analysis_data}

            Return JSON exactly in this format:
            {{
              "signal": "bullish" or "bearish" or "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_ben_graham_signal,
    )