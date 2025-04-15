import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary
import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools
from llm.models import LLM_ORDER, get_model_info
from utils.analysts import ANALYST_ORDER
from main import run_hedge_fund
from tools.api import get_prices, get_financial_metrics, ben_graham_metrics, buffett_metrics, valuation_metrics
from utils.display import print_backtest_results, format_backtest_row
from typing_extensions import Callable

init(autoreset=True)


class Backtester:
    def __init__(
        self,
        agent: Callable,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str = "gpt-4o",
        model_provider: str = "OpenAI",
        selected_analysts: list[str] = [],
        initial_margin_requirement: float = 0.5,
    ):
        """
        Initialize backtester for Indian stocks.

        :param agent: Trading agent (run_hedge_fund).
        :param tickers: List of tickers (without .NS).
        :param start_date: Start date (YYYY-MM-DD).
        :param end_date: End date (YYYY-MM-DD).
        :param initial_capital: Capital in INR.
        :param model_name: LLM model.
        :param model_provider: LLM provider.
        :param selected_analysts: Analyst names.
        :param initial_margin_requirement: Margin ratio.
        """
        self.agent = agent
        self.tickers = [ticker if ticker.endswith('.NS') else f"{ticker}.NS" for ticker in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.margin_ratio = initial_margin_requirement

        self.portfolio = {
            "cash": initial_capital,
            "margin_requirement": initial_margin_requirement,
            "margin_used": 0.0,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0,
                } for ticker in self.tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,
                    "short": 0.0,
                } for ticker in self.tickers
            }
        }
        self.margin_ratio = initial_margin_requirement
        self.portfolio_values = []
        self.last_prices = {}  # Store last available prices

    def execute_trade(self, ticker: str, action: str, quantity: float, current_price: float):
        """
        Execute trades with long/short support.

        :param ticker: Ticker (with .NS).
        :param action: buy, sell, short, cover.
        :param quantity: Shares to trade.
        :param current_price: Price (INR).
        :return: Shares traded.
        """
        if quantity <= 0 or not ticker.endswith('.NS'):
            return 0

        quantity = int(quantity)
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                old_shares = position["long"]
                old_cost_basis = position["long_cost_basis"]
                new_shares = quantity
                total_shares = old_shares + new_shares

                if total_shares > 0:
                    position["long_cost_basis"] = ((old_cost_basis * old_shares) + cost) / total_shares

                position["long"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                max_quantity = int(self.portfolio["cash"] / current_price)
                if max_quantity > 0:
                    cost = max_quantity * current_price
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity

                    if total_shares > 0:
                        position["long_cost_basis"] = ((old_cost_basis * old_shares) + cost) / total_shares

                    position["long"] += max_quantity
                    self.portfolio["cash"] -= cost
                    return max_quantity
                return 0

        elif action == "sell":
            quantity = min(quantity, position["long"])
            if quantity > 0:
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0 else 0
                realized_gain = (current_price - avg_cost_per_share) * quantity
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain

                position["long"] -= quantity
                self.portfolio["cash"] += quantity * current_price

                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0

                return quantity
            return 0

        elif action == "short":
            proceeds = quantity * current_price
            margin_required = proceeds * self.margin_ratio
            if margin_required <= self.portfolio["cash"]:
                old_short_shares = position["short"]
                old_cost_basis = position["short_cost_basis"]
                new_shares = quantity
                total_shares = old_short_shares + new_shares

                if total_shares > 0:
                    position["short_cost_basis"] = ((old_cost_basis * old_short_shares) + (current_price * new_shares)) / total_shares

                position["short"] += quantity
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= margin_required
                return quantity
            else:
                max_quantity = int(self.portfolio["cash"] / (current_price * self.margin_ratio)) if self.margin_ratio > 0 else 0
                if max_quantity > 0:
                    proceeds = current_price * max_quantity
                    margin_required = proceeds * self.margin_ratio

                    old_short_shares = position["short"]
                    old_cost_basis = position["short_cost_basis"]
                    total_shares = old_short_shares + max_quantity

                    if total_shares > 0:
                        position["short_cost_basis"] = ((old_cost_basis * old_short_shares) + (current_price * max_quantity)) / total_shares

                    position["short"] += max_quantity
                    position["short_margin_used"] += margin_required
                    self.portfolio["margin_used"] += margin_required
                    self.portfolio["cash"] += proceeds
                    self.portfolio["cash"] -= margin_required
                    return max_quantity
                return 0

        elif action == "cover":
            quantity = min(quantity, position["short"])
            if quantity > 0:
                cover_cost = quantity * current_price
                avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0
                realized_gain = (avg_short_price - current_price) * quantity
                self.portfolio["realized_gains"][ticker]["short"] += realized_gain

                margin_to_release = (quantity / position["short"]) * position["short_margin_used"] if position["short"] > 0 else 0
                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= cover_cost

                if position["short"] == 0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                return quantity
            return 0

        return 0

    def calculate_portfolio_value(self, current_prices):
        """
        Calculate portfolio value in INR.

        :param current_prices: Dict of ticker to price.
        :return: Total value.
        """
        total_value = self.portfolio["cash"]

        for ticker in self.tickers:
            position = self.portfolio["positions"][ticker]
            price = current_prices.get(ticker, 0)

            # Long positions
            total_value += position["long"] * price

            # Short positions: unrealized gain/loss
            if position["short"] > 0:
                unrealized_gain = position["short"] * (position["short_cost_basis"] - price)
                total_value += unrealized_gain

        return total_value

    def prefetch_data(self):
        """Pre-fetch data for agents."""
        print("\nPre-fetching data for Indian stocks...")
        start_date_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        prefetch_start = (start_date_dt - relativedelta(years=5)).strftime("%Y-%m-%d")

        for ticker in self.tickers:
            prices = get_prices(ticker, prefetch_start, self.end_date)
            if prices:
                self.last_prices[ticker] = prices[-1].close
            if 'ben_graham_agent' in self.selected_analysts:
                ben_graham_metrics(ticker, self.end_date, period="annual", limit=10)
            if 'warren_buffett_agent' in self.selected_analysts:
                buffett_metrics(ticker, self.end_date, period="annual", limit=5)
            if 'fundamentals_agent' in self.selected_analysts:
                get_financial_metrics(ticker, self.end_date, period="ttm", limit=10)
            if 'valuation_agent' in self.selected_analysts:
                valuation_metrics(ticker, self.end_date, period="ttm", limit=2)
            if 'technicals_agent' in self.selected_analysts:
                get_prices(ticker, prefetch_start, self.end_date)

        print("Data pre-fetch complete.")

    def run_backtest(self):
        """Run backtest."""
        self.prefetch_data()
        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []
        performance_metrics = {
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'max_drawdown': None
        }

        print("\nStarting backtest for Indian stocks...")

        if dates.size > 0:
            self.portfolio_values = [{"Date": dates[0], "Portfolio Value": self.initial_capital}]
        else:
            self.portfolio_values = []

        for current_date in dates:
            current_date_str = current_date.strftime("%Y-%m-%d")
            lookback_start = (current_date - relativedelta(years=1)).strftime("%Y-%m-%d")
            previous_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")

            if lookback_start >= current_date_str:
                continue

            try:
                current_prices = {}
                missing_data = False
                for ticker in self.tickers:
                    prices = get_prices(ticker, previous_date_str, current_date_str)
                    if prices:
                        current_prices[ticker] = prices[-1].close
                        self.last_prices[ticker] = prices[-1].close
                    elif ticker in self.last_prices:
                        print(f"Warning: No price data for {ticker} on {current_date_str}. Using last price: ₹{self.last_prices[ticker]:,.2f}")
                        current_prices[ticker] = self.last_prices[ticker]
                    else:
                        print(f"Error: No price data for {ticker} on {current_date_str} and no prior data.")
                        missing_data = True
                        break
                if missing_data:
                    continue

            except Exception as e:
                print(f"Error fetching prices for {current_date_str}: {e}")
                continue

            output = self.agent(
                tickers=[t.split('.NS')[0] for t in self.tickers],
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio,
                model_name=self.model_name,
                model_provider=self.model_provider,
                selected_analysts=self.selected_analysts,
                show_reasoning=False
            )
            decisions = output.get("decisions", {})
            analyst_signals = output.get("analyst_signals", {})

            executed_trades = {}
            for ticker in self.tickers:
                ticker_no_ns = ticker.split('.NS')[0]
                decision = decisions.get(ticker_no_ns, {"action": "hold", "quantity": 0})
                action = decision.get("action", "hold").lower()
                quantity = decision.get("quantity", 0)

                executed_quantity = self.execute_trade(ticker, action, quantity, current_prices[ticker])
                executed_trades[ticker] = executed_quantity

            total_value = self.calculate_portfolio_value(current_prices)
            total_position_value = sum(
                (self.portfolio["positions"][t]["long"] - self.portfolio["positions"][t]["short"]) * current_prices.get(t, 0)
                for t in self.tickers
            )

            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value
            })

            date_rows = []
            for ticker in self.tickers:
                ticker_no_ns = ticker.split('.NS')[0]
                ticker_signals = {
                    agent: signals.get(ticker_no_ns, {})
                    for agent, signals in analyst_signals.items()
                }

                bullish_count = sum(1 for s in ticker_signals.values() if s.get("signal", "").lower() == "bullish")
                bearish_count = sum(1 for s in ticker_signals.values() if s.get("signal", "").lower() == "bearish")
                neutral_count = sum(1 for s in ticker_signals.values() if s.get("signal", "").lower() == "neutral")

                pos = self.portfolio["positions"][ticker]
                net_position_value = (pos["long"] - pos["short"]) * current_prices.get(ticker, 0)

                action = decisions.get(ticker_no_ns, {}).get("action", "hold")
                quantity = executed_trades.get(ticker, 0)

                date_rows.append(
                    format_backtest_row(
                        date=current_date_str,
                        ticker=ticker_no_ns,
                        action=action,
                        quantity=quantity,
                        price=current_prices.get(ticker, 0),
                        shares_owned=pos["long"] - pos["short"],
                        position_value=net_position_value,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                )

            total_realized_gains = sum(
                self.portfolio["realized_gains"][t]["long"] + self.portfolio["realized_gains"][t]["short"]
                for t in self.tickers
            )
            portfolio_return = ((total_value - self.initial_capital) / self.initial_capital) * 100

            date_rows.append(
                format_backtest_row(
                    date=current_date_str,
                    ticker="",
                    action="",
                    quantity=0,
                    price=0,
                    shares_owned=0,
                    position_value=0,
                    bullish_count=0,
                    bearish_count=0,
                    neutral_count=0,
                    is_summary=True,
                    total_value=total_value,
                    return_pct=portfolio_return,
                    cash_balance=self.portfolio["cash"],
                    total_position_value=total_position_value,
                    sharpe_ratio=performance_metrics["sharpe_ratio"],
                    sortino_ratio=performance_metrics["sortino_ratio"],
                    max_drawdown=performance_metrics["max_drawdown"],
                )
            )

            table_rows.extend(date_rows)
            print_backtest_results(table_rows)

            if len(self.portfolio_values) > 1:
                self._update_performance_metrics(performance_metrics)

        return performance_metrics

    def _update_performance_metrics(self, performance_metrics):
        """Update performance metrics."""
        try:
            values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
            values_df["Daily Return"] = values_df["Portfolio Value"].pct_change().fillna(0)
            clean_returns = values_df["Daily Return"]

            if len(clean_returns) < 2:
                return

            daily_risk_free_rate = 0.06 / 252
            excess_returns = clean_returns - daily_risk_free_rate
            mean_excess_return = excess_returns.mean()
            std_excess_return = excess_returns.std()

            if std_excess_return > 1e-12:
                performance_metrics["sharpe_ratio"] = np.sqrt(252) * (mean_excess_return / std_excess_return)
            else:
                performance_metrics["sharpe_ratio"] = 0.0

            negative_returns = excess_returns[excess_returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                if downside_std > 1e-12:
                    performance_metrics["sortino_ratio"] = np.sqrt(252) * (mean_excess_return / downside_std)
                else:
                    performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0
            else:
                performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0

            rolling_max = values_df["Portfolio Value"].cummax()
            drawdown = (rolling_max - values_df["Portfolio Value"]) / rolling_max
            performance_metrics["max_drawdown"] = drawdown.max() * 100 if not drawdown.empty else 0
        except Exception as e:
            print(f"Error updating performance metrics: {e}")
            performance_metrics.update({"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None})

    def analyze_performance(self):
        """Analyze performance."""
        if not self.portfolio_values:
            print("No portfolio data.")
            return pd.DataFrame()

        try:
            performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")
            if performance_df.empty:
                print("No performance data.")
                return performance_df

            final_portfolio_value = performance_df["Portfolio Value"].iloc[-1]
            total_realized_gains = sum(
                self.portfolio["realized_gains"][t]["long"] + self.portfolio["realized_gains"][t]["short"]
                for t in self.tickers
            )
            total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100

            print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO PERFORMANCE SUMMARY (INR):{Style.RESET_ALL}")
            print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")
            print(f"Total Realized Gains/Losses: {Fore.GREEN if total_realized_gains >= 0 else Fore.RED}₹{total_realized_gains:,.2f}{Style.RESET_ALL}")

            try:
                plt.figure(figsize=(12, 6))
                plt.plot(performance_df.index, performance_df["Portfolio Value"], color="blue")
                plt.title("Portfolio Value Over Time (INR)")
                plt.ylabel("Portfolio Value (₹)")
                plt.xlabel("Date")
                plt.grid(True)
                plt.show()
            except Exception as e:
                print(f"Error plotting portfolio value: {e}")

            performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change().fillna(0)
            daily_rf = 0.06 / 252
            mean_daily_return = performance_df["Daily Return"].mean()
            std_daily_return = performance_df["Daily Return"].std()

            annualized_sharpe = np.sqrt(252) * ((mean_daily_return - daily_rf) / std_daily_return) if std_daily_return != 0 else 0
            print(f"Sharpe Ratio: {Fore.YELLOW}{annualized_sharpe:.2f}{Style.RESET_ALL}")

            negative_returns = performance_df["Daily Return"][performance_df["Daily Return"] < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                annualized_sortino = np.sqrt(252) * ((mean_daily_return - daily_rf) / downside_std) if downside_std != 0 else float('inf')
            else:
                annualized_sortino = float('inf') if mean_daily_return > daily_rf else 0
            print(f"Sortino Ratio: {Fore.YELLOW}{annualized_sortino:.2f}{Style.RESET_ALL}")

            rolling_max = performance_df["Portfolio Value"].cummax()
            drawdown = (rolling_max - performance_df["Portfolio Value"]) / rolling_max
            max_drawdown = drawdown.max() * 100 if not drawdown.empty else 0
            max_drawdown_date = drawdown.idxmax() if not drawdown.empty and pd.notnull(drawdown.idxmax()) else performance_df.index[0]
            print(f"Maximum Drawdown: {Fore.RED}{max_drawdown:.2f}%{Style.RESET_ALL} (on {max_drawdown_date.strftime('%Y-%m-%d')})")

            winning_days = len(performance_df[performance_df["Daily Return"] > 0])
            total_days = max(len(performance_df) - 1, 1)
            win_rate = (winning_days / total_days) * 100
            print(f"Win Rate: {Fore.GREEN}{win_rate:.2f}%{Style.RESET_ALL}")

            positive_returns = performance_df[performance_df["Daily Return"] > 0]["Daily Return"]
            negative_returns = performance_df[performance_df["Daily Return"] < 0]["Daily Return"]
            avg_win = positive_returns.mean() if not positive_returns.empty else 0
            avg_loss = abs(negative_returns.mean()) if not negative_returns.empty else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else (float('inf') if avg_win > 0 else 0)
            print(f"Win/Loss Ratio: {Fore.GREEN}{win_loss_ratio:.2f}{Style.RESET_ALL}")

            returns_binary = (performance_df["Daily Return"] > 0).astype(int)
            max_consecutive_wins = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 1), default=0)
            max_consecutive_losses = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 0), default=0)
            print(f"Max Consecutive Wins: {Fore.GREEN}{max_consecutive_wins}{Style.RESET_ALL}")
            print(f"Max Consecutive Losses: {Fore.RED}{max_consecutive_losses}{Style.RESET_ALL}")

            return performance_df

        except Exception as e:
            print(f"Error in performance analysis: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Indian stocks")
    parser.add_argument(
        "--tickers",  # Support --ticker as alias
        "--ticker",
        type=str,
        required=True,
        help="Comma-separated tickers (e.g., RELIANCE,INFY,TCS)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(years=1)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Capital in INR (default: 100000)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.5,
        help="Margin ratio (default: 0.5)",
    )

    args = parser.parse_args()
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    selected_analysts = questionary.checkbox(
        "Select analysts:",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\nPress Space to select.\nPress 'a' to toggle all.\nPress Enter to start.",
        validate=lambda x: len(x) > 0 or "Select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not selected_analysts:
        print("\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        print(f"\nSelected analysts: {', '.join(Fore.GREEN + c.title().replace('_', ' ') + Style.RESET_ALL for c in selected_analysts)}")

    model_choice = questionary.select(
        "Select LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
        style=questionary.Style([
            ("selected", "fg:green bold"),
            ("pointer", "fg:green bold"),
            ("highlighted", "fg:green"),
            ("answer", "fg:green bold"),
        ])
    ).ask()

    if not model_choice:
        print("\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        model_info = get_model_info(model_choice)
        model_provider = model_info.provider.value if model_info else "Unknown"
        print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_choice,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
    )

    performance_metrics = backtester.run_backtest()
    performance_df = backtester.analyze_performance()