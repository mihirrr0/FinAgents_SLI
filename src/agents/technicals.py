import math
import json
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_prices, prices_to_df
from utils.progress import progress

##### Technical Analyst #####
def technical_analyst_agent(state: AgentState):
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    technical_analysis = {}

    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Analyzing price data")

        prices = get_prices(ticker=ticker, start_date=start_date, end_date=end_date)
        if not prices:  # Ensure enough data for 6-month momentum
            progress.update_status("technical_analyst_agent", ticker, "Failed: Insufficient price data")
            continue

        prices_df = prices_to_df(prices)

        progress.update_status("technical_analyst_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        # Debugging: Print individual signals
        # print(f"{ticker} Signals: Trend={trend_signals['signal']}, MeanRev={mean_reversion_signals['signal']}, "
        #       f"Momentum={momentum_signals['signal']}, Vol={volatility_signals['signal']}, StatArb={stat_arb_signals['signal']}")

        strategy_weights = {
            "trend": 0.30,  # Increased weight for trend and momentum
            "mean_reversion": 0.15,
            "momentum": 0.30,
            "volatility": 0.15,
            "stat_arb": 0.10,
        }

        progress.update_status("technical_analyst_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )

        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }
        progress.update_status("technical_analyst_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

    return {"messages": state["messages"] + [message], "data": data}


def calculate_trend_signals(prices_df):
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)
    adx = calculate_adx(prices_df, 14)

    short_trend = ema_8 > ema_21  # Simplified: Only short-term trend required
    trend_strength = float(adx["adx"].iloc[-1])

    if short_trend.iloc[-1]:
        signal = "bullish"
        confidence = min((trend_strength - 15) / 35, 1.0)  # ADX > 15 starts confidence, 50 is max
    else:
        signal = "bearish"
        confidence = min((trend_strength - 15) / 35, 1.0)
    if trend_strength < 20:  # Weak trend overrides to neutral
        signal = "neutral"
        confidence = 0.6

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {"adx": float(adx["adx"].iloc[-1]), "trend_strength": trend_strength},
    }


def calculate_mean_reversion_signals(prices_df):
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    if z_score.iloc[-1] < -1.5 and price_vs_bb < 0.3:  # Relaxed thresholds
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 3, 1.0)
    elif z_score.iloc[-1] > 1.5 and price_vs_bb > 0.7:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.6

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df):
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]
    volume_confirmation = volume_momentum.iloc[-1] > 0.9  # Relaxed from 1.0

    if momentum_score > 0.03 and volume_confirmation:  # Lowered threshold
        signal = "bullish"
        confidence = min(abs(momentum_score) * 10, 1.0)  # Increased sensitivity
    elif momentum_score < -0.03 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 10, 1.0)
    else:
        signal = "neutral"
        confidence = 0.6

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
        },
    }


def calculate_volatility_signals(prices_df):
    returns = prices_df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if vol_z < -0.8:  # Simplified to z-score only
        signal = "bullish"
        confidence = min(abs(vol_z) / 2, 1.0)
    elif vol_z > 0.8:
        signal = "bearish"
        confidence = min(abs(vol_z) / 2, 1.0)
    else:
        signal = "neutral"
        confidence = 0.6

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def calculate_stat_arb_signals(prices_df):
    returns = prices_df["close"].pct_change()
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()
    hurst = calculate_hurst_exponent(prices_df["close"])

    if hurst < 0.45:  # Relaxed threshold, skew removed
        signal = "bullish"
        confidence = (0.5 - hurst) * 2.5  # Increased sensitivity
    elif hurst > 0.55:
        signal = "bearish"
        confidence = (hurst - 0.5) * 2.5
    else:
        signal = "neutral"
        confidence = 0.6

    return {
        "signal": signal,
        "confidence": min(confidence, 1.0),
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1]),
        },
    }


def weighted_signal_combination(signals, weights):
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]
        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    final_score = weighted_sum / total_confidence if total_confidence > 0 else 0

    if final_score > 0.1:  # Lowered threshold
        signal = "bullish"
    elif final_score < -0.1:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": abs(final_score)}


# Helper functions remain unchanged
def normalize_pandas(obj):
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]
    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
    df["smoothed_tr"] = df["tr"].ewm(span=period, adjust=False).mean()
    df["smoothed_plus_dm"] = df["plus_dm"].ewm(span=period, adjust=False).mean()
    df["smoothed_minus_dm"] = df["minus_dm"].ewm(span=period, adjust=False).mean()
    df["+di"] = 100 * (df["smoothed_plus_dm"] / df["smoothed_tr"])
    df["-di"] = 100 * (df["smoothed_minus_dm"] / df["smoothed_tr"])
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period, adjust=False).mean()
    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    lags = range(2, max_lag)
    rs_values = []
    for lag in lags:
        sub_series = [price_series[i:i + lag].values for i in range(0, len(price_series) - lag + 1, lag) if len(price_series[i:i + lag]) == lag]
        if not sub_series:
            continue
        means = np.mean(sub_series, axis=1)
        deviations = np.cumsum(sub_series - means[:, np.newaxis], axis=1)
        r = np.max(deviations, axis=1) - np.min(deviations, axis=1)
        s = np.std(sub_series, axis=1)
        rs_values.append(np.mean(r[s > 0] / s[s > 0]))
    tau = np.array(rs_values)
    if len(tau) < 2:
        return 0.5
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
    except (ValueError, RuntimeWarning):
        return 0.5