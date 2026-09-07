
import pandas as pd
import numpy as np


# ============================================================
# QX AI LIVE SCANNER V4 - CORE ANALYSIS ENGINE
# Signal only: CALL / PUT / NO TRADE
# No automatic trade execution
# ============================================================

MIN_CANDLES = 220

# Higher = stricter signal filter
SIGNAL_THRESHOLD = 9

# Minimum agreement required from 1M / 5M / 15M
MTF_REQUIRED = 2

# Minimum 5-second directional agreement
CONFIRM_RATIO = 0.67


# ============================================================
# BASIC INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def macd(series):

    fast = ema(series, 12)
    slow = ema(series, 26)

    macd_line = fast - slow
    signal_line = ema(macd_line, 9)

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def adx(df, period=14):

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    atr_value = atr(df, period).replace(0, np.nan)

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index
        ).ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index
        ).ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def stochastic(df, period=14):

    lowest = df["low"].rolling(period).min()
    highest = df["high"].rolling(period).max()

    denominator = (
        highest - lowest
    ).replace(0, np.nan)

    k = (
        100
        * (df["close"] - lowest)
        / denominator
    )

    d = k.rolling(3).mean()

    return k, d


# ============================================================
# FEATURE ENGINE
# ============================================================

def build_features(df):

    data = df.copy()

    for period in [9, 21, 50, 200]:
        data[f"ema{period}"] = ema(
            data["close"],
            period
        )

    data["rsi"] = rsi(
        data["close"],
        14
    )

    data["atr"] = atr(
        data,
        14
    )

    (
        data["macd"],
        data["macd_signal"],
        data["macd_hist"]
    ) = macd(
        data["close"]
    )

    data["adx"] = adx(
        data,
        14
    )

    data["stoch_k"], data["stoch_d"] = stochastic(
        data,
        14
    )

    # Bollinger Bands

    bb_mid = data["close"].rolling(20).mean()
    bb_std = data["close"].rolling(20).std()

    data["bb_mid"] = bb_mid
    data["bb_upper"] = bb_mid + (2 * bb_std)
    data["bb_lower"] = bb_mid - (2 * bb_std)

    return data


# ============================================================
# TREND / STRUCTURE
# ============================================================

def trend_analysis(df):

    last = df.iloc[-1]

    score = 0
    reasons = []

    # EMA structure

    if (
        last["ema9"]
        > last["ema21"]
        > last["ema50"]
    ):

        score += 3

        reasons.append(
            "EMA 9/21/50 structure is bullish"
        )

    elif (
        last["ema9"]
        < last["ema21"]
        < last["ema50"]
    ):

        score -= 3

        reasons.append(
            "EMA 9/21/50 structure is bearish"
        )

    else:

        reasons.append(
            "EMA structure is mixed"
        )

    # EMA 200

    if last["close"] > last["ema200"]:

        score += 1

        reasons.append(
            "Price is above EMA 200"
        )

    elif last["close"] < last["ema200"]:

        score -= 1

        reasons.append(
            "Price is below EMA 200"
        )

    # ADX trend quality

    if last["adx"] >= 25:

        reasons.append(
            "ADX shows usable trend strength"
        )

    elif last["adx"] < 20:

        score = int(score * 0.65)

        reasons.append(
            "Weak/choppy trend penalty applied"
        )

    return score, reasons


# ============================================================
# MOMENTUM
# ============================================================

def momentum_analysis(df):

    last = df.iloc[-1]

    score = 0
    reasons = []

    # MACD

    if last["macd_hist"] > 0:

        score += 2

        reasons.append(
            "MACD momentum is bullish"
        )

    elif last["macd_hist"] < 0:

        score -= 2

        reasons.append(
            "MACD momentum is bearish"
        )

    # RSI

    if last["rsi"] >= 55:

        score += 2

        reasons.append(
            "RSI supports bullish momentum"
        )

    elif last["rsi"] <= 45:

        score -= 2

        reasons.append(
            "RSI supports bearish momentum"
        )

    else:

        reasons.append(
            "RSI is neutral"
        )

    # Stochastic

    if (
        last["stoch_k"]
        > last["stoch_d"]
        and last["stoch_k"] < 85
    ):

        score += 1

        reasons.append(
            "Stochastic momentum is bullish"
        )

    elif (
        last["stoch_k"]
        < last["stoch_d"]
        and last["stoch_k"] > 15
    ):

        score -= 1

        reasons.append(
            "Stochastic momentum is bearish"
        )

    return score, reasons


# ============================================================
# RUNNING CANDLE / PRICE ACTION
# ============================================================

def running_candle_analysis(candles):

    if not candles:

        return 0, [
            "Running candle unavailable"
        ]

    current = candles[-1]

    open_price = float(
        current["open"]
    )

    high = float(
        current["high"]
    )

    low = float(
        current["low"]
    )

    close = float(
        current["close"]
    )

    candle_range = max(
        high - low,
        1e-12
    )

    body = abs(
        close - open_price
    )

    upper_wick = (
        high
        - max(open_price, close)
    )

    lower_wick = (
        min(open_price, close)
        - low
    )

    body_ratio = (
        body / candle_range
    )

    upper_ratio = (
        upper_wick / candle_range
    )

    lower_ratio = (
        lower_wick / candle_range
    )

    score = 0
    reasons = []

    # Strong bullish running candle

    if (
        close > open_price
        and body_ratio >= 0.55
    ):

        score += 2

        reasons.append(
            "Running candle has strong bullish body"
        )

    # Strong bearish running candle

    if (
        close < open_price
        and body_ratio >= 0.55
    ):

        score -= 2

        reasons.append(
            "Running candle has strong bearish body"
        )

    # Lower rejection

    if (
        lower_ratio >= 0.45
        and body_ratio < 0.45
    ):

        score += 1

        reasons.append(
            "Running candle shows lower-wick rejection"
        )

    # Upper rejection

    if (
        upper_ratio >= 0.45
        and body_ratio < 0.45
    ):

        score -= 1

        reasons.append(
            "Running candle shows upper-wick rejection"
        )

    # Previous candle comparison

    if len(candles) >= 2:

        previous = candles[-2]

        po = float(
            previous["open"]
        )

        pc = float(
            previous["close"]
        )

        # Bullish engulfing

        if (
            close > open_price
            and pc < po
            and close >= po
            and open_price <= pc
        ):

            score += 2

            reasons.append(
                "Bullish engulfing structure detected"
            )

        # Bearish engulfing

        if (
            close < open_price
            and pc > po
            and open_price >= pc
            and close <= po
        ):

            score -= 2

            reasons.append(
                "Bearish engulfing structure detected"
            )

    return score, reasons


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def support_resistance_analysis(df):

    if len(df) < 80:

        return 0, [
            "Support/resistance history insufficient"
        ]

    last = df.iloc[-1]

    lookback = df.iloc[-80:-1]

    resistance = lookback["high"].max()
    support = lookback["low"].min()

    current_price = float(
        last["close"]
    )

    current_atr = float(
        last["atr"]
    )

    if current_atr <= 0:

        return 0, [
            "ATR unavailable for support/resistance"
        ]

    score = 0
    reasons = []

    distance_resistance = abs(
        resistance - current_price
    )

    distance_support = abs(
        current_price - support
    )

    # Too close to resistance

    if distance_resistance < (
        0.35 * current_atr
    ):

        score -= 2

        reasons.append(
            "Price is too close to resistance"
        )

    # Too close to support

    elif distance_support < (
        0.35 * current_atr
    ):

        score += 2

        reasons.append(
            "Price is reacting near support"
        )

    else:

        reasons.append(
            "No immediate S/R conflict"
        )

    return score, reasons


# ============================================================
# VOLATILITY
# ============================================================

def volatility_analysis(df):

    last = df.iloc[-1]

    if (
        pd.isna(last["atr"])
        or last["atr"] <= 0
    ):

        return 0, [
            "Volatility data unavailable"
        ]

    bb_width = (
        last["bb_upper"]
        - last["bb_lower"]
    ) / max(
        abs(last["close"]),
        1e-12
    )

    score = 0
    reasons = []

    # Extremely compressed market

    if bb_width < 0.0015:

        score -= 2

        reasons.append(
            "Very low volatility — breakout guessing avoided"
        )

    # Excessive volatility

    elif bb_width > 0.015:

        score -= 1

        reasons.append(
            "High volatility — stricter filtering applied"
        )

    else:

        score += 1

        reasons.append(
            "Volatility is within acceptable range"
        )

    return score, reasons


# ============================================================
# TRUE 5-SECOND CONFIRMATION
# ============================================================

def five_second_confirmation(
    ticks,
    direction
):

    # We need real short-term observations.

    if (
    ticks is None
    or len(ticks) < 3
):

        return (
            False,
            0.0,
            "REAL 5-second confirmation unavailable"
        )

    prices = np.array(
        [
            float(x["price"])
            for x in ticks[-7:]
        ]
    )

    changes = np.diff(
        prices
    )

    if direction == "CALL":

        agreeing_moves = (
            changes > 0
        ).sum()

    else:

        agreeing_moves = (
            changes < 0
        ).sum()

    ratio = (
        agreeing_moves
        / max(len(changes), 1)
    )

    net_move = (
        prices[-1]
        - prices[0]
    )

    directional_move = (
        net_move > 0
        if direction == "CALL"
        else net_move < 0
    )

    confirmed = (
        ratio >= CONFIRM_RATIO
        and directional_move
    )

    if confirmed:

        return (
            True,
            ratio,
            f"5-second movement confirms {direction}"
        )

    return (
        False,
        ratio,
        "5-second movement conflicts with candidate direction"
    )


# ============================================================
# MULTI-TIMEFRAME ALIGNMENT
# ============================================================

def timeframe_alignment(
    score_1m,
    score_5m,
    score_15m,
    candidate
):

    scores = [
        score_1m,
        score_5m,
        score_15m
    ]

    if candidate == "CALL":

        agreement = sum(
            s > 0
            for s in scores
        )

    elif candidate == "PUT":

        agreement = sum(
            s < 0
            for s in scores
        )

    else:

        agreement = 0

    return (
        agreement >= MTF_REQUIRED,
        agreement
    )


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze(
    candles_1m,
    candles_5m,
    candles_15m,
    ticks_5s
):

    # --------------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------------

    if not candles_1m:

        return {
            "decision": "NO TRADE",
            "status": "DATA_NOT_CONNECTED",
            "score": 0,
            "confidence": 0,
            "reasons": [
                "1-minute live data unavailable"
            ]
        }

    if len(candles_1m) < MIN_CANDLES:

        return {
            "decision": "NO TRADE",
            "status": "INSUFFICIENT_DATA",
            "score": 0,
            "confidence": 0,
            "reasons": [
                f"Need at least {MIN_CANDLES} completed 1-minute candles"
            ]
        }

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    try:

        df1 = pd.DataFrame(
            candles_1m
        )

        df5 = pd.DataFrame(
            candles_5m
        )

        df15 = pd.DataFrame(
            candles_15m
        )

    except Exception:

        return {
            "decision": "NO TRADE",
            "status": "DATA_ERROR",
            "score": 0,
            "confidence": 0,
            "reasons": [
                "Market data could not be parsed"
            ]
        }

    # --------------------------------------------------------
    # FEATURE GENERATION
    # --------------------------------------------------------

    try:

        f1 = build_features(
            df1
        ).dropna().reset_index(
            drop=True
        )

        f5 = build_features(
            df5
        ).dropna().reset_index(
            drop=True
        )

        f15 = build_features(
            df15
        ).dropna().reset_index(
            drop=True
        )

    except Exception:

        return {
            "decision": "NO TRADE",
            "status": "INDICATOR_ERROR",
            "score": 0,
            "confidence": 0,
            "reasons": [
                "Indicator calculation failed"
            ]
        }

    if (
        len(f1) < 40
        or len(f5) < 40
        or len(f15) < 40
    ):

        return {
            "decision": "NO TRADE",
            "status": "INDICATORS_NOT_READY",
            "score": 0,
            "confidence": 0,
            "reasons": [
                "Multi-timeframe indicators are not warmed up"
            ]
        }

    # --------------------------------------------------------
    # INDIVIDUAL TIMEFRAME ANALYSIS
    # --------------------------------------------------------

    score_1m, why_1m = trend_analysis(
        f1
    )

    momentum_1m, momentum_why = momentum_analysis(
        f1
    )

    score_1m += momentum_1m

    score_5m, why_5m = trend_analysis(
        f5
    )

    score_5m += momentum_analysis(
        f5
    )[0]

    score_15m, why_15m = trend_analysis(
        f15
    )

    score_15m += momentum_analysis(
        f15
    )[0]

    # --------------------------------------------------------
    # RUNNING CANDLE
    # --------------------------------------------------------

    running_score, running_why = (
        running_candle_analysis(
            candles_1m
        )
    )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    sr_score, sr_why = (
        support_resistance_analysis(
            f1
        )
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    volatility_score, volatility_why = (
        volatility_analysis(
            f1
        )
    )

    # --------------------------------------------------------
    # FINAL RAW SCORE
    # --------------------------------------------------------

    total_score = (
        score_1m
        + score_5m
        + score_15m
        + running_score
        + sr_score
        + volatility_score
    )

    reasons = []

    reasons.extend(
        why_1m
    )

    reasons.extend(
        momentum_why
    )

    reasons.extend(
        why_5m
    )

    reasons.extend(
        why_15m
    )

    reasons.extend(
        running_why
    )

    reasons.extend(
        sr_why
    )

    reasons.extend(
        volatility_why
    )

    # --------------------------------------------------------
    # CANDIDATE DIRECTION
    # --------------------------------------------------------

    if total_score > 0:

        candidate = "CALL"

    elif total_score < 0:

        candidate = "PUT"

    else:

        candidate = "NO TRADE"

    # --------------------------------------------------------
    # MULTI-TIMEFRAME CONFIRMATION
    # --------------------------------------------------------

    aligned, agreement = (
        timeframe_alignment(
            score_1m,
            score_5m,
            score_15m,
            candidate
        )
    )

    reasons.append(
        f"MTF agreement: {agreement}/3"
    )

    # --------------------------------------------------------
    # 5 SECOND CONFIRMATION
    # --------------------------------------------------------

    if candidate != "NO TRADE":

        confirmed, confirm_ratio, confirm_reason = (
            five_second_confirmation(
                ticks_5s,
                candidate
            )
        )

    else:

        confirmed = False
        confirm_ratio = 0.0
        confirm_reason = (
            "No directional candidate"
        )

    reasons.append(
        confirm_reason
    )

    # --------------------------------------------------------
    # FINAL FILTER
    # --------------------------------------------------------

    if (
        candidate != "NO TRADE"
        and abs(total_score) >= SIGNAL_THRESHOLD
        and aligned
        and confirmed
    ):

        decision = candidate
        status = "SIGNAL"

    else:

        decision = "NO TRADE"
        status = "FILTERED"

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = int(
        50
        + abs(total_score) * 3
        + max(
            0,
            confirm_ratio - 0.67
        ) * 35
    )

    confidence = min(
        confidence,
        97
    )

    if decision == "NO TRADE":

        confidence = min(
            confidence,
            59
        )

    # --------------------------------------------------------
    # INDICATOR SNAPSHOT
    # --------------------------------------------------------

    last = f1.iloc[-1]

    indicators = {

        "ema9": round(
            float(last["ema9"]),
            8
        ),

        "ema21": round(
            float(last["ema21"]),
            8
        ),

        "ema50": round(
            float(last["ema50"]),
            8
        ),

        "ema200": round(
            float(last["ema200"]),
            8
        ),

        "rsi": round(
            float(last["rsi"]),
            2
        ),

        "adx": round(
            float(last["adx"]),
            2
        ),

        "atr": round(
            float(last["atr"]),
            8
        ),

        "macd_hist": round(
            float(last["macd_hist"]),
            8
        ),

        "stoch_k": round(
            float(last["stoch_k"]),
            2
        ),

        "bb_width": round(
            float(
                (
                    last["bb_upper"]
                    - last["bb_lower"]
                )
                / max(
                    abs(last["close"]),
                    1e-12
                )
            ),
            6
        )
    }

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "decision": decision,

        "status": status,

        "score": int(
            total_score
        ),

        "confidence": int(
            confidence
        ),

        "timeframe_scores": {

            "1m": int(
                score_1m
            ),

            "5m": int(
                score_5m
            ),

            "15m": int(
                score_15m
            )
        },

        "confirmation_ratio": round(
            float(confirm_ratio),
            2
        ),

        "indicators": indicators,

        "reasons": reasons
    }
