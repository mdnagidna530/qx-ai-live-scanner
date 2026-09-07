import pandas as pd
import numpy as np

# ============================================================
# QX AI LIVE SCANNER - ADAPTIVE ENSEMBLE ENGINE
# Signal only: CALL / PUT / NO TRADE
#
# 5-second confirmation is NOT used as a mandatory gate.
# The engine decides:
#   1) direction: CALL / PUT / NO TRADE
#   2) entry timing: CURRENT CANDLE / NEXT CANDLE / NO TRADE
#
# Confidence is a model score, NOT a guaranteed probability.
# ============================================================

MIN_CANDLES = 220
SIGNAL_THRESHOLD = 9.0
RANGE_THRESHOLD = 11.0
MTF_REQUIRED = 2
ENTRY_MIN = 2.5


# ============================================================
# BASIC HELPERS
# ============================================================

def safe(x, default=np.nan):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def sma(s, p):
    return s.rolling(p).mean()


def rsi(s, p=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)

    ag = gain.ewm(alpha=1 / p, adjust=False).mean()
    al = loss.ewm(alpha=1 / p, adjust=False).mean()

    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df, p=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / p, adjust=False).mean()


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def adx(df, p=14):
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index,
    )

    a = atr(df, p).replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / p, adjust=False).mean()
        / a
    )
    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / p, adjust=False).mean()
        / a
    )

    den = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / den
    value = dx.ewm(alpha=1 / p, adjust=False).mean()

    return value, plus_di, minus_di


def stochastic(df, p=14):
    lo = df["low"].rolling(p).min()
    hi = df["high"].rolling(p).max()
    den = (hi - lo).replace(0, np.nan)

    k = 100 * (df["close"] - lo) / den
    d = k.rolling(3).mean()
    return k, d


def cci(df, p=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mean = tp.rolling(p).mean()
    dev = tp.rolling(p).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )
    return (tp - mean) / (0.015 * dev.replace(0, np.nan))


# ============================================================
# FEATURE ENGINE
# ============================================================

def build_features(df):
    d = df.copy()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)

    for p in [9, 21, 50, 100, 200]:
        d[f"ema{p}"] = ema(d["close"], p)

    d["sma20"] = sma(d["close"], 20)
    d["sma50"] = sma(d["close"], 50)

    d["rsi"] = rsi(d["close"], 14)

    (
        d["macd"],
        d["macd_signal"],
        d["macd_hist"],
    ) = macd(d["close"])

    d["macd_hist_slope"] = d["macd_hist"].diff()

    d["adx"], d["plus_di"], d["minus_di"] = adx(d, 14)

    d["stoch_k"], d["stoch_d"] = stochastic(d, 14)

    d["roc9"] = d["close"].pct_change(9) * 100
    d["cci"] = cci(d, 20)
    d["atr"] = atr(d, 14)

    bb_mid = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()

    d["bb_mid"] = bb_mid
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_width"] = (
        (d["bb_upper"] - d["bb_lower"])
        / d["close"].abs().clip(lower=1e-12)
    )

    d["body"] = d["close"] - d["open"]
    d["range"] = (d["high"] - d["low"]).clip(lower=1e-12)
    d["body_abs"] = d["body"].abs()

    d["upper_wick"] = (
        d["high"] - d[["open", "close"]].max(axis=1)
    )
    d["lower_wick"] = (
        d[["open", "close"]].min(axis=1) - d["low"]
    )

    d["body_ratio"] = d["body_abs"] / d["range"]
    d["upper_wick_ratio"] = d["upper_wick"] / d["range"]
    d["lower_wick_ratio"] = d["lower_wick"] / d["range"]

    d["range_vs_atr"] = d["range"] / d["atr"].replace(0, np.nan)
    d["atr_pct"] = d["atr"] / d["close"].abs().clip(lower=1e-12) * 100

    return d


# ============================================================
# MARKET REGIME
# ============================================================

def detect_regime(df):
    last = df.iloc[-1]

    adx_v = safe(last["adx"])
    width = safe(last["bb_width"])
    e21 = safe(last["ema21"])
    e50 = safe(last["ema50"])
    a = safe(last["atr"])

    widths = df["bb_width"].dropna()

    if len(widths) >= 60:
        low_w = float(widths.tail(120).quantile(0.20))
        high_w = float(widths.tail(120).quantile(0.80))
    else:
        low_w = 0.0015
        high_w = 0.015

    separation = 0
    if np.isfinite(e21) and np.isfinite(e50) and np.isfinite(a) and a > 0:
        separation = abs(e21 - e50) / a

    if adx_v >= 25 and separation >= 0.20:
        if width >= high_w:
            return "TREND_HIGH_VOL", 2, [
                "Strong trend with elevated volatility"
            ]
        return "TREND_LOW_VOL", 3, [
            "Strong trend with controlled volatility"
        ]

    if adx_v < 20 and width <= low_w:
        return "RANGE_LOW_VOL", -2, [
            "Compressed range; breakout guessing avoided"
        ]

    if adx_v < 20 and width >= high_w:
        return "RANGE_HIGH_VOL", -3, [
            "Wide unstable range; stricter filtering applied"
        ]

    if adx_v >= 20:
        return "DEVELOPING_TREND", 1, [
            "Developing directional movement"
        ]

    return "RANGE", -1, [
        "Sideways/ranging market"
    ]


# ============================================================
# TREND EXPERT
# ============================================================

def trend_expert(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0.0
    reasons = []

    e9 = safe(last["ema9"])
    e21 = safe(last["ema21"])
    e50 = safe(last["ema50"])
    e200 = safe(last["ema200"])
    close = safe(last["close"])

    if e9 > e21 > e50:
        score += 3
        reasons.append("EMA 9/21/50 bullish alignment")
    elif e9 < e21 < e50:
        score -= 3
        reasons.append("EMA 9/21/50 bearish alignment")
    else:
        reasons.append("EMA structure is mixed")

    if close > e200:
        score += 1.5
        reasons.append("Price above EMA200")
    elif close < e200:
        score -= 1.5
        reasons.append("Price below EMA200")

    plus = safe(last["plus_di"])
    minus = safe(last["minus_di"])
    adx_v = safe(last["adx"])

    if plus > minus:
        score += 1
    elif minus > plus:
        score -= 1

    if adx_v >= 25:
        score *= 1.15
        reasons.append("ADX confirms usable trend strength")
    elif adx_v < 18:
        score *= 0.70
        reasons.append("Low ADX reduces trend confidence")

    prev_e21 = safe(prev["ema21"])
    if e21 > prev_e21:
        score += 0.5
    elif e21 < prev_e21:
        score -= 0.5

    return score, reasons


# ============================================================
# MOMENTUM EXPERT
# ============================================================

def momentum_expert(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0.0
    reasons = []

    hist = safe(last["macd_hist"])
    prev_hist = safe(prev["macd_hist"])
    r = safe(last["rsi"])
    k = safe(last["stoch_k"])
    d = safe(last["stoch_d"])
    roc_v = safe(last["roc9"])
    cci_v = safe(last["cci"])

    if hist > 0:
        score += 1.5
        reasons.append("MACD momentum bullish")
    elif hist < 0:
        score -= 1.5
        reasons.append("MACD momentum bearish")

    if hist > prev_hist:
        score += 0.75
    elif hist < prev_hist:
        score -= 0.75

    # RSI is interpreted as momentum first, not as a blind
    # overbought/oversold reversal trigger.
    if 52 <= r <= 68:
        score += 1.25
        reasons.append("RSI supports healthy bullish momentum")
    elif 32 <= r <= 48:
        score -= 1.25
        reasons.append("RSI supports bearish momentum")
    elif r > 75:
        score -= 0.75
        reasons.append("RSI is extremely stretched upward")
    elif r < 25:
        score += 0.75
        reasons.append("RSI is extremely stretched downward")

    if k > d:
        score += 0.75
    elif k < d:
        score -= 0.75

    if roc_v > 0:
        score += 0.75
    elif roc_v < 0:
        score -= 0.75

    if cci_v > 50:
        score += 0.5
    elif cci_v < -50:
        score -= 0.5

    return score, reasons


# ============================================================
# PRICE ACTION EXPERT
# ============================================================

def price_action_expert(df):
    if len(df) < 4:
        return 0.0, ["Price-action history insufficient"]

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    score = 0.0
    reasons = []

    close = safe(last["close"])
    op = safe(last["open"])
    body_ratio = safe(last["body_ratio"])
    upper = safe(last["upper_wick_ratio"])
    lower = safe(last["lower_wick_ratio"])

    pc = safe(prev["close"])
    po = safe(prev["open"])

    if close > op and body_ratio >= 0.60:
        score += 2
        reasons.append("Current candle has strong bullish body")
    elif close < op and body_ratio >= 0.60:
        score -= 2
        reasons.append("Current candle has strong bearish body")

    if lower >= 0.45 and body_ratio < 0.50:
        score += 1.25
        reasons.append("Lower-wick rejection favors buyers")

    if upper >= 0.45 and body_ratio < 0.50:
        score -= 1.25
        reasons.append("Upper-wick rejection favors sellers")

    if (
        close > op
        and pc < po
        and close >= po
        and op <= pc
    ):
        score += 2
        reasons.append("Bullish engulfing pattern")

    if (
        close < op
        and pc > po
        and op >= pc
        and close <= po
    ):
        score -= 2
        reasons.append("Bearish engulfing pattern")

    p2c = safe(prev2["close"])
    p2o = safe(prev2["open"])

    if close > op and pc > po and p2c > p2o:
        score += 1
        reasons.append("Three-bar bullish continuation")

    if close < op and pc < po and p2c < p2o:
        score -= 1
        reasons.append("Three-bar bearish continuation")

    prior_high = safe(df["high"].iloc[-21:-1].max())
    prior_low = safe(df["low"].iloc[-21:-1].min())
    high = safe(last["high"])
    low = safe(last["low"])

    if high > prior_high and close > prior_high:
        score += 2
        reasons.append("Bullish Donchian breakout")
    elif low < prior_low and close < prior_low:
        score -= 2
        reasons.append("Bearish Donchian breakdown")

    return score, reasons


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def structure_expert(df):
    if len(df) < 80:
        return 0.0, ["Support/resistance history insufficient"]

    last = df.iloc[-1]

    close = safe(last["close"])
    a = safe(last["atr"])

    if not np.isfinite(close) or not np.isfinite(a) or a <= 0:
        return 0.0, ["Structure data unavailable"]

    look = df.iloc[-80:-1]
    resistance = float(look["high"].max())
    support = float(look["low"].min())

    score = 0.0
    reasons = []

    dist_r = resistance - close
    dist_s = close - support

    if 0 <= dist_r <= 0.40 * a:
        score -= 1.75
        reasons.append("Price is close to resistance")
    elif dist_r < 0:
        score += 1
        reasons.append("Price is above recent resistance")

    if 0 <= dist_s <= 0.40 * a:
        score += 1.75
        reasons.append("Price is close to support")
    elif dist_s < 0:
        score -= 1
        reasons.append("Price is below recent support")

    upper = safe(last["bb_upper"])
    lower = safe(last["bb_lower"])

    if close > upper:
        score -= 0.75
        reasons.append("Price extended above Bollinger upper band")
    elif close < lower:
        score += 0.75
        reasons.append("Price extended below Bollinger lower band")

    return score, reasons


# ============================================================
# VOLATILITY EXPERT
# ============================================================

def volatility_expert(df):
    last = df.iloc[-1]

    width = safe(last["bb_width"])
    range_vs_atr = safe(last["range_vs_atr"])

    widths = df["bb_width"].dropna()

    if len(widths) >= 60:
        low_w = float(widths.tail(120).quantile(0.20))
        high_w = float(widths.tail(120).quantile(0.80))
    else:
        low_w = 0.0015
        high_w = 0.015

    if width <= low_w:
        regime = "LOW"
        score = -1.5
        reason = "Volatility compressed"
    elif width >= high_w:
        regime = "HIGH"
        score = -1
        reason = "Volatility elevated"
    else:
        regime = "NORMAL"
        score = 1
        reason = "Volatility within normal range"

    if range_vs_atr >= 2:
        score -= 1
        reason += "; current candle is unusually large"

    return score, regime, [reason]


# ============================================================
# DIVERGENCE / EXHAUSTION
# ============================================================

def divergence_expert(df):
    if len(df) < 35:
        return 0.0, []

    close_now = safe(df["close"].iloc[-1])
    close_old = safe(df["close"].iloc[-12])
    r_now = safe(df["rsi"].iloc[-1])
    r_old = safe(df["rsi"].iloc[-12])

    score = 0.0
    reasons = []

    if close_now > close_old and r_now < r_old - 3:
        score -= 1.25
        reasons.append("Bearish RSI divergence risk")

    elif close_now < close_old and r_now > r_old + 3:
        score += 1.25
        reasons.append("Bullish RSI divergence risk")

    return score, reasons


# ============================================================
# MULTI-TIMEFRAME
# ============================================================

def timeframe_score(df):
    t, tr = trend_expert(df)
    m, mr = momentum_expert(df)
    return t + m, tr + mr


def mtf_alignment(scores, candidate):
    if candidate == "CALL":
        n = sum(x > 0 for x in scores)
    elif candidate == "PUT":
        n = sum(x < 0 for x in scores)
    else:
        n = 0

    return n >= MTF_REQUIRED, n


# ============================================================
# CURRENT CANDLE VS NEXT CANDLE
# ============================================================

def entry_timing(candles, candidate, regime, total_score):
    if not candles:
        return "NO TRADE", 0.0, None, "Current candle unavailable"

    c = candles[-1]

    op = safe(c.get("open"))
    hi = safe(c.get("high"))
    lo = safe(c.get("low"))
    cl = safe(c.get("close"))
    ts = safe(c.get("timestamp"))

    if not all(np.isfinite(x) for x in [op, hi, lo, cl]):
        return "NO TRADE", 0.0, None, "Current candle invalid"

    rng = max(hi - lo, 1e-12)
    body_ratio = abs(cl - op) / rng
    close_pos = (cl - lo) / rng

    if candidate == "CALL":
        favorable = cl > op
        location = close_pos
    elif candidate == "PUT":
        favorable = cl > op
        location = 1 - close_pos
    else:
        return "NO TRADE", 0.0, None, "No directional candidate"

    q = 0.0
    reasons = []

    if favorable and body_ratio >= 0.55:
        q += 3
        reasons.append("Current candle body supports direction")
    elif body_ratio < 0.25:
        q -= 2
        reasons.append("Current candle is indecisive")

    if location >= 0.70:
        q += 2
        reasons.append("Candle closes near favorable side")
    elif location <= 0.35:
        q -= 2
        reasons.append("Candle closes against candidate")

    if body_ratio >= 0.75:
        q -= 1.5
        reasons.append("Current candle is already extended")

    if regime in {"RANGE", "RANGE_LOW_VOL", "RANGE_HIGH_VOL"}:
        q -= 1
        reasons.append("Range regime favors waiting")

    if abs(total_score) >= 14:
        q += 1

    if q >= ENTRY_MIN:
        entry = "CURRENT CANDLE"
    elif q >= 2.5:
        entry = "NEXT CANDLE"
    else:
        entry = "NO TRADE"

    progress = None
    if np.isfinite(ts):
        try:
            now = pd.Timestamp.now(tz="UTC").timestamp()
            # BiQuote timestamps are expected to be epoch seconds.
            elapsed = max(0.0, now - ts)
            progress = min(1.0, elapsed / 60.0)
        except Exception:
            progress = None

    return (
        entry,
        round(q, 2),
        round(progress, 2) if progress is not None else None,
        "; ".join(reasons),
    )


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_score(total, agreement, regime, entry_q, decision):
    base = 50 + min(abs(total), 24) * 1.65

    if agreement == 3:
        base += 8
    elif agreement == 2:
        base += 2
    else:
        base -= 8

    if regime in {"TREND_LOW_VOL", "DEVELOPING_TREND"}:
        base += 3
    elif regime in {"RANGE_HIGH_VOL", "RANGE_LOW_VOL"}:
        base -= 6

    base += max(-4, min(6, entry_q))

    out = int(max(1, min(95, base)))

    if decision == "NO TRADE":
        out = min(out, 59)

    return out


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze(
    candles_1m,
    candles_5m,
    candles_15m,
    ticks_5s=None,
):
    # ticks_5s is intentionally accepted for API compatibility.
    # It is NOT used as a mandatory confirmation gate.

    if not candles_1m:
        return {
            "decision": "NO TRADE",
            "signal": "NO TRADE",
            "status": "DATA_NOT_CONNECTED",
            "score": 0,
            "confidence": 0,
            "entry_timing": "NO TRADE",
            "entry_quality": 0,
            "confirmation_mode": "5S_DISABLED",
            "reasons": ["1-minute live data unavailable"],
        }

    if len(candles_1m) < MIN_CANDLES:
        return {
            "decision": "NO TRADE",
            "signal": "NO TRADE",
            "status": "INSUFFICIENT_DATA",
            "score": 0,
            "confidence": 0,
            "entry_timing": "NO TRADE",
            "entry_quality": 0,
            "confirmation_mode": "5S_DISABLED",
            "reasons": [
                f"Need at least {MIN_CANDLES} 1-minute candles"
            ],
        }

    try:
        f1 = build_features(pd.DataFrame(candles_1m))
        f5 = build_features(pd.DataFrame(candles_5m or []))
        f15 = build_features(pd.DataFrame(candles_15m or []))

        f1 = f1.dropna().reset_index(drop=True)
        f5 = f5.dropna().reset_index(drop=True)
        f15 = f15.dropna().reset_index(drop=True)

    except Exception as exc:
        return {
            "decision": "NO TRADE",
            "signal": "NO TRADE",
            "status": "INDICATOR_ERROR",
            "score": 0,
            "confidence": 0,
            "entry_timing": "NO TRADE",
            "entry_quality": 0,
            "confirmation_mode": "5S_DISABLED",
            "reasons": [
                "Indicator calculation failed",
                str(exc),
            ],
        }

    if len(f1) < 40 or len(f5) < 40 or len(f15) < 40:
        return {
            "decision": "NO TRADE",
            "signal": "NO TRADE",
            "status": "INDICATORS_NOT_READY",
            "score": 0,
            "confidence": 0,
            "entry_timing": "NO TRADE",
            "entry_quality": 0,
            "confirmation_mode": "5S_DISABLED",
            "reasons": [
                "Multi-timeframe indicators are not warmed up"
            ],
        }

    # Market regime
    regime, regime_score, regime_reasons = detect_regime(f1)

    # 1m / 5m / 15m experts
    s1, r1 = timeframe_score(f1)
    s5, r5 = timeframe_score(f5)
    s15, r15 = timeframe_score(f15)

    # Higher timeframes receive slightly more weight.
    mtf_score = (
        0.30 * s1
        + 0.35 * s5
        + 0.35 * s15
    )

    # 1m entry experts
    pa, pa_reasons = price_action_expert(f1)
    sr, sr_reasons = structure_expert(f1)
    vol, vol_regime, vol_reasons = volatility_expert(f1)
    div, div_reasons = divergence_expert(f1)

    if regime in {"TREND_LOW_VOL", "DEVELOPING_TREND"}:
        mtf_weight = 1.00
        pa_weight = 1.00
    elif regime == "TREND_HIGH_VOL":
        mtf_weight = 0.95
        pa_weight = 1.10
    elif regime in {"RANGE", "RANGE_LOW_VOL"}:
        mtf_weight = 0.85
        pa_weight = 0.85
    else:
        mtf_weight = 0.70
        pa_weight = 0.70

    total = (
        mtf_weight * mtf_score
        + pa_weight * pa
        + sr
        + vol
        + div
        + 0.75 * regime_score
    )

    if total > 0:
        candidate = "CALL"
    elif total < 0:
        candidate = "PUT"
    else:
        candidate = "NO TRADE"

    aligned, agreement = mtf_alignment(
        [s1, s5, s15],
        candidate,
    )

    # Decide whether the current candle is usable or the next one
    # is the cleaner entry.
    entry, entry_q, progress, entry_reason = entry_timing(
        candles_1m,
        candidate,
        regime,
        total,
    )

    threshold = (
        RANGE_THRESHOLD
        if regime in {"RANGE", "RANGE_LOW_VOL", "RANGE_HIGH_VOL"}
        else SIGNAL_THRESHOLD
    )

    decision = "NO TRADE"
    status = "FILTERED"

    if (
        candidate != "NO TRADE"
        and abs(total) >= threshold
        and aligned
        and entry in {"CURRENT CANDLE", "NEXT CANDLE"}
        and entry_q >= ENTRY_MIN
    ):
        decision = candidate
        status = "SIGNAL"

    reasons = []
    reasons.extend(regime_reasons)
    reasons.extend(r1)
    reasons.extend(r5)
    reasons.extend(r15)
    reasons.extend(pa_reasons)
    reasons.extend(sr_reasons)
    reasons.extend(vol_reasons)
    reasons.extend(div_reasons)

    reasons.append(f"MTF agreement: {agreement}/3")
    reasons.append(f"Entry timing: {entry}")

    if entry_reason:
        reasons.append(entry_reason)

    if not aligned:
        reasons.append("MTF direction is not sufficiently aligned")

    if abs(total) < threshold:
        reasons.append(
            f"Score {total:.1f} is below required {threshold:.1f}"
        )

    confidence = confidence_score(
        total,
        agreement,
        regime,
        entry_q,
        decision,
    )

    last = f1.iloc[-1]

    indicators = {
        "ema9": round(safe(last["ema9"]), 8),
        "ema21": round(safe(last["ema21"]), 8),
        "ema50": round(safe(last["ema50"]), 8),
        "ema200": round(safe(last["ema200"]), 8),
        "rsi": round(safe(last["rsi"]), 2),
        "adx": round(safe(last["adx"]), 2),
        "atr": round(safe(last["atr"]), 8),
        "macd_hist": round(safe(last["macd_hist"]), 8),
        "macd_hist_slope": round(safe(last["macd_hist_slope"]), 8),
        "stoch_k": round(safe(last["stoch_k"]), 2),
        "stoch_d": round(safe(last["stoch_d"]), 2),
        "roc9": round(safe(last["roc9"]), 4),
        "cci": round(safe(last["cci"]), 2),
        "bb_width": round(safe(last["bb_width"]), 6),
        "atr_pct": round(safe(last["atr_pct"]), 4),
    }

    return {
        "decision": decision,
        "signal": decision,
        "status": status,

        "score": int(round(total)),
        "raw_score": round(float(total), 2),
        "confidence": confidence,

        "entry_timing": entry,
        "entry_quality": entry_q,
        "candle_progress": progress,

        "market_regime": regime,
        "volatility_regime": vol_regime,

        "timeframe_scores": {
            "1m": int(round(s1)),
            "5m": int(round(s5)),
            "15m": int(round(s15)),
        },

        "mtf_agreement": int(agreement),

        # Kept only for compatibility with the old frontend.
        # It is deliberately not a signal gate.
        "confirmation_ratio": None,
        "confirmation_mode": "5S_DISABLED",

        "indicators": indicators,
        "reasons": reasons,
    }
