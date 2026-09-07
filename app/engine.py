
import pandas as pd
import numpy as np

# ============================================================
# QX AI LIVE SCANNER V6 - ADAPTIVE EVIDENCE ENGINE
#
# Signal only: CALL / PUT / NO TRADE
#
# Design:
#   - Uses REAL supplied candles only.
#   - No fabricated ticks/candles.
#   - No mandatory 5-second gate.
#   - Direction is selected by the stronger side of the
#     multi-factor evidence.
#   - Direction percentages are RELATIVE MODEL EVIDENCE,
#     NOT calibrated win probabilities.
#   - NO TRADE is reserved for missing/invalid data.
#
# IMPORTANT:
#   This is a decision-support model, not a guarantee of outcome.
# ============================================================

MIN_CANDLES = 220
MIN_TF_CANDLES = 80

# Expert weights. They sum to 1.0.
WEIGHTS = {
    "trend": 0.24,
    "momentum": 0.20,
    "price_action": 0.18,
    "structure": 0.14,
    "volatility": 0.08,
    "mtf": 0.16,
}

EPS = 1e-12


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
    f = ema(s, 12)
    sl = ema(s, 26)
    line = f - sl
    signal = ema(line, 9)
    return line, signal, line - signal


def adx(df, p=14):
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )

    a = atr(df, p).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / p, adjust=False).mean() / a
    minus_di = 100 * minus_dm.ewm(alpha=1 / p, adjust=False).mean() / a

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
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - mean) / (0.015 * dev.replace(0, np.nan))


def williams_r(df, p=14):
    hi = df["high"].rolling(p).max()
    lo = df["low"].rolling(p).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)


def supertrend(df, p=10, multiplier=3.0):
    # Lightweight, deterministic Supertrend implementation.
    a = atr(df, p)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a

    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = pd.Series(1.0, index=df.index)

    for i in range(1, len(df)):
        if df["close"].iloc[i - 1] <= final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(upper.iloc[i], final_upper.iloc[i - 1])
        else:
            final_upper.iloc[i] = upper.iloc[i]

        if df["close"].iloc[i - 1] >= final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(lower.iloc[i], final_lower.iloc[i - 1])
        else:
            final_lower.iloc[i] = lower.iloc[i]

        if direction.iloc[i - 1] > 0:
            direction.iloc[i] = -1 if df["close"].iloc[i] < final_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df["close"].iloc[i] > final_upper.iloc[i] else -1

    return direction, final_upper, final_lower


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
    d = d.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    for p in [9, 21, 50, 100, 200]:
        d[f"ema{p}"] = ema(d["close"], p)

    d["sma20"] = sma(d["close"], 20)
    d["sma50"] = sma(d["close"], 50)
    d["rsi"] = rsi(d["close"], 14)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd(d["close"])
    d["macd_hist_slope"] = d["macd_hist"].diff()

    d["adx"], d["plus_di"], d["minus_di"] = adx(d, 14)
    d["stoch_k"], d["stoch_d"] = stochastic(d, 14)
    d["williams_r"] = williams_r(d, 14)
    d["roc9"] = d["close"].pct_change(9) * 100
    d["cci"] = cci(d, 20)
    d["atr"] = atr(d, 14)

    bb_mid = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_mid"] = bb_mid
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["close"].abs().clip(lower=EPS)

    d["body"] = d["close"] - d["open"]
    d["range"] = (d["high"] - d["low"]).clip(lower=EPS)
    d["body_abs"] = d["body"].abs()
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]
    d["body_ratio"] = d["body_abs"] / d["range"]
    d["upper_wick_ratio"] = d["upper_wick"] / d["range"]
    d["lower_wick_ratio"] = d["lower_wick"] / d["range"]
    d["range_vs_atr"] = d["range"] / d["atr"].replace(0, np.nan)
    d["atr_pct"] = d["atr"] / d["close"].abs().clip(lower=EPS) * 100

    d["st_dir"], d["st_upper"], d["st_lower"] = supertrend(d)
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
        sample = widths.tail(120)
        low_w = float(sample.quantile(0.20))
        high_w = float(sample.quantile(0.80))
    else:
        low_w, high_w = 0.0015, 0.015

    separation = abs(e21 - e50) / a if all(
        np.isfinite(x) for x in [e21, e50, a]
    ) and a > 0 else 0

    if adx_v >= 25 and separation >= 0.20:
        if width >= high_w:
            return "TREND_HIGH_VOL", 0.55, ["Strong trend with elevated volatility"]
        return "TREND_LOW_VOL", 0.90, ["Strong trend with controlled volatility"]

    if adx_v < 20 and width <= low_w:
        return "RANGE_LOW_VOL", 0.35, ["Compressed range"]

    if adx_v < 20 and width >= high_w:
        return "RANGE_HIGH_VOL", 0.30, ["Wide unstable range"]

    if adx_v >= 20:
        return "DEVELOPING_TREND", 0.75, ["Developing directional movement"]

    return "RANGE", 0.45, ["Sideways/ranging market"]


# ============================================================
# NORMALIZED EXPERTS
# Each expert returns a score in approximately [-1, +1].
# Positive = CALL evidence, negative = PUT evidence.
# ============================================================

def trend_expert(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 0.0
    reasons = []

    e9, e21, e50, e100, e200 = [safe(last[x]) for x in ["ema9","ema21","ema50","ema100","ema200"]]
    close = safe(last["close"])
    plus, minus, adx_v = safe(last["plus_di"]), safe(last["minus_di"]), safe(last["adx"])

    if e9 > e21 > e50:
        score += 0.32
        reasons.append("EMA 9/21/50 bullish alignment")
    elif e9 < e21 < e50:
        score -= 0.32
        reasons.append("EMA 9/21/50 bearish alignment")
    else:
        reasons.append("EMA structure mixed")

    if e50 > e100 > e200:
        score += 0.22
        reasons.append("Long EMA structure bullish")
    elif e50 < e100 < e200:
        score -= 0.22
        reasons.append("Long EMA structure bearish")

    if close > e200:
        score += 0.16
        reasons.append("Price above EMA200")
    elif close < e200:
        score -= 0.16
        reasons.append("Price below EMA200")

    if plus > minus:
        score += 0.14
    elif minus > plus:
        score -= 0.14

    if adx_v >= 25:
        score *= 1.15
        reasons.append("ADX confirms usable trend strength")
    elif adx_v < 18:
        score *= 0.70
        reasons.append("Low ADX reduces trend confidence")

    prev_e21 = safe(prev["ema21"])
    if e21 > prev_e21:
        score += 0.08
    elif e21 < prev_e21:
        score -= 0.08

    return float(np.clip(score, -1, 1)), reasons


def momentum_expert(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 0.0
    reasons = []

    hist, prev_hist = safe(last["macd_hist"]), safe(prev["macd_hist"])
    r = safe(last["rsi"])
    k, d = safe(last["stoch_k"]), safe(last["stoch_d"])
    roc_v, cci_v = safe(last["roc9"]), safe(last["cci"])

    if hist > 0:
        score += 0.22
        reasons.append("MACD momentum bullish")
    elif hist < 0:
        score -= 0.22
        reasons.append("MACD momentum bearish")

    if hist > prev_hist:
        score += 0.10
    elif hist < prev_hist:
        score -= 0.10

    if 52 <= r <= 68:
        score += 0.16
        reasons.append("RSI healthy bullish momentum")
    elif 32 <= r <= 48:
        score -= 0.16
        reasons.append("RSI bearish momentum")
    elif r > 75:
        score -= 0.08
        reasons.append("RSI extremely stretched upward")
    elif r < 25:
        score += 0.08
        reasons.append("RSI extremely stretched downward")

    if k > d:
        score += 0.10
    elif k < d:
        score -= 0.10

    if roc_v > 0:
        score += 0.10
    elif roc_v < 0:
        score -= 0.10

    if cci_v > 50:
        score += 0.08
    elif cci_v < -50:
        score -= 0.08

    return float(np.clip(score, -1, 1)), reasons


def price_action_expert(df):
    if len(df) < 4:
        return 0.0, ["Price-action history insufficient"]

    last, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    score = 0.0
    reasons = []

    close, op = safe(last["close"]), safe(last["open"])
    br = safe(last["body_ratio"])
    upper, lower = safe(last["upper_wick_ratio"]), safe(last["lower_wick_ratio"])
    pc, po = safe(prev["close"]), safe(prev["open"])

    if close > op and br >= 0.60:
        score += 0.25
        reasons.append("Strong bullish candle body")
    elif close < op and br >= 0.60:
        score -= 0.25
        reasons.append("Strong bearish candle body")

    if lower >= 0.45 and br < 0.50:
        score += 0.16
        reasons.append("Lower-wick rejection favors buyers")
    if upper >= 0.45 and br < 0.50:
        score -= 0.16
        reasons.append("Upper-wick rejection favors sellers")

    if close > op and pc < po and close >= po and op <= pc:
        score += 0.24
        reasons.append("Bullish engulfing")
    elif close < op and pc > po and op >= pc and close <= po:
        score -= 0.24
        reasons.append("Bearish engulfing")

    p2c, p2o = safe(prev2["close"]), safe(prev2["open"])
    if close > op and pc > po and p2c > p2o:
        score += 0.12
        reasons.append("Three-bar bullish continuation")
    elif close < op and pc < po and p2c < p2o:
        score -= 0.12
        reasons.append("Three-bar bearish continuation")

    prior_high = safe(df["high"].iloc[-21:-1].max())
    prior_low = safe(df["low"].iloc[-21:-1].min())
    high, low = safe(last["high"]), safe(last["low"])

    if high > prior_high and close > prior_high:
        score += 0.23
        reasons.append("Bullish Donchian breakout")
    elif low < prior_low and close < prior_low:
        score -= 0.23
        reasons.append("Bearish Donchian breakdown")

    # Supertrend
    st = safe(last["st_dir"])
    if st > 0:
        score += 0.10
        reasons.append("Supertrend bullish")
    elif st < 0:
        score -= 0.10
        reasons.append("Supertrend bearish")

    return float(np.clip(score, -1, 1)), reasons


def structure_expert(df):
    if len(df) < 80:
        return 0.0, ["Structure history insufficient"]

    last = df.iloc[-1]
    close, a = safe(last["close"]), safe(last["atr"])
    if not np.isfinite(close) or not np.isfinite(a) or a <= 0:
        return 0.0, ["Structure data unavailable"]

    look = df.iloc[-80:-1]
    resistance = float(look["high"].max())
    support = float(look["low"].min())

    score = 0.0
    reasons = []

    dist_r = resistance - close
    dist_s = close - support

    if dist_r < 0:
        score += 0.22
        reasons.append("Price above recent resistance")
    elif 0 <= dist_r <= 0.40 * a:
        score -= 0.18
        reasons.append("Price close to resistance")

    if dist_s < 0:
        score -= 0.22
        reasons.append("Price below recent support")
    elif 0 <= dist_s <= 0.40 * a:
        score += 0.18
        reasons.append("Price close to support")

    upper, lower = safe(last["bb_upper"]), safe(last["bb_lower"])
    if close > upper:
        score -= 0.10
        reasons.append("Price extended above Bollinger upper band")
    elif close < lower:
        score += 0.10
        reasons.append("Price extended below Bollinger lower band")

    return float(np.clip(score, -1, 1)), reasons


def volatility_expert(df):
    last = df.iloc[-1]
    width = safe(last["bb_width"])
    rva = safe(last["range_vs_atr"])
    widths = df["bb_width"].dropna()

    if len(widths) >= 60:
        sample = widths.tail(120)
        low_w = float(sample.quantile(0.20))
        high_w = float(sample.quantile(0.80))
    else:
        low_w, high_w = 0.0015, 0.015

    if width <= low_w:
        return -0.10, "LOW", ["Volatility compressed"]
    if width >= high_w:
        return -0.06, "HIGH", ["Volatility elevated"]
    if rva >= 2:
        return -0.08, "HIGH", ["Current candle unusually large"]
    return 0.08, "NORMAL", ["Volatility within normal range"]


# ============================================================
# MULTI-TIMEFRAME EVIDENCE
# ============================================================

def timeframe_score(df):
    t, tr = trend_expert(df)
    m, mr = momentum_expert(df)
    # Trend + momentum, balanced inside each timeframe.
    return float(np.clip(0.58 * t + 0.42 * m, -1, 1)), tr + mr


def mtf_expert(f1, f5, f15):
    s1, r1 = timeframe_score(f1)
    s5, r5 = timeframe_score(f5)
    s15, r15 = timeframe_score(f15)

    score = 0.25 * s1 + 0.375 * s5 + 0.375 * s15
    return float(np.clip(score, -1, 1)), [s1, s5, s15], r1 + r5 + r15


# ============================================================
# DIRECTION PERCENT
# ============================================================

def direction_percent(total_score):
    # Relative evidence share, deliberately NOT called probability.
    x = float(np.clip(total_score, -1, 1))
    call = 50.0 + 50.0 * x
    put = 100.0 - call
    return round(call, 1), round(put, 1)


# ============================================================
# ENTRY TIMING
# ============================================================

def entry_timing(candles, candidate, regime, total_score):
    if not candles:
        return "NO TRADE", 0.0, None, "Current candle unavailable"

    c = candles[-1]
    op, hi, lo, cl, ts = [safe(c.get(k)) for k in ["open","high","low","close","timestamp"]]

    if not all(np.isfinite(x) for x in [op, hi, lo, cl]):
        return "NO TRADE", 0.0, None, "Current candle invalid"

    rng = max(hi - lo, EPS)
    body_ratio = abs(cl - op) / rng
    close_pos = (cl - lo) / rng

    if candidate == "CALL":
        favorable = cl > op
        location = close_pos
    elif candidate == "PUT":
        # FIX: bearish candle is close < open.
        favorable = cl < op
        location = 1.0 - close_pos
    else:
        return "NO TRADE", 0.0, None, "No directional candidate"

    q = 5.0
    reasons = []

    if favorable:
        q += 1.8
        reasons.append("Current candle body supports direction")
    else:
        q -= 1.8
        reasons.append("Current candle body opposes direction")

    if location >= 0.70:
        q += 1.4
        reasons.append("Candle closes near favorable side")
    elif location <= 0.35:
        q -= 1.4
        reasons.append("Candle closes against candidate")

    if body_ratio < 0.25:
        q -= 1.6
        reasons.append("Current candle is indecisive")

    if body_ratio >= 0.75:
        q -= 1.3
        reasons.append("Current candle is already extended")

    if regime in {"RANGE", "RANGE_LOW_VOL", "RANGE_HIGH_VOL"}:
        q -= 0.7
        reasons.append("Range regime favors waiting for cleaner entry")

    if abs(total_score) >= 0.70:
        q += 0.8
        reasons.append("Directional evidence is strong")

    q = float(np.clip(q, 0, 10))

    # Current = clean live candle. Next = direction remains strong but
    # current candle is too weak/extended for a clean immediate entry.
    if q >= 6.2 and body_ratio < 0.75:
        entry = "CURRENT CANDLE"
    elif q >= 3.5:
        entry = "NEXT CANDLE"
    else:
        entry = "NEXT CANDLE"

    progress = None
    if np.isfinite(ts):
        try:
            now = pd.Timestamp.now(tz="UTC").timestamp()
            elapsed = max(0.0, now - ts)
            progress = min(1.0, elapsed / 60.0)
        except Exception:
            progress = None

    return entry, round(q, 2), round(progress, 2) if progress is not None else None, "; ".join(reasons)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def _error(status, reason):
    return {
        "decision": "NO TRADE",
        "signal": "NO TRADE",
        "status": status,
        "score": 0,
        "raw_score": 0.0,
        "confidence": 0,
        "direction_percent": {"CALL": 0.0, "PUT": 0.0},
        "entry_timing": "NO TRADE",
        "entry_quality": 0.0,
        "confirmation_mode": "5S_DISABLED",
        "reasons": [reason],
    }


def analyze(candles_1m, candles_5m, candles_15m, ticks_5s=None):
    if not candles_1m:
        return _error("DATA_NOT_CONNECTED", "1-minute live data unavailable")

    if len(candles_1m) < MIN_CANDLES:
        return _error(
            "INSUFFICIENT_DATA",
            f"Need at least {MIN_CANDLES} 1-minute candles",
        )

    try:
        f1 = build_features(pd.DataFrame(candles_1m))
        f5 = build_features(pd.DataFrame(candles_5m or []))
        f15 = build_features(pd.DataFrame(candles_15m or []))
    except Exception as exc:
        return _error("INDICATOR_ERROR", f"Indicator calculation failed: {exc}")

    if len(f1) < MIN_TF_CANDLES or len(f5) < MIN_TF_CANDLES or len(f15) < MIN_TF_CANDLES:
        return _error(
            "INDICATORS_NOT_READY",
            "Multi-timeframe indicator history is not warmed up",
        )

    try:
        regime, regime_strength, regime_reasons = detect_regime(f1)

        trend, trend_reasons = trend_expert(f1)
        momentum, momentum_reasons = momentum_expert(f1)
        pa, pa_reasons = price_action_expert(f1)
        structure, structure_reasons = structure_expert(f1)
        vol, vol_regime, vol_reasons = volatility_expert(f1)
        mtf, tf_scores, mtf_reasons = mtf_expert(f1, f5, f15)

        # Regime adaptively adjusts expert influence without creating
        # a hard signal veto.
        trend_w = WEIGHTS["trend"]
        momentum_w = WEIGHTS["momentum"]
        pa_w = WEIGHTS["price_action"]
        structure_w = WEIGHTS["structure"]
        vol_w = WEIGHTS["volatility"]
        mtf_w = WEIGHTS["mtf"]

        if regime in {"TREND_LOW_VOL", "DEVELOPING_TREND"}:
            trend_w *= 1.12
            mtf_w *= 1.10
        elif regime == "TREND_HIGH_VOL":
            pa_w *= 1.08
            vol_w *= 0.90
        elif regime in {"RANGE", "RANGE_LOW_VOL"}:
            structure_w *= 1.12
            pa_w *= 0.92
            trend_w *= 0.88
        elif regime == "RANGE_HIGH_VOL":
            pa_w *= 0.90
            trend_w *= 0.85
            vol_w *= 0.85

        raw = (
            trend_w * trend
            + momentum_w * momentum
            + pa_w * pa
            + structure_w * structure
            + vol_w * vol
            + mtf_w * mtf
        )

        # Normalize by effective weights so the score remains comparable.
        eff = trend_w + momentum_w + pa_w + structure_w + vol_w + mtf_w
        total = float(np.clip(raw / max(eff, EPS), -1, 1))

        call_pct, put_pct = direction_percent(total)

        if call_pct > put_pct:
            candidate = "CALL"
        elif put_pct > call_pct:
            candidate = "PUT"
        else:
            candidate = "NO TRADE"

        entry, entry_q, progress, entry_reason = entry_timing(
            candles_1m, candidate, regime, total
        )

        # User-requested behavior: once live/valid data and indicators
        # are ready, the stronger directional side is selected.
        decision = candidate if candidate != "NO TRADE" else "NO TRADE"
        status = "SIGNAL" if decision != "NO TRADE" else "FILTERED"

        # Confidence is a strength score, NOT a calibrated probability.
        edge = abs(call_pct - put_pct)
        confidence = int(np.clip(50 + edge * 0.85 + entry_q * 1.2 + (regime_strength - 0.5) * 10, 1, 95))

        reasons = []
        reasons.extend(regime_reasons)
        reasons.extend(trend_reasons)
        reasons.extend(momentum_reasons)
        reasons.extend(mtf_reasons)
        reasons.extend(pa_reasons)
        reasons.extend(structure_reasons)
        reasons.extend(vol_reasons)
        reasons.append(f"MTF scores: 1M {tf_scores[0]:+.2f}, 5M {tf_scores[1]:+.2f}, 15M {tf_scores[2]:+.2f}")
        reasons.append(f"Direction evidence: CALL {call_pct:.1f}% / PUT {put_pct:.1f}%")
        reasons.append(f"Entry timing: {entry}")
        reasons.append(f"Entry quality: {entry_q:.2f}/10")
        if entry_reason:
            reasons.append(entry_reason)

        last = f1.iloc[-1]
        indicators = {
            "ema9": round(safe(last["ema9"]), 8),
            "ema21": round(safe(last["ema21"]), 8),
            "ema50": round(safe(last["ema50"]), 8),
            "ema100": round(safe(last["ema100"]), 8),
            "ema200": round(safe(last["ema200"]), 8),
            "rsi": round(safe(last["rsi"]), 2),
            "adx": round(safe(last["adx"]), 2),
            "plus_di": round(safe(last["plus_di"]), 2),
            "minus_di": round(safe(last["minus_di"]), 2),
            "atr": round(safe(last["atr"]), 8),
            "macd_hist": round(safe(last["macd_hist"]), 8),
            "macd_hist_slope": round(safe(last["macd_hist_slope"]), 8),
            "stoch_k": round(safe(last["stoch_k"]), 2),
            "stoch_d": round(safe(last["stoch_d"]), 2),
            "williams_r": round(safe(last["williams_r"]), 2),
            "roc9": round(safe(last["roc9"]), 4),
            "cci": round(safe(last["cci"]), 2),
            "bb_width": round(safe(last["bb_width"]), 6),
            "atr_pct": round(safe(last["atr_pct"]), 4),
            "supertrend": "BULLISH" if safe(last["st_dir"]) > 0 else "BEARISH",
        }

        return {
            "decision": decision,
            "signal": decision,
            "status": status,
            "score": int(round(total * 100)),
            "raw_score": round(total, 4),
            "confidence": confidence,
            "direction_percent": {
                "CALL": call_pct,
                "PUT": put_pct,
            },
            "entry_timing": entry,
            "entry_quality": entry_q,
            "candle_progress": progress,
            "market_regime": regime,
            "volatility_regime": vol_regime,
            "timeframe_scores": {
                "1m": int(round(tf_scores[0] * 100)),
                "5m": int(round(tf_scores[1] * 100)),
                "15m": int(round(tf_scores[2] * 100)),
            },
            "mtf_agreement": int(sum(
                (s > 0 if candidate == "CALL" else s < 0)
                for s in tf_scores
            )) if candidate != "NO TRADE" else 0,
            "confirmation_ratio": None,
            "confirmation_mode": "5S_DISABLED",
            "indicators": indicators,
            "reasons": reasons,
        }

    except Exception as exc:
        return _error("ANALYSIS_ERROR", f"Analysis failed: {exc}")
