from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

from .auth import login, require
from .engine import analyze
from .live_feed import LiveFeed


app = FastAPI(
    title="QX AI Live Scanner V4",
    version="4.1"
)


# ============================================================
# LIVE FEED MANAGER
# ============================================================

_FEEDS: dict[str, LiveFeed] = {}
_FEEDS_LOCK = asyncio.Lock()


async def get_live_feed(symbol: str) -> LiveFeed:
    symbol = symbol.upper().strip()

    if not symbol:
        raise HTTPException(
            status_code=400,
            detail="Market symbol is required"
        )

    async with _FEEDS_LOCK:
        feed = _FEEDS.get(symbol)

        if feed is None:
            feed = LiveFeed(symbol)
            _FEEDS[symbol] = feed
            await feed.start()

    return feed


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    # Start the default live feed.
    await get_live_feed("EURUSD")


# ============================================================
# DATA MODELS
# ============================================================

class Candle(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0


class Tick(BaseModel):
    timestamp: float
    price: float


class LoginRequest(BaseModel):
    password: str


class ScanRequest(BaseModel):
    # Live mode
    symbol: Optional[str] = None

    # Manual / compatibility mode
    candles_1m: List[Candle] = Field(default_factory=list)
    candles_5m: List[Candle] = Field(default_factory=list)
    candles_15m: List[Candle] = Field(default_factory=list)
    ticks_5s: List[Tick] = Field(default_factory=list)


# ============================================================
# WEB INTERFACE
# ============================================================

@app.get("/")
def home():
    return FileResponse(
        "static/index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "ok": True,
        "service": "QX AI Live Scanner V4",
        "version": "4.1",
        "mode": "SIGNAL_ONLY",
        "automatic_trading": False,
        "live_feed": True
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/api/login")
def api_login(
    request: LoginRequest
):

    try:
        token = login(
            request.password
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid password"
        )

    return {
        "success": True,
        "token": token
    }


# ============================================================
# LIVE DATA
# ============================================================

@app.get("/api/live")
async def live_data(
    symbol: str = Query(
        default="EURUSD",
        min_length=1
    ),
    x_session: str | None = Header(
        default=None
    )
):

    try:
        require(x_session)
    except PermissionError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc)
        )

    feed = await get_live_feed(symbol)

    data = feed.get_data()

    return {
        "success": True,
        "source": "BiQuote",
        "signal_only": True,
        "automatic_trading": False,
        "data": data
    }


# ============================================================
# LIVE SCAN
# ============================================================

@app.post("/api/scan")
async def scan(
    request: ScanRequest,
    x_session: str | None = Header(
        default=None
    )
):

    # --------------------------------------------------------
    # SECURITY CHECK
    # --------------------------------------------------------

    try:
        require(x_session)
    except PermissionError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc)
        )

    # --------------------------------------------------------
    # LIVE MODE
    # --------------------------------------------------------

    if request.symbol:

        feed = await get_live_feed(
            request.symbol
        )

        live = feed.get_data()

        if not live["connected"]:
            return {
                "signal": "NO TRADE",
                "reason": "LIVE DATA NOT CONNECTED",
                "source": "BiQuote",
                "symbol": live["symbol"],
                "live_data": False
            }

        if live["last_price"] is None:
            return {
                "signal": "NO TRADE",
                "reason": "WAITING FOR LIVE PRICE",
                "source": "BiQuote",
                "symbol": live["symbol"],
                "live_data": False
            }

        # ----------------------------------------------------
        # Analyze REAL received live data
        # ----------------------------------------------------

        result = analyze(
            candles_1m=live["candles_1m"],
            candles_5m=live["candles_5m"],
            candles_15m=live["candles_15m"],
            ticks_5s=live["ticks_5s"]
        )

        # Add live-feed metadata without changing
        # the analysis engine's decision.
        result["source"] = "BiQuote"
        result["symbol"] = live["symbol"]
        result["live_data"] = True
        result["last_price"] = live["last_price"]
        result["last_tick_age"] = live["last_tick_age"]
        result["feed_connected"] = live["connected"]

        return result

    # --------------------------------------------------------
    # MANUAL / COMPATIBILITY MODE
    # --------------------------------------------------------

    candles_1m = [
        candle.model_dump()
        for candle in request.candles_1m
    ]

    candles_5m = [
        candle.model_dump()
        for candle in request.candles_5m
    ]

    candles_15m = [
        candle.model_dump()
        for candle in request.candles_15m
    ]

    ticks_5s = [
        tick.model_dump()
        for tick in request.ticks_5s
    ]

    result = analyze(
        candles_1m=candles_1m,
        candles_5m=candles_5m,
        candles_15m=candles_15m,
        ticks_5s=ticks_5s
    )

    result["live_data"] = False

    return result
