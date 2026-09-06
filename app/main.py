from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from .auth import login, require
from .engine import analyze


app = FastAPI(
    title="QX AI Live Scanner V4",
    version="4.0"
)


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
    candles_1m: List[Candle]
    candles_5m: List[Candle]
    candles_15m: List[Candle]
    ticks_5s: List[Tick]


# ============================================================
# WEB INTERFACE
# ============================================================

@app.get("/")
def home():
    return FileResponse(
        "app/static/index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "ok": True,
        "service": "QX AI Live Scanner V4",
        "mode": "SIGNAL_ONLY",
        "automatic_trading": False
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/api/login")
def api_login(
    request: LoginRequest
):

    token = login(
        request.password
    )

    return {
        "success": True,
        "token": token
    }


# ============================================================
# LIVE SCAN
# ============================================================

@app.post("/api/scan")
def scan(
    request: ScanRequest,
    x_session: str | None = Header(
        default=None
    )
):

    # --------------------------------------------------------
    # SECURITY CHECK
    # --------------------------------------------------------

    require(
        x_session
    )

    # --------------------------------------------------------
    # Convert Pydantic models to dictionaries
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

    # --------------------------------------------------------
    # RUN AI ANALYSIS ENGINE
    # --------------------------------------------------------

    result = analyze(
        candles_1m=candles_1m,
        candles_5m=candles_5m,
        candles_15m=candles_15m,
        ticks_5s=ticks_5s
    )

    return result
