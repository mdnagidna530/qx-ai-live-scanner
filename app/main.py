from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import logging

from .auth import login, require
from .users import (
    admin_login, require_admin, create_user, login_user, require_user,
    list_users, extend_user, set_active, get_user
)
from .engine import analyze
from .live_feed import LiveFeed

logger = logging.getLogger(__name__)

app = FastAPI(title="QX AI Live Scanner V7", version="7.0")

_FEEDS: dict[str, LiveFeed] = {}
_FEEDS_LOCK = asyncio.Lock()

async def get_live_feed(symbol: str) -> LiveFeed:
    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Market symbol is required")
    async with _FEEDS_LOCK:
        feed = _FEEDS.get(symbol)
        if feed is None:
            feed = LiveFeed(symbol)
            _FEEDS[symbol] = feed
            await feed.start()
    return feed

async def _start_default_feed() -> None:
    try:
        await get_live_feed("EURUSD")
        logger.info("Default EURUSD live feed startup completed")
    except Exception:
        logger.exception("Default live feed startup failed")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_start_default_feed())

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

class UserLoginRequest(BaseModel):
    username: str
    password: str
    device_id: Optional[str] = None

class AdminLoginRequest(BaseModel):
    password: str

class CreateUserRequest(BaseModel):
    username: str
    password: str
    days: int = Field(ge=1, le=3650)
    max_devices: int = Field(default=1, ge=1, le=5)

class ExtendUserRequest(BaseModel):
    days: int = Field(ge=1, le=3650)

class ActiveRequest(BaseModel):
    active: bool

class ScanRequest(BaseModel):
    symbol: Optional[str] = None
    candles_1m: List[Candle] = Field(default_factory=list)
    candles_5m: List[Candle] = Field(default_factory=list)
    candles_15m: List[Candle] = Field(default_factory=list)
    ticks_5s: List[Tick] = Field(default_factory=list)

def _require_access(token: str | None):
    # Accept both the legacy web-scanner session and subscription users.
    try:
        return require_user(token)
    except PermissionError:
        try:
            require(token)
            return {"username": "legacy"}
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc))

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "QX AI Live Scanner V7",
        "version": "7.0",
        "mode": "SIGNAL_ONLY",
        "automatic_trading": False,
        "live_feed": True,
        "subscription_api": True,
    }

# Legacy single-password login retained so the current web scanner keeps working.
@app.post("/api/login")
def api_login(request: LoginRequest):
    try:
        token = login(request.password)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"success": True, "token": token, "mode": "legacy"}

# New customer login for the Android app.
@app.post("/api/user/login")
def api_user_login(request: UserLoginRequest):
    try:
        token, user = login_user(request.username, request.password, request.device_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    return {"success": True, "token": token, "user": user}

@app.get("/api/user/me")
def api_user_me(x_session: str | None = Header(default=None)):
    try:
        user = require_user(x_session)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    return {"success": True, "user": user}

# Admin API: create, list, extend and suspend/activate paid users.
@app.post("/api/admin/login")
def api_admin_login(request: AdminLoginRequest):
    try:
        token = admin_login(request.password)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid admin password")
    return {"success": True, "token": token}

def _admin(token: str | None):
    try:
        require_admin(token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

@app.get("/api/admin/users")
def api_admin_users(x_admin_session: str | None = Header(default=None)):
    _admin(x_admin_session)
    return {"success": True, "users": list_users()}

@app.post("/api/admin/users")
def api_admin_create_user(request: CreateUserRequest, x_admin_session: str | None = Header(default=None)):
    _admin(x_admin_session)
    try:
        user = create_user(request.username, request.password, request.days, request.max_devices)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "user": user}

@app.post("/api/admin/users/{user_id}/extend")
def api_admin_extend_user(user_id: int, request: ExtendUserRequest, x_admin_session: str | None = Header(default=None)):
    _admin(x_admin_session)
    try:
        user = extend_user(user_id, request.days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "user": user}

@app.post("/api/admin/users/{user_id}/active")
def api_admin_set_active(user_id: int, request: ActiveRequest, x_admin_session: str | None = Header(default=None)):
    _admin(x_admin_session)
    try:
        user = set_active(user_id, request.active)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"success": True, "user": user}

@app.get("/api/live")
async def live_data(symbol: str = Query(default="EURUSD", min_length=1), x_session: str | None = Header(default=None)):
    _require_access(x_session)
    feed = await get_live_feed(symbol)
    data = feed.get_data()
    return {"success": True, "source": "BiQuote", "signal_only": True, "automatic_trading": False, "data": data}

@app.post("/api/scan")
async def scan(request: ScanRequest, x_session: str | None = Header(default=None)):
    _require_access(x_session)
    if request.symbol:
        feed = await get_live_feed(request.symbol)
        live = feed.get_data()
        if not live["connected"]:
            return {"signal": "NO TRADE", "reason": "LIVE DATA NOT CONNECTED", "source": "BiQuote", "symbol": live["symbol"], "live_data": False}
        if live["last_price"] is None:
            return {"signal": "NO TRADE", "reason": "WAITING FOR LIVE PRICE", "source": "BiQuote", "symbol": live["symbol"], "live_data": False}
        result = analyze(candles_1m=live["candles_1m"], candles_5m=live["candles_5m"], candles_15m=live["candles_15m"], ticks_5s=live["ticks_5s"], symbol=live["symbol"])
        result["source"] = "BiQuote"
        result["symbol"] = live["symbol"]
        result["live_data"] = True
        result["last_price"] = live["last_price"]
        result["last_tick_age"] = live["last_tick_age"]
        result["feed_connected"] = live["connected"]
        return result

    candles_1m = [candle.model_dump() for candle in request.candles_1m]
    candles_5m = [candle.model_dump() for candle in request.candles_5m]
    candles_15m = [candle.model_dump() for candle in request.candles_15m]
    ticks_5s = [tick.model_dump() for tick in request.ticks_5s]
    result = analyze(candles_1m=candles_1m, candles_5m=candles_5m, candles_15m=candles_15m, ticks_5s=ticks_5s, symbol=request.symbol or "")
    result["live_data"] = False
    return result
