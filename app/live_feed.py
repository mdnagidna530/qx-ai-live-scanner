from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime
from typing import Any

import requests
from pysignalr.client import SignalRClient

BIQUOTE_HUB = "https://biquote.io/hubs/tick"
BIQUOTE_API = "https://biquote.io/api"
DEFAULT_SYMBOL = "EURUSD"


class LiveFeed:
    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        self.symbol = symbol.upper().strip()
        self.ticks_5s: deque[dict[str, float]] = deque(maxlen=5000)
        self.candles_1m: list[dict[str, Any]] = []
        self.candles_5m: list[dict[str, Any]] = []
        self.candles_15m: list[dict[str, Any]] = []
        self.connected = False
        self.last_tick_time = 0.0
        self.last_price: float | None = None
        self.price_source: str | None = None
        self.market_state: str | None = None
        self.stale = True
        self.stream_error: str | None = None
        self._client: SignalRClient | None = None
        self._task: asyncio.Task | None = None

    def load_history(self) -> None:
        for interval, target in (
            ("1m", self.candles_1m),
            ("5m", self.candles_5m),
            ("15m", self.candles_15m),
        ):
            r = requests.get(
                f"{BIQUOTE_API}/{self.symbol}/ohlc",
                params={"interval": interval, "limit": 500},
                timeout=15,
            )
            r.raise_for_status()
            bars = r.json().get("bars", [])
            target.clear()
            for bar in reversed(bars):
                target.append({
                    "timestamp": self._timestamp(bar.get("openTime")),
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(bar.get("tickVolume") or bar.get("volume") or 0),
                })

    def refresh_snapshot(self) -> bool:
        try:
            r = requests.get(
                f"{BIQUOTE_API}/{self.symbol}",
                params={"allowStale": "false"},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            price = float(data["mid"])
            if price <= 0:
                return False
            self.last_price = price
            self.price_source = data.get("source")
            self.market_state = data.get("marketState")
            self.stale = bool(data.get("stale", False))
            return True
        except Exception as exc:
            self.stream_error = f"REST snapshot: {exc}"
            return False

    def _handle_tick(self, tick: dict[str, Any]) -> None:
        if str(tick.get("symbol", "")).upper() != self.symbol:
            return
        try:
            price = float(tick.get("mid"))
        except (TypeError, ValueError):
            return
        if price <= 0:
            return
        timestamp = self._timestamp(tick.get("timestamp")) or time.time()
        self.ticks_5s.append({"timestamp": timestamp, "price": price})
        self.last_price = price
        self.last_tick_time = timestamp
        self.price_source = tick.get("source")
        self.market_state = tick.get("marketState") or "open"
        self.stale = False
        self.stream_error = None
        self._update_running_candles(timestamp, price)

    def _update_running_candles(self, timestamp: float, price: float) -> None:
        self._update_timeframe(self.candles_1m, timestamp, price, 60)
        self._update_timeframe(self.candles_5m, timestamp, price, 300)
        self._update_timeframe(self.candles_15m, timestamp, price, 900)

    def _update_timeframe(self, candles, timestamp, price, seconds) -> None:
        bucket = int(timestamp // seconds) * seconds
        if not candles:
            candles.append(self._new_candle(bucket, price))
            return
        current = candles[-1]
        if int(current["timestamp"]) != bucket:
            candles.append(self._new_candle(bucket, price))
            if len(candles) > 600:
                del candles[:-600]
            return
        current["high"] = max(float(current["high"]), price)
        current["low"] = min(float(current["low"]), price)
        current["close"] = price
        current["volume"] = float(current.get("volume", 0)) + 1.0

    @staticmethod
    def _new_candle(timestamp: int, price: float) -> dict[str, Any]:
        return {
            "timestamp": timestamp, "open": price, "high": price,
            "low": price, "close": price, "volume": 1.0
        }

    async def connect(self) -> None:
        try:
            await asyncio.to_thread(self.load_history)
        except Exception as exc:
            self.stream_error = f"History: {exc}"
        try:
            client = SignalRClient(BIQUOTE_HUB)
            self._client = client
            client.on("ReceiveTick", self._receive_tick)
            client.on_open(self._on_open)
            self.connected = False
            await client.run()
        except Exception as exc:
            self.connected = False
            self.stream_error = f"SignalR: {exc}"
        finally:
            self.connected = False

    def _receive_tick(self, message: Any) -> None:
        if isinstance(message, list):
            if not message:
                return
            message = message[0]
        if isinstance(message, dict):
            self._handle_tick(message)

    def _on_open(self) -> None:
        self.connected = True
        self.stream_error = None
        if self._client is not None:
            self._client.send("Subscribe", [[self.symbol]])

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self.connect())

    def get_data(self) -> dict[str, Any]:
        now = time.time()
        age = now - self.last_tick_time if self.last_tick_time else None
        recent = self._recent_5_seconds()
        return {
            "symbol": self.symbol,
            "connected": self.connected,
            "last_price": self.last_price,
            "last_tick_age": age,
            "price_source": self.price_source,
            "market_state": self.market_state,
            "stale": self.stale,
            "stream_error": self.stream_error,
            "has_real_5s_ticks": bool(recent),
            "candles_1m": list(self.candles_1m),
            "candles_5m": list(self.candles_5m),
            "candles_15m": list(self.candles_15m),
            "ticks_5s": recent,
        }

    def _recent_5_seconds(self) -> list[dict[str, float]]:
        cutoff = time.time() - 5.0
        return [t for t in self.ticks_5s if t["timestamp"] >= cutoff]

    @staticmethod
    def _timestamp(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            n = float(value)
            return n / 1000 if n > 100_000_000_000 else n
        text = str(value).strip()
        try:
            n = float(text)
            return n / 1000 if n > 100_000_000_000 else n
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0


def get_live_snapshot(symbol: str = DEFAULT_SYMBOL) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    r = requests.get(
        f"{BIQUOTE_API}/{symbol}",
        params={"allowStale": "false"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "symbol": symbol,
        "price": float(data["mid"]),
        "timestamp": data.get("timestamp"),
        "market_state": data.get("marketState"),
        "stale": bool(data.get("stale", False)),
        "source": data.get("source"),
    }
