
"""
QX AI Live Scanner V4
Verified live market-data adapter.

Source:
BiQuote public market-data feed.

IMPORTANT:
- Signal-only
- No order execution
- No fabricated ticks
- Uses real received ticks only
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import requests
from pysignalr.client import SignalRClient


BIQUOTE_HUB = "https://biquote.io/hubs/tick"
BIQUOTE_API = "https://biquote.io/api"

DEFAULT_SYMBOL = "EURUSD"


@dataclass
class LiveTick:
    timestamp: float
    price: float
    symbol: str


class LiveFeed:
    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        self.symbol = symbol.upper().strip()

        self.ticks_5s: deque[dict[str, float]] = deque(maxlen=3000)

        self.candles_1m: list[dict[str, Any]] = []
        self.candles_5m: list[dict[str, Any]] = []
        self.candles_15m: list[dict[str, Any]] = []

        self.connected = False
        self.last_tick_time = 0.0
        self.last_price: float | None = None

        self._client: SignalRClient | None = None
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------
    # REST bootstrap
    # ---------------------------------------------------------

    def load_history(self) -> None:
        """
        Load historical 1m / 5m / 15m candles.
        The current open candle is also returned by the source,
        but we keep the data as received and mark it appropriately.
        """

        for interval, target in (
            ("1m", self.candles_1m),
            ("5m", self.candles_5m),
            ("15m", self.candles_15m),
        ):
            response = requests.get(
                f"{BIQUOTE_API}/{self.symbol}/ohlc",
                params={
                    "interval": interval,
                    "limit": 500,
                },
                timeout=15,
            )

            response.raise_for_status()

            payload = response.json()
            bars = payload.get("bars", [])

            target.clear()

            for bar in reversed(bars):
                target.append(
                    {
                        "timestamp": self._timestamp(bar.get("openTime")),
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": float(
                            bar.get("volume")
                            or bar.get("tickVolume")
                            or 0
                        ),
                    }
                )

    # ---------------------------------------------------------
    # Live tick handling
    # ---------------------------------------------------------

    def _handle_tick(self, tick: dict[str, Any]) -> None:
        symbol = str(tick.get("symbol", "")).upper()

        if symbol != self.symbol:
            return

        price = tick.get("mid")

        if price is None:
            return

        try:
            price = float(price)
        except (TypeError, ValueError):
            return

        if price <= 0:
            return

        timestamp = self._timestamp(
            tick.get("timestamp")
        )

        if timestamp <= 0:
            timestamp = time.time()

        item = {
            "timestamp": timestamp,
            "price": price,
        }

        self.ticks_5s.append(item)

        self.last_price = price
        self.last_tick_time = timestamp

        self._update_running_candles(
            timestamp,
            price,
        )

    # ---------------------------------------------------------
    # Running candles
    # ---------------------------------------------------------

    def _update_running_candles(
        self,
        timestamp: float,
        price: float,
    ) -> None:

        self._update_timeframe(
            self.candles_1m,
            timestamp,
            price,
            60,
        )

        self._update_timeframe(
            self.candles_5m,
            timestamp,
            price,
            300,
        )

        self._update_timeframe(
            self.candles_15m,
            timestamp,
            price,
            900,
        )

    def _update_timeframe(
        self,
        candles: list[dict[str, Any]],
        timestamp: float,
        price: float,
        seconds: int,
    ) -> None:

        bucket = int(timestamp // seconds) * seconds

        if not candles:
            candles.append(
                {
                    "timestamp": bucket,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1.0,
                }
            )
            return

        current = candles[-1]

        if int(current["timestamp"]) != bucket:
            candles.append(
                {
                    "timestamp": bucket,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1.0,
                }
            )

            # Keep memory bounded.
            if len(candles) > 600:
                del candles[:-600]

            return

        current["high"] = max(
            float(current["high"]),
            price,
        )

        current["low"] = min(
            float(current["low"]),
            price,
        )

        current["close"] = price
        current["volume"] = (
            float(current.get("volume", 0))
            + 1.0
        )

    # ---------------------------------------------------------
    # SignalR connection
    # ---------------------------------------------------------

    async def connect(self) -> None:
        self.load_history()

        client = SignalRClient(BIQUOTE_HUB)
        self._client = client

        client.on(
            "ReceiveTick",
            self._receive_tick,
        )

        client.on_open(
            self._on_open,
        )

        self.connected = False

        await client.run()

    def _receive_tick(self, message: Any) -> None:
        """
        pysignalr may return the event arguments as a list.
        """

        if isinstance(message, list):
            if not message:
                return
            message = message[0]

        if not isinstance(message, dict):
            return

        self._handle_tick(message)

    def _on_open(self) -> None:
        self.connected = True

        if self._client is not None:
            self._client.send(
                "Subscribe",
                [[self.symbol]],
            )

    async def start(self) -> None:
        if self._task and not self._task.done():
            return

        self._task = asyncio.create_task(
            self.connect()
        )

    # ---------------------------------------------------------
    # Data access
    # ---------------------------------------------------------

    def get_data(self) -> dict[str, Any]:
        now = time.time()

        age = (
            now - self.last_tick_time
            if self.last_tick_time
            else None
        )

        return {
            "symbol": self.symbol,
            "connected": self.connected,
            "last_price": self.last_price,
            "last_tick_age": age,
            "candles_1m": list(self.candles_1m),
            "candles_5m": list(self.candles_5m),
            "candles_15m": list(self.candles_15m),
            "ticks_5s": self._recent_5_seconds(),
        }

    def _recent_5_seconds(self) -> list[dict[str, float]]:
        if not self.ticks_5s:
            return []

        cutoff = time.time() - 5.0

        return [
            tick
            for tick in self.ticks_5s
            if tick["timestamp"] >= cutoff
        ]

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _timestamp(value: Any) -> float:
        if value is None:
            return 0.0

        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()

        try:
            return float(text)
        except ValueError:
            pass

        try:
            from datetime import datetime

            text = text.replace("Z", "+00:00")

            return datetime.fromisoformat(
                text
            ).timestamp()

        except Exception:
            return 0.0


def get_live_snapshot(
    symbol: str = DEFAULT_SYMBOL,
) -> dict[str, Any]:
    """
    Synchronous REST snapshot.

    Used for connectivity testing before the
    continuous SignalR stream is enabled.
    """

    symbol = symbol.upper().strip()

    response = requests.get(
        f"{BIQUOTE_API}/{symbol}",
        params={
            "allowStale": "false",
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    return {
        "symbol": symbol,
        "price": float(data["mid"]),
        "timestamp": data.get("timestamp"),
        "market_state": data.get("marketState"),
        "stale": bool(data.get("stale", False)),
        "source": data.get("source"),
    }
