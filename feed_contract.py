"""
QX AI Live Scanner V4
Live data contract.

IMPORTANT:
This module defines the format expected from a REAL,
authorized market-data source.

The scanner must NEVER manufacture 5-second ticks.
"""

from typing import TypedDict


class Candle(TypedDict):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class Tick(TypedDict):
    timestamp: float
    price: float


def validate_candle(candle: dict) -> bool:
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    if not required.issubset(candle.keys()):
        return False

    try:
        float(candle["open"])
        float(candle["high"])
        float(candle["low"])
        float(candle["close"])
        float(candle["volume"])
        int(candle["timestamp"])
    except (TypeError, ValueError):
        return False

    return (
        candle["high"] >= candle["low"]
        and candle["high"] >= max(
            candle["open"],
            candle["close"]
        )
        and candle["low"] <= min(
            candle["open"],
            candle["close"]
        )
    )


def validate_tick(tick: dict) -> bool:
    required = {
        "timestamp",
        "price",
    }

    if not required.issubset(tick.keys()):
        return False

    try:
        float(tick["timestamp"])
        price = float(tick["price"])
    except (TypeError, ValueError):
        return False

    return price > 0
