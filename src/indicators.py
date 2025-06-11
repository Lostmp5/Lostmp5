# Copyright (c) 2024.
# Licensed under the MIT License.
"""Indicator calculations."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append simple technical indicators to ``df``."""
    logger.info("Calculating technical indicators")
    data = df.copy()
    data["sma20"] = data["close"].rolling(20).mean()
    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["rsi"] = 100 - (100 / (1 + rs))
    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    high_low = data[["high", "close"]].max(axis=1) - data[["low", "close"]].min(axis=1)
    data["atr"] = high_low.rolling(14).mean()
    data.dropna(inplace=True)
    return data
