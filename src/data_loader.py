# Copyright (c) 2024.
# Licensed under the MIT License.
"""Data ingestion utilities."""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import yfinance as yf  # type: ignore

logger = logging.getLogger(__name__)


def fetch_data(symbol: str, *, interval: str, lookback: int = 365) -> pd.DataFrame:
    """Download OHLCV data via yfinance."""
    logger.info("Fetching %s data at %s interval for %s days", symbol, interval, lookback)

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback)

    raw = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=False,
    )
    df: pd.DataFrame = pd.DataFrame(raw)
    if df.empty:
        msg = f"No data returned for {symbol}"
        logger.error(msg)
        raise ValueError(msg)
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    df.columns = df.columns.get_level_values(0).str.lower()
    df.columns.name = None
    return df
