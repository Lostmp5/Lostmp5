# Copyright (c) 2024.
# Licensed under the MIT License.
"""Simple walk-forward backtesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series[float]
    cagr: float
    max_drawdown: float
    win_rate: float


def backtest(df: pd.DataFrame) -> BacktestResult:
    """Basic back-test assuming next-day execution."""
    logger.info("Running backtest over %d rows", len(df))
    returns = df["close"].pct_change().shift(-1).iloc[:-1]
    strat_ret = returns * df["signal"].iloc[:-1] * df["size"].iloc[:-1]
    equity_curve = (1 + strat_ret).cumprod()
    if equity_curve.empty:
        return BacktestResult(pd.Series(dtype=float), 0.0, 0.0, 0.0)
    cagr = equity_curve.iloc[-1] ** (365 / len(equity_curve)) - 1
    rolling_max = equity_curve.cummax()
    max_dd = ((equity_curve - rolling_max) / rolling_max).min()
    win_rate = (strat_ret > 0).mean()
    return BacktestResult(equity_curve, cagr, max_dd, win_rate)
