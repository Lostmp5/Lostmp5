# Copyright (c) 2024.
# Licensed under the MIT License.
"""Meta-strategy for position sizing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_signals(
    df: pd.DataFrame, predictions: np.ndarray, *, risk_pct: float
) -> pd.DataFrame:
    """Combine indicator data and model predictions into trading signals."""
    logger.info("Generating trading signals with risk %%: %s", risk_pct)

    df_sig = df.copy().iloc[-len(predictions) :].copy()
    df_sig["pred"] = predictions
    df_sig["signal"] = np.where(df_sig["pred"] > 0, 1, -1)
    df_sig["size"] = risk_pct / 100
    return df_sig
