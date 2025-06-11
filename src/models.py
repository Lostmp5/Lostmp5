# Copyright (c) 2024.
# Licensed under the MIT License.
"""Forecasting models."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


def predict_returns(df: pd.DataFrame, *, gpu: bool = False) -> Tuple[np.ndarray, xgb.XGBRegressor]:
    """Train an XGBoost model to forecast next-period returns.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with indicators.
    gpu: bool, optional
        Use GPU acceleration if ``True``.

    Returns
    -------
    tuple[np.ndarray, xgb.XGBRegressor]
        (predictions, fitted model)
    """
    logger.info("Training XGBoost model (gpu=%s)", gpu)

    df_feat = df.copy()
    df_feat["return"] = df_feat["close"].pct_change().shift(-1)
    df_feat.dropna(inplace=True)

    X = df_feat.drop(columns=["return"]).values
    y = df_feat["return"].values

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="reg:squarederror",
        tree_method="gpu_hist" if gpu else "hist",
        n_jobs=1,
    )
    model.fit(X, y)

    preds = model.predict(X)
    return preds, model
