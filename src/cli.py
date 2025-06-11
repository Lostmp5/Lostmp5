# Copyright (c) 2024.
# Licensed under the MIT License.
"""Command line interface for the trading system."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from .data_loader import fetch_data
from .indicators import add_indicators
from .models import predict_returns
from .strategy import generate_signals
from .backtest import backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Market analysis and trade recommendation system")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--lookback", type=int, default=365)
    parser.add_argument("--risk_pct", type=float, default=2.0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = fetch_data(args.symbol, interval=args.interval, lookback=args.lookback)
    df_ind = add_indicators(df)
    preds, _ = predict_returns(df_ind, gpu=args.gpu)
    signals = generate_signals(df_ind, preds, risk_pct=args.risk_pct)

    result_json = {}

    if args.backtest:
        bt = backtest(signals)
        result_json = {
            "cagr": bt.cagr,
            "max_drawdown": bt.max_drawdown,
            "win_rate": bt.win_rate,
        }
        if args.plot:
            reports = Path("reports")
            reports.mkdir(exist_ok=True)
            bt.equity_curve.plot(title="Equity Curve")
            plt.savefig(reports / "equity_curve.png")

    print(json.dumps(result_json, indent=2))
    print("Educational purposes only—no investment advice.")


if __name__ == "__main__":
    main()
