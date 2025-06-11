# Market Analysis & Trade-Recommendation System

This project provides a command-line tool to fetch market data, compute technical indicators, train a simple forecast model and optionally backtest a naive strategy. It is intended **for educational purposes only—no investment advice**.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Example CLI call:

```bash
python -m src.cli --symbol ETH-USD --interval 1h --lookback 120 --risk_pct 1 --gpu --plot --backtest
```

## Indicator Glossary

Indicators include SMA, EMA, RSI, MACD and ATR.

## Caveats

- Forecasting and strategy logic are simplified for demonstration.
- Ensure you understand all trading risks before using in production.
