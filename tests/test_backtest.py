from src.data_loader import fetch_data
from src.indicators import add_indicators
from src.models import predict_returns
from src.strategy import generate_signals
from src.backtest import backtest

def test_backtest():
    df = fetch_data('AAPL', interval='1d', lookback=60)
    df_ind = add_indicators(df)
    preds, _ = predict_returns(df_ind, gpu=False)
    signals = generate_signals(df_ind, preds, risk_pct=1)
    bt = backtest(signals)
    assert bt.cagr is not None
    assert bt.equity_curve.iloc[-1] > 0
