from src.data_loader import fetch_data
from src.indicators import add_indicators
from src.models import predict_returns
from src.strategy import generate_signals

def test_generate_signals():
    df = fetch_data('AAPL', interval='1d', lookback=60)
    df_ind = add_indicators(df)
    preds, _ = predict_returns(df_ind, gpu=False)
    signals = generate_signals(df_ind, preds, risk_pct=1)
    assert 'signal' in signals.columns
    assert set(signals['signal'].unique()) <= {1, -1}
