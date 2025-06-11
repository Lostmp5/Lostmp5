from src.data_loader import fetch_data
from src.indicators import add_indicators
from src.models import predict_returns

def test_predict_returns():
    df = fetch_data('AAPL', interval='1d', lookback=60)
    df_ind = add_indicators(df)
    preds, model = predict_returns(df_ind, gpu=False)
    assert len(preds) == len(df_ind) - 1
