from src.data_loader import fetch_data
from src.indicators import add_indicators

def test_add_indicators():
    df = fetch_data('AAPL', interval='1d', lookback=10)
    df_ind = add_indicators(df)
    assert len(df_ind) < len(df)  # due to dropna
    assert 'sma20' in df_ind.columns
