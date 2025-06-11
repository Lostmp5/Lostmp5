from src.data_loader import fetch_data

def test_fetch_data():
    df = fetch_data('AAPL', interval='1d', lookback=5)
    assert not df.empty
    assert {'open', 'close', 'high', 'low', 'volume'}.issubset(df.columns)
