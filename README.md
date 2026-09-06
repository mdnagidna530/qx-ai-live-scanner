# QX AI Live Scanner V4
Signal-only scanner for manual trading. Password: `12345678`.
It does not place trades and never fabricates 5-second data.
Pipeline: live ticks -> running candles -> 1m/5m/15m analysis -> price action -> trend/momentum/volatility -> true 5s confirmation -> CALL/PUT/NO TRADE.
Without a real authorized live feed it stays NO TRADE/DATA NOT CONNECTED.
Run: `pip install -r requirements.txt` then `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
Docker files are included. See app/feed_contract.py for the live-data contract.
