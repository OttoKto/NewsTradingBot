import websocket
import json
def on_message(ws, message):
    print(f"Received: {message}")
    data = json.loads(message)
    if data.get("event_type") == "tweet":
        print(f"Tweet: {data.get('tweets', [])}")
def on_error(ws, error):
    print(f"Error: {error}")
def on_close(ws, code, msg):
    print(f"Closed: code={code}, msg={msg}")
def on_open(ws):
    print("Connected!")
ws = websocket.WebSocketApp(
    "wss://ws.twitterapi.io/twitter/tweet/websocket",
    header={"x-api-key": "d13e7230ecab4060871c64f8f3501413", "User-Agent": "TradingBot/1.0"},
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
    on_open=on_open
)
ws.run_forever(ping_interval=40, ping_timeout=30, reconnect=90)