import asyncio
import hashlib
import hmac
import time
import aiohttp
import ujson


class Websocket:
    base_url: str
    recv_window = 5000
    reconnect_timeout = 30
    ping_interval = 180

    def __init__(self, category='linear', streams=None, on_message=None, on_open=None, on_close=None, on_error=None,
                 testnet=False, api_key=None, secret_key=None):
        if isinstance(streams, str):
            streams = [streams]
        self.streams = streams
        self.on_message = on_message
        self.on_open = on_open
        self.on_close = on_close
        self.on_error = on_error
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.working = True
        self.connected = False
        self.ws = None
        if testnet:
            self.base_url = f'wss://stream-testnet.bybit.com/v5/'
        else:
            self.base_url = f'wss://stream.bybit.com/v5/'
        if api_key and secret_key:
            self.base_url += 'private'
        else:
            self.base_url += f'public/{category}'

    async def run(self):
        asyncio.create_task(self._run())
        if self.api_key and self.secret_key:
            asyncio.create_task(self.ping_service())

    async def _run(self):
        while self.working:
            self.connected = False
            try:
                while self.working:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.ws_connect(self.base_url) as self.ws:
                                self.connected = True
                                auth = await self.auth()
                                if self.streams and not auth:
                                    await self._subscribe(self.streams)
                                if self.on_open:
                                    await self.on_open(self.ws)
                                async for msg in self.ws:
                                    if msg.type == aiohttp.WSMsgType.TEXT:
                                        data = msg.data
                                        try:
                                            data = ujson.loads(data)
                                        except ujson.JSONDecodeError:
                                            pass
                                        try:
                                            if not self.working:
                                                break
                                            if auth and data.get('op') == 'auth' and data.get('success'):
                                                await self._subscribe(self.streams)
                                            if self.on_message:
                                                await self.on_message(self.ws, data)
                                        except Exception as e:
                                            if self.on_error:
                                                try:
                                                    await self.on_error(self.ws, e)
                                                except:
                                                    pass
                                    elif msg.type == aiohttp.WSMsgType.ERROR:
                                        if self.on_error:
                                            try:
                                                await self.on_error(self.ws, msg)
                                            except:
                                                pass
                                        break
                    except aiohttp.ClientConnectionError as e:
                        if self.on_error:
                            try:
                                await self.on_error(self.ws, e)
                            except:
                                pass
                    except asyncio.CancelledError:
                        break
            except Exception as e:
                if self.on_error:
                    try:
                        await self.on_error(self.ws, e)
                    except:
                        pass
            finally:
                if not self.connected:
                    await asyncio.sleep(self.reconnect_timeout)

    async def close(self):
        self.working = False
        if self.ws:
            await self.ws.close()

    async def subscribe(self, streams: list):
        if isinstance(streams, str):
            streams = [streams]
        self.streams.extend(streams)
        await self._subscribe(streams)

    async def _subscribe(self, streams: list):
        msg = {
            "op": "subscribe",
            "args": streams
        }
        await self.ws.send_str(ujson.dumps(msg))

    async def unsubscribe(self, streams: list):
        if isinstance(streams, str):
            streams = [streams]
        if self.streams:
            self.streams = [stream for stream in self.streams if stream not in streams]
        msg = {
            "op": "unsubscribe",
            "args": streams
        }
        await self.ws.send_str(ujson.dumps(msg))

    async def auth(self):
        if not self.api_key or not self.secret_key:
            return
        timestamp = int(time.time() * 1000 + self.recv_window)
        data = f"GET/realtime{timestamp}"
        signature = hmac.new(bytes(self.secret_key, "utf-8"), data.encode("utf-8"), hashlib.sha256).hexdigest()
        msg = {
            "op": "auth",
            "args": [self.api_key, timestamp, signature]
        }
        await self.ws.send_str(ujson.dumps(msg))
        return True

    async def ping_service(self):
        msg = ujson.dumps({"op": "ping"})
        while True:
            try:
                if not self.working:
                    break
                if self.connected:
                    await self.ws.send_str(msg)
            except Exception as e:
                if self.on_error:
                    try:
                        await self.on_error(self.ws, e)
                    except:
                        pass
            finally:
                await asyncio.sleep(self.ping_interval)
