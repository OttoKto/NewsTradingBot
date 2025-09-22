import hashlib
import hmac
import time
from urllib.parse import urlencode
import aiohttp
import requests
import ujson
from .error import ClientException, ServerException
import certifi
import ssl
import asyncio

class Client:
    base_url: str
    recv_window = 5000
    MAX_RETRIES = 5
    RETRY_DELAY = 3  # seconds

    def __init__(self, api_key=None, secret_key=None, testnet=False, demo=False, asynced=False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.demo = demo
        if demo:
            self.base_url = "https://api-demo.bybit.com"
        elif testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

        self.asynced = False
        self.headers = {
            "Content-Type": "application/json;charset=utf-8"
        }
        if api_key and secret_key:
            self.headers["X-BAPI-API-KEY"] = api_key
            self.headers['X-BAPI-RECV-WINDOW'] = str(self.recv_window)
        if asynced:
            self._set_async()
        else:
            self.session = requests.Session()
            self.session.headers.update(self.headers)

    def _set_async(self):
        if not self.asynced:
            self.asynced = True
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                connector=aiohttp.TCPConnector(ssl=ssl_context)
            )
            self.close = self._close_async

    def close(self):
        if self.session:
            self.session.close()

    async def _close_async(self):
        if self.session:
            await self.session.close()

    async def __aenter__(self):
        self._set_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_async()

    def _get_signature(self, method, data, timestamp):
        param_str = f"{timestamp}{self.api_key}{self.recv_window}"
        param_str += ujson.dumps(data) if method != 'get' else '&'.join([f'{k}={v}' for k, v in data.items()])
        return hmac.new(bytes(self.secret_key, "utf-8"), param_str.encode("utf-8"), hashlib.sha256).hexdigest()

    def _prepare_data(self, method, data, sign):
        new_data = {}
        for key, value in data.items():
            if value and key != 'self':
                if key.startswith('_'):
                    key = key[1:]
                if isinstance(value, list):
                    new_data[key] = ujson.dumps(value)
                else:
                    new_data[key] = str(value)
        if sign:
            timestamp = int(time.time() * 1000)
            self.headers['X-BAPI-SIGN'] = self._get_signature(method, new_data, timestamp)
            self.headers['X-BAPI-TIMESTAMP'] = str(timestamp)
        else:
            self.headers.pop('X-BAPI-SIGN', None)
            self.headers.pop('X-BAPI-TIMESTAMP', None)
        self.session.headers.update(self.headers)
        return new_data

    def request(self, method, url, data=None, sign=False):
        if self.asynced:
            return self._request_async_with_retry(method.lower(), url, data, sign)
        else:
            return self._request_sync_with_retry(method.lower(), url, data, sign)

    def _request_sync_with_retry(self, method, url, data, sign):
        for attempt in range(self.MAX_RETRIES):
            try:
                data = self._prepare_data(method, data, sign)
                kwargs = {'data': ujson.dumps(data)} if method != 'get' else {'params': data}
                with getattr(self.session, method)(self.base_url + url, **kwargs) as response:
                    return self._response(response.status_code, response.headers, response.text)
            except (ClientException, ServerException, requests.RequestException) as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(self.RETRY_DELAY)
        return None

    async def _request_async_with_retry(self, method, url, data, sign):
        for attempt in range(self.MAX_RETRIES):
            try:
                data = self._prepare_data(method, data, sign)
                kwargs = {'data': ujson.dumps(data)} if method != 'get' else {'params': data}
                async with getattr(self.session, method)(self.base_url + url, **kwargs) as response:
                    return self._response(response.status, dict(response.headers), await response.text())
            except (ClientException, ServerException, aiohttp.ClientError) as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(self.RETRY_DELAY)
        return None

    @staticmethod
    def _response(code, headers, text):
        if 400 <= code < 500:
            raise ClientException(code, text, headers)
        if code >= 500:
            raise ServerException(code, text)
        try:
            data = ujson.loads(text)
            if ret_code := data.get('retCode'):
                raise ClientException(code, text, headers, ret_code, data.get('retMsg'))
        except ujson.JSONDecodeError:
            data = text
        return data

    def instruments_info(self, category=None, symbol=""):
        return self.request('get', '/v5/market/instruments-info', locals())

    def account_info(self):
        return self.request('get', '/v5/account/info', locals(), sign=True)

    def position_info(self, category, symbol=None, baseCoin=None, settleCoin=None, limit=None, cursor=None):
        return self.request('get', '/v5/position/list', locals(), sign=True)

    def create_order(self, category, symbol, side, orderType, qty=None, timeInForce="", orderFilter="", price="", stopLoss="", takeProfit="", triggerDirection="", triggerPrice="", marketUnit="", orderQty=""):
        return self.request('post', '/v5/order/create', locals(), sign=True)

    def amend_order(self, category, symbol, orderId, stop_loss=None, price=""):
        return self.request('post', '/v5/order/amend', locals(), sign=True)

    def set_trading_stop(self, category, symbol, tpslMode="Full", takeProfit="", stopLoss="", tpOrderType="Market", slOrderType="", trailingStop="", slTriggerBy="", positionIdx=""):
        return self.request('post', "/v5/position/trading-stop", locals(), sign=True)

    def get_wallet_balance(self, accountType="UNIFIED", coin="USDT"):
        return self.request('get', '/v5/account/wallet-balance', locals(), sign=True)

    def cancel_order(self, symbol, orderId, category):
        return self.request("post", "/v5/order/cancel", locals(), sign=True)

    def get_closed_pnl(self, category, startTime, endTime, symbol=""):
        return self.request("get", "/v5/position/closed-pnl", locals(), sign=True)

    def get_open_orders(self, category, symbol):
        return self.request("get", "/v5/order/realtime", locals(), sign=True)

    def get_tickers(self, category):
        return self.request("get", "/v5/market/tickers", locals(), sign=False)

    def get_klines(self, symbol, interval, start="", end="", limit="", category="linear"):
        return self.request("get", "/v5/market/kline", locals(), sign=False)

    def set_leverage(self, symbol, buyLeverage="", sellLeverage="", category="linear"):
        return self.request("post", "/v5/position/set-leverage", locals(), sign=True)

    def close_all_orders(self, symbol="", category="linear"):
        return self.request("post", "/v5/order/cancel-all", locals(), sign=True)

    def get_mark_price_kline(self, symbol, interval, start="", end="", limit="", category="linear"):
        return self.request("get", "/v5/market/mark-price-kline", locals(), sign=False)

    def get_server_time(self):
        """Получить текущее серверное время Bybit"""
        return self.request("get", "/v5/market/time", {}, sign=False)