import websocket
import json
import threading
import logging
import requests
from time import sleep
import sqlite3
from datetime import datetime
from paths import get_trading_log_path, get_database_path

# Настройка логирования
logging.basicConfig(
    filename=get_trading_log_path(),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

DB_NAME = get_database_path()

def store_tweet(tweet_data):
    """Сохранение данных твита в SQLite."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        try:
            # Проверка обязательных полей
            if not all(key in tweet_data for key in ['id', 'text', 'created_at']):
                logger.warning(f"Неполные данные твита: {tweet_data}")
                return
            cursor.execute('''
                INSERT OR REPLACE INTO tweets (
                    id, timestamp, text, author_id, author_username,
                    retweet_count, like_count, reply_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tweet_data['id'],
                tweet_data['created_at'],
                tweet_data['text'],
                tweet_data.get('author_id', 'Unknown'),
                tweet_data.get('author_username', 'Unknown'),
                tweet_data.get('retweet_count', 0),
                tweet_data.get('like_count', 0),
                tweet_data.get('reply_count', 0)
            ))
            conn.commit()
            logger.debug(f"Сохранён твит: {tweet_data['id']} от {tweet_data.get('author_username', 'Unknown')}")
        except sqlite3.IntegrityError:
            logger.warning(f"Дубликат твита ID {tweet_data['id']} проигнорирован")
        except Exception as e:
            logger.error(f"Ошибка при сохранении твита: {str(e)}")

class RealTimeTwitterAPIParser:
    def __init__(self, api_key: str, accounts: list, rule_id: str = None):
        if not accounts or len(accounts) > 50:
            accounts = accounts[:50] if len(accounts) > 50 else accounts
            logger.warning("Лимит 50 аккаунтов, использованы первые 50")
        self.api_key = api_key
        self.accounts = accounts
        self.rule_id = rule_id
        self.tweet_count = 0
        self.monthly_limit = 1500
        self.ws = None
        self.stream_thread = None
        logger.info(f"Парсер инициализирован для {len(self.accounts)} аккаунтов")

    def create_filter_rule(self, retries=3, backoff=3):
        """Создание правила фильтрации через REST API с повторными попытками."""
        if not self.api_key.isascii():
            logger.error("API-ключ содержит не-ASCII символы: %s", self.api_key)
            return None
        if any(not username.isascii() for username in self.accounts):
            logger.error("Обнаружены не-ASCII символы в именах аккаунтов: %s", self.accounts)
            return None

        url = "https://api.twitterapi.io/oapi/tweet_filter/add_rule"
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "tag": "twitter_accounts_filter",
            "value": " OR ".join([f"from:{username}" for username in self.accounts]),
            "interval_seconds": 100
        }

        if len(payload["value"]) > 255:
            logger.error("Фильтр превышает лимит в 255 символов: %s", payload["value"])
            return None

        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    self.rule_id = data.get("rule_id")
                    logger.info(f"Правило создано с ID: {self.rule_id}")
                    return self.rule_id
                else:
                    logger.error(f"Ошибка создания правила: %s, Ответ сервера: %s", data.get("msg"), response.text)
                    return None
            except requests.RequestException as e:
                logger.error(f"Попытка {attempt + 1}/{retries} не удалась: {str(e)}")
                if attempt < retries - 1:
                    sleep(backoff * (2 ** attempt))  # Экспоненциальная задержка
                else:
                    logger.error("Не удалось создать правило после всех попыток")
                    return None

    def on_message(self, ws, message):
        """Обработка входящих твитов."""
        try:
            if self.tweet_count >= self.monthly_limit:
                logger.warning("Лимит 1,500 твитов достигнут, остановка")
                self.stop_real_time_parsing()
                return
            data = json.loads(message)
            event_type = data.get("event_type")

            if event_type == "connected":
                logger.info("Соединение успешно установлено")
                return
            if event_type == "ping":
                logger.debug("Получен ping от сервера")
                return
            if event_type == "tweet":
                tweets = data.get("tweets", [])
                for tweet in tweets:
                    if self.tweet_count >= self.monthly_limit:
                        logger.warning("Лимит 1,500 твитов достигнут, остановка")
                        self.stop_real_time_parsing()
                        return
                    tweet_data = {
                        "id": tweet.get("id"),
                        "text": tweet.get("text", ""),
                        "author_id": tweet.get("author", {}).get("id"),
                        "author_username": tweet.get("author", {}).get("username", "Unknown"),
                        "created_at": tweet.get("created_at", ""),
                        "retweet_count": tweet.get("retweet_count", 0),
                        "like_count": tweet.get("like_count", 0),
                        "reply_count": tweet.get("reply_count", 0)
                    }
                    store_tweet(tweet_data)
                    self.tweet_count += 1
                    logger.debug(
                        f"Получен твит от @{tweet_data['author_username']}: {tweet_data['text']}")
        except json.JSONDecodeError:
            logger.error("Ошибка декодирования JSON")
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {str(e)}")

    def on_error(self, ws, error):
        """Обработка ошибок."""
        logger.error(f"Ошибка WebSocket: {str(error)}")
        if isinstance(error, websocket.WebSocketBadStatusException):
            logger.error(f"Сервер вернул ошибку: {str(error)}. Проверьте API-ключ и URL.")
        elif isinstance(error, websocket.WebSocketTimeoutException):
            logger.error("Таймаут соединения. Проверьте сеть или сервер.")
        elif isinstance(error, ConnectionRefusedError):
            logger.error("Соединение отклонено. Проверьте адрес сервера.")

    def on_close(self, ws, close_status_code, close_msg):
        """Обработка закрытия соединения."""
        logger.info(f"Соединение закрыто: код={close_status_code}, сообщение={close_msg}")
        if close_status_code == 404:
            logger.error("WebSocket эндпоинт не найден. Проверьте URL или конфигурацию API.")
            return
        if close_status_code == 401:
            logger.error("Ошибка аутентификации. Проверьте API-ключ.")
            return
        logger.info("Переподключение через 90 секунд...")
        sleep(90)
        self.start_real_time_parsing()

    def on_open(self, ws):
        """Обработка открытия соединения."""
        logger.info("Соединение открыто")
        if self.rule_id:
            filter_data = {"rule_id": self.rule_id}
            logger.info(f"Отправка rule_id: {self.rule_id}")
            ws.send(json.dumps(filter_data))
        else:
            logger.error("ID правила не задано, стриминг невозможен")

    def start_real_time_parsing(self):
        """Запуск WebSocket клиента для стриминга."""
        if hasattr(self, 'stream_thread') and self.stream_thread is not None and self.stream_thread.is_alive():
            logger.warning("Стриминг уже активен")
            return
        if not self.rule_id:
            rule_id = self.create_filter_rule()
            if not rule_id:
                logger.error("Не удалось создать правило, стриминг невозможен")
                return
        headers = {
            "X-API-Key": self.api_key,
            "Connection": "Upgrade",
            "Upgrade": "websocket",
            "Sec-WebSocket-Version": "13"
        }
        self.ws = websocket.WebSocketApp(
            "wss://ws.twitterapi.io/twitter/tweet/websocket",
            header=headers,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        self.stream_thread = threading.Thread(target=self.ws.run_forever,
                                             kwargs={"ping_interval": 40, "ping_timeout": 30})
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info("Стриминг запущен")

    def stop_real_time_parsing(self):
        """Остановка стриминга."""
        if hasattr(self, 'ws') and self.ws:
            self.ws.close()
            if hasattr(self, 'stream_thread') and self.stream_thread is not None:
                self.stream_thread.join()
            logger.info("Стриминг остановлен")