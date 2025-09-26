import warnings_config  # Отключаем предупреждения
import json
import logging
import asyncio
from datetime import datetime
import sqlite3
import requests
import websockets
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import os
import warnings
from paths import get_trading_log_path, get_database_path, get_sentiment_csv_path

# Configure logging
logging.basicConfig(
    filename=get_trading_log_path(),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Correct database path
DB_NAME = get_database_path()

# Initialize FinBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device} {'(' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Crypto keywords for filtering
crypto_keywords = {
    "BTC": [r"\bbitcoin\b", r"\bbtc\b", r"\b#bitcoin\b", r"\b#btc\b"],
    "ETH": [r"\bethereum\b", r"\beth\b", r"\b#ethereum\b", r"\b#eth\b"],
    "SOL": [r"\bsolana\b", r"\bsol\b", r"\b#solana\b", r"\b#sol\b"]
}


def get_relevant_cryptos(text):
    """Identify relevant cryptocurrencies in tweet text."""
    relevant_cryptos = []
    text_lower = text.lower()
    for crypto, keywords in crypto_keywords.items():
        for keyword in keywords:
            if re.search(keyword, text_lower):
                relevant_cryptos.append(crypto)
                logger.debug(f"Tweet matched {crypto} with keyword '{keyword}': {text[:50]}...")
                break
    return relevant_cryptos if relevant_cryptos else ["general"]


def get_sentiment(text):
    """Compute sentiment score and probabilities using FinBERT."""
    logger.debug(f"Computing sentiment for tweet: {text[:50]}...")
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        probs = probs[0].cpu().tolist()
        positive, negative, neutral = probs[0], probs[1], probs[2]
        denominator = positive + negative + neutral
        sentiment_score = (positive - negative) / denominator if denominator > 0 else 0
        logger.debug(
            f"Sentiment computed: score={sentiment_score:.4f}, positive={positive:.4f}, negative={negative:.4f}, neutral={neutral:.4f}")
        return sentiment_score, probs
    except Exception as e:
        logger.error(f"Error computing sentiment: {text[:50]}... - {e}")
        return 0.0, [0.0, 0.0, 0.0]


def init_tweets_table():
    """Create tweets table matching trading_bot.py schema."""
    try:
        db_dir = os.path.dirname(DB_NAME)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Created directory: {db_dir}")

        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tweets (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    text TEXT,
                    author_id TEXT,
                    author_username TEXT,
                    retweet_count INTEGER,
                    like_count INTEGER,
                    reply_count INTEGER,
                    sentiment_score REAL,
                    probs TEXT,
                    crypto TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets (timestamp, crypto)')
            conn.commit()
            logger.info(f"Initialized tweets table in {DB_NAME}")
    except Exception as e:
        logger.error(f"Error initializing tweets table: {str(e)}", exc_info=True)


def load_tweets_from_csv():
    """Load tweets from sentiment_BTC.csv into tweets table, avoiding duplicates."""
    try:
        csv_path = get_sentiment_csv_path('BTC')
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}, creating sample CSV")
            sample_data = {
                'id': ['12345', '12346'],
                'created_at': ['2025-09-07 12:00:00', '2025-09-07 12:01:00'],
                'text': ['Bitcoin is pumping! #BTC', 'BTC looks bearish today'],
                'sentiment_score': [0.5, -0.3],
                'username': ['woonomic', 'CoinDesk'],
                'probs': [json.dumps([0.6, 0.1, 0.3]), json.dumps([0.2, 0.5, 0.3])]
            }
            pd.DataFrame(sample_data).to_csv(csv_path, index=False)
            logger.info(f"Created sample CSV at {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"CSV file is empty: {csv_path}")
            return 0

        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            inserted = 0
            for _, row in df.iterrows():
                tweet_id = str(row['id'])
                timestamp = str(row['created_at'])
                text = str(row['text'])
                sentiment_score = float(row['sentiment_score']) if 'sentiment_score' in row else 0.0
                probs = row['probs'] if 'probs' in row and pd.notna(row['probs']) else json.dumps([0.0, 0.0, 0.0])
                username = str(row['username']) if 'username' in row else 'Unknown'

                relevant_cryptos = get_relevant_cryptos(text)
                for crypto in relevant_cryptos:
                    unique_id = f"{tweet_id}_{crypto}"
                    cursor.execute('SELECT COUNT(*) FROM tweets WHERE id = ?', (unique_id,))
                    if cursor.fetchone()[0] == 0:
                        cursor.execute('''
                            INSERT INTO tweets (
                                id, timestamp, text, author_id, author_username,
                                retweet_count, like_count, reply_count, sentiment_score, probs, crypto
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            unique_id,
                            timestamp,
                            text,
                            'Unknown',
                            username,
                            0,
                            0,
                            0,
                            sentiment_score,
                            probs,
                            crypto
                        ))
                        inserted += 1
            conn.commit()
            logger.info(f"Loaded {inserted} tweets from {csv_path} into {DB_NAME}")
            return inserted
    except Exception as e:
        logger.error(f"Error loading tweets from CSV: {str(e)}", exc_info=True)
        return 0


def store_tweet(tweet_data):
    """Save tweet data to SQLite with sentiment score and probs."""
    logger.debug(f"Tweet received: id={tweet_data.get('id')}, text={tweet_data.get('text', '')[:50]}...")
    try:
        if not all(key in tweet_data for key in ['id', 'text', 'created_at']):
            logger.warning(f"Incomplete tweet data: {tweet_data}")
            return

        relevant_cryptos = get_relevant_cryptos(tweet_data['text'])
        if not relevant_cryptos:
            logger.debug(f"Skipping tweet not relevant to crypto: {tweet_data['text'][:50]}...")
            return

        try:
            timestamp = datetime.utcfromtimestamp(int(tweet_data['created_at']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            try:
                timestamp = datetime.strptime(tweet_data['created_at'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                #logger.error(f"Invalid timestamp format: {tweet_data['created_at']}")
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        sentiment_score, probs = get_sentiment(tweet_data['text'])

        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            for crypto in relevant_cryptos:
                unique_id = f"{tweet_data['id']}_{crypto}"
                cursor.execute('''
                    INSERT OR REPLACE INTO tweets (
                        id, timestamp, text, author_id, author_username,
                        retweet_count, like_count, reply_count, sentiment_score, probs, crypto
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    unique_id,
                    timestamp,
                    tweet_data['text'],
                    tweet_data.get('author_id', 'Unknown'),
                    tweet_data.get('author_username', 'Unknown'),
                    tweet_data.get('retweet_count', 0),
                    tweet_data.get('like_count', 0),
                    tweet_data.get('reply_count', 0),
                    sentiment_score,
                    json.dumps(probs),
                    crypto
                ))
            conn.commit()
            logger.info(
                f"Tweet saved: id={tweet_data['id']} ({','.join(relevant_cryptos)}), "
                f"author=@{tweet_data.get('author_username', 'Unknown')}, sentiment={sentiment_score:.4f}"
            )

    except Exception as e:
        logger.error(f"Error saving tweet: {str(e)}", exc_info=True)


def get_rolling_sentiment(crypto, window_hours=48):
    """Calculate the rolling sentiment score for a cryptocurrency over the last window_hours."""
    logger.debug(f"Calculating rolling sentiment for {crypto} over {window_hours} hours")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            window_seconds = window_hours * 3600
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            query = '''
                SELECT timestamp, sentiment_score
                FROM tweets
                WHERE crypto = ? 
                  AND timestamp <= ? 
                  AND timestamp >= datetime(?, ?)
                ORDER BY timestamp DESC
            '''
            cursor.execute(query, (crypto, current_time, current_time, f'-{window_seconds} seconds'))
            tweets = cursor.fetchall()

            if not tweets:
                logger.warning(f"No tweets found for {crypto} in the last {window_hours} hours")
                return 0.0

            total_weight, weighted_sentiment = 0.0, 0.0
            for timestamp, sentiment_score in tweets:
                tweet_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                time_diff = (datetime.now() - tweet_time).total_seconds()
                weight = 1.0 - (time_diff / window_seconds) if time_diff <= window_seconds else 0.0
                weighted_sentiment += sentiment_score * weight
                total_weight += weight

            rolling_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
            logger.info(f"Rolling sentiment for {crypto} over {window_hours}h: {rolling_sentiment:.4f} "
                        f"(tweets={len(tweets)})")

            cursor.execute('''
                INSERT OR REPLACE INTO rolling_sentiment (timestamp, crypto, sentiment_score)
                VALUES (?, ?, ?)
            ''', (current_time, crypto, rolling_sentiment))
            conn.commit()

            return rolling_sentiment

    except Exception as e:
        logger.error(f"Error calculating rolling sentiment for {crypto}: {str(e)}", exc_info=True)
        return 0.0

class RealTimeTwitterAPIParser:
    def __init__(self, api_key: str, accounts: list, monthly_limit=1500):
        if not accounts or len(accounts) > 50:
            accounts = accounts[:50] if len(accounts) > 50 else accounts
            logger.warning("Лимит 50 аккаунтов, использованы первые 50")
        self.api_key = api_key
        self.accounts = accounts
        self.tweet_count = 0
        self.monthly_limit = monthly_limit
        self.should_reconnect = True
        self.ws = None
        self.rule_id = None
        init_tweets_table()  # Initialize table on startup
        logger.info(f"Parser initialized for {len(self.accounts)} accounts")

    async def create_filter_rule(self, retries=3, backoff=3):
        """Create a filter rule via REST API with retries."""
        url = "https://api.twitterapi.io/oapi/tweet_filter/add_rule"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "TradingBot/1.0"
        }
        payload = {
            "tag": "twitter_accounts_filter",
            "value": " OR ".join([f"from:{username}" for username in self.accounts]),
            "interval_seconds": 100
        }
        if len(payload["value"]) > 255:
            logger.error("Filter exceeds 255 characters limit: %s", payload["value"])
            return None

        for attempt in range(retries):
            try:
                with requests.Session() as session:
                    response = await asyncio.to_thread(
                        lambda: session.post(url, headers=headers, json=payload, timeout=10)
                    )
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    rule_id = data.get("rule_id")
                    logger.info(f"Rule created with ID: {rule_id}")
                    activated = await self.change_state_of_filter_rule(1, rule_id)
                    if activated:
                        return rule_id
                    else:
                        return None
                else:
                    logger.error(f"Error creating rule: {data.get('msg')}, Response: {response.text}")
                    return None
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1}/{retries} failed: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                else:
                    logger.error("Failed to create rule after all attempts")
                    return None
        return None

    async def change_state_of_filter_rule(self, state, rule_id, retries=3, backoff=3):
        """Activate an existing filter rule by setting is_effect=1."""
        url = "https://api.twitterapi.io/oapi/tweet_filter/update_rule"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "TradingBot/1.0"
        }
        payload = {
            "rule_id": rule_id,
            "tag": "twitter_accounts_filter",
            "value": " OR ".join([f"from:{username}" for username in self.accounts]),
            "interval_seconds": 100,
            "is_effect": state
        }

        for attempt in range(retries):
            try:
                with requests.Session() as session:
                    response = await asyncio.to_thread(
                        lambda: session.post(url, headers=headers, json=payload, timeout=10)
                    )
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    logger.info(f"Rule {rule_id} activated successfully")
                    return True
                else:
                    logger.error(f"Error activating rule: {data.get('msg')}, Response: {response.text}")
                    return False
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1}/{retries} failed to activate rule: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                else:
                    return False
        return False

    async def on_message(self, ws, message):
        """Handle incoming tweet messages."""
        try:
            data = json.loads(message)
            event_type = data.get("event_type")

            # Игнорируем ping/pong и connected события
            if event_type in ("ping", "pong", "connected"):
                return

            # Обработка твитов
            if event_type == "tweet":
                tweets = data.get("tweets", [])
                for tweet in tweets:
                    if self.tweet_count >= self.monthly_limit:
                        logger.warning("Monthly limit reached during processing, stopping")
                        self.should_reconnect = False
                        await self.stop_real_time_parsing()
                        return

                    tweet_data = {
                        "id": tweet.get("id"),
                        "text": tweet.get("text", ""),
                        "author_id": tweet.get("author", {}).get("id", "Unknown"),
                        "author_username": tweet.get("author", {}).get("username", "Unknown"),
                        "created_at": tweet.get("created_at", ""),
                        "retweet_count": tweet.get("retweet_count", 0),
                        "like_count": tweet.get("like_count", 0),
                        "reply_count": tweet.get("reply_count", 0)
                    }

                    store_tweet(tweet_data)
                    self.tweet_count += 1
                    logger.info(
                        f"Tweet received and processed: id={tweet_data['id']}, "
                        f"author=@{tweet_data['author_username']}, total_count={self.tweet_count}"
                    )

        except json.JSONDecodeError:
            logger.error("Error decoding JSON message")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)

    async def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {str(error)}")
        if isinstance(error, websockets.exceptions.ConnectionClosed):
            logger.error(f"Connection closed: {str(error)}")
        elif isinstance(error, websockets.exceptions.InvalidStatusCode):
            logger.error(f"Invalid status code: {str(error)}. Check API key or URL.")
            if "403" in str(error):
                logger.error("403 Forbidden. Check API key or Cloudflare restrictions.")
                self.should_reconnect = False
        elif isinstance(error, asyncio.TimeoutError):
            logger.error("Connection timeout. Check network or server.")

    async def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure."""
        logger.info(f"WebSocket closed: code={close_status_code}, message={close_msg}")
        if close_status_code in (401, 403):
            logger.error(f"Authentication or permission error: {close_status_code}")
            self.should_reconnect = False
        elif self.should_reconnect:
            logger.info("Reconnecting in 30 seconds...")
            await asyncio.sleep(30)
            await self.start_real_time_parsing()

    async def on_open(self, ws):
        """Handle WebSocket opening."""
        logger.info("WebSocket connection opened")
        logger.info("Waiting for server to send data...")

    async def start_real_time_parsing(self):
        """Start the WebSocket client for streaming."""
        # Проверяем активное соединение
        if self.ws:
            try:
                if self.ws.open:
                    logger.warning("Streaming already active")
                    return
            except AttributeError:
                logger.warning("Streaming already active (ws.open не найден)")
                return

        if not hasattr(self, 'rule_id') or not self.rule_id:
            self.rule_id = await self.create_filter_rule()
            if not self.rule_id:
                logger.error("Failed to create filter rule, streaming not possible")
                self.should_reconnect = False
                return

        headers = [
            ("x-api-key", self.api_key),
            ("User-Agent", "TradingBot/1.0")
        ]
        logger.info(f"Starting WebSocket with headers: {headers}")

        while self.should_reconnect:
            try:
                self.ws = await websockets.connect(
                    "wss://ws.twitterapi.io/twitter/tweet/websocket",
                    additional_headers=headers,
                    ping_interval=20,  # каждые 20 секунд отправляем ping
                    ping_timeout=10  # ждём pong максимум 10 секунд
                )
                logger.info("WebSocket connection established")
                await self.on_open(self.ws)

                while self.should_reconnect:
                    try:
                        message = await self.ws.recv()
                        await self.on_message(self.ws, message)
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.warning(f"WebSocket closed unexpectedly: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error receiving message: {e}", exc_info=True)
                        await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error starting WebSocket: {str(e)}", exc_info=True)
                await asyncio.sleep(30)

            finally:
                if self.ws:
                    try:
                        if getattr(self.ws, 'open', False):
                            await self.ws.close()
                            logger.info("WebSocket closed in finally block")
                    except Exception as e:
                        logger.error(f"Error closing WebSocket: {e}", exc_info=True)

                if self.should_reconnect:
                    logger.info("Reconnecting in 30 seconds...")
                    await asyncio.sleep(30)

    async def stop_real_time_parsing(self):
        """Stop the WebSocket client."""
        try:
            if self.ws:
                try:
                    if getattr(self.ws, 'open', False):
                        await self.ws.close()
                        if self.rule_id:
                            await self.change_state_of_filter_rule(0, self.rule_id)
                        logger.info("WebSocket client closed")
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}", exc_info=True)
            self.should_reconnect = False
            logger.info("Twitter parsing stopped")
        except Exception as e:
            logger.error(f"Error stopping real-time parsing: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Example usage
    api_key = ""
    accounts = ["woonomic", "WuBlockchain", "CoinDesk", "Cointelegraph"]
    parser = RealTimeTwitterAPIParser(api_key, accounts)
    asyncio.run(parser.start_real_time_parsing())