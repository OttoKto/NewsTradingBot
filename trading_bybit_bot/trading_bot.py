import warnings_config  # Отключаем предупреждения
import pandas as pd
import numpy as np
import datetime
import asyncio
import jax
import jax.numpy as jnp
import logging
import sqlite3
import json
import warnings
from bybit_api.client import Client
from indicators.smart_signals_index4 import calculate_indicators, add_signals, calculate_index, optimize_weights_jax, optimize_strategy_two_stage, get_binance_data_by_requests, run_strategy_tp_ATR
from tweet_analyzer.real_time_analyzer import init_tweets_table, load_tweets_from_csv, get_rolling_sentiment, RealTimeTwitterAPIParser
from settings import capital, symbol, twitter_accounts
from config import bybit_key, bybit_secret, twitter_key
from paths import get_trading_log_path, get_database_path

jax.config.update('jax_platform_name', 'cpu')

logging.basicConfig(
    filename=get_trading_log_path(),
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

DB_NAME = get_database_path()

def init_db():
    """Инициализация SQLite базы данных."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimized_parameters (
                timestamp TEXT PRIMARY KEY,
                weights_0_2 TEXT,
                weights_0_5 TEXT,
                take_profit_atr REAL,
                stop_loss_atr REAL,
                ind_entry_0_2 REAL,
                ind_entry_0_5 REAL,
                leverage REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                side TEXT,
                quantity REAL,
                take_profit REAL,
                stop_loss REAL,
                pnl REAL,
                reason TEXT,
                capital_after REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TEXT PRIMARY KEY,
                total_trades INTEGER,
                win_trades INTEGER,
                win_rate REAL,
                current_capital REAL
            )
        ''')
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rolling_sentiment (
                timestamp TEXT,
                crypto TEXT,
                sentiment_score REAL,
                PRIMARY KEY (timestamp, crypto)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets (timestamp, crypto)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rolling_sentiment ON rolling_sentiment (timestamp, crypto)')
        conn.commit()

def store_trade(entry_time, exit_time, entry_price, exit_price, side, quantity, take_profit, stop_loss, pnl, reason, capital_after):
    """Сохранение деталей сделки."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (
                entry_time, exit_time, entry_price, exit_price, side, quantity,
                take_profit, stop_loss, pnl, reason, capital_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry_time.isoformat() if entry_time else None,
            exit_time.isoformat() if exit_time else None,
            entry_price, exit_price, side, quantity,
            take_profit, stop_loss, pnl, reason, capital_after
        ))
        conn.commit()
        logger.info(f"Сохранена сделка: {side}, вход={entry_price}, выход={exit_price}, PNL={pnl}")

def store_performance_metrics(timestamp, total_trades, win_trades, win_rate, current_capital):
    """Сохранение метрик производительности."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO performance_metrics (
                timestamp, total_trades, win_trades, win_rate, current_capital
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(), total_trades, win_trades, win_rate, current_capital
        ))
        conn.commit()
        logger.info(f"Сохранены метрики: сделок={total_trades}, винрейт={win_rate:.2f}%")

async def twitter_parsing_loop(api_key, accounts):
    """Асинхронная обёртка для парсера Twitter."""
    parser = RealTimeTwitterAPIParser(api_key, accounts)
    try:
        while True:
            await parser.start_real_time_parsing()  # Await the async method
            await asyncio.sleep(3600)  # Check periodically
            if not parser.should_reconnect:  # Check if parser intends to stop
                logger.warning("Twitter parsing stopped, attempting restart")
                parser = RealTimeTwitterAPIParser(api_key, accounts)  # Reinitialize
    except asyncio.CancelledError:
        logger.info("Twitter parsing loop cancelled")
        await parser.stop_real_time_parsing()
    except Exception as e:
        logger.error(f"Ошибка в twitter_parsing_loop: {str(e)}", exc_info=True)
        await asyncio.sleep(3600)
        await parser.start_real_time_parsing()  # Retry on error
    finally:
        await parser.stop_real_time_parsing()
        logger.info("Цикл парсинга Twitter завершён")

async def optimization_loop(api_key, secret_key, symbol, testnet=True, demo=False):
    client = Client(api_key, secret_key, testnet=testnet, demo=demo, asynced=True)

    interval = "1h"
    settings = {
        'RSI_window': 14, 'CCI_window': 20, 'MFI_window': 14, 'ADX_window': 14, 'ATR_window': 14,
        'CMO_window': 14, 'Bollinger_window': 20, 'Bollinger_window_dev': 2, 'LSMA_window': 25,
        'LSMA_offset': 0, 'EMA_window': 9, 'MACD_window_slow': 26, 'MACD_window_fast': 12,
        'MACD_window_sign': 9, 'Keltner_window': 20, 'Keltner_multiplier': 2, 'HMA_window': 9,
        'Supertrend_length': 10, 'Supertrend_multiplier': 3, 'Chaikin_ema_short_window': 3,
        'Chaikin_ema_long_window': 10
    }
    signal_columns = [
        "RSI_Signal", "RSI_Breakout", "CCI_Signal", "CCI_Breakout", "MFI_Signal", "MFI_Breakout",
        "CMO_Signal", "CMO_Breakout", "LSMA_Signal", "LSMA_Breakout", "EMA_Signal", "EMA_Breakout",
        "MACD_Signal", "MACD_Breakout", "HMA_Signal", "BOP_Trade_Signal", "BOP_Breakout",
        "Supertrend_Signal", "Supertrend_Breakout", "Chaikin_Signal", "Chaikin_Breakout",
        "Stoch_Signal", "OBV_Signal", "VWAP_Signal", "Parabolic_SAR_Signal", "Ichimoku_Signal"
    ]

    try:
        while True:
            current_time = datetime.datetime.now()
            logger.info("Начало месячной оптимизации")

            try:
                end = current_time.strftime('%Y-%m-%d %H:%M:%S')
                start = (current_time - datetime.timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')

                logger.debug(f"Fetching data from {start} to {end}")
                for attempt in range(3):
                    try:
                        historical_df = get_binance_data_by_requests(
                            ticker=symbol,
                            interval=interval,
                            start=start,
                            end=end
                        )
                        logger.info(f"Получено {len(historical_df)} записей исторических данных с Binance")
                        break
                    except Exception as e:
                        logger.error(f"Попытка {attempt + 1}/3 получения исторических данных не удалась: {str(e)}")
                        if attempt < 2:
                            await asyncio.sleep(3 * (2 ** attempt))
                        else:
                            logger.error("Не удалось получить исторические данные")
                            await asyncio.sleep(3600)
                            continue

                if historical_df.empty:
                    logger.error("Получен пустой DataFrame с историческими данными")
                    await asyncio.sleep(3600)
                    continue

                required_columns = ['open', 'high', 'low', 'close', 'volume', 'qav']
                if not all(col in historical_df.columns for col in required_columns):
                    logger.error(f"Отсутствуют требуемые столбцы в данных: {historical_df.columns}")
                    await asyncio.sleep(3600)
                    continue
                if historical_df.index.duplicated().any():
                    logger.warning("Обнаружены дубликаты временных меток, удаление...")
                    historical_df = historical_df[~historical_df.index.duplicated(keep='last')]
                if historical_df.isna().any().any():
                    logger.warning("Обнаружены пропущенные значения, заполнение нулями...")
                    historical_df = historical_df.fillna(0)

                historical_df = historical_df.rename(columns={'qav': 'turnover'})
                historical_df = historical_df[['open', 'high', 'low', 'close', 'volume', 'turnover']]
                historical_df = historical_df.astype(float)

                logger.debug("Вычисление индикаторов...")
                historical_df = calculate_indicators(historical_df, settings)
                logger.debug("Добавление сигналов...")
                historical_df = add_signals(historical_df)
                historical_df['diff_curve'] = historical_df['close'].pct_change() * 100
                historical_df['rolling_sentiment_score'] = 0  # Placeholder; could integrate real sentiment if needed

                logger.debug("Оптимизация весов JAX...")
                signals = jnp.array(historical_df[signal_columns].fillna(0).values)
                close = jnp.array(historical_df['close'].values)
                weights_0_2, weights_0_5 = optimize_weights_jax(signals, close)
                logger.debug(f"Веса 0.2: {weights_0_2}")
                logger.debug(f"Веса 0.5: {weights_0_5}")

                logger.debug("Вычисление индексов...")
                index_values_0_2 = calculate_index(weights_0_2, signals)
                historical_df['Indicator_Index_0_2'] = np.array(index_values_0_2)
                index_values_0_5 = calculate_index(weights_0_5, signals)
                historical_df['Indicator_Index_0_5'] = np.array(index_values_0_5)

                logger.debug("Оптимизация стратегии (двухэтапная)...")
                best_params, best_capital, best_df = optimize_strategy_two_stage(historical_df)
                logger.info(f"Оптимизированные параметры: {best_params}")
                store_optimized_parameters(current_time, weights_0_2, weights_0_5, best_params)

            except Exception as e:
                logger.error(f"Оптимизация не удалась: {str(e)}", exc_info=True)

            next_month = (current_time + datetime.timedelta(days=30)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            sleep_seconds = (next_month - current_time).total_seconds()
            logger.debug(f"Ожидание следующей оптимизации: {sleep_seconds} секунд")
            await asyncio.sleep(sleep_seconds)

    except asyncio.CancelledError:
        logger.info("Optimization loop cancelled")
    except KeyboardInterrupt:
        logger.info("Optimization loop interrupted")
    except Exception as e:
        logger.error(f"Ошибка в optimization_loop: {str(e)}", exc_info=True)
    finally:
        await client.close()
        logger.info("Цикл оптимизации завершён")

def store_optimized_parameters(timestamp, weights_0_2, weights_0_5, params):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimized_parameters (
                timestamp TEXT,
                weights_0_2 TEXT,
                weights_0_5 TEXT,
                take_profit_atr REAL,
                stop_loss_atr REAL,
                ind_entry_0_2 REAL,
                ind_entry_0_5 REAL,
                leverage REAL
            )
        ''')
        take_profit, stop_loss, ind_entry_0_2, ind_entry_0_5, leverage = [
            float(x) if isinstance(x, (int, float, np.floating, np.integer)) else float(x[0]) for x in params
        ]
        weights_0_2_list = weights_0_2.tolist() if hasattr(weights_0_2, 'tolist') else list(weights_0_2)
        weights_0_5_list = weights_0_5.tolist() if hasattr(weights_0_5, 'tolist') else list(weights_0_5)
        cursor.execute('''
            INSERT INTO optimized_parameters (timestamp, weights_0_2, weights_0_5, take_profit_atr, stop_loss_atr, ind_entry_0_2, ind_entry_0_5, leverage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(),
            json.dumps(weights_0_2_list),
            json.dumps(weights_0_5_list),
            take_profit,
            stop_loss,
            ind_entry_0_2,
            ind_entry_0_5,
            leverage
        ))
        conn.commit()
        logger.info(f"[{timestamp}] Сохранены оптимизированные параметры: tp={take_profit:.2f}, sl={stop_loss:.2f}, e2={ind_entry_0_2:.2f}, e5={ind_entry_0_5:.2f}, leverage={leverage:.2f}")
    except Exception as e:
        logger.error(f"[{timestamp}] Ошибка сохранения параметров: {str(e)}", exc_info=True)
    finally:
        conn.close()

def load_latest_parameters():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimized_parameters (
                timestamp TEXT,
                weights_0_2 TEXT,
                weights_0_5 TEXT,
                take_profit_atr REAL,
                stop_loss_atr REAL,
                ind_entry_0_2 REAL,
                ind_entry_0_5 REAL,
                leverage REAL
            )
        ''')
        cursor.execute('''
            SELECT weights_0_2, weights_0_5, take_profit_atr, stop_loss_atr, ind_entry_0_2, ind_entry_0_5, leverage
            FROM optimized_parameters
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        row = cursor.fetchone()
        if row:
            try:
                weights_0_2 = jnp.array(json.loads(row[0]))
                weights_0_5 = jnp.array(json.loads(row[1]))
                params = (
                    float(row[2]) if row[2] is not None else 12.0,
                    float(row[3]) if row[3] is not None else 3.0,
                    float(row[4]) if row[4] is not None else 0.4,
                    float(row[5]) if row[5] is not None else 0.9,
                    float(row[6]) if row[6] is not None else 2.0
                )
                logger.info(f"Loaded parameters: tp={params[0]:.2f}, sl={params[1]:.2f}, e2={params[2]:.2f}, e5={params[3]:.2f}, leverage={params[4]:.2f}")
                return weights_0_2, weights_0_5, params
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing parameters: {str(e)}", exc_info=True)
                return None, None, None
        else:
            logger.error("No parameters found in database, using defaults")
            return None, None, (12.0, 3.0, 0.4, 0.9, 2.0)
    except Exception as e:
        logger.error(f"Ошибка загрузки параметров: {str(e)}", exc_info=True)
        return None, None, None
    finally:
        conn.close()

import math

async def trading_loop(api_key, secret_key, symbol, capital, testnet=True, demo=False):
    client = Client(api_key, secret_key, testnet=testnet, demo=demo, asynced=True)
    commission_rate = 0.001
    interval = 60
    settings = {
        'RSI_window': 14, 'CCI_window': 20, 'MFI_window': 14, 'ADX_window': 14, 'ATR_window': 14,
        'CMO_window': 14, 'Bollinger_window': 20, 'Bollinger_window_dev': 2, 'LSMA_window': 25,
        'LSMA_offset': 0, 'EMA_window': 9, 'MACD_window_slow': 26, 'MACD_window_fast': 12,
        'MACD_window_sign': 9, 'Keltner_window': 20, 'Keltner_multiplier': 2, 'HMA_window': 9,
        'Supertrend_length': 10, 'Supertrend_multiplier': 3, 'Chaikin_ema_short_window': 3,
        'Chaikin_ema_long_window': 10
    }
    signal_columns = [
        "RSI_Signal", "RSI_Breakout", "CCI_Signal", "CCI_Breakout", "MFI_Signal", "MFI_Breakout",
        "CMO_Signal", "CMO_Breakout", "LSMA_Signal", "LSMA_Breakout", "EMA_Signal", "EMA_Breakout",
        "MACD_Signal", "MACD_Breakout", "HMA_Signal", "BOP_Trade_Signal", "BOP_Breakout",
        "Supertrend_Signal", "Supertrend_Breakout", "Chaikin_Signal", "Chaikin_Breakout",
        "Stoch_Signal", "OBV_Signal", "VWAP_Signal", "Parabolic_SAR_Signal", "Ichimoku_Signal"
    ]
    position = None

    try:
        # Initialize tweets table and load existing tweets from CSV
        init_tweets_table()
        tweets_loaded = load_tweets_from_csv()
        if tweets_loaded == 0:
            logger.warning("No tweets loaded from CSV, sentiment will default to 0.0")

        # Fetch instrument info for precision
        instrument_info = await client.instruments_info(category="linear", symbol=symbol)
        instrument = instrument_info['result']['list'][0]
        min_qty = float(instrument['lotSizeFilter']['minOrderQty'])
        qty_step = float(instrument['lotSizeFilter']['qtyStep'])
        tick_size = float(instrument['priceFilter']['tickSize'])
        logger.info(
            f"[{datetime.datetime.now()}] Instrument info: min_qty={min_qty}, qty_step={qty_step}, tick_size={tick_size}")

        # Check for existing position
        position_response = await client.position_info(category="linear", symbol=symbol)
        logger.debug(f"[{datetime.datetime.now()}] Raw position response: {position_response}")
        position_list = position_response.get('result', {}).get('list', [])
        if position_list and float(position_list[0].get('size', 0)) > 0:
            position = position_list[0]
            position['entry_time'] = str(datetime.datetime.now())  # Set for compatibility
            logger.info(
                f"[{datetime.datetime.now()}] Found existing position: side={position.get('side', 'N/A')}, qty={position.get('size', '0')}, entry_price={position.get('avgPrice', 'N/A')}, tp={position.get('takeProfit', 'N/A')}, sl={position.get('stopLoss', 'N/A')}")
        else:
            logger.info(f"[{datetime.datetime.now()}] No active position found")

        while True:
            try:
                current_time = datetime.datetime.now()
                next_hour = (current_time + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                sleep_seconds = (next_hour - current_time).total_seconds()
                logger.info(
                    f"[{current_time}] Waiting for next hourly candle at {next_hour}, sleeping for {sleep_seconds} seconds")

                await asyncio.sleep(sleep_seconds)

                logger.info(f"[{next_hour}] Processing new candle")
                logger.debug(f"[{next_hour}] Fetching latest Kline data for {symbol}")
                latest_data = await client.get_klines(symbol=symbol, interval=interval, limit=50)
                if not latest_data.get('result', {}).get('list'):
                    logger.error(f"[{next_hour}] No Kline data received from Bybit")
                    continue

                kline = pd.DataFrame(latest_data['result']['list'],
                                     columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                if kline.empty:
                    logger.error(f"[{next_hour}] Kline DataFrame is empty")
                    continue

                logger.info(f"[{next_hour}] Kline data received, shape: {kline.shape}")
                kline['timestamp'] = pd.to_datetime(kline['timestamp'].astype(int), unit='ms')
                kline.set_index('timestamp', inplace=True)
                kline = kline.astype(float)
                logger.debug(f"[{next_hour}] Kline columns: {kline.columns.tolist()}")

                required_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                if not all(col in kline.columns for col in required_columns):
                    logger.error(f"[{next_hour}] Missing required columns in Kline data: {kline.columns}")
                    continue
                if kline.isna().any().any():
                    logger.warning(f"[{next_hour}] NaN values in Kline data, filling with 0")
                    kline = kline.fillna(0)

                logger.debug(f"[{next_hour}] Calculating indicators")
                try:
                    kline = calculate_indicators(kline, settings)
                except Exception as e:
                    logger.error(f"[{next_hour}] Error in calculate_indicators: {str(e)}", exc_info=True)
                    continue

                logger.debug(f"[{next_hour}] Adding signals")
                try:
                    kline = add_signals(kline)
                except Exception as e:
                    logger.error(f"[{next_hour}] Error in add_signals: {str(e)}", exc_info=True)
                    continue

                logger.info(f"[{next_hour}] Signal columns: {list(kline.columns)}")
                if not all(col in kline.columns for col in signal_columns):
                    missing = [col for col in signal_columns if col not in kline.columns]
                    logger.error(f"[{next_hour}] Missing signal columns: {missing}")
                    continue

                weights_0_2, weights_0_5, params = load_latest_parameters()
                if params is None:
                    logger.error(f"[{next_hour}] No optimized parameters found, skipping trade")
                    continue
                logger.info(
                    f"[{next_hour}] Loaded parameters: tp={params[0]:.2f}, sl={params[1]:.2f}, e2={params[2]:.2f}, e5={params[3]:.2f}, leverage={params[4]:.2f}")

                logger.debug(f"[{next_hour}] Calculating index values")
                signals = jnp.array(kline[signal_columns].fillna(0).values)
                logger.info(
                    f"[{next_hour}] Signals shape: {signals.shape}, Weights_0_2 shape: {weights_0_2.shape}, Weights_0_5 shape: {weights_0_5.shape}")
                try:
                    ind_0_2 = float(calculate_index(weights_0_2, signals)[-1])
                    ind_0_5 = float(calculate_index(weights_0_5, signals)[-1])
                except Exception as e:
                    logger.error(f"[{next_hour}] Error in calculate_index: {str(e)}", exc_info=True)
                    continue

                # Use 2-day (48-hour) rolling sentiment
                symbol_short = symbol.replace("USDT", "")  # BTCUSDT -> BTC
                sentiment = get_rolling_sentiment(symbol_short, window_hours=48)
                if sentiment == 0.0:
                    logger.warning(f"[{next_hour}] No sentiment data for BTC over 48 hours, defaulting to 0.0")
                current_price = float(kline['close'].iloc[-1])
                atr = float(kline['ATR'].iloc[-1]) if 'ATR' in kline.columns else 1.0
                logger.info(
                    f"[{next_hour}] Decision inputs: ind_0_2={ind_0_2:.4f}, ind_0_5={ind_0_5:.4f}, sentiment={sentiment:.4f}, price={current_price:.2f}, ATR={atr:.2f}")

                take_profit = float(params[0])
                stop_loss = float(params[1])
                ind_entry_0_2 = float(params[2])
                ind_entry_0_5 = float(params[3])
                leverage = float(params[4])

                # Check if position is still open
                if position is not None:
                    position_response = await client.position_info(category="linear", symbol=symbol)
                    logger.debug(f"[{next_hour}] Raw position response: {position_response}")
                    position_list = position_response.get('result', {}).get('list', [])
                    if not position_list or float(position_list[0].get('size', 0)) == 0:
                        logger.info(
                            f"[{next_hour}] Position closed: side={position.get('side', 'N/A')}, resetting position")
                        with sqlite3.connect(DB_NAME) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE trades
                                SET exit_time = ?, exit_price = ?, pnl = ?, reason = ?
                                WHERE entry_time = ? AND exit_time IS NULL
                            ''', (
                                str(next_hour),
                                current_price,
                                (current_price - float(position.get('avgPrice', 0))) * float(
                                    position.get('size', 0)) * (-1 if position.get('side', 'N/A') == 'Sell' else 1),
                                'Closed by API check',
                                position.get('entry_time')
                            ))
                            conn.commit()
                        position = None

                # Log decision logic with adjusted sentiment thresholds
                long_condition = (ind_0_2 >= ind_entry_0_2 or ind_0_5 >= ind_entry_0_5) and sentiment <= 0.5
                short_condition = (ind_0_2 <= -ind_entry_0_2 or ind_0_5 <= -ind_entry_0_5) and sentiment >= -0.5
                logger.info(
                    f"[{next_hour}] Long condition: (ind_0_2={ind_0_2:.4f} >= {ind_entry_0_2:.2f} OR ind_0_5={ind_0_5:.4f} >= {ind_entry_0_5:.2f}) AND sentiment={sentiment:.4f} <= 0.5 -> {long_condition}")
                logger.info(
                    f"[{next_hour}] Short condition: (ind_0_2={ind_0_2:.4f} <= -{ind_entry_0_2:.2f} OR ind_0_5={ind_0_5:.4f} <= -{ind_entry_0_5:.2f}) AND sentiment={sentiment:.4f} >= -0.5 -> {short_condition}")

                if position is None:
                    # Calculate position size
                    volume = (capital * leverage) / current_price
                    volume = math.floor(volume / qty_step) * qty_step
                    volume = max(volume, min_qty)
                    logger.info(
                        f"[{next_hour}] Calculated volume: {volume}, price: {current_price}, leverage: {leverage}")

                    # Set leverage
                    leverage_str = str(int(leverage))
                    logger.info(f"[{next_hour}] Setting leverage: {leverage_str}x for {symbol}")
                    try:
                        leverage_response = await client.set_leverage(
                            symbol=symbol,
                            buyLeverage=leverage_str,
                            sellLeverage=leverage_str,
                            category="linear"
                        )
                        if leverage_response.get('retCode') == 0:
                            logger.info(f"[{next_hour}] Leverage set successfully: {leverage_str}x")
                        elif leverage_response.get('retCode') == 110043:
                            logger.info(f"[{next_hour}] Leverage already set to {leverage_str}x, proceeding")
                        else:
                            logger.error(f"[{next_hour}] Failed to set leverage: {leverage_response.get('retMsg')}")
                            continue
                    except Exception as e:
                        if "leverage not modified" in str(e):
                            logger.info(f"[{next_hour}] Leverage already set to {leverage_str}x, proceeding")
                        else:
                            logger.error(f"[{next_hour}] Error setting leverage: {str(e)}", exc_info=True)
                            continue

                    # Round TP/SL to tick_size
                    tp_price_long = round(
                        math.floor((current_price + take_profit * atr / leverage) / tick_size) * tick_size, 8)
                    sl_price_long = round(
                        math.floor((current_price - stop_loss * atr / leverage) / tick_size) * tick_size, 8)
                    tp_price_short = round(
                        math.floor((current_price - take_profit * atr / leverage) / tick_size) * tick_size, 8)
                    sl_price_short = round(
                        math.floor((current_price + stop_loss * atr / leverage) / tick_size) * tick_size, 8)
                    logger.info(
                        f"[{next_hour}] TP/SL: long_tp={tp_price_long}, long_sl={sl_price_long}, short_tp={tp_price_short}, short_sl={sl_price_short}")

                    if long_condition and volume >= min_qty:
                        logger.info(
                            f"[{next_hour}] Открываем длинную позицию: цена={current_price:.2f}, объём={volume:.6f}, TP={tp_price_long:.2f}, SL={sl_price_long:.2f}")
                        order_response = await client.create_order(
                            category="linear",
                            symbol=symbol,
                            side="Buy",
                            orderType="Market",
                            qty=str(volume),
                            takeProfit=str(tp_price_long),
                            stopLoss=str(sl_price_long)
                        )
                        if order_response.get('retCode') == 0:
                            position = order_response
                            position['side'] = 'Buy'
                            position['size'] = str(volume)
                            position['avgPrice'] = str(current_price)
                            position['takeProfit'] = str(tp_price_long)
                            position['stopLoss'] = str(sl_price_long)
                            position['entry_time'] = str(current_time)
                            logger.info(f"[{next_hour}] Открыта длинная позиция: {order_response}")
                            with sqlite3.connect(DB_NAME) as conn:
                                cursor = conn.cursor()
                                cursor.execute('''
                                    INSERT INTO trades (
                                        entry_time, exit_time, entry_price, exit_price, side,
                                        quantity, take_profit, stop_loss, pnl, reason, capital_after
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    str(current_time),
                                    None,
                                    current_price,
                                    None,
                                    "Buy",
                                    volume,
                                    tp_price_long,
                                    sl_price_long,
                                    0.0,
                                    None,
                                    capital * (1 - commission_rate * leverage)
                                ))
                                conn.commit()
                        else:
                            logger.error(
                                f"[{next_hour}] Не удалось открыть длинную позицию: {order_response.get('retMsg')}")
                    elif short_condition and volume >= min_qty:
                        logger.info(
                            f"[{next_hour}] Открываем короткую позицию: цена={current_price:.2f}, объём={volume:.6f}, TP={tp_price_short:.2f}, SL={sl_price_short:.2f}")
                        order_response = await client.create_order(
                            category="linear",
                            symbol=symbol,
                            side="Sell",
                            orderType="Market",
                            qty=str(volume),
                            takeProfit=str(tp_price_short),
                            stopLoss=str(sl_price_short)
                        )
                        if order_response.get('retCode') == 0:
                            position = order_response
                            position['side'] = 'Sell'
                            position['size'] = str(volume)
                            position['avgPrice'] = str(current_price)
                            position['takeProfit'] = str(tp_price_short)
                            position['stopLoss'] = str(sl_price_short)
                            position['entry_time'] = str(current_time)
                            logger.info(f"[{next_hour}] Открыта короткая позиция: {order_response}")
                            with sqlite3.connect(DB_NAME) as conn:
                                cursor = conn.cursor()
                                cursor.execute('''
                                    INSERT INTO trades (
                                        entry_time, exit_time, entry_price, exit_price, side,
                                        quantity, take_profit, stop_loss, pnl, reason, capital_after
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    str(current_time),
                                    None,
                                    current_price,
                                    None,
                                    "Sell",
                                    volume,
                                    tp_price_short,
                                    sl_price_short,
                                    0.0,
                                    None,
                                    capital * (1 - commission_rate * leverage)
                                ))
                                conn.commit()
                        else:
                            logger.error(
                                f"[{next_hour}] Не удалось открыть короткую позицию: {order_response.get('retMsg')}")
                    else:
                        logger.info(
                            f"[{next_hour}] No trade: Long condition={long_condition}, Short condition={short_condition}, volume={volume}")
                else:
                    logger.info(
                        f"[{next_hour}] Position already open, skipping trade: side={position.get('side', 'N/A')}, qty={position.get('size', '0')}")

            except Exception as e:
                logger.error(f"[{next_hour}] Ошибка обработки данных Kline: {str(e)}", exc_info=True)

    except asyncio.CancelledError:
        logger.info("Trading loop cancelled")
    except KeyboardInterrupt:
        logger.info("Trading loop interrupted")
    except Exception as e:
        logger.error(f"Ошибка в trading_loop: {str(e)}", exc_info=True)
    finally:
        await client.close()
        logger.info("Торговый цикл завершён")

async def main(bybit_api_key, bybit_secret_key, twitter_api_key, twitter_accounts, testnet=True, demo=False):
    """Запуск всех циклов."""
    init_db()
    try:

        await asyncio.gather(
            optimization_loop(bybit_api_key, bybit_secret_key, symbol, testnet, demo),
            trading_loop(bybit_api_key, bybit_secret_key, symbol, capital, testnet, demo),
            twitter_parsing_loop(twitter_api_key, twitter_accounts)
        )
    except asyncio.CancelledError:
        logger.info("Main loop cancelled")
    except KeyboardInterrupt:
        logger.info("Main loop interrupted")
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main(
        bybit_api_key=bybit_key,
        bybit_secret_key=bybit_secret,
        twitter_api_key=twitter_key,
        twitter_accounts=twitter_accounts,
        testnet=False,
        demo=False
    ))