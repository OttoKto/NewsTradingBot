import warnings_config  # Отключаем предупреждения
import requests
import datetime
import pandas as pd
import numpy as np
import ta
import pandas_ta_classic as pta
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool
import jax.numpy as jnp
from jax import grad
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
import logging
from paths import get_sentiment_csv_path, get_optimization_results_path, get_indicator_settings_path


def calculate_wma(data, window):
    weights = np.arange(1, window + 1)
    wma = data.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

def calculate_hull_moving_average(data, window):
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))

    # Step 1: Calculate WMA for the half-length period
    wma_half = calculate_wma(data, half_length)

    # Step 2: Calculate WMA for the full-length period
    wma_full = calculate_wma(data, window)

    # Step 3: Calculate the difference
    diff = 2 * wma_half - wma_full

    # Step 4: Calculate the WMA of the difference
    hma = calculate_wma(diff, sqrt_length)

    return hma

def calculate_keltner_channel(close, high, low, window=20, multiplier=2):
    middle_line = ta.trend.EMAIndicator(close=close, window=window).ema_indicator()

    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=10).average_true_range()
    upper_line = middle_line + (multiplier * atr)
    lower_line = middle_line - (multiplier * atr)

    return middle_line, upper_line, lower_line

def calculate_LSMA(data, length, offset=0): #offset parameter is working good only when 0 don't touch
    lsma = []
    for i in range(len(data)):
        if i < length + offset - 1:
            lsma.append(np.nan)
        else:
            y = data[i - length - offset + 1:i - offset + 1]
            x = np.arange(length)
            coeffs = np.polyfit(x, y, 1)
            lsma.append(coeffs[0] * (length - 1) + coeffs[1])

    lsma_series = pd.Series(lsma, index=data.index)
    return lsma_series

def calculate_CMO(data, window):
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    sum_gain = gain.rolling(window=window).sum()
    sum_loss = loss.rolling(window=window).sum()

    cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
    return cmo

def get_binance_data_by_requests(ticker='ETHUSDT', interval='4h', start='2020-01-01 00:00:00',
                                 end='2024-07-01 00:00:00'):
    """
    interval: str tick interval - 4h/1h/1d ...
    """
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
               'taker_base_vol', 'taker_quote_vol', 'ignore']
    usecols = ['open', 'high', 'low', 'close', 'volume', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol']
    start = int(datetime.datetime.timestamp(pd.to_datetime(start)) * 1000)
    end_u = int(datetime.datetime.timestamp(pd.to_datetime(end)) * 1000)
    df = pd.DataFrame()
    print(f'Downloading {interval} {ticker} ohlc-data ...', end=' ')
    while True:
        url = f'https://www.binance.com/api/v3/klines?symbol={ticker}&interval={interval}&limit=1000&startTime={start}#&endTime={end_u}'
        data = pd.DataFrame(requests.get(url, headers={'Cache-Control': 'no-cache', "Pragma": "no-cache"}).json(),
                            columns=columns, dtype=np.float64)
        start = int(data.open_time.tolist()[-1]) + 1
        # Смещение времени на 3 часа вперед
        data.index = [pd.to_datetime(x, unit='ms') + pd.Timedelta(hours=3) for x in data.open_time]
        data = data[usecols]
        df = pd.concat([df, data], axis=0)
        if int(pd.to_datetime(data.index[-1]).timestamp() * 1000) >= end_u:
            break
    print('Done.')
    df.index = pd.to_datetime(df.index)
    df = df.loc[:end]
    return df[['open', 'high', 'low', 'close', 'volume', 'qav']]




def read_indicator_settings(file_path):
    settings = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, value = line.strip().split('=')
            settings[name] = int(value)
    return settings




def add_signals(df):
    # Существующие сигналы
    df['RSI_Signal'] = np.where(
        (df['RSI'] < 30) & (df['RSI'] > df['RSI'].shift(1)),
        1,
        np.where((df['RSI'] > 70) & (df['RSI'] < df['RSI'].shift(1)), -1, 0)
    )
    df['RSI_Breakout'] = np.where(
        (df['RSI'] > 50) & (df['RSI'].shift(1) <= 50),
        1,
        np.where((df['RSI'] < 50) & (df['RSI'].shift(1) >= 50), -1, 0)
    )
    df['CCI_Signal'] = np.where(
        (df['CCI'] < -100) & (df['CCI'] > df['CCI'].shift(1)),
        1,
        np.where((df['CCI'] > 100) & (df['CCI'] < df['CCI'].shift(1)), -1, 0)
    )
    df['CCI_Breakout'] = np.where(
        (df['CCI'] > 0) & (df['CCI'].shift(1) <= 0),
        1,
        np.where((df['CCI'] < 0) & (df['CCI'].shift(1) >= 0), -1, 0)
    )
    df['MFI_Signal'] = np.where(
        (df['MFI'] < 20) & (df['MFI'] > df['MFI'].shift(1)),
        1,
        np.where((df['MFI'] > 80) & (df['MFI'] < df['MFI'].shift(1)), -1, 0)
    )
    df['MFI_Breakout'] = np.where(
        (df['MFI'] > 50) & (df['MFI'].shift(1) <= 50),
        1,
        np.where((df['MFI'] < 50) & (df['MFI'].shift(1) >= 50), -1, 0)
    )
    df['CMO_Signal'] = np.where(
        (df['CMO'] > 50) & (df['CMO'] > df['CMO'].shift(1)),
        1,
        np.where((df['CMO'] < -50) & (df['CMO'] < df['CMO'].shift(1)), -1, 0)
    )
    df['CMO_Breakout'] = np.where(
        (df['CMO'] > 0) & (df['CMO'].shift(1) <= 0),
        1,
        np.where((df['CMO'] < 0) & (df['CMO'].shift(1) >= 0), -1, 0)
    )
    df['LSMA_Signal'] = np.where(
        df['close'] > df['LSMA'],
        1,
        np.where(df['close'] < df['LSMA'], -1, 0)
    )
    df['LSMA_Breakout'] = np.where(
        (df['close'] > df['LSMA']) & (df['close'].shift(1) <= df['LSMA'].shift(1)),
        1,
        np.where((df['close'] < df['LSMA']) & (df['close'].shift(1) >= df['LSMA'].shift(1)), -1, 0)
    )
    df['EMA_Signal'] = np.where(
        df['close'] > df['EMA'],
        1,
        np.where(df['close'] < df['EMA'], -1, 0)
    )
    df['EMA_Breakout'] = np.where(
        (df['close'] > df['EMA']) & (df['close'].shift(1) <= df['EMA'].shift(1)),
        1,
        np.where((df['close'] < df['EMA']) & (df['close'].shift(1) >= df['EMA'].shift(1)), -1, 0)
    )
    df['MACD_Signal'] = np.where(
        df['MACD'] > df['MACD_Diff'],
        1,
        np.where(df['MACD'] < df['MACD_Diff'], -1, 0)
    )
    df['MACD_Breakout'] = np.where(
        (df['MACD'] > df['MACD_Diff']) & (df['MACD'].shift(1) <= df['MACD_Diff'].shift(1)),
        1,
        np.where((df['MACD'] < df['MACD_Diff']) & (df['MACD'].shift(1) >= df['MACD_Diff'].shift(1)), -1, 0)
    )
    df['HMA_Signal'] = np.where(
        df['close'] > df['HMA'],
        1,
        np.where(df['close'] < df['HMA'], -1, 0)
    )
    df['BOP_Signal'] = pd.to_numeric(df['BOP_Signal'], errors='coerce')
    df['BOP_Trade_Signal'] = np.where(
        df['BOP_Signal'] > 0,
        1,
        np.where(df['BOP_Signal'] < 0, -1, 0)
    )
    df['BOP_Breakout'] = np.where(
        (df['BOP_Trade_Signal'] == 1) & (df['BOP_Trade_Signal'].shift(1) != 1),
        1,
        np.where((df['BOP_Trade_Signal'] == -1) & (df['BOP_Trade_Signal'].shift(1) != -1), -1, 0)
    )
    df['Supertrend_Signal'] = np.where(
        df['Supertrend_Direction'] == 1,
        1,
        np.where(df['Supertrend_Direction'] == -1, -1, 0)
    )
    df['Supertrend_Breakout'] = np.where(
        (df['close'] > df['Supertrend']) & (df['close'].shift(1) <= df['Supertrend'].shift(1)),
        1,
        np.where((df['close'] < df['Supertrend']) & (df['close'].shift(1) >= df['Supertrend'].shift(1)), -1, 0)
    )
    df['Chaikin_Signal'] = np.where(
        df['Chaikin_Osc'] > 0,
        1,
        np.where(df['Chaikin_Osc'] < 0, -1, 0)
    )
    df['Chaikin_Breakout'] = np.where(
        (df['Chaikin_Osc'] > 0) & (df['Chaikin_Osc'].shift(1) <= 0),
        1,
        np.where((df['Chaikin_Osc'] < 0) & (df['Chaikin_Osc'].shift(1) >= 0), -1, 0)
    )

    # Новые сигналы
    # 1. Stochastic Oscillator Signals
    df['Stoch_Signal'] = np.where(
        (df['Stoch_K'] < 20) & (df['Stoch_K'] > df['Stoch_D']),
        1,
        np.where((df['Stoch_K'] > 80) & (df['Stoch_K'] < df['Stoch_D']), -1, 0)
    )

    # 2. OBV Signals (на основе пересечения с скользящей средней)
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()
    df['OBV_Signal'] = np.where(
        df['OBV'] > df['OBV_MA'],
        1,
        np.where(df['OBV'] < df['OBV_MA'], -1, 0)
    )

    # 3. VWAP Signals
    df['VWAP_Signal'] = np.where(
        df['close'] > df['VWAP'],
        1,
        np.where(df['close'] < df['VWAP'], -1, 0)
    )

    # 4. Parabolic SAR Signals
    df['Parabolic_SAR_Signal'] = np.where(
        df['close'] > df['Parabolic_SAR'],
        1,
        np.where(df['close'] < df['Parabolic_SAR'], -1, 0)
    )

    # 5. Ichimoku Cloud Signals
    df['Ichimoku_Signal'] = np.where(
        (df['close'] > df['Ichimoku_A']) & (df['close'] > df['Ichimoku_B']),
        1,
        np.where((df['close'] < df['Ichimoku_A']) & (df['close'] < df['Ichimoku_B']), -1, 0)
    )

    return df




def calculate_indicators(df, settings):
    """
    Рассчитывает индикаторы для DataFrame с данными и возвращает обновлённый DataFrame.
    """
    # Существующие индикаторы
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=settings['RSI_window']).rsi()
    df['CCI'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'],
                                      close=df['close'], window=settings['CCI_window']).cci()
    df['MFI'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'],
                                       close=df['close'], volume=df['volume'],
                                       window=settings['MFI_window']).money_flow_index()
    df['ADX'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'],
                                      close=df['close'], window=settings['ADX_window']).adx()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'],
                                               close=df['close'],
                                               window=settings['ATR_window']).average_true_range()
    df['CMO'] = calculate_CMO(df['close'], settings['CMO_window'])
    bollinger = ta.volatility.BollingerBands(close=df['close'],
                                             window=settings['Bollinger_window'],
                                             window_dev=settings['Bollinger_window_dev'])
    df['Bollinger_Signal'] = ((df['close'] > bollinger.bollinger_hband()) |
                              (df['close'] < bollinger.bollinger_lband())).astype(str)
    df['SMA'] = ta.trend.SMAIndicator(close=df['close'], window=7).sma_indicator()
    df['LSMA'] = calculate_LSMA(df['close'], settings['LSMA_window'], offset=settings['LSMA_offset'])
    df['EMA'] = ta.trend.EMAIndicator(close=df['close'],
                                      window=settings['EMA_window']).ema_indicator()
    macd = ta.trend.MACD(close=df['close'],
                         window_slow=settings['MACD_window_slow'],
                         window_fast=settings['MACD_window_fast'],
                         window_sign=settings['MACD_window_sign'])
    df['MACD'] = macd.macd()
    df['MACD_Diff'] = macd.macd_diff()
    df['MACD_Signal'] = macd.macd_signal()
    middle_line, upper_line, lower_line = calculate_keltner_channel(df['close'], df['high'], df['low'],
                                                                    window=settings['Keltner_window'],
                                                                    multiplier=settings['Keltner_multiplier'])
    df['Keltner_Signal'] = ((df['close'] > upper_line) | (df['close'] < lower_line)).astype(str)
    df['HMA'] = calculate_hull_moving_average(df['close'], window=settings['HMA_window'])
    df['HMA_Signal'] = np.where(df['close'] > df['HMA'], 'LONG', 'SHORT')
    df['BOP'] = pta.bop(open_=df['open'], high=df['high'],
                        low=df['low'], close=df['close'])
    df['BOP_Signal'] = np.where(df['BOP'] > 0, 'LONG', 'SHORT')
    supertrend = pta.supertrend(df['high'], df['low'], df['close'],
                                length=settings['Supertrend_length'],
                                multiplier=settings['Supertrend_multiplier'])
    df['Supertrend'] = supertrend[f'SUPERT_{settings["Supertrend_length"]}_{settings["Supertrend_multiplier"]}.0']
    df['Supertrend_Direction'] = supertrend[
        f'SUPERTd_{settings["Supertrend_length"]}_{settings["Supertrend_multiplier"]}.0']
    ad = ta.volume.AccDistIndexIndicator(high=df['high'], low=df['low'],
                                         close=df['close'], volume=df['volume']).acc_dist_index()
    ema3 = ta.trend.EMAIndicator(close=ad, window=settings['Chaikin_ema_short_window']).ema_indicator()
    ema10 = ta.trend.EMAIndicator(close=ad, window=settings['Chaikin_ema_long_window']).ema_indicator()
    df['Chaikin_Osc'] = ema3 - ema10

    # Новые индикаторы
    # 1. Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'],
                                             window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # 2. On-Balance Volume (OBV)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

    # 3. VWAP (упрощённый расчёт для дневных данных, для точности нужен внутридневной VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # 4. Parabolic SAR
    df['Parabolic_SAR'] = ta.trend.PSARIndicator(high=df['high'], low=df['low'],
                                                 close=df['close']).psar()

    # 5. Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'],
                                          window1=9, window2=26, window3=52)
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()
    df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()

    return df



def run_strategy_tp(df, initial_capital=100, commission_rate=0.001, stop_loss_pct=0.6, take_profit_pct=6, ind_entry=0.7, front_output=False):
    # Предварительное извлечение данных в массивы для быстрого доступа
    indicator = df['Indicator_Index'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    times = df.index.to_numpy()  # получаем numpy-массив с индексами

    n = len(df)
    position_arr = np.zeros(n, dtype=np.int8)
    equity_curve = np.empty(n, dtype=np.float64)
    diff_curve = np.empty(n, dtype=np.float64)

    trades = []
    capital = initial_capital
    position = 0  # 1 – лонг, -1 – шорт, 0 – нет позиции
    entry_price = 0.0
    entry_time = None
    entry_capital = 0.0
    total_commission = 0.0
    win_trades = 0
    total_trades = 0

    for i in range(n):
        current_signal = indicator[i]
        current_price = close[i]
        current_time = times[i]

        # Вычисление изменения цены (diff)
        if i == 0:
            diff_curve[i] = np.nan
        else:
            diff_curve[i] = ((current_price - close[i - 1]) / close[i - 1]) * 100

        if position == 0:
            if current_signal >= ind_entry:
                position = 1
                entry_price = current_price
                entry_time = current_time
                entry_capital = capital * (1 - commission_rate)
                total_commission += capital * commission_rate  # комиссия за вход
                position_arr[i] = position
            elif current_signal <= -ind_entry:
                position = -1
                entry_price = current_price
                entry_time = current_time
                entry_capital = capital * (1 - commission_rate)
                total_commission += capital * commission_rate
                position_arr[i] = position
            else:
                position_arr[i] = 0
        else:
            # Определяем условия стоп-лосса и тейк-профита
            stop_loss_hit = False
            take_profit_hit = False
            if position == 1:
                if low[i] <= entry_price * (1 - stop_loss_pct / 100):
                    stop_loss_hit = True
                elif high[i] >= entry_price * (1 + take_profit_pct / 100):
                    take_profit_hit = True
            elif position == -1:
                if high[i] >= entry_price * (1 + stop_loss_pct / 100):
                    stop_loss_hit = True
                elif low[i] <= entry_price * (1 - take_profit_pct / 100):
                    take_profit_hit = True

            # Если условие выхода выполнено, определяем цену выхода и причину
            if stop_loss_hit:
                slippage_multiplier = np.random.uniform(1, 1.01)
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct / 100 * slippage_multiplier)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct / 100 * slippage_multiplier)
                exit_reason = "Стоп-лосс"
            elif take_profit_hit:
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct / 100)
                else:
                    exit_price = entry_price * (1 - take_profit_pct / 100)
                exit_reason = "Тейк-профит"
            elif (position == 1 and current_signal <= -2) or (position == -1 and current_signal >= 2):
                exit_price = current_price
                exit_reason = "Сигнальный выход"
            else:
                position_arr[i] = position
                if position == 1:
                    equity_curve[i] = entry_capital * (1 + (current_price - entry_price) / entry_price)
                else:
                    equity_curve[i] = entry_capital * (1 + (entry_price - current_price) / entry_price)
                continue  # продолжаем, если условия выхода не выполнены

            # Фиксируем сделку
            exit_time = current_time
            trade_return = ((exit_price - entry_price) / entry_price) if position == 1 else ((entry_price - exit_price) / entry_price)
            final_capital = entry_capital * (1 + trade_return) * (1 - commission_rate)
            profit = final_capital - capital
            total_commission += final_capital * commission_rate  # комиссия за выход

            total_trades += 1
            if profit > 0:
                win_trades += 1

            trades.append({
                "Вход": entry_time,
                "Выход": exit_time,
                "Цена входа": entry_price,
                "Цена выхода": exit_price,
                "Доходность": f"{trade_return * 100:.2f}%",
                "Капитал до сделки": f"{capital:.2f}",
                "Капитал после сделки": f"{final_capital:.2f}",
                "Профит": f"{profit:.2f}",
                "Причина выхода": exit_reason
            })

            capital = final_capital
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_capital = 0.0
            position_arr[i] = 0

        # Обновляем equity_curve для текущего шага
        if position == 0:
            equity_curve[i] = capital
        else:
            if position == 1:
                equity_curve[i] = entry_capital * (1 + (current_price - entry_price) / entry_price)
            else:
                equity_curve[i] = entry_capital * (1 + (entry_price - current_price) / entry_price)

    # Записываем результаты обратно в DataFrame
    df['position'] = position_arr
    df['equity_curve'] = equity_curve
    df['diff_curve'] = diff_curve

    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    if front_output:
        print("\nСделки:")
        for trade in trades:
            print(
                f"🔹 {trade['Вход']} → {trade['Выход']} | Вход: {trade['Цена входа']:.2f} | Выход: {trade['Цена выхода']:.2f} | "
                f"Доходность: {trade['Доходность']} | Капитал: {trade['Капитал до сделки']} → {trade['Капитал после сделки']} | "
                f"Профит: {trade['Профит']} | Причина: {trade['Причина выхода']}"
            )
        print(f"\n🔹 Винрейт: {win_rate:.2f}%")
        print(f"🔹 Общая сумма уплаченной комиссии: {total_commission:.2f} USDT")

    return df



# Configure logging
logger = logging.getLogger(__name__)


def run_strategy_tp_ATR(df, initial_capital=100, commission_rate=0.001, stop_loss_ATR=0.5, take_profit_ATR=2,
                        ind_entry_0_2=0.7, ind_entry_0_5=0.7, leverage=1, sentiment_long=0.5, sentiment_short=-0.5,
                        front_output=False):
    # Determine the path to the sentiment file
    sentiment_file_path = get_sentiment_csv_path('BTC')

    # Load sentiment data
    try:
        sentiment_df = pd.read_csv(sentiment_file_path, parse_dates=['created_at'])
        sentiment_df['created_at'] = pd.to_datetime(sentiment_df['created_at'], utc=True).dt.tz_localize(None)
        sentiment_df = sentiment_df.sort_values('created_at')
        sentiment_df = sentiment_df[['created_at', 'rolling_sentiment_score']]
        logger.debug(f"Loaded sentiment data: {len(sentiment_df)} rows")
    except Exception as e:
        logger.error(f"Failed to load sentiment data: {str(e)}")
        return df

    # Create a copy of df to avoid modifying the input
    df = df.copy()

    # Ensure df index is datetime and timezone-naive
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.reset_index().rename(columns={'index': 'time'})

    # Drop any existing 'rolling_sentiment_score' to avoid conflicts
    if 'rolling_sentiment_score' in df.columns:
        logger.debug("Dropping existing 'rolling_sentiment_score' from df to avoid merge conflict")
        df = df.drop(columns=['rolling_sentiment_score'])

    # Use merge_asof for nearest previous sentiment score
    try:
        merged = pd.merge_asof(
            df.sort_values('time'),
            sentiment_df.rename(columns={'created_at': 'time'}),
            on='time',
            direction='backward'
        )
        merged.set_index('time', inplace=True)
        df = merged
        logger.debug(f"Columns after merge: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"merge_asof failed: {str(e)}")
        return df

    # Verify required columns
    required_columns = ['Indicator_Index_0_2', 'Indicator_Index_0_5', 'rolling_sentiment_score', 'close', 'high', 'low',
                        'ATR']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns after merge: {missing_columns}")
        return df

    # Extract data into arrays for fast access
    indicator_0_2 = df['Indicator_Index_0_2'].values
    indicator_0_5 = df['Indicator_Index_0_5'].values
    sentiment = df['rolling_sentiment_score'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    times = df.index.to_numpy()
    atr = df['ATR'].values
    n = len(df)
    position_arr = np.zeros(n, dtype=np.int8)
    equity_curve = np.empty(n, dtype=np.float64)
    diff_curve = np.empty(n, dtype=np.float64)

    trades = []
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_time = None
    entry_capital = 0.0
    total_commission = 0.0
    win_trades = 0
    total_trades = 0

    for i in range(n):
        current_signal_0_2 = indicator_0_2[i]
        current_signal_0_5 = indicator_0_5[i]
        current_sentiment = sentiment[i]
        current_price = close[i]
        current_time = pd.to_datetime(times[i])
        current_atr = atr[i]

        if current_time < pd.to_datetime(times[0]) + pd.Timedelta(days=14):
            equity_curve[i] = capital
            position_arr[i] = 0
            continue

        if i == 0:
            diff_curve[i] = np.nan
        else:
            diff_curve[i] = ((current_price - close[i - 1]) / close[i - 1]) * 100

        if position == 0:
            if pd.isna(current_sentiment):
                position_arr[i] = 0
                equity_curve[i] = capital
                continue

            if ((current_signal_0_2 >= ind_entry_0_2) or (
                    current_signal_0_5 >= ind_entry_0_5)) and current_sentiment <= sentiment_long:
                position = 1
                entry_price = current_price
                entry_time = current_time
                entry_capital = capital * (1 - (commission_rate * leverage))
                total_commission += capital * commission_rate * leverage
                position_arr[i] = position
            elif ((current_signal_0_2 <= -ind_entry_0_2) or (
                    current_signal_0_5 <= -ind_entry_0_5)) and current_sentiment >= sentiment_short:
                position = -1
                entry_price = current_price
                entry_time = current_time
                entry_capital = capital * (1 - (commission_rate * leverage))
                total_commission += capital * commission_rate * leverage
                position_arr[i] = position
            else:
                position_arr[i] = 0
                equity_curve[i] = capital
        else:
            stop_loss_hit = False
            take_profit_hit = False
            if position == 1:
                if low[i] <= entry_price - ((stop_loss_ATR / leverage) * current_atr):
                    stop_loss_hit = True
                elif high[i] >= entry_price + ((take_profit_ATR / leverage) * current_atr):
                    take_profit_hit = True
            elif position == -1:
                if high[i] >= entry_price + ((stop_loss_ATR / leverage) * current_atr):
                    stop_loss_hit = True
                elif low[i] <= entry_price - ((take_profit_ATR / leverage) * current_atr):
                    take_profit_hit = True

            if stop_loss_hit:
                slippage_multiplier = np.random.uniform(1, 1.01)
                if position == 1:
                    exit_price = entry_price - ((stop_loss_ATR / leverage) * current_atr) * slippage_multiplier
                else:
                    exit_price = entry_price + ((stop_loss_ATR / leverage) * current_atr) * slippage_multiplier
                exit_reason = "Стоп-лосс"
            elif take_profit_hit:
                if position == 1:
                    exit_price = entry_price + ((take_profit_ATR / leverage) * current_atr)
                else:
                    exit_price = entry_price - ((take_profit_ATR / leverage) * current_atr)
                exit_reason = "Тейк-профит"
            elif (position == 1 and current_signal_0_2 <= -2) or (position == -1 and current_signal_0_2 >= 2):
                exit_price = current_price
                exit_reason = "Сигнальный выход"
            else:
                position_arr[i] = position
                if position == 1:
                    equity_curve[i] = entry_capital * (1 + (((current_price - entry_price) / entry_price) * leverage))
                else:
                    equity_curve[i] = entry_capital * (1 + (((entry_price - current_price) / entry_price) * leverage))
                continue

            exit_time = current_time
            trade_return = ((exit_price - entry_price) / entry_price) * leverage if position == 1 else (
                                                                                                               (
                                                                                                                           entry_price - exit_price) / entry_price) * leverage
            final_capital = (entry_capital * (1 + trade_return) * (1 - commission_rate * leverage))
            profit = final_capital - capital
            total_commission += final_capital * commission_rate * leverage

            total_trades += 1
            if profit > 0:
                win_trades += 1

            formatted_entry_time = entry_time.strftime('%Y-%m-%d %H:%M:%S')
            formatted_exit_time = exit_time.strftime('%Y-%m-%d %H:%M:%S')

            trades.append({
                "Вход": formatted_entry_time,
                "Выход": formatted_exit_time,
                "Цена входа": entry_price,
                "Цена выхода": exit_price,
                "Доходность": f"{trade_return * 100:.2f}%",
                "Капитал до сделки": f"{capital:.2f}",
                "Капитал после сделки": f"{final_capital:.2f}",
                "Профит": f"{profit:.2f}",
                "Причина выхода": exit_reason
            })

            capital = final_capital
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_capital = 0.0
            position_arr[i] = 0

        if position == 0:
            equity_curve[i] = capital
        else:
            if position == 1:
                equity_curve[i] = entry_capital * (1 + (((current_price - entry_price) / entry_price) * leverage))
            else:
                equity_curve[i] = entry_capital * (1 + (((entry_price - current_price) / entry_price) * leverage))

    df['position'] = position_arr
    df['equity_curve'] = equity_curve
    df['diff_curve'] = diff_curve

    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    if front_output:
        logger.info("\nСделки:")
        for trade in trades:
            logger.info(
                f"🔹 {trade['Вход']} → {trade['Выход']} | Вход: {trade['Цена входа']:.2f} | Выход: {trade['Цена выхода']:.2f} | "
                f"Доходность: {trade['Доходность']} | Капитал: {trade['Капитал до сделки']} → {trade['Капитал после сделки']} | "
                f"Профит: {trade['Профит']} | Причина: {trade['Причина выхода']}"
            )
        logger.info(f"\n🔹 Винрейт: {win_rate:.2f}%")
        logger.info(f"🔹 Общая сумма уплаченной комиссии: {total_commission:.2f} USDT")

    return df


def process_combination(args):
    """Process a single parameter combination for grid optimization."""
    tp, sl, e2, e5, l, df, initial_capital, commission_rate = args
    try:
        logger.debug(f"Processing params: tp={tp}, sl={sl}, e2={e2}, e5={e5}, l={l}")
        result_df = run_strategy_tp_ATR(
            df,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            take_profit_ATR=tp,
            stop_loss_ATR=sl,
            ind_entry_0_2=e2,
            ind_entry_0_5=e5,
            leverage=l,
            front_output=False
        )
        final_capital = result_df['equity_curve'].iloc[-1]
        logger.debug(f"Final capital for params (tp={tp}, sl={sl}, e2={e2}, e5={e5}, l={l}): {final_capital}")
        return {
            "Take Profit %": tp,
            "Stop Loss %": sl,
            "Entry Threshold_0_2": e2,
            "Entry Threshold_0_5": e5,
            "Leverage": l,
            "Final Capital": final_capital,
            "result_df": result_df
        }
    except Exception as e:
        logger.error(f"Error processing params (tp={tp}, sl={sl}, e2={e2}, e5={e5}, l={l}): {str(e)}", exc_info=True)
        return None


def run_grid_optimization(df, tp_values, sl_values, entry_0_2_values, entry_0_5_values, leverage_values,
                          initial_capital=10000, commission_rate=0.001):
    """Run grid search over parameters, calling run_strategy_tp_ATR."""
    param_combinations = list(product(tp_values, sl_values, entry_0_2_values, entry_0_5_values, leverage_values))
    params_list = [(tp, sl, e2, e5, l, df, initial_capital, commission_rate)
                   for tp, sl, e2, e5, l in param_combinations]

    logger.info(f"Running grid optimization with {len(params_list)} combinations")
    results_list = Parallel(n_jobs=-1)(
        delayed(process_combination)(args) for args in tqdm(params_list, desc="Оптимизация")
    )

    results_list = [r for r in results_list if r is not None]
    if not results_list:
        logger.error("No valid results from grid optimization")
        raise ValueError("Grid optimization failed to produce valid results")

    return results_list


def optimize_strategy_two_stage(df, initial_capital=10000, commission_rate=0.001,
                                save_path=None):
    # Reduced grid for testing
    tp_values = [3, 5, 7, 9, 12]
    sl_values = [1, 2, 3, 5]
    entry_0_2_values = [0.3, 0.6, 0.9]
    entry_0_5_values = [0.3, 0.6, 0.9]
    leverage_values = [1, 2, 3]

    logger.info(
        f"Starting stage 1 optimization with {len(tp_values) * len(sl_values) * len(entry_0_2_values) * len(entry_0_5_values) * len(leverage_values)} combinations")

    try:
        stage1_results = run_grid_optimization(
            df, tp_values, sl_values, entry_0_2_values, entry_0_5_values, leverage_values,
            initial_capital, commission_rate
        )
        best_stage1 = max(stage1_results, key=lambda x: x["Final Capital"])
        logger.info(f"Лучший результат этапа 1: {best_stage1['Final Capital']:.2f}, параметры: {best_stage1}")
    except Exception as e:
        logger.error(f"Stage 1 optimization failed: {str(e)}", exc_info=True)
        return (5, 2, 0.7, 0.7, 1), 0, df

    def around(val, step=0.1, minv=0, maxv=10):
        return [round(x, 2) for x in np.clip([val - step, val, val + step], minv, maxv)]

    tp_best = best_stage1["Take Profit %"]
    sl_best = best_stage1["Stop Loss %"]
    e2_best = best_stage1["Entry Threshold_0_2"]
    e5_best = best_stage1["Entry Threshold_0_5"]
    l_best = [best_stage1["Leverage"]]

    tp_values = around(tp_best, 1, 2, 30)
    sl_values = around(sl_best, 0.5, 0.2, 10)
    entry_0_2_values = around(e2_best, 0.1, 0, 1)
    entry_0_5_values = around(e5_best, 0.1, 0, 1)

    logger.info(
        f"Starting stage 2 optimization with {len(tp_values) * len(sl_values) * len(entry_0_2_values) * len(entry_0_5_values) * len(l_best)} combinations")

    try:
        stage2_results = run_grid_optimization(
            df, tp_values, sl_values, entry_0_2_values, entry_0_5_values, l_best,
            initial_capital, commission_rate
        )
        best_stage2 = max(stage2_results, key=lambda x: x["Final Capital"])
        logger.info(f"Лучший результат этапа 2: {best_stage2['Final Capital']:.2f}, параметры: {best_stage2}")
    except Exception as e:
        logger.error(f"Stage 2 optimization failed: {str(e)}", exc_info=True)
        return (5, 2, 0.7, 0.7, 1), 0, df

    best_result = max(stage2_results, key=lambda x: x["Final Capital"])

    best_params = (
        best_result["Take Profit %"],
        best_result["Stop Loss %"],
        best_result["Entry Threshold_0_2"],
        best_result["Entry Threshold_0_5"],
        best_result["Leverage"]
    )
    best_capital = best_result["Final Capital"]
    best_df = best_result["result_df"]

    logger.info(f"ЛУЧШИЕ ПАРАМЕТРЫ: {best_params} → Капитал: {best_capital:.2f}")

    results_df = pd.DataFrame([{
        "Take Profit %": r["Take Profit %"],
        "Stop Loss %": r["Stop Loss %"],
        "Entry Threshold_0_2": r["Entry Threshold_0_2"],
        "Entry Threshold_0_5": r["Entry Threshold_0_5"],
        "Leverage": r["Leverage"],
        "Final Capital": r["Final Capital"]
    } for r in stage2_results])
    try:
        if save_path is None:
            save_path = get_optimization_results_path()
        results_df.to_csv(save_path, index=False)
        logger.info(f"Результаты сохранены в {save_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}", exc_info=True)

    return best_params, best_capital, best_df


# ---------------------------
# Функции на базе JAX
# ---------------------------


def loss_function(weights, signals, close, threshold):
    index_values = calculate_index(weights, signals)
    pct = jnp.diff(close) / close[:-1]
    # Для i>=2: desired сигнал и соответствующий индекс из index_values (сдвигаем на 1)
    desired = jnp.where(pct[1:] > 0, 1.0, -1.0)
    signal = index_values[1:-1]
    mask = jnp.abs(pct[1:]) > threshold
    return jnp.sum((signal[mask] - desired[mask]) ** 2)



def optimize_weights_jax(signals, close, learning_rate=0.05, epochs=200):
    """
    Оптимизация весов индикаторов с помощью градиентного спуска на базе JAX.
    Принимает:
      signals: jnp.array размера (n_samples, num_indicators)
      close: jnp.array цен закрытия (n_samples)
    Возвращает:
      best_weights: набор весов с минимальной функцией потерь
    """
    num_indicators = signals.shape[1]
    weights_0_2 = jnp.ones(num_indicators)
    loss_grad = grad(loss_function)

    # Инициализация переменных для отслеживания минимальной потери и лучших весов
    min_loss = float('inf')  # Начинаем с бесконечности
    best_weights_0_2 = weights_0_2  # Изначально лучшие веса — начальные

    for epoch in range(epochs):
        loss_val = loss_function(weights_0_2, signals, close, 0.02)
        gradients = loss_grad(weights_0_2, signals, close, 0.02)
        weights_0_2 = weights_0_2 - learning_rate * gradients

        # Обновляем минимальную потерю и лучшие веса, если текущая потеря меньше
        if loss_val < min_loss:
            min_loss = loss_val
            best_weights_0_2 = weights_0_2.copy()  # Сохраняем копию текущих весов

        #print(f"Epoch {epoch + 1}: Loss = {loss_val:.6f}, Weights = {weights_0_2}")

    print(f"Best Loss = {min_loss:.6f}, Best Weights = {best_weights_0_2}")

    num_indicators = signals.shape[1]
    weights_0_5 = jnp.ones(num_indicators)
    loss_grad = grad(loss_function)

    # Инициализация переменных для отслеживания минимальной потери и лучших весов
    min_loss = float('inf')  # Начинаем с бесконечности
    best_weights_0_5 = weights_0_5  # Изначально лучшие веса — начальные

    for epoch in range(epochs + 1000):
        loss_val = loss_function(weights_0_5, signals, close, 0.04)
        gradients = loss_grad(weights_0_5, signals, close, 0.04)
        weights_0_5 = weights_0_5 - learning_rate * gradients

        # Обновляем минимальную потерю и лучшие веса, если текущая потеря меньше
        if loss_val < min_loss:
            min_loss = loss_val
            best_weights_0_5 = weights_0_5.copy()  # Сохраняем копию текущих весов

        #print(f"Epoch {epoch + 1}: Loss = {loss_val:.6f}, Weights = {weights_0_5}")

    print(f"Best Loss = {min_loss:.6f}, Best Weights = {best_weights_0_5}")

    return best_weights_0_2, best_weights_0_5





def calculate_index(weights, signals):
    """
    Рассчитывает итоговый индекс как взвешенную сумму сигналов,
    нормализованную по сумме весов и пропущенную через гиперболический тангенс.
    """
    weighted_sum = jnp.dot(signals, weights)
    index_values = weighted_sum / jnp.sum(weights)
    return jnp.tanh(index_values)

if __name__ == '__main__':
    final_capital = 100

    # Начало исторических данных (начало тренировочного периода)
    initial_training_start = datetime.datetime(2023, 12, 1)
    training_length_months = 12  # 12 месяцев обучения
    trading_optimization_length_month = 12

    training_start = initial_training_start
    training_end = (pd.Timestamp(training_start) + pd.DateOffset(months=training_length_months)).to_pydatetime()
    today = datetime.datetime.today()

    capital_over_time = []
    test_dates = []


    settings = read_indicator_settings(get_indicator_settings_path())

    # Определяем колонки, используемые для расчёта индекса
    signal_columns = [
        "RSI_Signal", "RSI_Breakout",
        "CCI_Signal", "CCI_Breakout",
        "MFI_Signal", "MFI_Breakout",
        "CMO_Signal", "CMO_Breakout",
        "LSMA_Signal", "LSMA_Breakout",
        "EMA_Signal", "EMA_Breakout",
        "MACD_Signal", "MACD_Breakout",
        "HMA_Signal",
        "BOP_Trade_Signal", "BOP_Breakout",
        "Supertrend_Signal", "Supertrend_Breakout",
        "Chaikin_Signal", "Chaikin_Breakout",
        "Stoch_Signal",
        "OBV_Signal",
        "VWAP_Signal",
        "Parabolic_SAR_Signal",
        "Ichimoku_Signal"
    ]

    # Основной цикл (скользящее окно)
    while True:
        # Тестовый период: следующий месяц после тренировочного окна
        test_start = (training_end - pd.DateOffset(weeks=2)).to_pydatetime()
        test_end = (pd.Timestamp(test_start) + pd.DateOffset(months=1) + pd.DateOffset(weeks=2)).to_pydatetime()

        if test_end > today:
            break

        # Форматирование дат для API-запросов
        train_start_str = training_start.strftime('%Y-%m-%d 00:00:00')
        train_end_str = training_end.strftime('%Y-%m-%d 00:00:00')
        test_start_str = test_start.strftime('%Y-%m-%d 00:00:00')
        test_end_str = test_end.strftime('%Y-%m-%d 00:00:00')

        # 1. Обучение (тренировочный период)
        train_settings = {
            'ticker': 'BTCUSDT',
            'interval': '1h',
            'start': train_start_str,
            'end': train_end_str
        }
        train_data = get_binance_data_by_requests(
            ticker=train_settings['ticker'],
            interval=train_settings['interval'],
            start=train_settings['start'],
            end=train_settings['end']
        )
        train_data = calculate_indicators(train_data, settings)
        train_data = add_signals(train_data)

        # Извлекаем матрицу сигналов и цены закрытия для оптимизации JAX
        train_signals = jnp.array(train_data[signal_columns].values)
        train_close = jnp.array(train_data["close"].values)

        optimized_weights_0_2, optimized_weights_0_5 = optimize_weights_jax(train_signals, train_close, learning_rate=0.01, epochs=1000)
        # Рассчитываем индекс на основе оптимизированных весов
        index_values = calculate_index(optimized_weights_0_2, train_signals)
        # Преобразуем jax-массив в numpy для сохранения в DataFrame
        train_data['Indicator_Index_0_2'] = np.array(index_values)
        # Рассчитываем индекс на основе оптимизированных весов
        index_values = calculate_index(optimized_weights_0_5, train_signals)
        # Преобразуем jax-массив в numpy для сохранения в DataFrame
        train_data['Indicator_Index_0_5'] = np.array(index_values)

        # Определяем начало оптимизационного периода (последние trading_optimization_length_month месяцев)
        optimization_start = (pd.Timestamp(training_end) - pd.DateOffset(
            months=trading_optimization_length_month)).to_pydatetime()
        train_data_opt = train_data.loc[optimization_start:]

        best_params, best_capital, train_data_opt = optimize_strategy_two_stage(
            train_data_opt,
            initial_capital=100,
            commission_rate=(0.01 / 100)
        )

        # 2. Тестирование (следующий месяц)
        test_settings = {
            'ticker':'BTCUSDT',
            'interval': '1h',
            'start': test_start_str,
            'end': test_end_str
        }
        test_data = get_binance_data_by_requests(
            ticker=test_settings['ticker'],
            interval=test_settings['interval'],
            start=test_settings['start'],
            end=test_settings['end']
        )
        test_data = calculate_indicators(test_data, settings)
        test_data = add_signals(test_data)

        test_signals = jnp.array(test_data[signal_columns].values)
        test_close = jnp.array(test_data["close"].values)
        index_values_test_0_2 = calculate_index(optimized_weights_0_2, test_signals)
        test_data['Indicator_Index_0_2'] = np.array(index_values_test_0_2)
        index_values_test_0_5 = calculate_index(optimized_weights_0_5, test_signals)
        test_data['Indicator_Index_0_5'] = np.array(index_values_test_0_5)

        test_data = run_strategy_tp_ATR(
            df=test_data,
            initial_capital=final_capital,
            commission_rate=(0.01 / 100),
            stop_loss_ATR=best_params[1],
            take_profit_ATR=best_params[0],
            ind_entry_0_2=best_params[2],
            ind_entry_0_5=best_params[3],
            leverage=best_params[4],
            front_output=True
        )


        final_capital = test_data['equity_curve'].iloc[-1]

        capital_over_time.append(final_capital)
        test_dates.append(test_start)

        print(f"Training: {train_start_str} to {train_end_str} -> Test: {test_start_str} to {test_end_str} | Capital: {final_capital}")

        # 3. Обновление тренировочного окна (скользящее окно)
        training_start = (pd.Timestamp(training_start) + pd.DateOffset(months=1)).to_pydatetime()
        training_end = (pd.Timestamp(training_end) + pd.DateOffset(months=1)).to_pydatetime()

    # Построение графика изменения капитала
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, capital_over_time, marker='o')
    plt.xlabel('Дата начала тестового периода')
    plt.ylabel('Конечный капитал')
    plt.title('Изменение капитала (скользящее окно тренировки по 12 мес.)')
    plt.grid(True)
    plt.show()

