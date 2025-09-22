"""
Централизованный файл с путями для всех компонентов бота.
Все пути к файлам, директориям и ресурсам определены здесь.
"""

import os

# Базовые директории
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADING_BOT_DIR = os.path.join(BASE_DIR, 'trading_bybit_bot')
TWEET_ANALYZER_DIR = os.path.join(BASE_DIR, 'tweet_analyzer')
TWITTER_PARSER_DIR = os.path.join(BASE_DIR, 'twitter_parser')
INDICATORS_DIR = os.path.join(BASE_DIR, 'indicators')
TWEETS_DIR = os.path.join(TWITTER_PARSER_DIR, 'tweets')

# База данных
DATABASE_PATH = os.path.join(TRADING_BOT_DIR, 'trading_data.db')

# Лог файлы
TRADING_LOG_PATH = os.path.join(BASE_DIR, 'trading_log.log')
TWITTER_PARSER_LOG_PATH = os.path.join(TWITTER_PARSER_DIR, 'trading_log.log')

# CSV файлы с данными
SENTIMENT_BTC_CSV = os.path.join(TWEET_ANALYZER_DIR, 'sentiment_BTC.csv')
SENTIMENT_ETH_CSV = os.path.join(TWEET_ANALYZER_DIR, 'sentiment_ETH.csv')
SENTIMENT_SOL_CSV = os.path.join(TWEET_ANALYZER_DIR, 'sentiment_SOL.csv')
SENTIMENT_GENERAL_CSV = os.path.join(TWEET_ANALYZER_DIR, 'sentiment_general.csv')
HISTORICAL_TWEETS_CSV = os.path.join(TWITTER_PARSER_DIR, 'historical_tweets.csv')
OPTIMIZATION_RESULTS_CSV = os.path.join(TRADING_BOT_DIR, 'optimization_results.csv')

# JSON файлы
TWEETS_JSONL = os.path.join(TWEETS_DIR, 'tweets_2025_08.jsonl')

# Настройки индикаторов
INDICATOR_SETTINGS_TXT = os.path.join(INDICATORS_DIR, 'indicator_settings.txt')

# Функции для получения путей (для обратной совместимости)
def get_database_path():
    """Возвращает путь к базе данных"""
    return DATABASE_PATH

def get_trading_log_path():
    """Возвращает путь к лог файлу торгового бота"""
    return TRADING_LOG_PATH

def get_sentiment_csv_path(crypto='BTC'):
    """Возвращает путь к CSV файлу с данными сентимента для указанной криптовалюты"""
    sentiment_files = {
        'BTC': SENTIMENT_BTC_CSV,
        'ETH': SENTIMENT_ETH_CSV,
        'SOL': SENTIMENT_SOL_CSV,
        'general': SENTIMENT_GENERAL_CSV
    }
    return sentiment_files.get(crypto, SENTIMENT_BTC_CSV)

def get_historical_tweets_path():
    """Возвращает путь к файлу с историческими твитами"""
    return HISTORICAL_TWEETS_CSV

def get_optimization_results_path():
    """Возвращает путь к файлу с результатами оптимизации"""
    return OPTIMIZATION_RESULTS_CSV

def get_indicator_settings_path():
    """Возвращает путь к файлу настроек индикаторов"""
    return INDICATOR_SETTINGS_TXT

def get_tweets_jsonl_path():
    """Возвращает путь к JSONL файлу с твитами"""
    return TWEETS_JSONL

# Создание директорий при необходимости
def ensure_directories():
    """Создает необходимые директории, если они не существуют"""
    directories = [
        TRADING_BOT_DIR,
        TWEET_ANALYZER_DIR,
        TWITTER_PARSER_DIR,
        INDICATORS_DIR,
        TWEETS_DIR
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# Проверка существования файлов
def check_required_files():
    """Проверяет существование необходимых файлов"""
    required_files = [
        DATABASE_PATH,
        SENTIMENT_BTC_CSV,
        INDICATOR_SETTINGS_TXT
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: The following required files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True
