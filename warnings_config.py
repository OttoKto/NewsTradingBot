"""
Конфигурация для отключения предупреждений в проекте
"""
import warnings

def suppress_deprecated_warnings():
    """Отключает предупреждения о deprecated API"""
    # Отключаем предупреждения о deprecated pkg_resources
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta_classic")
    
    # Отключаем другие распространенные предупреждения
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Отключаем предупреждения JAX о TPU на Windows
    warnings.filterwarnings("ignore", message="Unable to initialize backend 'tpu'")

# Автоматически применяем настройки при импорте модуля
suppress_deprecated_warnings()
