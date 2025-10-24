import time
from functools import wraps
from typing import Callable
import logging

logger = logging.getLogger(__name__)

def retry_with_backoff(
    retries: int = 3,
    backoff_in_seconds: int = 1,
    max_backoff_in_seconds: int = 60,
    exceptions: tuple = (Exception,)
):
    """
    Decorator para retry com backoff exponencial.
    
    Args:
        retries: Número máximo de tentativas
        backoff_in_seconds: Tempo inicial entre tentativas
        max_backoff_in_seconds: Tempo máximo entre tentativas
        exceptions: Tupla de exceções que devem ser tratadas
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            curr_backoff = backoff_in_seconds
            
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt == retries:
                        logger.error(f"Max retries reached for {func.__name__}")
                        raise e
                    
                    logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}. "
                        f"Retrying in {curr_backoff} seconds..."
                    )
                    
                    time.sleep(curr_backoff)
                    curr_backoff = min(
                        curr_backoff * 2,
                        max_backoff_in_seconds
                    )
            
            return None
        return wrapper
    return decorator