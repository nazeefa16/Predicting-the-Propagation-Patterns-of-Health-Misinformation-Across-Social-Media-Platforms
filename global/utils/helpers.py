# utils/helpers.py
import os
import random
import logging
import numpy as np
import torch
import time
import functools
import traceback

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(debug_mode=False, log_file=None):
    """
    Setup logging configuration with file output option
    
    Args:
        debug_mode: Whether to enable debug logging
        log_file: Optional path to log file
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    # Reduce verbosity of external libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

def timing(func):
    """Decorator to measure and log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"TIMING: {func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def debug_this(func):
    """Decorator to add detailed logging to any function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        arg_str = ', '.join([str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()])
        logger.debug(f"ENTER: {func.__name__}({arg_str})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"EXIT: {func.__name__} → {type(result)}")
            return result
        except Exception as e:
            logger.error(f"ERROR in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper