# utils/helpers.py
import os
import random
import logging
import numpy as np
import torch
from pathlib import Path

def get_project_root() -> Path:
    """
    Get the project root directory
    
    Returns:
        Path to project root
    """
    # Assumes this file is in utils/ directory
    return Path(__file__).parent.parent

def get_abs_path(rel_path: str) -> str:
    """
    Convert a project-relative path to absolute path
    
    Args:
        rel_path: Relative path from project root
        
    Returns:
        Absolute path
    """
    return str(get_project_root() / rel_path)

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

def setup_logging(debug_mode: bool = False) -> None:
    """
    Setup logging configuration
    
    Args:
        debug_mode: Whether to enable debug level logging
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format
    )
    
    # Reduce verbosity of external libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

def timing_decorator(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"TIMING: {func.__name__} took {end - start:.2f} seconds")
        return result
    
    return wrapper