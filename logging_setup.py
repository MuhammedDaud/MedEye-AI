"""
╔════════════════════════════════════════════════════════════════════════════╗
║                            MEDEYE v2.0                                      ║
║                   Medical Eye Disease Detection System                       ║
║                                                                              ║
║  Developer: Muhammad Daud                                                    ║
║  Logging Module - Centralized logging configuration                         ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE, LOGS_DIR

# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(exist_ok=True)


def setup_logger(name, level=None):
    """
    Configure logger with file and console handlers.
    
    Args:
        name (str): Logger name (typically __name__)
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if level is None:
        level = LOG_LEVEL
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set logger level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (rotating)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB per file
            backupCount=5,  # Keep 5 backups
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    return logger


def get_logger(name):
    """
    Get or create logger for a module.
    
    Args:
        name (str): Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    return setup_logger(name)


# Create module-level loggers
data_logger = get_logger("data_utils")
model_logger = get_logger("model_utils")
training_logger = get_logger("training")
inference_logger = get_logger("inference")
error_logger = get_logger("errors")


def log_error_with_context(logger, error, context=""):
    """
    Log error with context information.
    
    Args:
        logger (logging.Logger): Logger instance
        error (Exception): Exception to log
        context (str): Additional context information
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=True)
