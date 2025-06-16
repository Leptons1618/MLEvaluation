"""
Logging configuration and utilities for ML Evaluation App
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging():
    """Set up comprehensive logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        log_dir / f"ml_evaluation_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # File handler for errors only
    error_handler = logging.FileHandler(
        log_dir / f"ml_evaluation_errors_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Console handler for development (only warnings and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Create specific loggers
    app_logger = logging.getLogger('ml_evaluation')
    model_logger = logging.getLogger('ml_evaluation.model')
    explanation_logger = logging.getLogger('ml_evaluation.explanation')
    user_study_logger = logging.getLogger('ml_evaluation.user_study')
    
    return app_logger, model_logger, explanation_logger, user_study_logger


def get_logger(name: str):
    """Get a logger with the specified name"""
    return logging.getLogger(f'ml_evaluation.{name}')
