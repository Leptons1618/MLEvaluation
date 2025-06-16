#!/usr/bin/env python3
"""
Simple test script to verify logging functionality
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime

def test_logging_setup():
    """Test the logging setup from main.py"""
    
    # Import the setup function
    sys.path.append(os.path.dirname(__file__))
    from main import setup_logging
    
    print("Testing logging setup...")
    
    # Initialize logging
    app_logger, model_logger, explanation_logger, user_study_logger = setup_logging()
    
    # Test all loggers
    app_logger.info("Application logger test - info level")
    app_logger.debug("Application logger test - debug level")
    app_logger.warning("Application logger test - warning level")
    app_logger.error("Application logger test - error level")
    
    model_logger.info("Model logger test - info level")
    model_logger.debug("Model logger test - debug level")
    
    explanation_logger.info("Explanation logger test - info level")
    explanation_logger.debug("Explanation logger test - debug level")
    
    user_study_logger.info("User study logger test - info level")
    user_study_logger.debug("User study logger test - debug level")
    
    print("\nLogging test completed!")
    print(f"Check the logs directory for log files:")
    
    log_dir = Path("logs")
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            print(f"  - {log_file}")
            # Show last few lines of each log file
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    print(f"    Last entry: {lines[-1].strip()}")
    else:
        print("  No logs directory found")

if __name__ == "__main__":
    test_logging_setup()
