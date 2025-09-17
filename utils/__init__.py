# utils/__init__.py

"""
Utilities Package for ORB Trading Strategy
Contains helper functions, authentication, logging, and common utilities
"""

from .enhanced_auth_helper import (
    FyersAuthManager,
    setup_auth_only,
    authenticate_fyers,
    test_authentication,
    update_pin_only,
    test_pin_input,
    show_environment_info
)

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json

__version__ = "2.0.0"
__author__ = "ORB Trading Strategy Team"

# Package exports
__all__ = [
    # Authentication utilities
    "FyersAuthManager",
    "setup_auth_only",
    "authenticate_fyers",
    "test_authentication",
    "update_pin_only",
    "test_pin_input",
    "show_environment_info",

    # Utility functions
    "setup_logging",
    "validate_environment",
    "format_currency",
    "format_percentage",
    "format_time_remaining",
    "calculate_risk_metrics",
    "validate_symbol",
    "sanitize_filename",
    "get_system_info",
    "measure_performance",
    "retry_on_failure",
    "safe_divide",
    "round_to_tick_size",

    # File and data utilities
    "save_to_json",
    "load_from_json",
    "backup_file",
    "ensure_directory",
    "get_file_size",
    "cleanup_old_files",

    # Configuration utilities
    "load_env_config",
    "validate_config_file",
    "merge_configs",
    "get_config_value",

    # Market utilities
    "is_market_open",
    "get_trading_holidays",
    "calculate_trading_days",
    "format_market_time",

    # Performance utilities
    "PerformanceTimer",
    "MemoryTracker",
    "ErrorCounter",
    "HealthMonitor"
]

logger = logging.getLogger(__name__)


# Logging utilities
def setup_logging(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path (optional)
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        enable_console: Enable console logging

    Returns:
        Configured logger instance
    """
    # Create logs directory if needed
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            ensure_directory(log_dir)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")
    return root_logger


def validate_environment() -> Dict[str, Any]:
    """Validate current environment and system requirements"""
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'system_info': {}
    }

    try:
        # Python version check
        python_version = sys.version_info
        validation['system_info']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"

        if python_version < (3, 9):
            validation['errors'].append("Python 3.9+ required")
            validation['valid'] = False

        # Platform info
        validation['system_info']['platform'] = sys.platform
        validation['system_info']['executable'] = sys.executable

        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            validation['system_info']['available_memory_gb'] = round(available_gb, 2)

            if available_gb < 1.0:
                validation['warnings'].append("Low available memory (<1GB)")
        except ImportError:
            validation['warnings'].append("psutil not available - cannot check memory")

        # Required packages check
        required_packages = [
            'pandas', 'numpy', 'requests', 'python-dotenv',
            'pytz', 'yfinance', 'fyers_apiv3'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            validation['errors'].append(f"Missing packages: {', '.join(missing_packages)}")
            validation['valid'] = False

        # Environment variables check
        required_env_vars = ['FYERS_CLIENT_ID', 'FYERS_SECRET_KEY']
        missing_env_vars = [var for var in required_env_vars if not os.environ.get(var)]

        if missing_env_vars:
            validation['warnings'].append(f"Missing environment variables: {', '.join(missing_env_vars)}")

        # Directory permissions check
        required_dirs = ['logs', 'data', 'backups']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name, exist_ok=True)
                except PermissionError:
                    validation['errors'].append(f"Cannot create {dir_name} directory")
                    validation['valid'] = False
            elif not os.access(dir_name, os.W_OK):
                validation['errors'].append(f"{dir_name} directory not writable")
                validation['valid'] = False

    except Exception as e:
        validation['errors'].append(f"Environment validation error: {str(e)}")
        validation['valid'] = False

    return validation


# Formatting utilities
def format_currency(amount: float, currency: str = "₹") -> str:
    """Format currency amount with proper formatting"""
    if abs(amount) >= 10000000:  # 1 crore
        return f"{currency}{amount / 10000000:.2f}Cr"
    elif abs(amount) >= 100000:  # 1 lakh
        return f"{currency}{amount / 100000:.2f}L"
    elif abs(amount) >= 1000:  # 1 thousand
        return f"{currency}{amount / 1000:.1f}K"
    else:
        return f"{currency}{amount:.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage with proper sign and decimal places"""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimal_places}f}%"


def format_time_remaining(seconds: int) -> str:
    """Format seconds into human readable time"""
    if seconds <= 0:
        return "0 seconds"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60