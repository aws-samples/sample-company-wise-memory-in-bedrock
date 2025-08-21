"""
Centralized logging configuration for the application.
"""

import logging
import sys
from typing import Optional

from .config import AppConfig


def setup_logging(config: Optional[AppConfig] = None) -> None:
    """
    Setup centralized logging configuration.

    Args:
        config: AppConfig instance, uses default if None
    """
    if config is None:
        from .config import config as default_config
        config = default_config

    # Configure root logger
    logging.basicConfig(level=getattr(logging, config.log_level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


def get_logger(name: str, config: Optional[AppConfig] = None) -> logging.Logger:
    """
    Get a logger with proper configuration.

    Args:
        name: Logger name (usually __name__)
        config: AppConfig instance, uses default if None

    Returns:
        Configured logger instance
    """
    if config is None:
        from .config import config as default_config
        config = default_config

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level.upper()))
    return logger
