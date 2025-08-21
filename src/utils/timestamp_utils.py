"""
Timestamp utilities for consistent time handling across the system.
"""

import time
from datetime import datetime
from typing import Optional


def to_seconds_str(timestamp: Optional[int] = None) -> str:
    """Convert timestamp to seconds string format.

    Args:
        timestamp: Unix timestamp in seconds (optional, uses current time if None)

    Returns:
        Seconds timestamp as string
    """
    if timestamp is None:
        timestamp = time.time()
    return str(int(timestamp))


def to_datetime(timestamp: Optional[int] = None) -> datetime:
    """Convert timestamp to datetime object.

    Args:
        timestamp: Unix timestamp in seconds (optional, uses current time if None)

    Returns:
        datetime object
    """
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp)
