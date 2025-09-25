# config/settings.py

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Sector(Enum):
    FMCG = "FMCG"
    IT = "IT"
    BANKING = "BANKING"
    AUTO = "AUTO"
    PHARMA = "PHARMA"
    METALS = "METALS"
    REALTY = "REALTY"
    ENERGY = "ENERGY"
    TELECOM = "TELECOM"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class FyersConfig:
    client_id: str
    secret_key: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    base_url: str = "https://api-t1.fyers.in/api/v3"


@dataclass
class ORBStrategyConfig:
    """Open Range Breakout Strategy Configuration"""
    # Portfolio settings
    portfolio_value: float = 5000
    risk_per_trade_pct: float = 1.0
    max_positions: int = 5

    # ORB specific parameters
    orb_period_minutes: int = 15  # Opening range period
    min_breakout_volume: float = 2.0  # Volume multiplier for breakouts
    min_range_size_pct: float = 0.5  # Minimum range size as % of price

    # Risk management
    stop_loss_pct: float = 1.0  # Stop loss as % from breakout level
    target_multiplier: float = 2.0  # Target as multiple of risk
    trailing_stop_pct: float = 0.5  # Trailing stop adjustment

    # Signal filtering
    min_confidence: float = 0.6
    min_volume_ratio: float = 1.5  # Current vs average volume
    max_gap_size: float = 3.0  # Max overnight gap to consider

    # Position management
    enable_trailing_stops: bool = True
    enable_partial_exits: bool = True
    partial_exit_pct: float = 50.0  # % to exit at first target


@dataclass
class TradingConfig:
    # Market hours (IST)
    market_start_hour: int = 9
    market_start_minute: int = 15
    market_end_hour: int = 15
    market_end_minute: int = 30

    # ORB specific timing
    orb_end_minute: int = 30  # ORB period ends at 9:30 AM
    signal_generation_end_hour: int = 14  # Stop generating signals at 2:00 PM
    signal_generation_end_minute: int = 0

    # Monitoring
    monitoring_interval: int = 10  # seconds
    position_update_interval: int = 5  # seconds