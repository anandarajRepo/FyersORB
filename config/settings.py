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
    """Dataclass Default Value - Fallback if env var not found"""
    """Open Range Breakout Strategy Configuration"""
    # Portfolio settings
    portfolio_value: float = 30000
    risk_per_trade_pct: float = 30.0
    max_positions: int = 3

    intraday_margin_multiplier: float = 1.0  # 5x intraday margin allowed

    # ORB specific parameters
    orb_period_minutes: int = 15  # Opening range period
    min_breakout_volume: float = 3.0  # Volume multiplier for breakouts
    min_range_size_pct: float = 1.0  # Minimum range size as % of price

    # Risk management
    stop_loss_pct: float = 1.5  # Stop loss as % from breakout level
    target_multiplier: float = 2.0  # Target as multiple of risk
    trailing_stop_pct: float = 1.5  # Trailing stop adjustment
    min_profit_for_trailing: float = 1.0  # Minimum profit % before trailing starts

    # Signal filtering
    min_confidence: float = 0.65
    min_volume_ratio: float = 1.5  # Current vs average volume
    max_gap_size: float = 3.0  # Max overnight gap to consider

    # Position management
    enable_trailing_stops: bool = True
    enable_partial_exits: bool = True
    partial_exit_pct: float = 50.0  # % to exit at first target

    # Fair Value Gap (FVG) settings
    enable_fvg_check: bool = False  # Enable/disable FVG filtering
    fvg_timeframe: str = "5"  # Timeframe for FVG detection (5min candles)
    fvg_lookback_candles: int = 20  # Number of candles to analyze for FVG
    fvg_min_gap_size_pct: float = 0.3  # Minimum gap size as % of price
    fvg_filter_mode: str = "STRICT"  # STRICT: block trades in FVG, LENIENT: reduce confidence, ALIGNED: only allow FVG-aligned trades

    # Momentum screening settings
    enable_momentum_filter: bool = True  # Enable/disable momentum-based stock filtering
    min_momentum_score: float = 50.0  # Minimum composite momentum score (0-100) to qualify
    momentum_top_n: int = 15  # Only consider top N momentum stocks for ORB trading
    momentum_lookback_days: int = 30  # Days of history for momentum calculation
    momentum_confidence_boost: float = 0.10  # Extra confidence added for high-momentum stocks (>75 score)


@dataclass
class TradingConfig:
    # Market hours (IST)
    market_start_hour: int = 9
    market_start_minute: int = 15
    market_end_hour: int = 15
    market_end_minute: int = 30

    # ORB specific timing
    orb_end_minute: int = 30  # ORB period ends at 9:30 AM
    signal_generation_end_hour: int = 15  # Stop generating signals at 2:00 PM
    signal_generation_end_minute: int = 0

    # Monitoring
    monitoring_interval: int = 1  # seconds
    position_update_interval: int = 5  # seconds