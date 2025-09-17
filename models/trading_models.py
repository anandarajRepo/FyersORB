# models/trading_models.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from config.settings import Sector, SignalType


@dataclass
class LiveQuote:
    """Real-time quote data from WebSocket"""
    symbol: str
    ltp: float  # Last traded price
    open_price: float
    high_price: float
    low_price: float
    volume: int
    previous_close: float
    timestamp: datetime
    change: float = 0.0
    change_pct: float = 0.0

    def __post_init__(self):
        if self.previous_close > 0:
            self.change = self.ltp - self.previous_close
            self.change_pct = (self.change / self.previous_close) * 100


@dataclass
class OpenRange:
    """Opening range data for ORB strategy"""
    symbol: str
    high: float
    low: float
    range_size: float
    range_pct: float
    volume: int
    start_time: datetime
    end_time: datetime

    def __post_init__(self):
        self.range_size = self.high - self.low
        if self.low > 0:
            self.range_pct = (self.range_size / self.low) * 100


@dataclass
class ORBSignal:
    """Open Range Breakout Signal"""
    symbol: str
    sector: Sector
    signal_type: SignalType  # LONG or SHORT
    breakout_price: float
    range_high: float
    range_low: float
    range_size: float

    # Entry parameters
    entry_price: float
    stop_loss: float
    target_price: float

    # Signal quality metrics
    confidence: float
    volume_ratio: float
    breakout_volume: int
    momentum_score: float

    # Timing
    timestamp: datetime
    orb_period_end: datetime

    # Risk metrics
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float = field(init=False)

    def __post_init__(self):
        if self.risk_amount > 0:
            self.risk_reward_ratio = self.reward_amount / self.risk_amount


@dataclass
class Position:
    """Trading position with ORB-specific features"""
    symbol: str
    sector: Sector
    signal_type: SignalType

    # Position details
    entry_price: float
    quantity: int
    stop_loss: float
    target_price: float

    # ORB specific
    breakout_price: float
    range_high: float
    range_low: float

    # Timing
    entry_time: datetime
    orb_signal_time: datetime

    # Tracking
    highest_price: float = 0.0  # For trailing stops
    lowest_price: float = 0.0  # For trailing stops
    current_stop_loss: float = 0.0  # Dynamic stop loss

    # Orders
    order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    target_order_id: Optional[str] = None

    # Performance
    unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    def __post_init__(self):
        self.current_stop_loss = self.stop_loss
        self.highest_price = self.entry_price if self.signal_type == SignalType.LONG else 0.0
        self.lowest_price = self.entry_price if self.signal_type == SignalType.SHORT else float('inf')

    def update_price_extremes(self, current_price: float):
        """Update price extremes for trailing stop calculation"""
        if self.signal_type == SignalType.LONG:
            self.highest_price = max(self.highest_price, current_price)

            # Calculate favorable and adverse excursion
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                current_price - self.entry_price
            )
            self.max_adverse_excursion = min(
                self.max_adverse_excursion,
                current_price - self.entry_price
            )
        else:  # SHORT
            self.lowest_price = min(self.lowest_price, current_price)

            # Calculate favorable and adverse excursion
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                self.entry_price - current_price
            )
            self.max_adverse_excursion = min(
                self.max_adverse_excursion,
                self.entry_price - current_price
            )


@dataclass
class TradeResult:
    """Completed trade result"""
    symbol: str
    sector: Sector
    signal_type: SignalType

    # Trade details
    entry_price: float
    exit_price: float
    quantity: int

    # Timing
    entry_time: datetime
    exit_time: datetime
    holding_period: float  # in minutes

    # ORB specific
    breakout_price: float
    range_size: float

    # Performance
    gross_pnl: float
    net_pnl: float = field(init=False)

    # Metrics
    exit_reason: str  # "TARGET", "STOP_LOSS", "TRAILING_STOP", "TIME_EXIT"
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Performance
    commission: float = 0.0

    def __post_init__(self):
        self.net_pnl = self.gross_pnl - self.commission
        self.holding_period = (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_portfolio_risk: float = 0.0

    # ORB specific
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    # Timing metrics
    avg_holding_period: float = 0.0  # minutes
    avg_time_to_signal: float = 0.0  # minutes after ORB period

    # Recent performance
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    def update_metrics(self, trades: List[TradeResult]):
        """Update metrics from trade list"""
        if not trades:
            return

        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        self.losing_trades = self.total_trades - self.winning_trades
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0

        self.total_pnl = sum(t.net_pnl for t in trades)
        self.gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        self.gross_loss = sum(t.net_pnl for t in trades if t.net_pnl < 0)

        # ORB specific metrics
        long_trades_list = [t for t in trades if t.signal_type == SignalType.LONG]
        short_trades_list = [t for t in trades if t.signal_type == SignalType.SHORT]

        self.long_trades = len(long_trades_list)
        self.short_trades = len(short_trades_list)

        if self.long_trades > 0:
            self.long_win_rate = (sum(1 for t in long_trades_list if t.net_pnl > 0) / self.long_trades) * 100

        if self.short_trades > 0:
            self.short_win_rate = (sum(1 for t in short_trades_list if t.net_pnl > 0) / self.short_trades) * 100

        # Timing metrics
        if trades:
            self.avg_holding_period = sum(t.holding_period for t in trades) / len(trades)


@dataclass
class MarketState:
    """Current market state for strategy decisions"""
    timestamp: datetime

    # Market indicators
    nifty_change_pct: float = 0.0
    market_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    volatility_regime: str = "NORMAL"  # HIGH, NORMAL, LOW

    # Volume indicators
    market_volume_ratio: float = 1.0  # Current vs average

    # ORB specific market state
    orb_period_active: bool = False
    signal_generation_active: bool = True

    # Risk indicators
    max_positions_reached: bool = False
    daily_loss_limit_hit: bool = False