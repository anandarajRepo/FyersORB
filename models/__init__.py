# models/__init__.py

"""
Trading Models Package for ORB Strategy
Contains all data models, trading signals, positions, and performance tracking classes
Updated to use centralized symbol management
"""

from .trading_models import (
    # Core data models
    LiveQuote,
    OpenRange,

    # Signal and position models
    ORBSignal,
    Position,
    TradeResult,

    # Performance and metrics
    StrategyMetrics,
    MarketState,

    # Utility functions
    create_orb_signal_from_symbol,
    create_position_from_signal,
    create_trade_result_from_position,
    get_category_summary,
    validate_signal_quality,
    calculate_portfolio_risk
)

from datetime import datetime
from typing import List, Dict, Optional
from config.settings import SignalType
from config.symbols import symbol_manager

__version__ = "2.0.0"
__author__ = "ORB Trading Strategy Team"

# Package exports
__all__ = [
    # Core data models
    "LiveQuote",
    "OpenRange",

    # Trading models
    "ORBSignal",
    "Position",
    "TradeResult",

    # Analytics models
    "StrategyMetrics",
    "MarketState",

    # Utility functions
    "create_sample_quote",
    "create_sample_opening_range",
    "create_sample_orb_signal",
    "create_sample_position",
    "calculate_position_pnl",
    "validate_trade_result",
    "create_orb_signal_from_symbol",
    "create_position_from_signal",
    "create_trade_result_from_position",
    "get_category_summary",
    "validate_signal_quality",
    "calculate_portfolio_risk"
]


# Utility functions for creating and validating models (UPDATED)
def create_sample_quote(symbol: str, price: float = 100.0) -> LiveQuote:
    """Create a sample LiveQuote for testing"""
    return LiveQuote(
        symbol=symbol,
        ltp=price,
        open_price=price * 0.98,
        high_price=price * 1.02,
        low_price=price * 0.97,
        volume=10000,
        previous_close=price * 0.99,
        timestamp=datetime.now()
    )


def create_sample_opening_range(symbol: str, base_price: float = 100.0) -> OpenRange:
    """Create a sample OpenRange for testing"""
    high = base_price * 1.01
    low = base_price * 0.99

    return OpenRange(
        symbol=symbol,
        high=high,
        low=low,
        range_size=high - low,
        range_pct=((high - low) / low) * 100,
        volume=50000,
        start_time=datetime.now().replace(hour=9, minute=15),
        end_time=datetime.now().replace(hour=9, minute=20)
    )


def create_sample_orb_signal(
        symbol: str,
        signal_type: SignalType = SignalType.LONG,
        entry_price: float = 100.0,
        confidence: float = 0.8
) -> ORBSignal:
    """Create a sample ORB signal for testing - UPDATED to use centralized symbols"""
    opening_range = create_sample_opening_range(symbol, entry_price * 0.99)

    if signal_type == SignalType.LONG:
        breakout_price = opening_range.high
        stop_loss = breakout_price * 0.99
        target_price = breakout_price * 1.02
    else:
        breakout_price = opening_range.low
        stop_loss = breakout_price * 1.01
        target_price = breakout_price * 0.98

    # Get category from centralized symbol manager
    category = symbol

    return ORBSignal(
        symbol=symbol,
        category=category,  # UPDATED: Use category instead of sector
        signal_type=signal_type,
        breakout_price=breakout_price,
        range_high=opening_range.high,
        range_low=opening_range.low,
        range_size=opening_range.range_size,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        confidence=confidence,
        volume_ratio=2.5,
        breakout_volume=75000,
        momentum_score=0.8,
        timestamp=datetime.now(),
        orb_period_end=datetime.now().replace(hour=9, minute=20),
        risk_amount=abs(entry_price - stop_loss),
        reward_amount=abs(target_price - entry_price)
    )


def create_sample_position(
        symbol: str,
        signal_type: SignalType = SignalType.LONG,
        entry_price: float = 100.0,
        quantity: int = 100
) -> Position:
    """Create a sample position for testing - UPDATED to use centralized symbols"""
    signal = create_sample_orb_signal(symbol, signal_type, entry_price)

    return Position(
        symbol=symbol,
        category=signal.category,  # UPDATED: Use category instead of sector
        signal_type=signal_type,
        entry_price=entry_price,
        quantity=quantity if signal_type == SignalType.LONG else -quantity,
        stop_loss=signal.stop_loss,
        target_price=signal.target_price,
        breakout_price=signal.breakout_price,
        range_high=signal.range_high,
        range_low=signal.range_low,
        entry_time=datetime.now(),
        orb_signal_time=signal.timestamp,
        order_id=f"TEST_{symbol}_{int(datetime.now().timestamp())}"
    )


def calculate_position_pnl(position: Position, current_price: float) -> float:
    """Calculate current P&L for a position"""
    if position.quantity > 0:  # Long position
        return (current_price - position.entry_price) * position.quantity
    else:  # Short position
        return (position.entry_price - current_price) * abs(position.quantity)


def validate_live_quote(quote: LiveQuote) -> bool:
    """Validate a LiveQuote object"""
    if not quote.symbol:
        return False
    if quote.ltp <= 0:
        return False
    if quote.volume < 0:
        return False
    if quote.previous_close <= 0:
        return False
    return True


def validate_orb_signal(signal: ORBSignal) -> bool:
    """Validate an ORB signal"""
    if not signal.symbol:
        return False
    if signal.confidence < 0 or signal.confidence > 1:
        return False
    if signal.entry_price <= 0:
        return False
    if signal.range_size <= 0:
        return False
    if signal.risk_amount <= 0:
        return False
    return True


def validate_trade_result(trade: TradeResult) -> bool:
    """Validate a trade result"""
    if not trade.symbol:
        return False
    if trade.quantity <= 0:
        return False
    if trade.entry_price <= 0 or trade.exit_price <= 0:
        return False
    if trade.entry_time >= trade.exit_time:
        return False
    return True


# Data conversion utilities (UPDATED)
def quote_to_dict(quote: LiveQuote) -> Dict:
    """Convert LiveQuote to dictionary"""
    return {
        'symbol': quote.symbol,
        'ltp': quote.ltp,
        'open_price': quote.open_price,
        'high_price': quote.high_price,
        'low_price': quote.low_price,
        'volume': quote.volume,
        'previous_close': quote.previous_close,
        'change': quote.change,
        'change_pct': quote.change_pct,
        'timestamp': quote.timestamp.isoformat()
    }


def signal_to_dict(signal: ORBSignal) -> Dict:
    """Convert ORBSignal to dictionary - UPDATED"""
    return {
        'symbol': signal.symbol,
        'category': signal.category.value,  # UPDATED: Use category instead of sector
        'signal_type': signal.signal_type.value,
        'breakout_price': signal.breakout_price,
        'entry_price': signal.entry_price,
        'stop_loss': signal.stop_loss,
        'target_price': signal.target_price,
        'confidence': signal.confidence,
        'volume_ratio': signal.volume_ratio,
        'momentum_score': signal.momentum_score,
        'risk_reward_ratio': signal.risk_reward_ratio,
        'timestamp': signal.timestamp.isoformat()
    }


def position_to_dict(position: Position) -> Dict:
    """Convert Position to dictionary - UPDATED"""
    return {
        'symbol': position.symbol,
        'category': position.category.value,  # UPDATED: Use category instead of sector
        'signal_type': position.signal_type.value,
        'entry_price': position.entry_price,
        'quantity': position.quantity,
        'stop_loss': position.current_stop_loss,
        'target_price': position.target_price,
        'unrealized_pnl': position.unrealized_pnl,
        'entry_time': position.entry_time.isoformat(),
        'order_id': position.order_id
    }


# Model factory class (UPDATED)
class ModelFactory:
    """Factory for creating model instances"""

    @staticmethod
    def create_test_portfolio() -> List[Position]:
        """Create a test portfolio with sample positions - UPDATED"""
        symbols = ['TCS', 'INFY', 'HDFCBANK', 'RELIANCE']  # Use display symbols
        positions = []

        for i, symbol in enumerate(symbols):
            signal_type = SignalType.LONG if i % 2 == 0 else SignalType.SHORT
            position = create_sample_position(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=100 + i * 10,
                quantity=100 + i * 50
            )
            positions.append(position)

        return positions

    @staticmethod
    def create_test_signals() -> List[ORBSignal]:
        """Create test signals for different scenarios - UPDATED"""
        signals = []

        # High confidence long signal
        signals.append(create_sample_orb_signal(
            'TCS', SignalType.LONG, 3500.0, 0.9
        ))

        # Medium confidence short signal
        signals.append(create_sample_orb_signal(
            'HDFCBANK', SignalType.SHORT, 1600.0, 0.7
        ))

        # Low confidence signal (should be filtered out)
        signals.append(create_sample_orb_signal(
            'RELIANCE', SignalType.LONG, 2400.0, 0.4
        ))

        return signals


# Export factory instance
model_factory = ModelFactory()


# Statistics and analysis utilities (UPDATED)
def calculate_portfolio_metrics(positions: List[Position], current_quotes: Dict[str, LiveQuote]) -> Dict:
    """Calculate portfolio-level metrics"""
    total_unrealized = 0.0
    total_invested = 0.0
    long_positions = 0
    short_positions = 0
    category_distribution = {}

    for position in positions:
        total_invested += abs(position.entry_price * position.quantity)

        if position.quantity > 0:
            long_positions += 1
        else:
            short_positions += 1

        # Track category distribution
        category = position.category.value
        category_distribution[category] = category_distribution.get(category, 0) + 1

        # Calculate unrealized P&L if quote available
        if position.symbol in current_quotes:
            current_price = current_quotes[position.symbol].ltp
            total_unrealized += calculate_position_pnl(position, current_price)

    return {
        'total_positions': len(positions),
        'long_positions': long_positions,
        'short_positions': short_positions,
        'total_invested': total_invested,
        'total_unrealized_pnl': total_unrealized,
        'unrealized_pnl_pct': (total_unrealized / total_invested * 100) if total_invested > 0 else 0,
        'category_distribution': category_distribution  # UPDATED: Show category distribution
    }


def analyze_trade_results(trades: List[TradeResult]) -> Dict:
    """Analyze completed trade results"""
    if not trades:
        return {'total_trades': 0}

    winning_trades = [t for t in trades if t.net_pnl > 0]
    losing_trades = [t for t in trades if t.net_pnl <= 0]

    total_pnl = sum(t.net_pnl for t in trades)
    avg_holding_period = sum(t.holding_period for t in trades) / len(trades)

    # Category analysis
    category_performance = {}
    for trade in trades:
        category = trade.category.value
        if category not in category_performance:
            category_performance[category] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}

        category_performance[category]['trades'] += 1
        category_performance[category]['total_pnl'] += trade.net_pnl
        if trade.net_pnl > 0:
            category_performance[category]['wins'] += 1

    # Calculate win rates by category
    for category_data in category_performance.values():
        category_data['win_rate'] = (category_data['wins'] / category_data['trades']) * 100

    return {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) * 100,
        'total_pnl': total_pnl,
        'avg_win': sum(t.net_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
        'avg_loss': sum(t.net_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        'avg_holding_period': avg_holding_period,
        'profit_factor': abs(sum(t.net_pnl for t in winning_trades) / sum(t.net_pnl for t in losing_trades)) if losing_trades else float('inf'),
        'category_performance': category_performance  # UPDATED: Category-wise performance
    }


# Helper function to get symbol info for display
def get_symbol_display_info(symbol: str) -> Dict:
    """Get display information for a symbol"""
    from config.symbols import convert_to_fyers_format

    return {
        'display_symbol': symbol,
        'company_name': symbol,
        'category': 'GENERAL',
        'fyers_symbol': convert_to_fyers_format(symbol) or f'NSE:{symbol}-EQ'
    }


# Add the helper function to exports
__all__.append("get_symbol_display_info")