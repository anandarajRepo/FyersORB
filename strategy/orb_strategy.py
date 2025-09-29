# strategy/orb_strategy.py

"""
Open Range Breakout (ORB) Strategy Implementation - Updated with Centralized Symbol Management
Complete strategy with WebSocket integration, risk management, and performance tracking
Uses centralized symbol configuration - no hardcoded mappings or sector limitations
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from config.settings import FyersConfig, ORBStrategyConfig, TradingConfig, SignalType
from config.websocket_config import WebSocketConfig
from config.symbols import (
    symbol_manager, get_orb_symbols,
    validate_orb_symbol
)
from models.trading_models import (
    Position, ORBSignal, LiveQuote, OpenRange, TradeResult,
    StrategyMetrics, MarketState, create_orb_signal_from_symbol,
    create_position_from_signal, create_trade_result_from_position,
    get_category_summary, validate_signal_quality, calculate_portfolio_risk
)
from services.fyers_websocket_service import HybridORBDataService
from services.analysis_service import ORBTechnicalAnalysisService
from services.market_timing_service import MarketTimingService

logger = logging.getLogger(__name__)


class ORBStrategy:
    """Complete Open Range Breakout Strategy with Centralized Symbol Management"""

    def __init__(self, fyers_config: FyersConfig, strategy_config: ORBStrategyConfig,
                 trading_config: TradingConfig, ws_config: WebSocketConfig):

        # Configuration
        self.fyers_config = fyers_config
        self.strategy_config = strategy_config
        self.trading_config = trading_config
        self.ws_config = ws_config

        # Services
        self.data_service = HybridORBDataService(fyers_config, ws_config)
        self.analysis_service = ORBTechnicalAnalysisService(self.data_service)
        self.timing_service = MarketTimingService(trading_config)

        # Strategy state
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[TradeResult] = []
        self.metrics = StrategyMetrics()
        self.market_state = MarketState(timestamp=datetime.now())

        # ORB specific state
        self.orb_completed = False
        self.signals_generated_today = []
        self.daily_pnl = 0.0
        self.max_daily_loss = self.strategy_config.portfolio_value * 0.02  # 2% max daily loss

        # Get trading universe from centralized symbol manager
        self.trading_symbols = get_orb_symbols()
        logger.info(f"ORB Strategy initialized with {len(self.trading_symbols)} symbols from centralized manager")

        # Real-time data tracking
        self.live_quotes: Dict[str, LiveQuote] = {}
        self.opening_ranges: Dict[str, OpenRange] = {}

        # Add data callback
        self.data_service.add_data_callback(self._on_live_data_update)

    async def initialize(self) -> bool:
        """Initialize strategy and data connections"""
        try:
            logger.info("Initializing ORB Strategy with centralized symbol management...")

            # Connect to data service
            if not self.data_service.connect():
                logger.error("Failed to connect to data service")
                return False

            # Subscribe to all symbols from centralized manager
            symbols = get_orb_symbols()
            if not self.data_service.subscribe_symbols(symbols):
                logger.error("Failed to subscribe to symbols")
                return False

            # Initialize market state
            self._update_market_state()

            # Log strategy configuration
            logger.info(f"ORB Strategy initialized successfully:")
            logger.info(f"  Total symbols: {len(symbols)}")
            logger.info(f"  Max positions: {self.strategy_config.max_positions}")
            logger.info(f"  Risk per trade: {self.strategy_config.risk_per_trade_pct}%")
            category_dist = symbol_manager.get_category_distribution()
            readable_categories = {cat.value: count for cat, count in category_dist.items()}
            logger.info(f"  Symbol categories: {readable_categories}")
            # logger.info(f"  Symbol categories: {symbol_manager.get_category_distribution()}")
            logger.info("  Position allocation: Based on signal quality only (no category limits)")

            return True

        except Exception as e:
            logger.error(f"Strategy initialization failed: {e}")
            return False

    def _on_live_data_update(self, symbol: str, live_quote: LiveQuote):
        """Handle real-time data updates"""
        try:
            # Validate symbol using centralized manager
            if not validate_orb_symbol(symbol):
                logger.warning(f"Received data for invalid symbol: {symbol}")
                return

            # Update internal storage
            self.live_quotes[symbol] = live_quote

            # Update position tracking if we have a position
            if symbol in self.positions:
                self._update_position_tracking(symbol, live_quote)

            # Log significant movements during ORB period
            if self._is_orb_period():
                logger.debug(f"ORB Live: {symbol} - Rs.{live_quote.ltp:.2f} "
                             f"H:{live_quote.high_price:.2f} L:{live_quote.low_price:.2f}")

        except Exception as e:
            logger.error(f"Error handling live data update for {symbol}: {e}")

    def _update_position_tracking(self, symbol: str, live_quote: LiveQuote):
        """Update position tracking with current price"""
        try:
            position = self.positions[symbol]
            current_price = live_quote.ltp

            # Update price extremes for trailing stops
            position.update_price_extremes(current_price)

            # Calculate unrealized P&L
            if position.signal_type == SignalType.LONG:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)

            # Update trailing stop if enabled
            if self.strategy_config.enable_trailing_stops:
                new_stop = self.analysis_service.calculate_trailing_stop(
                    position.signal_type,
                    position.entry_price,
                    current_price,
                    position.highest_price,
                    position.lowest_price,
                    self.strategy_config.trailing_stop_pct
                )

                # Only update if trailing stop is better
                if position.signal_type == SignalType.LONG and new_stop > position.current_stop_loss:
                    position.current_stop_loss = new_stop
                    logger.info(f"Trailing stop updated for {symbol}: Rs.{new_stop:.2f}")
                elif position.signal_type == SignalType.SHORT and new_stop < position.current_stop_loss:
                    position.current_stop_loss = new_stop
                    logger.info(f"Trailing stop updated for {symbol}: Rs.{new_stop:.2f}")

        except Exception as e:
            logger.error(f"Error updating position tracking for {symbol}: {e}")

    async def run_strategy_cycle(self):
        """Main strategy execution cycle"""
        try:
            # Update market state
            self._update_market_state()

            # Check if we're in trading hours
            if not self.timing_service.is_trading_time():
                return

            # NEW: Check if ORB period just ended and mark as completed
            if not self.orb_completed and not self._is_orb_period():
                now = datetime.now()
                orb_end = now.replace(hour=9, minute=30, second=0, microsecond=0)
                if now > orb_end:
                    self.orb_completed = True
                    self.opening_ranges = self.data_service.get_all_opening_ranges()
                    logger.info(f"ORB period completed - {len(self.opening_ranges)} ranges finalized")

            # Monitor existing positions
            await self._monitor_positions()

            # Process ORB logic
            if self._is_orb_period():
                await self._process_orb_period()
            elif self._should_generate_signals():
                logger.info(f"Generating new signals: {self._should_generate_signals()}")
                await self._scan_for_breakouts()

            # Update strategy metrics
            self._update_strategy_metrics()

            # Log current status
            self._log_strategy_status()

        except Exception as e:
            logger.error(f"Error in strategy cycle: {e}")

    def _is_orb_period(self) -> bool:
        """Check if we're currently in ORB period"""
        now = datetime.now()
        orb_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        orb_end = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return orb_start <= now <= orb_end

    def _should_generate_signals(self) -> bool:
        """Check if we should generate new signals"""
        if not self.orb_completed:
            return False

        if len(self.positions) >= self.strategy_config.max_positions:
            return False

        # Check daily loss limit
        if self.daily_pnl < -abs(self.max_daily_loss):
            logger.warning("Daily loss limit reached - no new signals")
            return False

        # Only generate signals until 2 PM
        now = datetime.now()
        if now.hour >= 14:
            return False

        return True

    async def _process_orb_period(self):
        """Process data during ORB period"""
        try:
            # Update opening ranges from data service
            self.opening_ranges = self.data_service.get_all_opening_ranges()

            # Log ORB progress
            completed_ranges = len([r for r in self.opening_ranges.values() if r.range_size > 0])
            logger.debug(f"ORB Progress: {completed_ranges}/{len(self.trading_symbols)} ranges calculated")

        except Exception as e:
            logger.error(f"Error processing ORB period: {e}")

    async def _scan_for_breakouts(self):
        """Scan for breakout signals after ORB period"""
        try:
            # Mark ORB as completed if not already done
            if not self.orb_completed:
                self.orb_completed = True
                self.opening_ranges = self.data_service.get_all_opening_ranges()
                logger.info(f"ORB period completed - {len(self.opening_ranges)} ranges calculated")

            # Look for breakout signals
            new_signals = []

            for symbol in self.trading_symbols:
                # Skip if we already have a position in this symbol
                if symbol in self.positions:
                    continue

                # Get current data
                live_quote = self.live_quotes.get(symbol)
                opening_range = self.opening_ranges.get(symbol)

                if not live_quote or not opening_range:
                    continue

                # Check for breakouts
                is_breakout, signal_type, breakout_level = self.data_service.is_breakout_detected(
                    symbol, live_quote.ltp
                )

                logger.info(f"Breakout detected for {symbol}: {is_breakout}, {signal_type}, {breakout_level}, {opening_range}, {live_quote}")

                if is_breakout:
                    signal = await self._evaluate_breakout_signal(
                        symbol, signal_type, breakout_level, opening_range, live_quote
                    )
                    logger.info(f"Evaluating breakout signal for {symbol}: {signal}")

                    if signal:
                        new_signals.append(signal)
                        logger.info(f"Appending new signal for {symbol}: {signal}")

            # Sort signals by confidence and execute top ones
            new_signals.sort(key=lambda x: x.confidence, reverse=True)
            logger.info(f"Sorted signals: {new_signals}")

            # Execute signals up to position limit
            available_slots = self.strategy_config.max_positions - len(self.positions)
            logger.info(f"Available slots: {available_slots}")
            for signal in new_signals[:available_slots]:
                if validate_signal_quality(signal, self.strategy_config.min_confidence):
                    await self._execute_signal(signal)

        except Exception as e:
            logger.error(f"Error scanning for breakouts: {e}")

    async def _evaluate_breakout_signal(self, symbol: str, signal_type: str,
                                        breakout_level: float, opening_range: OpenRange,
                                        live_quote: LiveQuote) -> Optional[ORBSignal]:
        """Evaluate a potential breakout signal"""
        try:
            signal_type_enum = SignalType.LONG if signal_type == 'LONG' else SignalType.SHORT

            # Validate the breakout
            is_valid, confidence, quality_scores = self.analysis_service.validate_breakout_signal(
                symbol, opening_range, live_quote.ltp, signal_type_enum, self.strategy_config.min_confidence
            )

            if not is_valid:
                return None

            # Calculate entry parameters
            entry_price = live_quote.ltp
            stop_loss = self.analysis_service.calculate_stop_loss_level(
                signal_type_enum, breakout_level, opening_range, self.strategy_config.stop_loss_pct
            )
            target_price = self.analysis_service.calculate_target_price(
                signal_type_enum, entry_price, stop_loss, self.strategy_config.target_multiplier
            )

            # Calculate risk metrics
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = abs(target_price - entry_price)

            # Create signal using centralized symbol management
            signal = create_orb_signal_from_symbol(
                symbol=symbol,
                signal_type=signal_type_enum,
                breakout_price=breakout_level,
                range_high=opening_range.high,
                range_low=opening_range.low,
                range_size=opening_range.range_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                confidence=confidence,
                volume_ratio=quality_scores.get('volume_score', 0.0) * 3,  # Convert back to ratio
                breakout_volume=live_quote.volume,
                momentum_score=quality_scores.get('momentum_score', 0.0),
                timestamp=datetime.now(),
                orb_period_end=datetime.now().replace(hour=9, minute=30, second=0, microsecond=0),
                risk_amount=risk_amount,
                reward_amount=reward_amount
            )

            # Get symbol info for logging
            symbol_info = symbol_manager.get_symbol_info(symbol)
            category = symbol_info.category.value if symbol_info else "UNKNOWN"

            logger.info(f"ORB Signal: {symbol} ({category}) {signal_type_enum.value} - "
                        f"Entry: Rs.{entry_price:.2f}, SL: Rs.{stop_loss:.2f}, "
                        f"Target: Rs.{target_price:.2f}, Confidence: {confidence:.2f}")

            return signal

        except Exception as e:
            logger.error(f"Error evaluating breakout signal for {symbol}: {e}")
            return None

    async def _execute_signal(self, signal: ORBSignal) -> bool:
        """Execute a trading signal"""
        try:
            # Calculate position size
            quantity = self.analysis_service.calculate_position_size(
                self.strategy_config.portfolio_value,
                self.strategy_config.risk_per_trade_pct,
                signal.entry_price,
                signal.stop_loss
            )

            if quantity <= 0:
                logger.warning(f"Invalid quantity calculated for {signal.symbol}")
                return False

            # Create position using utility function
            position = create_position_from_signal(
                signal=signal,
                quantity=quantity if signal.signal_type == SignalType.LONG else -quantity,
                order_id=f"ORB_{signal.symbol}_{int(datetime.now().timestamp())}"
            )

            # Store position
            self.positions[signal.symbol] = position
            self.signals_generated_today.append(signal)

            logger.info(f"ORB Position Opened: {signal.symbol} ({signal.category.value}) {signal.signal_type.value} - "
                        f"Qty: {abs(quantity)}, Entry: Rs.{signal.entry_price:.2f}, "
                        f"Range: Rs.{signal.range_low:.2f}-{signal.range_high:.2f}")

            # In real implementation, place actual orders here
            # await self._place_entry_order(position)
            # await self._place_stop_loss_order(position)

            return True

        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False

    async def _monitor_positions(self):
        """Monitor existing positions for exits"""
        try:
            positions_to_close = []

            for symbol, position in self.positions.items():
                live_quote = self.live_quotes.get(symbol)
                if not live_quote:
                    continue

                current_price = live_quote.ltp

                # Check stop loss
                if self._should_exit_on_stop_loss(position, current_price):
                    positions_to_close.append((symbol, "STOP_LOSS", position.unrealized_pnl))

                # Check target
                elif self._should_exit_on_target(position, current_price):
                    if self.strategy_config.enable_partial_exits:
                        # Partial exit at target, move stop to breakeven
                        await self._partial_exit(position, current_price)
                    else:
                        positions_to_close.append((symbol, "TARGET", position.unrealized_pnl))

                # Check time-based exit (end of day)
                elif self._should_exit_on_time():
                    positions_to_close.append((symbol, "TIME_EXIT", position.unrealized_pnl))

            # Close positions
            for symbol, reason, pnl in positions_to_close:
                await self._close_position(symbol, reason, pnl)

        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")

    def _should_exit_on_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if position should be closed due to stop loss"""
        if position.signal_type == SignalType.LONG:
            return current_price <= position.current_stop_loss
        else:
            return current_price >= position.current_stop_loss

    def _should_exit_on_target(self, position: Position, current_price: float) -> bool:
        """Check if position should be closed due to target reached"""
        if position.signal_type == SignalType.LONG:
            return current_price >= position.target_price
        else:
            return current_price <= position.target_price

    def _should_exit_on_time(self) -> bool:
        """Check if positions should be closed due to time"""
        now = datetime.now()
        # Close all positions 15 minutes before market close
        market_close = now.replace(hour=15, minute=15, second=0, microsecond=0)
        return now >= market_close

    async def _partial_exit(self, position: Position, current_price: float):
        """Execute partial exit and move stop to breakeven"""
        try:
            # Calculate partial quantity (50% by default)
            partial_qty = int(abs(position.quantity) * self.strategy_config.partial_exit_pct / 100)

            if partial_qty > 0:
                # Calculate P&L for partial exit
                if position.signal_type == SignalType.LONG:
                    partial_pnl = (current_price - position.entry_price) * partial_qty
                else:
                    partial_pnl = (position.entry_price - current_price) * partial_qty

                # Update position quantity
                if position.quantity > 0:
                    position.quantity -= partial_qty
                else:
                    position.quantity += partial_qty

                # Move stop to breakeven
                position.current_stop_loss = position.entry_price

                # Update daily P&L
                self.daily_pnl += partial_pnl

                logger.info(f"Partial Exit: {position.symbol} - Qty: {partial_qty}, "
                            f"P&L: Rs.{partial_pnl:.2f}, Stop moved to breakeven")

        except Exception as e:
            logger.error(f"Error in partial exit for {position.symbol}: {e}")

    async def _close_position(self, symbol: str, reason: str, pnl: float):
        """Close a position and record the trade"""
        try:
            position = self.positions[symbol]
            current_price = self.live_quotes[symbol].ltp

            # Create trade result using utility function
            trade_result = create_trade_result_from_position(position, current_price, reason)

            # Store completed trade
            self.completed_trades.append(trade_result)

            # Update daily P&L
            self.daily_pnl += trade_result.net_pnl

            # Remove position
            del self.positions[symbol]

            logger.info(f"Position Closed: {symbol} - {reason} - P&L: Rs.{trade_result.net_pnl:.2f}")

            # In real implementation, place closing orders here
            # await self._place_exit_order(position, current_price)

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")

    def _update_market_state(self):
        """Update current market state"""
        try:
            now = datetime.now()

            # Update basic state
            self.market_state.timestamp = now
            self.market_state.orb_period_active = self._is_orb_period()
            self.market_state.signal_generation_active = self._should_generate_signals()
            self.market_state.max_positions_reached = len(self.positions) >= self.strategy_config.max_positions
            self.market_state.daily_loss_limit_hit = self.daily_pnl < -abs(self.max_daily_loss)

            # Update market trend (placeholder - could be enhanced with index analysis)
            self.market_state.market_trend = "NEUTRAL"
            self.market_state.volatility_regime = "NORMAL"

        except Exception as e:
            logger.error(f"Error updating market state: {e}")

    def _update_strategy_metrics(self):
        """Update strategy performance metrics"""
        try:
            self.metrics.update_metrics(self.completed_trades)
            self.metrics.daily_pnl = self.daily_pnl

            # Calculate unrealized P&L
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())

            # Update recent performance
            today_trades = [t for t in self.completed_trades
                            if t.exit_time.date() == datetime.now().date()]
            self.metrics.daily_pnl = sum(t.net_pnl for t in today_trades) + total_unrealized

        except Exception as e:
            logger.error(f"Error updating strategy metrics: {e}")

    def _log_strategy_status(self):
        """Log current strategy status"""
        try:
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())

            # Count positions by type
            long_positions = sum(1 for p in self.positions.values() if p.signal_type == SignalType.LONG)
            short_positions = len(self.positions) - long_positions

            # Count positions by category (for informational purposes)
            category_summary = get_category_summary(list(self.positions.values()))

            # Calculate portfolio risk
            risk_metrics = calculate_portfolio_risk(list(self.positions.values()), self.strategy_config.portfolio_value)

            logger.info(f"ORB Strategy Status:")
            logger.info(f"  Positions: {len(self.positions)}/{self.strategy_config.max_positions} "
                        f"(Long: {long_positions}, Short: {short_positions})")
            logger.info(f"  Daily P&L: Rs.{self.daily_pnl:.2f}")
            logger.info(f"  Unrealized P&L: Rs.{total_unrealized:.2f}")
            logger.info(f"  Total P&L: Rs.{self.metrics.total_pnl:.2f}")
            logger.info(f"  ORB Completed: {self.orb_completed}")
            logger.info(f"  Signals Today: {len(self.signals_generated_today)}")
            logger.info(f"  Portfolio Risk: {risk_metrics['risk_percentage']:.1f}%")

            if category_summary:
                category_info = ", ".join([f"{cat.value}: {info['count']}"
                                           for cat, info in category_summary.items()])
                logger.info(f"  Category Distribution: {category_info}")

        except Exception as e:
            logger.error(f"Error logging strategy status: {e}")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())

            # Position details
            position_details = []
            for symbol, pos in self.positions.items():
                current_quote = self.live_quotes.get(symbol)
                current_price = current_quote.ltp if current_quote else 0

                # Get symbol info for display
                symbol_info = symbol_manager.get_symbol_info(symbol)

                position_details.append({
                    'symbol': symbol,
                    'company_name': symbol_info.company_name if symbol_info else symbol,
                    'category': pos.category.value,
                    'signal_type': pos.signal_type.value,
                    'entry_price': pos.entry_price,
                    'current_price': current_price,
                    'quantity': pos.quantity,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'entry_time': pos.entry_time.strftime('%H:%M:%S'),
                    'stop_loss': pos.current_stop_loss,
                    'target_price': pos.target_price,
                    'breakout_price': pos.breakout_price,
                    'range_size': pos.range_high - pos.range_low,
                })

            # Opening ranges summary
            orb_summary = []
            for symbol, orb_range in self.opening_ranges.items():
                symbol_info = symbol_manager.get_symbol_info(symbol)
                orb_summary.append({
                    'symbol': symbol,
                    'company_name': symbol_info.company_name if symbol_info else symbol,
                    'range_high': orb_range.high,
                    'range_low': orb_range.low,
                    'range_size': orb_range.range_size,
                    'range_pct': orb_range.range_pct,
                    'volume': orb_range.volume
                })

            # Portfolio risk analysis
            risk_metrics = calculate_portfolio_risk(list(self.positions.values()), self.strategy_config.portfolio_value)

            # Category distribution
            category_summary = get_category_summary(list(self.positions.values()))

            return {
                'strategy_name': 'Open Range Breakout (ORB) - Centralized Symbol Management',
                'timestamp': datetime.now().isoformat(),
                'total_pnl': self.metrics.total_pnl,
                'daily_pnl': self.daily_pnl,
                'unrealized_pnl': total_unrealized,
                'active_positions': len(self.positions),
                'max_positions': self.strategy_config.max_positions,
                'completed_trades': len(self.completed_trades),
                'win_rate': self.metrics.win_rate,
                'orb_completed': self.orb_completed,
                'signals_generated_today': len(self.signals_generated_today),
                'websocket_connected': self.data_service.is_connected,
                'using_fallback': getattr(self.data_service, 'using_fallback', False),
                'symbol_management': {
                    'total_universe_size': symbol_manager.get_trading_universe_size(),
                    'category_distribution': symbol_manager.get_category_distribution(),
                    'centralized_management': True,
                    'no_category_limits': True
                },
                'risk_metrics': risk_metrics,
                'category_summary': {cat.value: info for cat, info in category_summary.items()},
                'market_state': {
                    'orb_period_active': self.market_state.orb_period_active,
                    'signal_generation_active': self.market_state.signal_generation_active,
                    'max_positions_reached': self.market_state.max_positions_reached,
                    'daily_loss_limit_hit': self.market_state.daily_loss_limit_hit,
                },
                'opening_ranges': orb_summary,
                'positions': position_details,
                'performance_metrics': {
                    'total_trades': self.metrics.total_trades,
                    'winning_trades': self.metrics.winning_trades,
                    'losing_trades': self.metrics.losing_trades,
                    'long_trades': self.metrics.long_trades,
                    'short_trades': self.metrics.short_trades,
                    'long_win_rate': self.metrics.long_win_rate,
                    'short_win_rate': self.metrics.short_win_rate,
                    'avg_holding_period': self.metrics.avg_holding_period,
                }
            }

        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}

    def get_symbol_analysis(self) -> Dict:
        """Get analysis of symbol performance and usage"""
        try:
            # Analyze completed trades by symbol
            symbol_performance = {}
            for trade in self.completed_trades:
                if trade.symbol not in symbol_performance:
                    symbol_performance[trade.symbol] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0.0,
                        'category': trade.category.value
                    }

                perf = symbol_performance[trade.symbol]
                perf['trades'] += 1
                perf['total_pnl'] += trade.net_pnl
                if trade.net_pnl > 0:
                    perf['wins'] += 1

            # Calculate win rates and add symbol info
            for symbol, perf in symbol_performance.items():
                perf['win_rate'] = (perf['wins'] / perf['trades']) * 100 if perf['trades'] > 0 else 0

                # Add symbol info
                symbol_info = symbol_manager.get_symbol_info(symbol)
                if symbol_info:
                    perf['company_name'] = symbol_info.company_name
                    perf['fyers_symbol'] = symbol_info.fyers_symbol

            return {
                'symbol_performance': symbol_performance,
                'total_symbols_traded': len(symbol_performance),
                'total_universe_size': symbol_manager.get_trading_universe_size(),
                'utilization_rate': len(symbol_performance) / symbol_manager.get_trading_universe_size() * 100
            }

        except Exception as e:
            logger.error(f"Error in symbol analysis: {e}")
            return {'error': str(e)}

    def validate_trading_universe(self) -> Dict:
        """Validate current trading universe against centralized manager"""
        try:
            validation_result = {
                'valid': True,
                'universe_size': len(self.trading_symbols),
                'manager_size': symbol_manager.get_trading_universe_size(),
                'invalid_symbols': [],
                'missing_symbols': [],
                'sync_status': 'unknown'
            }

            # Check if our symbols are valid
            for symbol in self.trading_symbols:
                if not validate_orb_symbol(symbol):
                    validation_result['invalid_symbols'].append(symbol)
                    validation_result['valid'] = False

            # Check if we're missing any symbols from the manager
            manager_symbols = get_orb_symbols()
            for symbol in manager_symbols:
                if symbol not in self.trading_symbols:
                    validation_result['missing_symbols'].append(symbol)

            # Determine sync status
            if validation_result['universe_size'] == validation_result['manager_size']:
                if not validation_result['invalid_symbols'] and not validation_result['missing_symbols']:
                    validation_result['sync_status'] = 'perfectly_synced'
                else:
                    validation_result['sync_status'] = 'size_match_but_differences'
            else:
                validation_result['sync_status'] = 'size_mismatch'

            return validation_result

        except Exception as e:
            logger.error(f"Error validating trading universe: {e}")
            return {'error': str(e), 'valid': False}

    def reset_daily_data(self):
        """Reset daily data for new trading day"""
        try:
            self.orb_completed = False
            self.signals_generated_today.clear()
            self.daily_pnl = 0.0
            self.opening_ranges.clear()

            # Reset data service daily data
            if hasattr(self.data_service, 'reset_daily_data'):
                self.data_service.reset_daily_data()

            logger.info("Daily ORB strategy data reset completed")

        except Exception as e:
            logger.error(f"Error resetting daily data: {e}")

    async def run(self):
        """Main strategy execution loop"""
        logger.info("Starting Open Range Breakout Strategy with Centralized Symbol Management")
        logger.info(f"Trading Universe: {symbol_manager.get_trading_universe_size()} symbols")

        if not await self.initialize():
            logger.error("Strategy initialization failed")
            return

        try:
            while True:
                # Check if we should reset for new day
                now = datetime.now()
                if now.hour == 9 and now.minute < 15:
                    # Reset daily data before market opens
                    self.reset_daily_data()

                # Check trading hours
                if not self.timing_service.is_trading_time():
                    logger.debug("Outside trading hours, sleeping...")
                    await asyncio.sleep(300)  # 5 minutes
                    continue

                # Run strategy cycle
                await self.run_strategy_cycle()

                # Sleep until next cycle (more frequent during ORB period)
                if self._is_orb_period():
                    await asyncio.sleep(5)  # 5 seconds during ORB
                else:
                    await asyncio.sleep(self.trading_config.monitoring_interval)

        except KeyboardInterrupt:
            logger.info("ORB Strategy stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in ORB strategy: {e}")
        finally:
            # Cleanup
            self.data_service.disconnect()
            logger.info("ORB Strategy disconnected and cleanup completed")

    def get_data_service_info(self) -> Dict:
        """Get information about data service configuration"""
        try:
            service_info = {
                'service_type': type(self.data_service).__name__,
                'is_connected': self.data_service.is_connected,
                'subscribed_symbols_count': len(getattr(self.data_service, 'subscribed_symbols', [])),
                'symbol_management': 'Centralized Symbol Manager',
                'symbol_validation': 'Automatic via centralized manager'
            }

            # Add service-specific info if available
            if hasattr(self.data_service, 'get_service_info'):
                service_info.update(self.data_service.get_service_info())

            return service_info

        except Exception as e:
            logger.error(f"Error getting data service info: {e}")
            return {'error': str(e)}