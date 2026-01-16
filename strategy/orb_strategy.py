# strategy/orb_strategy.py

"""
Open Range Breakout (ORB) Strategy Implementation - Event-Based with State Tracking
Complete strategy with WebSocket integration, risk management, and performance tracking
Uses centralized symbol configuration and transition detection for efficient signal generation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum

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
from strategy.order_manager import OrderManager

logger = logging.getLogger(__name__)


# NEW: Breakout state enumeration
class BreakoutState(Enum):
    """Enumeration for symbol breakout states"""
    NO_BREAKOUT = "NO_BREAKOUT"  # Price within range
    UPSIDE_BREAKOUT = "UPSIDE_BREAKOUT"  # Price broken above range
    DOWNSIDE_BREAKOUT = "DOWNSIDE_BREAKOUT"  # Price broken below range


class ORBStrategy:
    """Complete Open Range Breakout Strategy with Event-Based Transition Detection"""

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

        # ADD THIS LINE:
        self.order_manager = OrderManager(fyers_config)

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

        # ===== NEW: State tracking for breakout transitions =====
        # Track breakout state for each symbol
        self.breakout_states: Dict[str, BreakoutState] = {}

        # Track symbols that have new breakout transitions (event queue)
        self.pending_breakout_symbols: Set[str] = set()

        # Initialize all symbols to NO_BREAKOUT state
        for symbol in self.trading_symbols:
            self.breakout_states[symbol] = BreakoutState.NO_BREAKOUT

        logger.info(f"Initialized breakout state tracking for {len(self.trading_symbols)} symbols")
        # ========================================================

        # Add data callback
        self.data_service.add_data_callback(self._on_live_data_update)

    async def initialize(self) -> bool:
        """Initialize strategy and data connections"""
        try:
            logger.info("Initializing ORB Strategy with event-based transition detection...")

            # ADD THIS - Verify broker connection
            if not self.order_manager.verify_broker_connection():
                logger.error("Failed to verify broker connection")
                return False

            logger.info("Broker connection verified")

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
            logger.info(f"  Signal processing: Event-based (transition detection)")
            logger.info("  Position allocation: Based on signal quality only (no category limits)")

            return True

        except Exception as e:
            logger.error(f"Strategy initialization failed: {e}")
            return False

    def _on_live_data_update(self, symbol: str, live_quote: LiveQuote):
        """
        Handle real-time data updates with transition detection
        NEW: Detects breakout state transitions and queues symbols for signal generation
        """
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

            # ===== NEW: Transition detection logic =====
            # Only detect transitions after ORB period completes
            if self.orb_completed and symbol in self.opening_ranges:
                self._detect_breakout_transition(symbol, live_quote)
            # ===========================================

            # Log significant movements during ORB period
            if self._is_orb_period():
                logger.debug(f"ORB Live: {symbol} - Rs.{live_quote.ltp:.2f} "
                             f"H:{live_quote.high_price:.2f} L:{live_quote.low_price:.2f}")

        except Exception as e:
            logger.error(f"Error handling live data update for {symbol}: {e}")

    def _detect_breakout_transition(self, symbol: str, live_quote: LiveQuote):
        """
        NEW: Detect if a symbol transitions between breakout states
        Only queues symbols when a NEW transition occurs (not already broken out)
        """
        try:
            # Skip if we already have a position in this symbol
            if symbol in self.positions:
                return

            # Get current breakout state
            current_state = self.breakout_states.get(symbol, BreakoutState.NO_BREAKOUT)

            # Get opening range
            opening_range = self.opening_ranges.get(symbol)
            if not opening_range:
                return

            # Determine new state based on current price
            current_price = live_quote.ltp
            new_state = BreakoutState.NO_BREAKOUT

            if current_price > opening_range.high:
                new_state = BreakoutState.UPSIDE_BREAKOUT
            elif current_price < opening_range.low:
                new_state = BreakoutState.DOWNSIDE_BREAKOUT
            else:
                new_state = BreakoutState.NO_BREAKOUT

            # Check for state transition
            if new_state != current_state:
                # State changed - this is a transition event
                logger.info(f"Breakout Transition: {symbol} {current_state.value} → {new_state.value} "
                            f"(Price: Rs.{current_price:.2f}, Range: Rs.{opening_range.low:.2f}-{opening_range.high:.2f})")

                # Update state
                self.breakout_states[symbol] = new_state

                # Queue symbol for signal generation if it's a breakout transition
                if new_state in [BreakoutState.UPSIDE_BREAKOUT, BreakoutState.DOWNSIDE_BREAKOUT]:
                    self.pending_breakout_symbols.add(symbol)
                    logger.info(f"Queued {symbol} for signal evaluation (new {new_state.value})")
                elif new_state == BreakoutState.NO_BREAKOUT:
                    # Price came back into range - remove from pending queue
                    self.pending_breakout_symbols.discard(symbol)
                    logger.debug(f"{symbol} returned to range, removed from queue")

        except Exception as e:
            logger.error(f"Error detecting breakout transition for {symbol}: {e}")

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

                # Only update if trailing stop is better AND position is profitable
                should_modify = False
                if position.signal_type == SignalType.LONG:
                    # For LONG: trail only if price is above entry and new stop is better
                    if current_price > position.entry_price and new_stop > position.current_stop_loss:
                        should_modify = True
                elif position.signal_type == SignalType.SHORT:
                    # For SHORT: trail only if price is below entry and new stop is better
                    if current_price < position.entry_price and new_stop < position.current_stop_loss:
                        should_modify = True

                if should_modify:
                    # Queue modification for processing in async monitoring cycle
                    position.pending_stop_modification = new_stop
                    logger.info(f"Trailing stop update queued for {symbol}: "
                                f"Rs.{position.current_stop_loss:.2f} → Rs.{new_stop:.2f}")

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

            # Check if ORB period just ended and mark as completed
            if not self.orb_completed and not self._is_orb_period():
                now = datetime.now()
                orb_end = now.replace(hour=9, minute=30, second=0, microsecond=0)
                if now > orb_end:
                    self.orb_completed = True
                    self.opening_ranges = self.data_service.get_all_opening_ranges()
                    logger.info(f"ORB period completed - {len(self.opening_ranges)} ranges finalized")

                    # NEW: Initialize breakout states when ORB completes
                    self._initialize_breakout_states()

            # Monitor existing positions
            await self._monitor_positions()

            # Process ORB logic
            if self._is_orb_period():
                await self._process_orb_period()
            elif self._should_generate_signals():
                # NEW: Only scan symbols with pending breakout transitions
                await self._scan_for_breakouts()

            # Update strategy metrics
            self._update_strategy_metrics()

            # Log current status (less frequent to reduce noise)
            if datetime.now().minute % 5 == 0:  # Log every 5 minutes
                self._log_strategy_status()

        except Exception as e:
            logger.error(f"Error in strategy cycle: {e}")

    def _initialize_breakout_states(self):
        """
        NEW: Initialize breakout states for all symbols when ORB period completes
        Sets initial state based on current price relative to opening range
        """
        try:
            logger.info("Initializing breakout states after ORB completion...")

            initialized_count = 0
            for symbol in self.trading_symbols:
                opening_range = self.opening_ranges.get(symbol)
                live_quote = self.live_quotes.get(symbol)

                if opening_range and live_quote:
                    current_price = live_quote.ltp

                    # Determine initial state
                    if current_price > opening_range.high:
                        initial_state = BreakoutState.UPSIDE_BREAKOUT
                        # If already broken out at ORB end, queue for evaluation
                        self.pending_breakout_symbols.add(symbol)
                    elif current_price < opening_range.low:
                        initial_state = BreakoutState.DOWNSIDE_BREAKOUT
                        # If already broken out at ORB end, queue for evaluation
                        self.pending_breakout_symbols.add(symbol)
                    else:
                        initial_state = BreakoutState.NO_BREAKOUT

                    self.breakout_states[symbol] = initial_state
                    initialized_count += 1

                    if initial_state != BreakoutState.NO_BREAKOUT:
                        logger.info(f"Initial breakout detected: {symbol} - {initial_state.value} "
                                    f"(Price: Rs.{current_price:.2f}, Range: Rs.{opening_range.low:.2f}-{opening_range.high:.2f})")

            logger.info(f"Initialized {initialized_count} symbols, {len(self.pending_breakout_symbols)} already broken out")

        except Exception as e:
            logger.error(f"Error initializing breakout states: {e}")

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

        # Only generate signals until 3 PM
        now = datetime.now()
        if now.hour >= 15:
            return False

        return True

    async def _process_orb_period(self):
        """Process data during ORB period"""
        try:
            # Update opening ranges from data service
            self.opening_ranges = self.data_service.get_all_opening_ranges()

            # Log ORB progress (less frequently)
            if datetime.now().second % 30 == 0:  # Log every 30 seconds
                completed_ranges = len([r for r in self.opening_ranges.values() if r.range_size > 0])
                logger.debug(f"ORB Progress: {completed_ranges}/{len(self.trading_symbols)} ranges calculated")

        except Exception as e:
            logger.error(f"Error processing ORB period: {e}")

    async def _scan_for_breakouts(self):
        """
        MODIFIED: Scan for breakout signals - now only processes pending transitions
        This is the key change: instead of checking all symbols repeatedly,
        we only check symbols that have had a state transition
        """
        try:
            # NEW: Check if there are any pending breakout symbols
            if not self.pending_breakout_symbols:
                # No new transitions - nothing to do
                return

            logger.info(f"Processing {len(self.pending_breakout_symbols)} symbols with new breakout transitions")

            # Mark ORB as completed if not already done
            if not self.orb_completed:
                self.orb_completed = True
                self.opening_ranges = self.data_service.get_all_opening_ranges()
                logger.info(f"ORB period completed - {len(self.opening_ranges)} ranges calculated")

            # NEW: Process only symbols with pending breakout transitions
            new_signals = []
            symbols_to_process = list(self.pending_breakout_symbols)  # Create a copy to iterate

            for symbol in symbols_to_process:
                # Skip if we already have a position in this symbol
                if symbol in self.positions:
                    self.pending_breakout_symbols.discard(symbol)
                    continue

                # Get current data
                live_quote = self.live_quotes.get(symbol)
                opening_range = self.opening_ranges.get(symbol)
                breakout_state = self.breakout_states.get(symbol)

                if not live_quote or not opening_range or breakout_state == BreakoutState.NO_BREAKOUT:
                    # No longer in breakout state or missing data
                    self.pending_breakout_symbols.discard(symbol)
                    continue

                # Determine signal type from breakout state
                if breakout_state == BreakoutState.UPSIDE_BREAKOUT:
                    signal_type = SignalType.LONG
                    breakout_level = opening_range.high
                elif breakout_state == BreakoutState.DOWNSIDE_BREAKOUT:
                    signal_type = SignalType.SHORT
                    breakout_level = opening_range.low
                else:
                    # Should not happen, but handle it
                    self.pending_breakout_symbols.discard(symbol)
                    continue

                # Evaluate the breakout signal
                signal = await self._evaluate_breakout_signal(
                    symbol, signal_type, breakout_level, opening_range, live_quote
                )

                if signal:
                    new_signals.append(signal)
                    logger.info(f"Signal generated for {symbol}: {signal_type.value}")
                else:
                    logger.debug(f"Signal rejected for {symbol} (failed validation)")

                # Remove from pending queue after processing (whether signal generated or not)
                self.pending_breakout_symbols.discard(symbol)

            # Sort signals by confidence and execute top ones
            new_signals.sort(key=lambda x: x.confidence, reverse=True)

            if new_signals:
                logger.info(f"Generated {len(new_signals)} valid signals from transition events")

            # Execute signals up to position limit
            available_slots = self.strategy_config.max_positions - len(self.positions)
            for signal in new_signals[:available_slots]:
                if validate_signal_quality(signal, self.strategy_config.min_confidence):
                    await self._execute_signal(signal)

        except Exception as e:
            logger.error(f"Error scanning for breakouts: {e}")

    async def _evaluate_breakout_signal(self, symbol: str, signal_type: SignalType,
                                        breakout_level: float, opening_range: OpenRange,
                                        live_quote: LiveQuote) -> Optional[ORBSignal]:
        """Evaluate a potential breakout signal"""
        try:
            # Validate the breakout
            is_valid, confidence, quality_scores = self.analysis_service.validate_breakout_signal(
                symbol, opening_range, live_quote.ltp, signal_type, self.strategy_config.min_confidence
            )

            logger.debug(f"Signal validation for {symbol}: valid={is_valid}, confidence={confidence:.2f}")

            if not is_valid:
                return None

            # Calculate entry parameters
            entry_price = live_quote.ltp
            stop_loss = self.analysis_service.calculate_stop_loss_level(
                signal_type, breakout_level, opening_range, self.strategy_config.stop_loss_pct
            )
            target_price = self.analysis_service.calculate_target_price(
                signal_type, entry_price, stop_loss, self.strategy_config.target_multiplier
            )

            # Calculate risk metrics
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = abs(target_price - entry_price)

            # Get symbol info for logging
            category = "GENERAL"  # Default category since sector limits are disabled

            # Create signal using centralized symbol management
            signal = create_orb_signal_from_symbol(
                symbol=symbol,
                signal_type=signal_type,
                category=category,
                breakout_price=breakout_level,
                range_high=opening_range.high,
                range_low=opening_range.low,
                range_size=opening_range.range_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                confidence=confidence,
                volume_ratio=quality_scores.get('volume_score', 0.0) * 3,
                breakout_volume=live_quote.volume,
                momentum_score=quality_scores.get('momentum_score', 0.0),
                timestamp=datetime.now(),
                orb_period_end=datetime.now().replace(hour=9, minute=30, second=0, microsecond=0),
                risk_amount=risk_amount,
                reward_amount=reward_amount
            )

            logger.info(f"ORB Signal: {symbol} ({category}) {signal_type.value} - "
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

            logger.info(f"ORB Position Opened: {signal.symbol} ({signal.category}) {signal.signal_type.value} - "
                        f"Qty: {abs(quantity)}, Entry: Rs.{signal.entry_price:.2f}, "
                        f"Range: Rs.{signal.range_low:.2f}-{signal.range_high:.2f}")

            logger.info(f"Placing orders for {signal.symbol} ({signal.category}) {signal.signal_type.value} - "
                        f"Qty: {abs(quantity)}, Entry: Rs.{signal.entry_price:.2f}")

            # PLACE ACTUAL ORDERS
            entry_placed = await self.order_manager.place_entry_order(position)
            if not entry_placed:
                logger.error(f"Failed to place entry order for {signal.symbol}")
                return False

            # Place stop loss order
            sl_placed = await self.order_manager.place_stop_loss_order(position)
            if not sl_placed:
                logger.warning(f"Failed to place stop loss order for {signal.symbol}")
                # Consider cancelling entry order here

            # Optionally place target order
            target_placed = await self.order_manager.place_target_order(position)
            if not target_placed:
                logger.warning(f"Failed to place target order for {signal.symbol}")

            # Store position only after orders are placed
            self.positions[signal.symbol] = position
            self.signals_generated_today.append(signal)

            logger.info(f"Position Opened: {signal.symbol} - "
                        f"Entry Order: {position.order_id}, "
                        f"SL Order: {position.sl_order_id}, "
                        f"Target Order: {position.target_order_id}")

            return True

        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False

    async def _monitor_positions(self):
        """Monitor existing positions for exits and process pending modifications"""
        try:
            positions_to_close = []

            for symbol, position in self.positions.items():
                live_quote = self.live_quotes.get(symbol)
                if not live_quote:
                    continue

                current_price = live_quote.ltp

                # NEW: Process pending trailing stop modifications
                if position.pending_stop_modification is not None:
                    try:
                        success = await self.order_manager.modify_stop_loss(
                            position,
                            position.pending_stop_modification
                        )

                        if success:
                            old_stop = position.current_stop_loss
                            new_stop = position.pending_stop_modification
                            logger.info(f"Trailing stop modified for {symbol}: "
                                        f"Rs.{old_stop:.2f} → Rs.{new_stop:.2f}")
                        else:
                            logger.warning(f"Failed to modify trailing stop for {symbol}")
                    except Exception as e:
                        logger.error(f"Error modifying trailing stop for {symbol}: {e}")
                    finally:
                        # Always clear the pending modification
                        position.pending_stop_modification = None

                # Check stop loss
                if self._should_exit_on_stop_loss(position, current_price):
                    positions_to_close.append((symbol, "STOP_LOSS", position.unrealized_pnl))

                # Check target
                elif self._should_exit_on_target(position, current_price):
                    if self.strategy_config.enable_partial_exits:
                        await self._partial_exit(position, current_price)
                    else:
                        positions_to_close.append((symbol, "TARGET", position.unrealized_pnl))

                # Check time-based exit
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
        # Close all positions 20 minutes before market close
        market_close = now.replace(hour=15, minute=10, second=0, microsecond=0)
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

                logger.info(f"Partial Exit: {position.symbol} - Qty: {partial_qty}, "f"P&L: Rs.{partial_pnl:.2f}, Stop moved to breakeven")

        except Exception as e:
            logger.error(f"Error in partial exit for {position.symbol}: {e}")

    async def _close_position(self, symbol: str, reason: str, pnl: float):
        """Close a position and record the trade"""
        try:
            position = self.positions[symbol]
            current_price = self.live_quotes[symbol].ltp

            logger.info(f"Closing position {symbol} - Reason: {reason}")

            # PLACE ACTUAL EXIT ORDER
            exit_placed = await self.order_manager.place_exit_order(position, current_price)

            if not exit_placed:
                logger.error(f"Failed to place exit order for {symbol}")
                # Don't remove position if exit order failed
                return

            # Create trade result using utility function
            trade_result = create_trade_result_from_position(position, current_price, reason)

            # Store completed trade
            self.completed_trades.append(trade_result)

            # Update daily P&L
            self.daily_pnl += trade_result.net_pnl

            # Remove position
            del self.positions[symbol]

            logger.info(f"Position Closed: {symbol} - {reason} - "f"P&L: Rs.{trade_result.net_pnl:.2f}")

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

            # Count positions by category
            category_summary = get_category_summary(list(self.positions.values()))

            # Calculate portfolio risk
            risk_metrics = calculate_portfolio_risk(list(self.positions.values()), self.strategy_config.portfolio_value)

            # NEW: Count breakout states
            upside_breakouts = sum(1 for state in self.breakout_states.values()
                                   if state == BreakoutState.UPSIDE_BREAKOUT)
            downside_breakouts = sum(1 for state in self.breakout_states.values()
                                     if state == BreakoutState.DOWNSIDE_BREAKOUT)
            in_range = sum(1 for state in self.breakout_states.values()
                           if state == BreakoutState.NO_BREAKOUT)

            logger.info(f"ORB Strategy Status:")
            logger.info(f"  Positions: {len(self.positions)}/{self.strategy_config.max_positions} "
                        f"(Long: {long_positions}, Short: {short_positions})")
            logger.info(f"  Daily P&L: Rs.{self.daily_pnl:.2f}")
            logger.info(f"  Unrealized P&L: Rs.{total_unrealized:.2f}")
            logger.info(f"  Total P&L: Rs.{self.metrics.total_pnl:.2f}")
            logger.info(f"  ORB Completed: {self.orb_completed}")
            logger.info(f"  Signals Today: {len(self.signals_generated_today)}")
            logger.info(f"  Portfolio Risk: {risk_metrics['risk_percentage']:.1f}%")
            logger.info(f"  Breakout States: {upside_breakouts} | {downside_breakouts} | {in_range}")
            logger.info(f"  Pending Transitions: {len(self.pending_breakout_symbols)}")

            if category_summary:
                category_info = ", ".join([f"{cat}: {info['count']}"
                                           for cat, info in category_summary.items()])
                logger.info(f"  Category Distribution: {category_info}")

        except Exception as e:
            logger.error(f"Error logging strategy status: {e}")

    def reset_daily_data(self):
        """
        MODIFIED: Reset daily data for new trading day with state tracking
        NEW: Clears breakout states and pending transitions
        """
        try:
            logger.info("Resetting daily ORB strategy data...")

            # Original resets
            self.orb_completed = False
            self.signals_generated_today.clear()
            self.daily_pnl = 0.0
            self.opening_ranges.clear()

            # NEW: Reset breakout state tracking
            for symbol in self.trading_symbols:
                self.breakout_states[symbol] = BreakoutState.NO_BREAKOUT

            # NEW: Clear pending breakout queue
            self.pending_breakout_symbols.clear()

            logger.info(f"  Breakout states reset: {len(self.breakout_states)} symbols")
            logger.info(f"  Pending transitions cleared")

            # Reset data service daily data
            if hasattr(self.data_service, 'reset_daily_data'):
                self.data_service.reset_daily_data()

            logger.info("Daily ORB strategy data reset completed")

        except Exception as e:
            logger.error(f"Error resetting daily data: {e}")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary with state tracking info"""
        try:
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())

            # Position details
            position_details = []
            for symbol, pos in self.positions.items():
                current_quote = self.live_quotes.get(symbol)
                current_price = current_quote.ltp if current_quote else 0

                position_details.append({
                    'symbol': symbol,
                    'company_name': symbol,
                    'category': pos.category,
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
                orb_summary.append({
                    'symbol': symbol,
                    'company_name': symbol,
                    'range_high': orb_range.high,
                    'range_low': orb_range.low,
                    'range_size': orb_range.range_size,
                    'range_pct': orb_range.range_pct,
                    'volume': orb_range.volume,
                    'breakout_state': self.breakout_states.get(symbol, BreakoutState.NO_BREAKOUT).value
                })

            # Portfolio risk analysis
            risk_metrics = calculate_portfolio_risk(list(self.positions.values()), self.strategy_config.portfolio_value)

            # Category distribution
            category_summary = get_category_summary(list(self.positions.values()))

            # NEW: Breakout state statistics
            breakout_stats = {
                'upside_breakouts': sum(1 for s in self.breakout_states.values() if s == BreakoutState.UPSIDE_BREAKOUT),
                'downside_breakouts': sum(1 for s in self.breakout_states.values() if s == BreakoutState.DOWNSIDE_BREAKOUT),
                'in_range': sum(1 for s in self.breakout_states.values() if s == BreakoutState.NO_BREAKOUT),
                'pending_transitions': len(self.pending_breakout_symbols),
                'pending_symbols': list(self.pending_breakout_symbols)
            }

            return {
                'strategy_name': 'Open Range Breakout (ORB) - Event-Based Transition Detection',
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
                'signal_processing': 'Event-Based (Transition Detection)',
                'breakout_statistics': breakout_stats,  # NEW
                'symbol_management': {
                    'total_universe_size': symbol_manager.get_trading_universe_size(),
                    'centralized_management': True,
                    'no_category_limits': True
                },
                'risk_metrics': risk_metrics,
                'category_summary': {cat: info for cat, info in category_summary.items()},
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
                        'category': trade.category
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
                from config.symbols import convert_to_fyers_format
                perf['company_name'] = symbol
                perf['fyers_symbol'] = convert_to_fyers_format(symbol)

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

    async def run(self):
        """Main strategy execution loop"""
        logger.info("Starting Open Range Breakout Strategy with Event-Based Transition Detection")
        logger.info(f"Trading Universe: {symbol_manager.get_trading_universe_size()} symbols")
        logger.info(f"Signal Processing: Event-driven (no repeated polling)")

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
                    await asyncio.sleep(5)  # 1 seconds during ORB
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
                'symbol_validation': 'Automatic via centralized manager',
                'signal_processing': 'Event-Based Transition Detection'
            }

            # Add service-specific info if available
            if hasattr(self.data_service, 'get_service_info'):
                service_info.update(self.data_service.get_service_info())

            return service_info

        except Exception as e:
            logger.error(f"Error getting data service info: {e}")
            return {'error': str(e)}