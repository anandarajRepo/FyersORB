# services/fyers_websocket_service.py

"""
Enhanced Fyers WebSocket service for Open Range Breakout strategy
Uses centralized symbol management - no hardcoded mappings
"""

import logging
import threading
import time
import asyncio
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta
from queue import Queue
from collections import defaultdict

# Import the official Fyers API
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

from config.settings import FyersConfig
from config.websocket_config import WebSocketConfig
from config.symbols import symbol_manager, convert_to_fyers_format, convert_from_fyers_format
from models.trading_models import LiveQuote, OpenRange

logger = logging.getLogger(__name__)


class ORBWebSocketService:
    """Enhanced WebSocket service with centralized symbol management"""

    def __init__(self, fyers_config: FyersConfig, ws_config: WebSocketConfig):
        self.fyers_config = fyers_config
        self.ws_config = ws_config

        # Connection state
        self.is_connected = False
        self.reconnect_count = 0

        # Data management
        self.subscribed_symbols = set()
        self.live_quotes: Dict[str, LiveQuote] = {}
        self.data_callbacks: List[Callable] = []

        # ORB specific data storage
        self.opening_ranges: Dict[str, OpenRange] = {}
        self.orb_data_cache: Dict[str, List[LiveQuote]] = defaultdict(list)
        self.daily_high_low: Dict[str, Dict[str, float]] = defaultdict(lambda: {'high': 0, 'low': float('inf')})

        # Fyers WebSocket instance
        self.fyers_socket = None

        # ORB timing control
        self.orb_period_start = None
        self.orb_period_end = None
        self.is_orb_period_active = False

    def connect(self) -> bool:
        """Connect using official Fyers WebSocket with ORB enhancements"""
        try:
            logger.info("Connecting to Fyers WebSocket for ORB strategy...")

            # Create Fyers WebSocket instance
            self.fyers_socket = data_ws.FyersDataSocket(
                access_token=self.fyers_config.access_token,
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                reconnect_retry=self.ws_config.max_reconnect_attempts,
                on_message=self._on_message,
                on_connect=self._on_open,
                on_close=self._on_close,
                on_error=self._on_error
            )

            # Start connection in background thread
            self._start_connection_thread()

            # Wait for connection
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < self.ws_config.connection_timeout:
                time.sleep(0.1)

            if self.is_connected:
                logger.info("Fyers WebSocket connected successfully for ORB strategy")
                self._initialize_orb_timing()
                return True
            else:
                logger.error("Fyers WebSocket connection timeout")
                return False

        except Exception as e:
            logger.error(f"Error connecting to Fyers WebSocket: {e}")
            return False

    def _initialize_orb_timing(self):
        """Initialize ORB period timing"""
        now = datetime.now()

        # Set ORB period start (9:15 AM)
        self.orb_period_start = now.replace(hour=9, minute=15, second=0, microsecond=0)

        # Set ORB period end (9:30 AM)
        self.orb_period_end = now.replace(hour=9, minute=30, second=0, microsecond=0)

        # Check if we're currently in ORB period
        self.is_orb_period_active = self.orb_period_start <= now <= self.orb_period_end

        logger.info(f"ORB timing initialized - Active: {self.is_orb_period_active}")

    def _start_connection_thread(self):
        """Start WebSocket connection in background thread"""

        def run_connection():
            try:
                self.fyers_socket.connect()
            except Exception as e:
                logger.error(f"Connection thread error: {e}")

        connection_thread = threading.Thread(target=run_connection)
        connection_thread.daemon = True
        connection_thread.start()

    def disconnect(self):
        """Disconnect from Fyers WebSocket"""
        try:
            self.is_connected = False
            if self.fyers_socket:
                self.fyers_socket.close_connection()
            logger.info("Fyers WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """Subscribe to symbols using centralized symbol management"""
        try:
            if not self.is_connected:
                logger.error("WebSocket not connected")
                return False

            # Convert display symbols to Fyers format using centralized manager
            fyers_symbols = []
            valid_symbols = []

            for symbol in symbols:
                fyers_symbol = convert_to_fyers_format(symbol)
                if fyers_symbol:
                    fyers_symbols.append(fyers_symbol)
                    valid_symbols.append(symbol)
                    self.subscribed_symbols.add(symbol)
                    # Initialize ORB data cache for each symbol
                    self.orb_data_cache[symbol] = []
                else:
                    logger.warning(f"Unknown symbol: {symbol} - skipping")

            if not fyers_symbols:
                logger.error("No valid symbols to subscribe")
                return False

            # Subscribe using official API
            self.fyers_socket.subscribe(symbols=fyers_symbols, data_type="SymbolUpdate")

            logger.info(f"Subscribed to {len(valid_symbols)} symbols for ORB strategy")
            logger.debug(f"Valid symbols: {valid_symbols}")
            logger.debug(f"Fyers symbols: {fyers_symbols}")
            return True

        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
            return False

    def get_opening_range(self, symbol: str) -> Optional[OpenRange]:
        """Get calculated opening range for a symbol"""
        return self.opening_ranges.get(symbol)

    def get_all_opening_ranges(self) -> Dict[str, OpenRange]:
        """Get all calculated opening ranges"""
        return self.opening_ranges.copy()

    def is_breakout_detected(self, symbol: str, current_price: float) -> tuple:
        """
        Check if price breaks out of opening range
        Returns: (is_breakout, signal_type, breakout_level)
        """
        if symbol not in self.opening_ranges:
            return False, None, None

        range_data = self.opening_ranges[symbol]

        # Check for upside breakout
        if current_price > range_data.high:
            return True, 'LONG', range_data.high

        # Check for downside breakout
        elif current_price < range_data.low:
            return True, 'SHORT', range_data.low

        return False, None, None

    def get_live_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get latest live quote"""
        return self.live_quotes.get(symbol)

    def get_all_live_quotes(self) -> Dict[str, LiveQuote]:
        """Get all live quotes"""
        return self.live_quotes.copy()

    def add_data_callback(self, callback: Callable):
        """Add callback for data updates"""
        self.data_callbacks.append(callback)

    def _on_open(self):
        """WebSocket opened"""
        self.is_connected = True
        self.reconnect_count = 0
        logger.info("Fyers WebSocket opened for ORB strategy")

    def _on_close(self, message):
        """WebSocket closed"""
        self.is_connected = False
        logger.warning(f"Fyers WebSocket closed: {message}")

    def _on_error(self, message):
        """WebSocket error"""
        self.is_connected = False
        logger.error(f"Fyers WebSocket error: {message}")

    def _on_message(self, message):
        """Handle incoming message from Fyers WebSocket"""
        try:
            # Process the message data
            self._process_fyers_data(message)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.debug(f"Message content: {message}")

    def _process_fyers_data(self, data):
        """Process data from Fyers WebSocket with centralized symbol conversion"""
        try:
            # Handle different data formats from Fyers
            if isinstance(data, dict):
                symbol_data = data.get('symbol', '')

                # Convert back to display format using centralized manager
                display_symbol = convert_from_fyers_format(symbol_data)

                if display_symbol and isinstance(data, dict):
                    # Create LiveQuote from Fyers data
                    live_quote = LiveQuote(
                        symbol=display_symbol,
                        ltp=float(data.get('ltp', data.get('last_price', 0))),
                        open_price=float(data.get('open_price', data.get('open', 0))),
                        high_price=float(data.get('high_price', data.get('high', 0))),
                        low_price=float(data.get('low_price', data.get('low', 0))),
                        volume=int(data.get('volume', data.get('vol_traded_today', 0))),
                        previous_close=float(data.get('prev_close_price', data.get('prev_close', 0))),
                        timestamp=datetime.now()
                    )

                    # Update storage
                    self.live_quotes[display_symbol] = live_quote

                    # ORB-specific processing
                    self._process_orb_data(display_symbol, live_quote)

                    # Update daily high/low tracking
                    self._update_daily_extremes(display_symbol, live_quote)

                    # Notify callbacks
                    for callback in self.data_callbacks:
                        try:
                            callback(display_symbol, live_quote)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                    # Detailed logging for active ORB period
                    if self.is_orb_period_active:
                        logger.debug(f"ORB Data - {display_symbol}: Rs.{live_quote.ltp:.2f} "
                                     f"H:{live_quote.high_price:.2f} L:{live_quote.low_price:.2f} "
                                     f"Vol:{live_quote.volume}")

        except Exception as e:
            logger.error(f"Error processing Fyers data: {e}")
            logger.debug(f"Data: {data}")

    def _process_orb_data(self, symbol: str, live_quote: LiveQuote):
        """Process ORB-specific data during opening range period"""
        try:
            now = datetime.now()

            # Check if we're in ORB period
            if self.orb_period_start <= now <= self.orb_period_end:
                self.is_orb_period_active = True

                # Cache ORB period data
                self.orb_data_cache[symbol].append(live_quote)

                # Calculate current opening range
                orb_quotes = self.orb_data_cache[symbol]
                if orb_quotes:
                    current_high = max(q.high_price for q in orb_quotes)
                    current_low = min(q.low_price for q in orb_quotes)
                    total_volume = sum(q.volume for q in orb_quotes)

                    # Update opening range
                    self.opening_ranges[symbol] = OpenRange(
                        symbol=symbol,
                        high=current_high,
                        low=current_low,
                        range_size=current_high - current_low,
                        range_pct=((current_high - current_low) / current_low) * 100 if current_low > 0 else 0,
                        volume=total_volume,
                        start_time=self.orb_period_start,
                        end_time=self.orb_period_end
                    )

            elif now > self.orb_period_end and self.is_orb_period_active:
                # ORB period just ended
                self.is_orb_period_active = False
                self._finalize_opening_ranges()

        except Exception as e:
            logger.error(f"Error processing ORB data for {symbol}: {e}")

    def _finalize_opening_ranges(self):
        """Finalize opening ranges when ORB period ends"""
        try:
            logger.info("ORB period ended - Finalizing opening ranges")

            for symbol in self.subscribed_symbols:
                if symbol in self.orb_data_cache and self.orb_data_cache[symbol]:
                    orb_quotes = self.orb_data_cache[symbol]

                    final_high = max(q.high_price for q in orb_quotes)
                    final_low = min(q.low_price for q in orb_quotes)
                    total_volume = orb_quotes[-1].volume  # Latest cumulative volume

                    self.opening_ranges[symbol] = OpenRange(
                        symbol=symbol,
                        high=final_high,
                        low=final_low,
                        range_size=final_high - final_low,
                        range_pct=((final_high - final_low) / final_low) * 100 if final_low > 0 else 0,
                        volume=total_volume,
                        start_time=self.orb_period_start,
                        end_time=self.orb_period_end
                    )

                    logger.info(f"ORB finalized - {symbol}: H:{final_high:.2f} L:{final_low:.2f} "
                                f"Range:{final_high - final_low:.2f} ({self.opening_ranges[symbol].range_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error finalizing opening ranges: {e}")

    def _update_daily_extremes(self, symbol: str, live_quote: LiveQuote):
        """Update daily high/low tracking"""
        try:
            current_high = self.daily_high_low[symbol]['high']
            current_low = self.daily_high_low[symbol]['low']

            self.daily_high_low[symbol]['high'] = max(current_high, live_quote.high_price)
            self.daily_high_low[symbol]['low'] = min(current_low, live_quote.low_price)

        except Exception as e:
            logger.error(f"Error updating daily extremes for {symbol}: {e}")

    def get_daily_high_low(self, symbol: str) -> tuple:
        """Get daily high and low for a symbol"""
        data = self.daily_high_low.get(symbol, {'high': 0, 'low': float('inf')})
        return data['high'], data['low']

    def reset_daily_data(self):
        """Reset daily data for new trading day"""
        self.daily_high_low.clear()
        self.orb_data_cache.clear()
        self.opening_ranges.clear()
        logger.info("Daily ORB data reset completed")


# Enhanced fallback service with centralized symbol management
class ORBFallbackDataService:
    """Fallback service using REST API with centralized symbol management"""

    def __init__(self, fyers_config: FyersConfig, ws_config: WebSocketConfig):
        self.fyers_config = fyers_config
        self.ws_config = ws_config

        # State
        self.is_connected = False
        self.subscribed_symbols = set()
        self.live_quotes: Dict[str, LiveQuote] = {}
        self.data_callbacks: List[Callable] = []

        # ORB data
        self.opening_ranges: Dict[str, OpenRange] = {}
        self.orb_data_cache: Dict[str, List[LiveQuote]] = defaultdict(list)

        # Fyers model for REST API
        self.fyers = fyersModel.FyersModel(
            client_id=fyers_config.client_id,
            token=fyers_config.access_token
        )

        # Threading
        self.polling_thread = None
        self.stop_event = threading.Event()

    def connect(self) -> bool:
        """Start fallback data service with ORB support"""
        try:
            logger.info("Starting ORB fallback REST API data service...")

            # Test API connection
            profile = self.fyers.get_profile()
            if profile.get('s') != 'ok':
                logger.error("Fyers API authentication failed")
                return False

            logger.info(f"Connected to Fyers API for ORB strategy - User: {profile.get('data', {}).get('name', 'Unknown')}")

            # Start polling
            self.is_connected = True
            self._start_polling()

            return True

        except Exception as e:
            logger.error(f"ORB Fallback connection error: {e}")
            return False

    def _start_polling(self):
        """Start ORB-aware polling thread"""
        self.polling_thread = threading.Thread(target=self._poll_data_orb)
        self.polling_thread.daemon = True
        self.polling_thread.start()
        logger.info("ORB fallback polling started")

    def _poll_data_orb(self):
        """Poll for data with ORB period awareness"""
        while not self.stop_event.is_set() and self.is_connected:
            try:
                if self.subscribed_symbols:
                    self._fetch_quotes_orb()

                # Adjust polling frequency based on ORB period
                now = datetime.now()
                orb_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
                orb_end = now.replace(hour=9, minute=30, second=0, microsecond=0)

                if orb_start <= now <= orb_end:
                    # More frequent polling during ORB period
                    time.sleep(2)
                else:
                    # Normal polling outside ORB period
                    time.sleep(5)

            except Exception as e:
                logger.error(f"ORB Polling error: {e}")
                time.sleep(10)

    def _fetch_quotes_orb(self):
        """Fetch quotes with ORB processing using centralized symbols"""
        try:
            symbols = list(self.subscribed_symbols)
            # Convert to Fyers format using centralized manager
            fyers_symbols = [convert_to_fyers_format(s) for s in symbols if convert_to_fyers_format(s)]

            # Limit to 25 symbols per request
            symbol_chunks = [fyers_symbols[i:i + 25] for i in range(0, len(fyers_symbols), 25)]

            for chunk in symbol_chunks:
                data = {"symbols": ",".join(chunk)}
                response = self.fyers.quotes(data)

                if response.get('s') == 'ok':
                    self._process_rest_quotes_orb(response.get('d', {}))
                else:
                    logger.debug(f"API response: {response}")

                time.sleep(1)

        except Exception as e:
            logger.debug(f"Fetch ORB quotes error: {e}")

    def _process_rest_quotes_orb(self, data: dict):
        """Process quotes from REST API with centralized symbol conversion"""
        try:
            for fyers_symbol, quote_data in data.items():
                # Convert back to display format using centralized manager
                display_symbol = convert_from_fyers_format(fyers_symbol)

                if display_symbol and isinstance(quote_data, dict):
                    live_quote = LiveQuote(
                        symbol=display_symbol,
                        ltp=float(quote_data.get('lp', 0)),
                        open_price=float(quote_data.get('open_price', 0)),
                        high_price=float(quote_data.get('high_price', 0)),
                        low_price=float(quote_data.get('low_price', 0)),
                        volume=int(quote_data.get('volume', 0)),
                        previous_close=float(quote_data.get('prev_close_price', 0)),
                        timestamp=datetime.now()
                    )

                    # Update storage
                    old_quote = self.live_quotes.get(display_symbol)
                    self.live_quotes[display_symbol] = live_quote

                    # ORB processing (similar to WebSocket service)
                    self._process_orb_data_fallback(display_symbol, live_quote)

                    # Only notify callbacks if price changed significantly
                    if not old_quote or abs(old_quote.ltp - live_quote.ltp) > 0.01:
                        for callback in self.data_callbacks:
                            try:
                                callback(display_symbol, live_quote)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Process REST ORB quotes error: {e}")

    def _process_orb_data_fallback(self, symbol: str, live_quote: LiveQuote):
        """Process ORB data in fallback mode"""
        try:
            now = datetime.now()
            orb_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            orb_end = now.replace(hour=9, minute=30, second=0, microsecond=0)

            if orb_start <= now <= orb_end:
                # Cache ORB period data
                self.orb_data_cache[symbol].append(live_quote)

                # Calculate current opening range
                orb_quotes = self.orb_data_cache[symbol]
                if orb_quotes:
                    current_high = max(q.high_price for q in orb_quotes)
                    current_low = min(q.low_price for q in orb_quotes)

                    self.opening_ranges[symbol] = OpenRange(
                        symbol=symbol,
                        high=current_high,
                        low=current_low,
                        range_size=current_high - current_low,
                        range_pct=((current_high - current_low) / current_low) * 100 if current_low > 0 else 0,
                        volume=live_quote.volume,
                        start_time=orb_start,
                        end_time=orb_end
                    )

        except Exception as e:
            logger.error(f"Error processing fallback ORB data for {symbol}: {e}")

    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """Subscribe to symbols in fallback mode with validation"""
        valid_symbols = []
        for symbol in symbols:
            if convert_to_fyers_format(symbol):  # Validate using centralized manager
                valid_symbols.append(symbol)
            else:
                logger.warning(f"Invalid symbol for fallback mode: {symbol}")

        self.subscribed_symbols.update(valid_symbols)
        logger.info(f"Subscribed to {len(valid_symbols)} symbols in fallback mode: {valid_symbols}")
        return len(valid_symbols) > 0

    def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
        return True

    def add_data_callback(self, callback: Callable):
        """Add data callback"""
        self.data_callbacks.append(callback)

    def get_live_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get live quote"""
        return self.live_quotes.get(symbol)

    def get_all_live_quotes(self) -> Dict[str, LiveQuote]:
        """Get all live quotes"""
        return self.live_quotes.copy()

    def get_opening_range(self, symbol: str) -> Optional[OpenRange]:
        """Get opening range for symbol"""
        return self.opening_ranges.get(symbol)

    def get_all_opening_ranges(self) -> Dict[str, OpenRange]:
        """Get all opening ranges"""
        return self.opening_ranges.copy()

    def disconnect(self):
        """Disconnect fallback service"""
        self.stop_event.set()
        self.is_connected = False
        logger.info("ORB Fallback service disconnected")


# Hybrid service for ORB strategy with centralized symbol management
class HybridORBDataService:
    """Hybrid service that tries WebSocket first, falls back to REST API with centralized symbols"""

    def __init__(self, fyers_config: FyersConfig, ws_config: WebSocketConfig):
        self.fyers_config = fyers_config
        self.ws_config = ws_config

        # Try WebSocket first, fallback to REST
        self.primary_service = None
        self.fallback_service = None
        self.using_fallback = False

        # State
        self.is_connected = False
        self.subscribed_symbols = set()
        self.data_callbacks = []

    def connect(self) -> bool:
        """Try WebSocket first, fallback to REST API for ORB strategy"""
        logger.info("Attempting hybrid ORB connection (WebSocket -> REST fallback)")

        # Try WebSocket first
        try:
            self.primary_service = ORBWebSocketService(self.fyers_config, self.ws_config)

            if self.primary_service.connect():
                logger.info("Using WebSocket service for ORB strategy")
                self.is_connected = True
                self._setup_callbacks(self.primary_service)
                return True
        except Exception as e:
            logger.warning(f"ORB WebSocket failed: {e}")

        # Fallback to REST API
        try:
            logger.info("Falling back to REST API polling for ORB strategy...")
            self.fallback_service = ORBFallbackDataService(self.fyers_config, self.ws_config)

            if self.fallback_service.connect():
                logger.info("Using REST API fallback for ORB strategy")
                self.using_fallback = True
                self.is_connected = True
                self._setup_callbacks(self.fallback_service)
                return True
        except Exception as e:
            logger.error(f"ORB Fallback also failed: {e}")

        logger.error("All ORB connection methods failed")
        return False

    def _setup_callbacks(self, service):
        """Setup callbacks for the active service"""
        for callback in self.data_callbacks:
            service.add_data_callback(callback)

    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """Subscribe using active service with validation"""
        active_service = self.fallback_service if self.using_fallback else self.primary_service
        if active_service:
            # Validate symbols using centralized manager
            valid_symbols = [s for s in symbols if convert_to_fyers_format(s)]
            invalid_symbols = [s for s in symbols if not convert_to_fyers_format(s)]

            if invalid_symbols:
                logger.warning(f"Invalid symbols ignored: {invalid_symbols}")

            if valid_symbols:
                self.subscribed_symbols.update(valid_symbols)
                return active_service.subscribe_symbols(valid_symbols)
        return False

    def add_data_callback(self, callback: Callable):
        """Add callback"""
        self.data_callbacks.append(callback)
        active_service = self.fallback_service if self.using_fallback else self.primary_service
        if active_service:
            active_service.add_data_callback(callback)

    def get_live_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get live quote"""
        active_service = self.fallback_service if self.using_fallback else self.primary_service
        return active_service.get_live_quote(symbol) if active_service else None

    def get_opening_range(self, symbol: str) -> Optional[OpenRange]:
        """Get opening range"""
        active_service = self.fallback_service if self.using_fallback else self.primary_service
        return active_service.get_opening_range(symbol) if active_service else None

    def get_all_opening_ranges(self) -> Dict[str, OpenRange]:
        """Get all opening ranges"""
        active_service = self.fallback_service if self.using_fallback else self.primary_service
        return active_service.get_all_opening_ranges() if active_service else {}

    def is_breakout_detected(self, symbol: str, current_price: float) -> tuple:
        """Check for breakout using active service"""
        active_service = self.fallback_service if self.using_fallback else self.primary_service
        if hasattr(active_service, 'is_breakout_detected'):
            return active_service.is_breakout_detected(symbol, current_price)
        return False, None, None

    def disconnect(self):
        """Disconnect active service"""
        if self.primary_service:
            self.primary_service.disconnect()
        if self.fallback_service:
            self.fallback_service.disconnect()
        self.is_connected = False

    def get_service_info(self) -> Dict[str, any]:
        """Get information about current service configuration"""
        return {
            'using_fallback': self.using_fallback,
            'is_connected': self.is_connected,
            'subscribed_symbols_count': len(self.subscribed_symbols),
            'total_available_symbols': symbol_manager.get_trading_universe_size(),
            'service_type': 'REST API' if self.using_fallback else 'WebSocket',
            'symbol_validation': 'Centralized Symbol Manager'
        }