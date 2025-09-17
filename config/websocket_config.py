# config/websocket_config.py

"""
WebSocket Configuration for ORB Trading Strategy
Comprehensive configuration for Fyers WebSocket connections with fallback options
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class WebSocketMode(Enum):
    """WebSocket subscription modes"""
    QUOTES = "quotes"  # Basic quote data
    DEPTH = "depth"  # Market depth data
    FULL = "full"  # Complete market data
    LITE = "lite"  # Minimal data for low bandwidth


class ReconnectionStrategy(Enum):
    """Reconnection strategies"""
    EXPONENTIAL_BACKOFF = "exponential"  # Exponential backoff delays
    FIXED_INTERVAL = "fixed"  # Fixed interval retries
    IMMEDIATE = "immediate"  # Immediate reconnection attempts


@dataclass
class WebSocketConfig:
    """Comprehensive WebSocket configuration for ORB strategy"""

    # Core WebSocket Settings
    websocket_url: str = "wss://api-t1.fyers.in/socket/v2/dataSock"
    fallback_rest_url: str = "https://api-t1.fyers.in/api/v3"

    # Connection Management
    reconnect_interval: int = 5  # Base reconnection interval (seconds)
    max_reconnect_attempts: int = 10  # Maximum reconnection attempts
    connection_timeout: int = 30  # Connection timeout (seconds)
    ping_interval: int = 30  # WebSocket ping interval (seconds)
    pong_timeout: int = 10  # Pong response timeout (seconds)

    # Data Subscription Settings
    subscription_mode: WebSocketMode = WebSocketMode.QUOTES  # Data subscription mode
    enable_heartbeat: bool = True  # Enable WebSocket heartbeat
    buffer_size: int = 8192  # WebSocket buffer size

    # Reconnection Strategy
    reconnection_strategy: ReconnectionStrategy = ReconnectionStrategy.EXPONENTIAL_BACKOFF
    max_backoff_delay: int = 300  # Maximum backoff delay (5 minutes)
    backoff_multiplier: float = 1.5  # Backoff multiplier for exponential strategy

    # Performance Settings
    enable_compression: bool = True  # Enable WebSocket compression
    max_message_size: int = 1048576  # Maximum message size (1MB)
    queue_size: int = 1000  # Internal message queue size

    # ORB Strategy Specific Settings
    orb_data_retention_minutes: int = 30  # How long to retain ORB period data
    enable_data_validation: bool = True  # Validate incoming data
    symbol_subscription_batch_size: int = 25  # Symbols per subscription batch

    # Fallback Configuration
    enable_rest_fallback: bool = True  # Enable REST API fallback
    rest_polling_interval: int = 5  # REST API polling interval (seconds)
    rest_request_timeout: int = 10  # REST API request timeout
    rest_max_retries: int = 3  # Maximum REST API retries

    # Health Monitoring
    enable_connection_monitoring: bool = True  # Monitor connection health
    health_check_interval: int = 60  # Health check interval (seconds)
    max_missed_heartbeats: int = 3  # Maximum missed heartbeats before reconnect

    # Data Quality Settings
    enable_duplicate_filtering: bool = True  # Filter duplicate messages
    enable_stale_data_detection: bool = True  # Detect stale data
    max_data_age_seconds: int = 10  # Maximum acceptable data age

    # Logging Configuration
    enable_websocket_logging: bool = True  # Enable WebSocket specific logging
    log_all_messages: bool = False  # Log all WebSocket messages (debug only)
    log_connection_events: bool = True  # Log connection/disconnection events
    log_subscription_events: bool = True  # Log symbol subscription events

    # Advanced Settings
    enable_message_compression: bool = False  # Enable individual message compression
    custom_headers: Optional[Dict[str, str]] = None  # Custom WebSocket headers
    enable_ssl_verification: bool = True  # Enable SSL certificate verification

    def __post_init__(self):
        """Validate and process configuration after initialization"""
        # Load from environment variables if available
        self._load_from_environment()

        # Validate configuration
        self._validate_config()

        # Set derived configurations
        self._set_derived_config()

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Connection settings
        self.reconnect_interval = int(os.environ.get('WS_RECONNECT_INTERVAL', self.reconnect_interval))
        self.max_reconnect_attempts = int(os.environ.get('WS_MAX_RECONNECT_ATTEMPTS', self.max_reconnect_attempts))
        self.connection_timeout = int(os.environ.get('WS_CONNECTION_TIMEOUT', self.connection_timeout))
        self.ping_interval = int(os.environ.get('WS_PING_INTERVAL', self.ping_interval))

        # Performance settings
        self.buffer_size = int(os.environ.get('WS_BUFFER_SIZE', self.buffer_size))
        self.queue_size = int(os.environ.get('WS_QUEUE_SIZE', self.queue_size))

        # ORB specific settings
        self.orb_data_retention_minutes = int(os.environ.get('ORB_DATA_RETENTION_MINUTES', self.orb_data_retention_minutes))
        self.symbol_subscription_batch_size = int(os.environ.get('WS_SYMBOL_BATCH_SIZE', self.symbol_subscription_batch_size))

        # Fallback settings
        self.enable_rest_fallback = os.environ.get('ENABLE_REST_FALLBACK', 'true').lower() == 'true'
        self.rest_polling_interval = int(os.environ.get('REST_POLLING_INTERVAL', self.rest_polling_interval))
        self.rest_request_timeout = int(os.environ.get('REST_REQUEST_TIMEOUT', self.rest_request_timeout))

        # Logging settings
        self.enable_websocket_logging = os.environ.get('ENABLE_WS_LOGGING', 'true').lower() == 'true'
        self.log_all_messages = os.environ.get('WS_LOG_ALL_MESSAGES', 'false').lower() == 'true'
        self.enable_connection_monitoring = os.environ.get('WS_ENABLE_MONITORING', 'true').lower() == 'true'

        # Advanced settings
        self.enable_compression = os.environ.get('WS_ENABLE_COMPRESSION', 'true').lower() == 'true'
        self.enable_ssl_verification = os.environ.get('WS_SSL_VERIFICATION', 'true').lower() == 'true'

        # Subscription mode from environment
        mode_str = os.environ.get('WS_SUBSCRIPTION_MODE', 'quotes').lower()
        try:
            self.subscription_mode = WebSocketMode(mode_str)
        except ValueError:
            self.subscription_mode = WebSocketMode.QUOTES

        # Reconnection strategy from environment
        strategy_str = os.environ.get('WS_RECONNECTION_STRATEGY', 'exponential').lower()
        try:
            self.reconnection_strategy = ReconnectionStrategy(strategy_str)
        except ValueError:
            self.reconnection_strategy = ReconnectionStrategy.EXPONENTIAL_BACKOFF

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate intervals
        if self.reconnect_interval < 1:
            raise ValueError("Reconnect interval must be at least 1 second")

        if self.max_reconnect_attempts < 1:
            raise ValueError("Max reconnect attempts must be at least 1")

        if self.connection_timeout < 5:
            raise ValueError("Connection timeout must be at least 5 seconds")

        if self.ping_interval < 10:
            raise ValueError("Ping interval must be at least 10 seconds")

        # Validate buffer and queue sizes
        if self.buffer_size < 1024:
            raise ValueError("Buffer size must be at least 1024 bytes")

        if self.queue_size < 100:
            raise ValueError("Queue size must be at least 100")

        # Validate ORB settings
        if self.orb_data_retention_minutes < 15:
            raise ValueError("ORB data retention must be at least 15 minutes")

        if self.symbol_subscription_batch_size < 1 or self.symbol_subscription_batch_size > 50:
            raise ValueError("Symbol subscription batch size must be between 1 and 50")

        # Validate fallback settings
        if self.rest_polling_interval < 1:
            raise ValueError("REST polling interval must be at least 1 second")

        if self.rest_request_timeout < 5:
            raise ValueError("REST request timeout must be at least 5 seconds")

        # Validate monitoring settings
        if self.health_check_interval < 30:
            raise ValueError("Health check interval must be at least 30 seconds")

        if self.max_missed_heartbeats < 1:
            raise ValueError("Max missed heartbeats must be at least 1")

    def _set_derived_config(self):
        """Set derived configuration parameters"""
        # Set custom headers if not provided
        if self.custom_headers is None:
            self.custom_headers = {
                'User-Agent': 'ORB-Trading-Strategy/2.0',
                'Accept': 'application/json',
                'Connection': 'Upgrade'
            }

        # Adjust settings based on subscription mode
        if self.subscription_mode == WebSocketMode.LITE:
            self.buffer_size = min(self.buffer_size, 4096)
            self.queue_size = min(self.queue_size, 500)
        elif self.subscription_mode == WebSocketMode.FULL:
            self.buffer_size = max(self.buffer_size, 16384)
            self.queue_size = max(self.queue_size, 2000)

    def get_reconnect_delay(self, attempt: int) -> int:
        """Calculate reconnect delay based on strategy and attempt number"""
        if self.reconnection_strategy == ReconnectionStrategy.IMMEDIATE:
            return 0
        elif self.reconnection_strategy == ReconnectionStrategy.FIXED_INTERVAL:
            return self.reconnect_interval
        else:  # EXPONENTIAL_BACKOFF
            delay = self.reconnect_interval * (self.backoff_multiplier ** (attempt - 1))
            return min(int(delay), self.max_backoff_delay)

    def get_subscription_data_type(self) -> str:
        """Get Fyers-specific data type for subscription"""
        mode_mapping = {
            WebSocketMode.QUOTES: "SymbolUpdate",
            WebSocketMode.DEPTH: "MarketDepth",
            WebSocketMode.FULL: "FullMarketData",
            WebSocketMode.LITE: "LiteMarketData"
        }
        return mode_mapping.get(self.subscription_mode, "SymbolUpdate")

    def should_enable_compression(self) -> bool:
        """Determine if compression should be enabled based on settings"""
        return self.enable_compression and self.subscription_mode in [WebSocketMode.FULL, WebSocketMode.DEPTH]

    def get_health_check_config(self) -> Dict[str, Any]:
        """Get health check configuration"""
        return {
            'enabled': self.enable_connection_monitoring,
            'interval': self.health_check_interval,
            'max_missed_heartbeats': self.max_missed_heartbeats,
            'ping_interval': self.ping_interval,
            'pong_timeout': self.pong_timeout
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration for WebSocket"""
        return {
            'enable_websocket_logging': self.enable_websocket_logging,
            'log_all_messages': self.log_all_messages,
            'log_connection_events': self.log_connection_events,
            'log_subscription_events': self.log_subscription_events
        }

    def get_fallback_config(self) -> Dict[str, Any]:
        """Get REST API fallback configuration"""
        return {
            'enabled': self.enable_rest_fallback,
            'base_url': self.fallback_rest_url,
            'polling_interval': self.rest_polling_interval,
            'request_timeout': self.rest_request_timeout,
            'max_retries': self.rest_max_retries
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            'buffer_size': self.buffer_size,
            'queue_size': self.queue_size,
            'max_message_size': self.max_message_size,
            'enable_compression': self.should_enable_compression(),
            'symbol_batch_size': self.symbol_subscription_batch_size
        }

    def get_data_quality_config(self) -> Dict[str, Any]:
        """Get data quality configuration"""
        return {
            'enable_validation': self.enable_data_validation,
            'enable_duplicate_filtering': self.enable_duplicate_filtering,
            'enable_stale_detection': self.enable_stale_data_detection,
            'max_data_age_seconds': self.max_data_age_seconds,
            'orb_data_retention_minutes': self.orb_data_retention_minutes
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'websocket_url': self.websocket_url,
            'reconnect_interval': self.reconnect_interval,
            'max_reconnect_attempts': self.max_reconnect_attempts,
            'connection_timeout': self.connection_timeout,
            'ping_interval': self.ping_interval,
            'subscription_mode': self.subscription_mode.value,
            'reconnection_strategy': self.reconnection_strategy.value,
            'enable_rest_fallback': self.enable_rest_fallback,
            'enable_compression': self.enable_compression,
            'buffer_size': self.buffer_size,
            'queue_size': self.queue_size,
            'health_check': self.get_health_check_config(),
            'logging': self.get_logging_config(),
            'fallback': self.get_fallback_config(),
            'performance': self.get_performance_config(),
            'data_quality': self.get_data_quality_config()
        }

    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"""WebSocketConfig(
    url={self.websocket_url},
    mode={self.subscription_mode.value},
    reconnect_attempts={self.max_reconnect_attempts},
    fallback_enabled={self.enable_rest_fallback},
    compression_enabled={self.enable_compression}
)"""


# Predefined configuration profiles for different use cases
class WebSocketProfiles:
    """Predefined WebSocket configuration profiles"""

    @staticmethod
    def development() -> WebSocketConfig:
        """Development profile with detailed logging and shorter timeouts"""
        return WebSocketConfig(
            reconnect_interval=3,
            max_reconnect_attempts=5,
            connection_timeout=15,
            ping_interval=20,
            subscription_mode=WebSocketMode.QUOTES,
            enable_websocket_logging=True,
            log_all_messages=True,
            log_connection_events=True,
            health_check_interval=30,
            rest_polling_interval=3
        )

    @staticmethod
    def production() -> WebSocketConfig:
        """Production profile optimized for stability and performance"""
        return WebSocketConfig(
            reconnect_interval=5,
            max_reconnect_attempts=15,
            connection_timeout=30,
            ping_interval=30,
            subscription_mode=WebSocketMode.QUOTES,
            enable_websocket_logging=True,
            log_all_messages=False,
            log_connection_events=True,
            health_check_interval=60,
            rest_polling_interval=5,
            buffer_size=16384,
            queue_size=2000
        )

    @staticmethod
    def high_frequency() -> WebSocketConfig:
        """High-frequency trading profile with minimal latency"""
        return WebSocketConfig(
            reconnect_interval=1,
            max_reconnect_attempts=20,
            connection_timeout=10,
            ping_interval=15,
            subscription_mode=WebSocketMode.FULL,
            enable_compression=True,
            buffer_size=32768,
            queue_size=5000,
            rest_polling_interval=1,
            health_check_interval=30,
            symbol_subscription_batch_size=50
        )

    @staticmethod
    def low_bandwidth() -> WebSocketConfig:
        """Low bandwidth profile for limited connectivity"""
        return WebSocketConfig(
            reconnect_interval=10,
            max_reconnect_attempts=8,
            connection_timeout=45,
            ping_interval=60,
            subscription_mode=WebSocketMode.LITE,
            enable_compression=True,
            buffer_size=4096,
            queue_size=500,
            rest_polling_interval=10,
            health_check_interval=120,
            symbol_subscription_batch_size=10
        )

    @staticmethod
    def orb_optimized() -> WebSocketConfig:
        """ORB strategy optimized profile"""
        return WebSocketConfig(
            reconnect_interval=5,
            max_reconnect_attempts=10,
            connection_timeout=30,
            ping_interval=30,
            subscription_mode=WebSocketMode.QUOTES,
            enable_rest_fallback=True,
            enable_compression=True,
            buffer_size=8192,
            queue_size=1000,
            orb_data_retention_minutes=30,
            enable_data_validation=True,
            enable_duplicate_filtering=True,
            symbol_subscription_batch_size=25,
            rest_polling_interval=5,
            health_check_interval=60,
            max_data_age_seconds=10
        )


# Factory function to create configuration based on profile
def create_websocket_config(profile: str = "orb_optimized") -> WebSocketConfig:
    """Create WebSocket configuration based on profile name"""
    profile_map = {
        'development': WebSocketProfiles.development,
        'production': WebSocketProfiles.production,
        'high_frequency': WebSocketProfiles.high_frequency,
        'low_bandwidth': WebSocketProfiles.low_bandwidth,
        'orb_optimized': WebSocketProfiles.orb_optimized
    }

    profile_func = profile_map.get(profile.lower())
    if not profile_func:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(profile_map.keys())}")

    return profile_func()


# Example usage and testing
if __name__ == "__main__":
    print("WebSocket Configuration Test")
    print("=" * 50)

    # Test default configuration
    print("Default Configuration:")
    default_config = WebSocketConfig()
    print(default_config)

    print(f"\nConfiguration Dictionary:")
    config_dict = default_config.to_dict()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")

    # Test different profiles
    print(f"\nTesting Configuration Profiles:")
    profiles = ['development', 'production', 'high_frequency', 'low_bandwidth', 'orb_optimized']

    for profile_name in profiles:
        try:
            config = create_websocket_config(profile_name)
            print(f"{profile_name}: {config.subscription_mode.value} mode, "
                  f"{config.max_reconnect_attempts} max attempts, "
                  f"fallback: {config.enable_rest_fallback}")
        except Exception as e:
            print(f"{profile_name}: {e}")

    # Test reconnect delay calculation
    print(f"\nReconnect Delay Testing (Exponential Backoff):")
    config = WebSocketProfiles.orb_optimized()
    for attempt in range(1, 6):
        delay = config.get_reconnect_delay(attempt)
        print(f"  Attempt {attempt}: {delay} seconds")