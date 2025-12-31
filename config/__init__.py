# config/__init__.py

"""
Configuration Package for ORB Trading Strategy
Provides all configuration classes and settings for the Open Range Breakout strategy
"""

from .settings import (
    FyersConfig,
    ORBStrategyConfig,
    TradingConfig,
    SignalType
)

from .websocket_config import (
    WebSocketConfig,
    WebSocketMode,
    ReconnectionStrategy,
    WebSocketProfiles,
    create_websocket_config
)

# Import centralized symbol management
from .symbols import (
    symbol_manager,
    get_orb_symbols,
    get_orb_fyers_symbols,
    convert_to_fyers_format,
    convert_from_fyers_format,
    validate_orb_symbol,
    get_symbol_mappings
)

__version__ = "2.0.0"
__author__ = "ORB Trading Strategy Team"

# Package metadata
__all__ = [
    # Core configuration classes
    "FyersConfig",
    "ORBStrategyConfig",
    "TradingConfig",

    # Enums
    "SignalType",

    # WebSocket configuration
    "WebSocketConfig",
    "WebSocketMode",
    "ReconnectionStrategy",
    "WebSocketProfiles",
    "create_websocket_config",

    # Symbol management
    "symbol_manager",
    "get_orb_symbols",
    "get_orb_fyers_symbols",
    "convert_to_fyers_format",
    "convert_from_fyers_format",
    "validate_orb_symbol",
    "get_symbol_mappings",
]


# Configuration shortcuts for common use cases
def get_default_config():
    """Get default configuration for ORB strategy"""
    return {
        'fyers': FyersConfig(
            client_id="",
            secret_key="",
            access_token=None
        ),
        'strategy': ORBStrategyConfig(),
        'trading': TradingConfig(),
        'websocket': create_websocket_config("orb_optimized")
    }


def get_development_config():
    """Get development configuration with enhanced logging"""
    return {
        'fyers': FyersConfig(
            client_id="",
            secret_key="",
            access_token=None
        ),
        'strategy': ORBStrategyConfig(
            portfolio_value=30000,  # Smaller portfolio for testing
            max_positions=3,  # Fewer positions for testing
            risk_per_trade_pct=0.5  # Lower risk for testing
        ),
        'trading': TradingConfig(
            monitoring_interval=1  # More frequent monitoring
        ),
        'websocket': create_websocket_config("development")
    }


def get_production_config():
    """Get production configuration optimized for stability"""
    return {
        'fyers': FyersConfig(
            client_id="",
            secret_key="",
            access_token=None
        ),
        'strategy': ORBStrategyConfig(
            portfolio_value=30000,
            max_positions=3,
            risk_per_trade_pct=30.0
        ),
        'trading': TradingConfig(),
        'websocket': create_websocket_config("production")
    }


# Validation functions
def validate_fyers_config(config: FyersConfig) -> bool:
    """Validate Fyers configuration"""
    if not config.client_id:
        return False
    if not config.secret_key:
        return False
    if len(config.client_id) < 10:
        return False
    return True


def validate_strategy_config(config: ORBStrategyConfig) -> bool:
    """Validate strategy configuration"""
    if config.portfolio_value <= 0:
        return False
    if config.risk_per_trade_pct <= 0 or config.risk_per_trade_pct > 5:
        return False
    if config.max_positions <= 0 or config.max_positions > 20:
        return False
    if config.orb_period_minutes <= 0 or config.orb_period_minutes > 60:
        return False
    return True


# Configuration factory
class ConfigFactory:
    """Factory class for creating different configuration sets"""

    @staticmethod
    def create_config_set(environment: str = "default"):
        """Create a complete configuration set for given environment"""
        if environment.lower() == "development":
            return get_development_config()
        elif environment.lower() == "production":
            return get_production_config()
        else:
            return get_default_config()

    @staticmethod
    def create_custom_config(
            portfolio_value: float = 30000,
            risk_per_trade: float = 1.0,
            max_positions: int = 3,
            websocket_profile: str = "orb_optimized"
    ):
        """Create custom configuration with specified parameters"""
        return {
            'fyers': FyersConfig(
                client_id="",
                secret_key="",
                access_token=None
            ),
            'strategy': ORBStrategyConfig(
                portfolio_value=portfolio_value,
                risk_per_trade_pct=risk_per_trade,
                max_positions=max_positions
            ),
            'trading': TradingConfig(),
            'websocket': create_websocket_config(websocket_profile)
        }


# Export factory for easy access
config_factory = ConfigFactory()


# Symbol management utilities
def get_orb_trading_universe():
    """Get complete ORB trading universe information"""
    return {
        'total_symbols': symbol_manager.get_trading_universe_size(),
        'symbols': get_orb_symbols(),
        'fyers_symbols': get_orb_fyers_symbols(),
        'sample_mappings': {
            symbol: convert_to_fyers_format(symbol)
            for symbol in get_orb_symbols()[:5]
        }
    }


def validate_symbol_list(symbols: list) -> dict:
    """Validate a list of symbols for ORB trading"""
    validation_result = {
        'valid_symbols': [],
        'invalid_symbols': [],
        'total_count': len(symbols),
        'valid_count': 0,
        'invalid_count': 0
    }

    for symbol in symbols:
        if validate_orb_symbol(symbol):
            validation_result['valid_symbols'].append(symbol)
        else:
            validation_result['invalid_symbols'].append(symbol)

    validation_result['valid_count'] = len(validation_result['valid_symbols'])
    validation_result['invalid_count'] = len(validation_result['invalid_symbols'])

    return validation_result