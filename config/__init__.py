# config/__init__.py

"""
Configuration Package for ORB Trading Strategy
Provides all configuration classes and settings for the Open Range Breakout strategy
"""

from .settings import (
    FyersConfig,
    ORBStrategyConfig,
    TradingConfig,
    Sector,
    SignalType
)

from .websocket_config import (
    WebSocketConfig,
    WebSocketMode,
    ReconnectionStrategy,
    WebSocketProfiles,
    create_websocket_config
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
    "Sector",
    "SignalType",

    # WebSocket configuration
    "WebSocketConfig",
    "WebSocketMode",
    "ReconnectionStrategy",
    "WebSocketProfiles",
    "create_websocket_config",
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
            portfolio_value=100000,  # Smaller portfolio for testing
            max_positions=3,  # Fewer positions for testing
            risk_per_trade_pct=0.5  # Lower risk for testing
        ),
        'trading': TradingConfig(
            monitoring_interval=5  # More frequent monitoring
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
            portfolio_value=1000000,
            max_positions=5,
            risk_per_trade_pct=1.0
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
            portfolio_value: float = 1000000,
            risk_per_trade: float = 1.0,
            max_positions: int = 5,
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