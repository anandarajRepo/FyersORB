# strategy/__init__.py

"""
Strategy Package for ORB Trading System
Contains the main Open Range Breakout strategy implementation and related utilities
"""

from .orb_strategy import ORBStrategy

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

__version__ = "2.0.0"
__author__ = "ORB Trading Strategy Team"

# Package exports
__all__ = [
    "ORBStrategy",
    "create_orb_strategy",
    "validate_strategy_config",
    "get_strategy_metrics",
    "StrategyFactory",
    "BacktestRunner",
    "PerformanceAnalyzer"
]

logger = logging.getLogger(__name__)


# Strategy factory function
def create_orb_strategy(fyers_config, strategy_config, trading_config, ws_config):
    """
    Create and configure an ORB strategy instance

    Args:
        fyers_config: Fyers API configuration
        strategy_config: ORB strategy configuration
        trading_config: Trading session configuration
        ws_config: WebSocket configuration

    Returns:
        Configured ORBStrategy instance
    """
    try:
        logger.info("Creating ORB strategy instance")

        strategy = ORBStrategy(
            fyers_config=fyers_config,
            strategy_config=strategy_config,
            trading_config=trading_config,
            ws_config=ws_config
        )

        logger.info("ORB strategy instance created successfully")
        return strategy

    except Exception as e:
        logger.error(f"Error creating ORB strategy: {e}")
        raise


def validate_strategy_config(strategy_config) -> Dict[str, Any]:
    """
    Validate strategy configuration parameters

    Args:
        strategy_config: Strategy configuration to validate

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }

    try:
        # Portfolio validation
        if strategy_config.portfolio_value <= 0:
            validation_result['errors'].append("Portfolio value must be positive")
            validation_result['valid'] = False
        elif strategy_config.portfolio_value < 100000:
            validation_result['warnings'].append("Portfolio value is quite low for ORB strategy")

        # Risk validation
        if strategy_config.risk_per_trade_pct <= 0 or strategy_config.risk_per_trade_pct > 5:
            validation_result['errors'].append("Risk per trade must be between 0.1% and 5%")
            validation_result['valid'] = False
        elif strategy_config.risk_per_trade_pct > 2:
            validation_result['warnings'].append("Risk per trade is high (>2%)")

        # Position validation
        if strategy_config.max_positions <= 0 or strategy_config.max_positions > 20:
            validation_result['errors'].append("Max positions must be between 1 and 20")
            validation_result['valid'] = False
        elif strategy_config.max_positions > 10:
            validation_result['warnings'].append("High number of max positions for ORB strategy")

        # ORB specific validation
        if strategy_config.orb_period_minutes <= 0 or strategy_config.orb_period_minutes > 60:
            validation_result['errors'].append("ORB period must be between 1 and 60 minutes")
            validation_result['valid'] = False
        elif strategy_config.orb_period_minutes != 15:
            validation_result['warnings'].append("Non-standard ORB period (15 minutes is typical)")

        if strategy_config.min_breakout_volume < 1.0:
            validation_result['warnings'].append("Low breakout volume requirement may generate false signals")

        if strategy_config.min_range_size_pct <= 0 or strategy_config.min_range_size_pct > 5:
            validation_result['errors'].append("Min range size must be between 0.1% and 5%")
            validation_result['valid'] = False

        # Risk management validation
        if strategy_config.stop_loss_pct <= 0 or strategy_config.stop_loss_pct > 5:
            validation_result['errors'].append("Stop loss percentage must be between 0.1% and 5%")
            validation_result['valid'] = False

        if strategy_config.target_multiplier <= 0 or strategy_config.target_multiplier > 10:
            validation_result['errors'].append("Target multiplier must be between 0.1 and 10")
            validation_result['valid'] = False
        elif strategy_config.target_multiplier < 1.5:
            validation_result['warnings'].append("Low target multiplier may not provide good risk-reward")

        # Signal filtering validation
        if strategy_config.min_confidence < 0 or strategy_config.min_confidence > 1:
            validation_result['errors'].append("Min confidence must be between 0 and 1")
            validation_result['valid'] = False
        elif strategy_config.min_confidence < 0.5:
            validation_result['warnings'].append("Low confidence threshold may generate poor quality signals")

        # Calculate total risk exposure
        max_risk_per_position = strategy_config.portfolio_value * strategy_config.risk_per_trade_pct / 100
        max_total_risk = max_risk_per_position * strategy_config.max_positions
        max_risk_percentage = (max_total_risk / strategy_config.portfolio_value) * 100

        if max_risk_percentage > 15:
            validation_result['warnings'].append(f"High total risk exposure: {max_risk_percentage:.1f}%")

        # Recommendations
        if validation_result['valid']:
            validation_result['recommendations'].extend([
                "Start with paper trading to validate strategy",
                "Monitor performance closely during first week",
                "Adjust parameters based on market conditions",
                f"Current max risk exposure: {max_risk_percentage:.1f}% of portfolio"
            ])

    except Exception as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Validation error: {str(e)}")

    return validation_result


def get_strategy_metrics(strategy: ORBStrategy) -> Dict[str, Any]:
    """
    Get comprehensive strategy performance metrics

    Args:
        strategy: ORB strategy instance

    Returns:
        Dictionary with strategy metrics
    """
    try:
        if hasattr(strategy, 'get_performance_summary'):
            return strategy.get_performance_summary()
        else:
            return {
                'error': 'Strategy does not support performance summary',
                'strategy_type': type(strategy).__name__
            }
    except Exception as e:
        return {
            'error': f'Error getting strategy metrics: {str(e)}',
            'strategy_type': type(strategy).__name__
        }


# Strategy factory class
class StrategyFactory:
    """Factory for creating and managing strategy instances"""

    @staticmethod
    def create_paper_trading_strategy(fyers_config, strategy_config, trading_config, ws_config):
        """Create strategy configured for paper trading"""
        # Modify config for paper trading
        paper_config = strategy_config
        paper_config.portfolio_value = min(paper_config.portfolio_value, 100000)  # Limit size
        paper_config.max_positions = min(paper_config.max_positions, 3)  # Fewer positions
        paper_config.risk_per_trade_pct = min(paper_config.risk_per_trade_pct, 0.5)  # Lower risk

        strategy = create_orb_strategy(fyers_config, paper_config, trading_config, ws_config)
        logger.info("Paper trading strategy created")
        return strategy

    @staticmethod
    def create_live_trading_strategy(fyers_config, strategy_config, trading_config, ws_config):
        """Create strategy configured for live trading"""
        # Validate configuration for live trading
        validation = validate_strategy_config(strategy_config)

        if not validation['valid']:
            raise ValueError(f"Invalid configuration for live trading: {validation['errors']}")

        if validation['warnings']:
            logger.warning(f"Configuration warnings: {validation['warnings']}")

        strategy = create_orb_strategy(fyers_config, strategy_config, trading_config, ws_config)
        logger.info("Live trading strategy created")
        return strategy

    @staticmethod
    def create_demo_strategy():
        """Create strategy with demo/test configuration"""
        from config import FyersConfig, ORBStrategyConfig, TradingConfig, create_websocket_config

        # Demo configurations
        demo_fyers_config = FyersConfig(
            client_id="DEMO_CLIENT_ID",
            secret_key="DEMO_SECRET_KEY",
            access_token="DEMO_ACCESS_TOKEN"
        )

        demo_strategy_config = ORBStrategyConfig(
            portfolio_value=100000,  # 1 lakh demo portfolio
            risk_per_trade_pct=0.5,  # Conservative risk
            max_positions=2,  # Limited positions
            orb_period_minutes=15,
            min_confidence=0.7  # Higher confidence threshold
        )

        demo_trading_config = TradingConfig()
        demo_ws_config = create_websocket_config("development")

        strategy = create_orb_strategy(
            demo_fyers_config,
            demo_strategy_config,
            demo_trading_config,
            demo_ws_config
        )

        logger.info("Demo strategy created")
        return strategy


# Backtesting framework
class BacktestRunner:
    """Backtesting framework for ORB strategy"""

    def __init__(self, strategy_config, start_date: datetime, end_date: datetime):
        self.strategy_config = strategy_config
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}

    def run_backtest(self, historical_data: Dict) -> Dict[str, Any]:
        """
        Run backtest on historical data

        Args:
            historical_data: Dictionary with historical price data

        Returns:
            Backtest results and metrics
        """
        # Placeholder implementation - would integrate with actual historical data
        backtest_results = {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'status': 'not_implemented',
            'message': 'Backtesting framework is a placeholder - historical data integration needed'
        }

        logger.info("Backtest run completed (placeholder)")
        return backtest_results

    def optimize_parameters(self, parameter_ranges: Dict) -> Dict[str, Any]:
        """
        Optimize strategy parameters using historical data

        Args:
            parameter_ranges: Dictionary with parameter ranges to test

        Returns:
            Optimization results
        """
        optimization_results = {
            'best_parameters': self.strategy_config.__dict__,
            'best_performance': 0.0,
            'parameter_sensitivity': {},
            'status': 'not_implemented',
            'message': 'Parameter optimization framework is a placeholder'
        }

        logger.info("Parameter optimization completed (placeholder)")
        return optimization_results


# Performance analysis utilities
class PerformanceAnalyzer:
    """Analyzes strategy performance and generates reports"""

    @staticmethod
    def analyze_trades(trades: List) -> Dict[str, Any]:
        """Analyze list of completed trades"""
        if not trades:
            return {'total_trades': 0, 'message': 'No trades to analyze'}

        # Basic trade analysis
        winning_trades = [t for t in trades if getattr(t, 'net_pnl', 0) > 0]
        losing_trades = [t for t in trades if getattr(t, 'net_pnl', 0) <= 0]

        total_pnl = sum(getattr(t, 'net_pnl', 0) for t in trades)

        analysis = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'total_pnl': total_pnl,
            'average_trade': total_pnl / len(trades) if trades else 0,
            'profit_factor': 0.0
        }

        # Calculate profit factor
        gross_profit = sum(getattr(t, 'net_pnl', 0) for t in winning_trades)
        gross_loss = abs(sum(getattr(t, 'net_pnl', 0) for t in losing_trades))

        if gross_loss > 0:
            analysis['profit_factor'] = gross_profit / gross_loss

        # Calculate average holding periods
        if hasattr(trades[0], 'holding_period'):
            analysis['avg_holding_period'] = sum(t.holding_period for t in trades) / len(trades)

        return analysis

    @staticmethod
    def generate_performance_report(strategy: ORBStrategy) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            performance_summary = get_strategy_metrics(strategy)

            report = {
                'timestamp': datetime.now().isoformat(),
                'strategy_type': 'Open Range Breakout',
                'performance_summary': performance_summary,
                'recommendations': [],
                'warnings': []
            }

            # Add performance-based recommendations
            if 'win_rate' in performance_summary:
                win_rate = performance_summary['win_rate']
                if win_rate < 50:
                    report['warnings'].append(f"Low win rate: {win_rate:.1f}%")
                    report['recommendations'].append("Consider tightening entry criteria")
                elif win_rate > 70:
                    report['recommendations'].append("Excellent win rate - consider increasing position size")

            if 'daily_pnl' in performance_summary:
                daily_pnl = performance_summary['daily_pnl']
                if daily_pnl < 0:
                    report['warnings'].append(f"Negative daily P&L: ₹{daily_pnl:.2f}")
                    report['recommendations'].append("Review recent trades and market conditions")

            return report

        except Exception as e:
            return {
                'error': f"Error generating performance report: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }


# Strategy monitoring utilities
def monitor_strategy_health(strategy: ORBStrategy) -> Dict[str, Any]:
    """Monitor overall strategy health and status"""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'healthy': False,
        'issues': [],
        'recommendations': []
    }

    try:
        # Check if strategy is properly initialized
        if not hasattr(strategy, 'data_service'):
            health_status['issues'].append("Data service not initialized")
            return health_status

        if not hasattr(strategy, 'analysis_service'):
            health_status['issues'].append("Analysis service not initialized")
            return health_status

        # Check data connectivity
        if hasattr(strategy.data_service, 'is_connected'):
            if not strategy.data_service.is_connected:
                health_status['issues'].append("Data service not connected")
                health_status['recommendations'].append("Check internet connection and API credentials")

        # Check position status
        if hasattr(strategy, 'positions'):
            active_positions = len(strategy.positions)
            max_positions = getattr(strategy.strategy_config, 'max_positions', 5)

            if active_positions >= max_positions:
                health_status['recommendations'].append("At maximum position limit")

        # Check market timing
        if hasattr(strategy, 'timing_service'):
            if not strategy.timing_service.is_trading_time():
                health_status['recommendations'].append("Outside trading hours")

        # If no critical issues found, mark as healthy
        if not health_status['issues']:
            health_status['healthy'] = True

    except Exception as e:
        health_status['issues'].append(f"Health check error: {str(e)}")

    return health_status


# Export utility instances
strategy_factory = StrategyFactory()
performance_analyzer = PerformanceAnalyzer()


# Quick start functions
def quick_start_paper_trading():
    """Quick start function for paper trading setup"""
    try:
        demo_strategy = strategy_factory.create_demo_strategy()
        logger.info("Demo strategy created for paper trading")
        return demo_strategy
    except Exception as e:
        logger.error(f"Error creating demo strategy: {e}")
        return None


def validate_strategy_readiness(strategy: ORBStrategy) -> bool:
    """Validate if strategy is ready for trading"""
    try:
        health = monitor_strategy_health(strategy)
        config_validation = validate_strategy_config(strategy.strategy_config)

        return health['healthy'] and config_validation['valid']

    except Exception as e:
        logger.error(f"Error validating strategy readiness: {e}")
        return False


# Add to exports
__all__.extend([
    "strategy_factory",
    "performance_analyzer",
    "monitor_strategy_health",
    "quick_start_paper_trading",
    "validate_strategy_readiness"
])