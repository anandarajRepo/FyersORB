# services/__init__.py

"""
Services Package for ORB Trading Strategy
Contains all service classes for data management, analysis, and market operations
"""

from .fyers_websocket_service import (
    ORBWebSocketService,
    ORBFallbackDataService,
    HybridORBDataService
)

from .analysis_service import (
    ORBTechnicalAnalysisService
)

from .market_timing_service import (
    MarketTimingService
)

from typing import Optional, Dict, Any
from datetime import datetime
import logging

__version__ = "2.0.0"
__author__ = "ORB Trading Strategy Team"

# Package exports
__all__ = [
    # WebSocket services
    "ORBWebSocketService",
    "ORBFallbackDataService",
    "HybridORBDataService",

    # Analysis services
    "ORBTechnicalAnalysisService",

    # Market services
    "MarketTimingService",

    # Utility functions
    "create_data_service",
    "create_analysis_service",
    "create_timing_service",
    "validate_service_health",
]

logger = logging.getLogger(__name__)


# Service factory functions
def create_data_service(fyers_config, ws_config, service_type: str = "hybrid"):
    """
    Create appropriate data service based on type

    Args:
        fyers_config: Fyers API configuration
        ws_config: WebSocket configuration
        service_type: Type of service ('websocket', 'fallback', 'hybrid')

    Returns:
        Configured data service instance
    """
    service_type = service_type.lower()

    try:
        if service_type == "websocket":
            logger.info("Creating WebSocket-only data service")
            return ORBWebSocketService(fyers_config, ws_config)

        elif service_type == "fallback":
            logger.info("Creating REST API fallback data service")
            return ORBFallbackDataService(fyers_config, ws_config)

        elif service_type == "hybrid":
            logger.info("Creating hybrid data service (WebSocket + REST fallback)")
            return HybridORBDataService(fyers_config, ws_config)

        else:
            logger.warning(f"Unknown service type '{service_type}', defaulting to hybrid")
            return HybridORBDataService(fyers_config, ws_config)

    except Exception as e:
        logger.error(f"Error creating data service: {e}")
        logger.info("Falling back to REST API service")
        return ORBFallbackDataService(fyers_config, ws_config)


def create_analysis_service(data_service):
    """
    Create technical analysis service

    Args:
        data_service: Data service instance for analysis

    Returns:
        Configured analysis service instance
    """
    try:
        logger.info("Creating ORB technical analysis service")
        return ORBTechnicalAnalysisService(data_service)
    except Exception as e:
        logger.error(f"Error creating analysis service: {e}")
        raise


def create_timing_service(trading_config):
    """
    Create market timing service

    Args:
        trading_config: Trading configuration

    Returns:
        Configured timing service instance
    """
    try:
        logger.info("Creating market timing service")
        return MarketTimingService(trading_config)
    except Exception as e:
        logger.error(f"Error creating timing service: {e}")
        raise


def validate_service_health(service) -> Dict[str, Any]:
    """
    Validate health of a service instance

    Args:
        service: Service instance to validate

    Returns:
        Dictionary with health status and details
    """
    health_status = {
        'service_type': type(service).__name__,
        'timestamp': datetime.now().isoformat(),
        'healthy': False,
        'details': {}
    }

    try:
        # Check if service has basic required attributes
        required_methods = ['connect', 'disconnect']

        for method in required_methods:
            if not hasattr(service, method):
                health_status['details'][f'missing_{method}'] = True
                return health_status

        # Check connection status for data services
        if hasattr(service, 'is_connected'):
            health_status['details']['connected'] = service.is_connected

        # Check subscribed symbols for data services
        if hasattr(service, 'subscribed_symbols'):
            health_status['details']['subscribed_symbols'] = len(service.subscribed_symbols)

        # Check for WebSocket specific attributes
        if hasattr(service, 'using_fallback'):
            health_status['details']['using_fallback'] = service.using_fallback

        # Check for analysis service capabilities
        if isinstance(service, ORBTechnicalAnalysisService):
            health_status['details']['analysis_methods'] = [
                'calculate_breakout_strength',
                'validate_breakout_signal',
                'calculate_position_size'
            ]

        # Check for timing service capabilities
        if isinstance(service, MarketTimingService):
            health_status['details']['current_phase'] = service.get_current_market_phase()
            health_status['details']['is_trading_time'] = service.is_trading_time()
            health_status['details']['is_orb_period'] = service.is_orb_period()

        health_status['healthy'] = True
        logger.debug(f"Service health check passed for {type(service).__name__}")

    except Exception as e:
        health_status['details']['error'] = str(e)
        logger.warning(f"Service health check failed for {type(service).__name__}: {e}")

    return health_status


# Service manager class
class ServiceManager:
    """Manages all services for the ORB strategy"""

    def __init__(self, fyers_config, strategy_config, trading_config, ws_config):
        self.fyers_config = fyers_config
        self.strategy_config = strategy_config
        self.trading_config = trading_config
        self.ws_config = ws_config

        # Service instances
        self.data_service = None
        self.analysis_service = None
        self.timing_service = None

        self._initialized = False

    def initialize_services(self, data_service_type: str = "hybrid") -> bool:
        """Initialize all required services"""
        try:
            logger.info("Initializing ORB strategy services...")

            # Create data service
            self.data_service = create_data_service(
                self.fyers_config,
                self.ws_config,
                data_service_type
            )

            # Create analysis service
            self.analysis_service = create_analysis_service(self.data_service)

            # Create timing service
            self.timing_service = create_timing_service(self.trading_config)

            self._initialized = True
            logger.info("All services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False

    def connect_services(self) -> bool:
        """Connect all services that require connections"""
        if not self._initialized:
            logger.error("Services not initialized")
            return False

        try:
            # Connect data service
            if self.data_service and hasattr(self.data_service, 'connect'):
                if not self.data_service.connect():
                    logger.error("Data service connection failed")
                    return False
                logger.info("Data service connected successfully")

            return True

        except Exception as e:
            logger.error(f"Service connection failed: {e}")
            return False

    def disconnect_services(self):
        """Disconnect all services gracefully"""
        try:
            if self.data_service and hasattr(self.data_service, 'disconnect'):
                self.data_service.disconnect()
                logger.info("Data service disconnected")

        except Exception as e:
            logger.error(f"Error disconnecting services: {e}")

    def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all services"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': True,
            'services': {}
        }

        services = {
            'data_service': self.data_service,
            'analysis_service': self.analysis_service,
            'timing_service': self.timing_service
        }

        for service_name, service in services.items():
            if service:
                service_health = validate_service_health(service)
                health_report['services'][service_name] = service_health

                if not service_health['healthy']:
                    health_report['overall_healthy'] = False
            else:
                health_report['services'][service_name] = {
                    'healthy': False,
                    'details': {'error': 'Service not initialized'}
                }
                health_report['overall_healthy'] = False

        return health_report

    def restart_data_service(self, service_type: str = "hybrid") -> bool:
        """Restart the data service with specified type"""
        try:
            logger.info("Restarting data service...")

            # Disconnect current service
            if self.data_service:
                self.data_service.disconnect()

            # Create new service
            self.data_service = create_data_service(
                self.fyers_config,
                self.ws_config,
                service_type
            )

            # Update analysis service with new data service
            self.analysis_service = create_analysis_service(self.data_service)

            # Connect new service
            if self.data_service.connect():
                logger.info("Data service restarted successfully")
                return True
            else:
                logger.error("Failed to connect restarted data service")
                return False

        except Exception as e:
            logger.error(f"Error restarting data service: {e}")
            return False


# Service diagnostics
def run_service_diagnostics(service_manager: ServiceManager) -> Dict[str, Any]:
    """Run comprehensive diagnostics on all services"""
    diagnostics = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'unknown',
        'services': {},
        'recommendations': []
    }

    try:
        # Get health report
        health_report = service_manager.get_service_health()
        diagnostics['services'] = health_report['services']

        # Analyze health status
        healthy_services = sum(1 for service in health_report['services'].values() if service['healthy'])
        total_services = len(health_report['services'])

        if healthy_services == total_services:
            diagnostics['overall_status'] = 'healthy'
        elif healthy_services > 0:
            diagnostics['overall_status'] = 'degraded'
            diagnostics['recommendations'].append("Some services are unhealthy - check logs")
        else:
            diagnostics['overall_status'] = 'critical'
            diagnostics['recommendations'].append("All services are unhealthy - restart required")

        # Service-specific recommendations
        if service_manager.data_service:
            data_health = health_report['services'].get('data_service', {})
            if not data_health.get('healthy', False):
                if data_health.get('details', {}).get('using_fallback', False):
                    diagnostics['recommendations'].append("Using REST API fallback - check WebSocket connection")
                else:
                    diagnostics['recommendations'].append("Data service unhealthy - restart data connection")

        # Check market timing
        if service_manager.timing_service:
            timing_health = health_report['services'].get('timing_service', {})
            if timing_health.get('healthy', False):
                current_phase = timing_health.get('details', {}).get('current_phase')
                if current_phase:
                    diagnostics['market_phase'] = current_phase

                    if current_phase == 'MARKET_HOLIDAY':
                        diagnostics['recommendations'].append("Market is closed - services in standby mode")
                    elif current_phase == 'ORB_PERIOD':
                        diagnostics['recommendations'].append("ORB period active - monitor range calculation")
                    elif current_phase == 'SIGNAL_GENERATION':
                        diagnostics['recommendations'].append("Optimal time for signal generation")

    except Exception as e:
        diagnostics['overall_status'] = 'error'
        diagnostics['error'] = str(e)
        diagnostics['recommendations'].append("Service diagnostics failed - check system health")

    return diagnostics


# Export service manager for easy access
def create_service_manager(fyers_config, strategy_config, trading_config, ws_config):
    """Create and return a configured service manager"""
    return ServiceManager(fyers_config, strategy_config, trading_config, ws_config)


# Performance monitoring utilities
def monitor_service_performance(service_manager: ServiceManager, duration_seconds: int = 60) -> Dict[str, Any]:
    """Monitor service performance over specified duration"""
    import time

    performance_data = {
        'monitoring_duration': duration_seconds,
        'start_time': datetime.now().isoformat(),
        'data_points': [],
        'summary': {}
    }

    start_time = time.time()
    end_time = start_time + duration_seconds

    try:
        while time.time() < end_time:
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'health': service_manager.get_service_health()
            }

            # Add connection status if available
            if service_manager.data_service and hasattr(service_manager.data_service, 'is_connected'):
                data_point['data_connected'] = service_manager.data_service.is_connected

            # Add subscription count if available
            if service_manager.data_service and hasattr(service_manager.data_service, 'subscribed_symbols'):
                data_point['subscribed_symbols'] = len(service_manager.data_service.subscribed_symbols)

            performance_data['data_points'].append(data_point)

            time.sleep(5)  # Sample every 5 seconds

        # Calculate summary statistics
        total_points = len(performance_data['data_points'])
        healthy_points = sum(1 for dp in performance_data['data_points']
                             if dp['health']['overall_healthy'])

        performance_data['summary'] = {
            'total_samples': total_points,
            'healthy_samples': healthy_points,
            'health_percentage': (healthy_points / total_points * 100) if total_points > 0 else 0,
            'end_time': datetime.now().isoformat()
        }

    except Exception as e:
        performance_data['error'] = str(e)

    return performance_data


# Logging utilities for services
def setup_service_logging(log_level: str = "INFO"):
    """Setup logging configuration for services"""
    import logging

    # Configure service-specific loggers
    service_loggers = [
        'services.fyers_websocket_service',
        'services.analysis_service',
        'services.market_timing_service'
    ]

    for logger_name in service_loggers:
        service_logger = logging.getLogger(logger_name)
        service_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add service-specific formatting if needed
        if not service_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {logger_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            service_logger.addHandler(handler)


# Service health monitoring decorator
def monitor_service_health(func):
    """Decorator to monitor service method health"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.debug(f"{func.__name__} executed successfully in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise

    return wrapper


# Export additional utilities
__all__.extend([
    "ServiceManager",
    "create_service_manager",
    "run_service_diagnostics",
    "monitor_service_performance",
    "setup_service_logging",
    "monitor_service_health"
])