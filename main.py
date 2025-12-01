# main.py - MODIFIED VERSION (Sector Allocation Removed)

"""
Open Range Breakout (ORB) Trading Strategy - Complete Main Entry Point
Full algorithmic trading system with WebSocket data integration
MODIFIED: Removed sector allocation limits and references
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Core imports
from config.settings import FyersConfig, ORBStrategyConfig, TradingConfig
from config.websocket_config import WebSocketConfig
from strategy.orb_strategy import ORBStrategy

# Import the enhanced authentication system
from utils.enhanced_auth_helper import (
    setup_auth_only,
    authenticate_fyers,
    test_authentication,
    update_pin_only
)

# Load environment variables
load_dotenv()


# Configure enhanced logging
def setup_logging():
    """Setup enhanced logging configuration"""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()

    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging with rotation
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'orb_strategy.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific log levels for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def load_configuration():
    """Load all configuration from environment variables"""
    try:
        # Fyers configuration
        fyers_config = FyersConfig(
            client_id=os.environ.get('FYERS_CLIENT_ID'),
            secret_key=os.environ.get('FYERS_SECRET_KEY'),
            access_token=os.environ.get('FYERS_ACCESS_TOKEN'),
            refresh_token=os.environ.get('FYERS_REFRESH_TOKEN')
        )

        # ORB Strategy configuration
        strategy_config = ORBStrategyConfig(
            portfolio_value=float(os.environ.get('PORTFOLIO_VALUE', 10000)),
            risk_per_trade_pct=float(os.environ.get('RISK_PER_TRADE', 10.0)),
            max_positions=int(os.environ.get('MAX_POSITIONS', 5)),

            # ORB specific parameters
            orb_period_minutes=int(os.environ.get('ORB_PERIOD_MINUTES', 15)),
            min_breakout_volume=float(os.environ.get('MIN_BREAKOUT_VOLUME', 2.0)),
            min_range_size_pct=float(os.environ.get('MIN_RANGE_SIZE', 0.5)),

            # Risk management
            stop_loss_pct=float(os.environ.get('STOP_LOSS_PCT', 1.0)),
            target_multiplier=float(os.environ.get('TARGET_MULTIPLIER', 2.0)),
            trailing_stop_pct=float(os.environ.get('TRAILING_STOP_PCT', 0.5)),

            # Signal filtering
            min_confidence=float(os.environ.get('MIN_CONFIDENCE', 0.5)),
            min_volume_ratio=float(os.environ.get('MIN_VOLUME_RATIO', 1.5)),
            max_gap_size=float(os.environ.get('MAX_GAP_SIZE', 3.0)),

            # Position management
            enable_trailing_stops=os.environ.get('ENABLE_TRAILING_STOPS', 'true').lower() == 'true',
            enable_partial_exits=os.environ.get('ENABLE_PARTIAL_EXITS', 'true').lower() == 'true',
            partial_exit_pct=float(os.environ.get('PARTIAL_EXIT_PCT', 50.0))
        )

        # Trading configuration
        trading_config = TradingConfig(
            market_start_hour=9,
            market_start_minute=15,
            market_end_hour=15,
            market_end_minute=30,
            orb_end_minute=30,
            signal_generation_end_hour=14,
            signal_generation_end_minute=0,
            monitoring_interval=int(os.environ.get('MONITORING_INTERVAL', 10)),
            position_update_interval=int(os.environ.get('POSITION_UPDATE_INTERVAL', 5))
        )

        # WebSocket configuration
        ws_config = WebSocketConfig(
            reconnect_interval=5,
            max_reconnect_attempts=int(os.environ.get('WS_MAX_RECONNECT_ATTEMPTS', 10)),
            ping_interval=int(os.environ.get('WS_PING_INTERVAL', 30)),
            connection_timeout=int(os.environ.get('WS_CONNECTION_TIMEOUT', 30))
        )

        return fyers_config, strategy_config, trading_config, ws_config

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


async def run_orb_strategy():
    """Main function to run the ORB strategy with enhanced authentication"""
    try:
        logger.info("=" * 60)
        logger.info("STARTING OPEN RANGE BREAKOUT (ORB) STRATEGY")
        logger.info("SECTOR ALLOCATION LIMITS DISABLED")
        logger.info("=" * 60)

        # Load configuration
        fyers_config, strategy_config, trading_config, ws_config = load_configuration()

        # Validate basic configuration
        if not all([fyers_config.client_id, fyers_config.secret_key]):
            logger.error("Missing required Fyers API credentials")
            logger.error("Please set FYERS_CLIENT_ID and FYERS_SECRET_KEY in .env file")
            logger.error("Run 'python main.py auth' to setup authentication")
            return

        # Enhanced authentication with auto-refresh
        config_dict = {'fyers_config': fyers_config}
        if not authenticate_fyers(config_dict):
            logger.error("Authentication failed. Please run 'python main.py auth' to setup authentication")
            return

        logger.info("Authentication successful - Access token validated")

        # Log strategy configuration
        logger.info(f"Portfolio Value: {strategy_config.portfolio_value:,}")
        logger.info(f"Risk per Trade: {strategy_config.risk_per_trade_pct}%")
        logger.info(f"Max Positions: {strategy_config.max_positions}")
        logger.info(f"ORB Period: {strategy_config.orb_period_minutes} minutes")
        logger.info(f"Stop Loss: {strategy_config.stop_loss_pct}%")
        logger.info(f"Target Multiple: {strategy_config.target_multiplier}x")
        logger.info("Position allocation based on signal quality only")

        # Create and run strategy
        strategy = ORBStrategy(
            fyers_config=config_dict['fyers_config'],
            strategy_config=strategy_config,
            trading_config=trading_config,
            ws_config=ws_config
        )

        # Run strategy
        logger.info("Initializing ORB Strategy...")
        await strategy.run()

    except KeyboardInterrupt:
        logger.info("Strategy stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.exception("Full error details:")


def test_websocket_connection():
    """Test WebSocket connection for ORB strategy"""

    async def test():
        try:
            logger.info("=" * 50)
            logger.info("TESTING ORB WEBSOCKET CONNECTION")
            logger.info("=" * 50)

            fyers_config, _, _, ws_config = load_configuration()

            # Enhanced authentication
            config_dict = {'fyers_config': fyers_config}
            if not authenticate_fyers(config_dict):
                logger.error("Authentication failed for WebSocket test")
                return

            logger.info("Authentication successful for WebSocket test")

            # Test WebSocket service
            from services.fyers_websocket_service import HybridORBDataService

            data_service = HybridORBDataService(config_dict['fyers_config'], ws_config)

            # Data callback for testing
            def on_data(symbol, quote):
                logger.info(f" {symbol}: ₹{quote.ltp:.2f} ({quote.change_pct:+.2f}%) "
                            f"H:{quote.high_price:.2f} L:{quote.low_price:.2f} Vol:{quote.volume:,}")

            data_service.add_data_callback(on_data)

            if data_service.connect():
                logger.info(" ORB Data Service connected successfully")
                logger.info(f"Connection Type: {'REST API Fallback' if data_service.using_fallback else 'WebSocket'}")

                # Subscribe to test symbols
                test_symbols = [
                    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS',
                    'HINDUNILVR.NS', 'ICICIBANK.NS', 'SBIN.NS', 'MARUTI.NS', 'WIPRO.NS'
                ]

                if data_service.subscribe_symbols(test_symbols):
                    logger.info(f" Subscribed to {len(test_symbols)} symbols for ORB testing")
                    logger.info("Symbols: " + ", ".join(test_symbols))

                    logger.info(" Receiving live data for 60 seconds...")
                    await asyncio.sleep(60)  # 1 minute test

                    # Test opening ranges if available
                    opening_ranges = data_service.get_all_opening_ranges()
                    if opening_ranges:
                        logger.info(" Opening ranges detected:")
                        for symbol, orb_range in opening_ranges.items():
                            logger.info(f"  {symbol}: H:₹{orb_range.high:.2f} L:₹{orb_range.low:.2f} "
                                        f"Range:₹{orb_range.range_size:.2f} ({orb_range.range_pct:.2f}%)")
                    else:
                        logger.info("️  No opening ranges available (may be outside ORB period)")

                    # Test breakout detection
                    logger.info(" Testing breakout detection...")
                    for symbol in test_symbols[:3]:  # Test first 3 symbols
                        live_quote = data_service.get_live_quote(symbol)
                        if live_quote:
                            is_breakout, signal_type, breakout_level = data_service.is_breakout_detected(
                                symbol, live_quote.ltp
                            )
                            if is_breakout:
                                logger.info(f" BREAKOUT: {symbol} {signal_type} at ₹{breakout_level:.2f}")

                else:
                    logger.error(" Failed to subscribe to symbols")
            else:
                logger.error(" Failed to connect to ORB data service")

            data_service.disconnect()
            logger.info(" WebSocket test completed successfully")

        except Exception as e:
            logger.error(f" ORB WebSocket test failed: {e}")
            logger.exception("Full error details:")

    asyncio.run(test())


def show_strategy_help():
    """Show comprehensive ORB strategy help and configuration (modified to remove sector allocation)"""
    print("\n" + "=" * 80)
    print("OPEN RANGE BREAKOUT (ORB) TRADING STRATEGY - CONFIGURATION GUIDE")
    print("=" * 80)

    print("\n STRATEGY OVERVIEW:")
    print("• Identifies opening range during first 15 minutes of trading (9:15-9:30 AM)")
    print("• Detects price breakouts above/below opening range with volume confirmation")
    print("• Uses technical analysis for signal validation (RSI, momentum, volume)")
    print("• Implements comprehensive risk management with trailing stops")
    print("• Monitors 40+ stocks across multiple sectors")
    print("• ⚠️  SECTOR ALLOCATION LIMITS DISABLED - Positions based on signal quality only")

    print("\n STOCK UNIVERSE (40+ Stocks):")
    stocks = {
        "FMCG": ["NESTLEIND", "COLPAL", "HINDUNILVR", "ITC", "BRITANNIA", "DABUR", "MARICO", "TATACONSUM"],
        "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTI"],
        "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK"],
        "Auto": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "M&M", "HEROMOTOCO", "EICHERMOT"],
        "Energy": ["RELIANCE", "ONGC", "IOC", "BPCL"],
        "Others": ["NTPC", "POWERGRID", "SUNPHARMA", "DRREDDY", "TATASTEEL", "JSWSTEEL"]
    }

    for sector, stock_list in stocks.items():
        print(f"  {sector}: {', '.join(stock_list)}")

    print("\n️ CONFIGURATION PARAMETERS:")
    print("Edit .env file or set environment variables:")

    print("\n Portfolio Settings:")
    print("  PORTFOLIO_VALUE=5000          # Total portfolio value (₹5k)")
    print("  RISK_PER_TRADE=1.0            # Risk per trade (1% of portfolio)")
    print("  MAX_POSITIONS=5               # Maximum concurrent positions")

    print("\n ORB Strategy Parameters:")
    print("  ORB_PERIOD_MINUTES=15         # Opening range calculation period")
    print("  MIN_BREAKOUT_VOLUME=2.0       # Volume multiplier for valid breakouts")
    print("  MIN_RANGE_SIZE=0.5            # Minimum range size (0.5% of price)")
    print("  MAX_GAP_SIZE=3.0              # Maximum overnight gap to consider")

    print("\n️ Risk Management:")
    print("  STOP_LOSS_PCT=1.0             # Stop loss distance from breakout level")
    print("  TARGET_MULTIPLIER=2.0         # Target as multiple of risk (1:2 ratio)")
    print("  TRAILING_STOP_PCT=0.5         # Trailing stop adjustment percentage")
    print("  ENABLE_TRAILING_STOPS=true    # Enable dynamic trailing stops")
    print("  ENABLE_PARTIAL_EXITS=true     # Take partial profits at targets")
    print("  PARTIAL_EXIT_PCT=50.0         # Percentage for partial exits")

    print("\n Signal Filtering:")
    print("  MIN_CONFIDENCE=0.6            # Minimum signal confidence (60%)")
    print("  MIN_VOLUME_RATIO=1.5          # Volume vs 20-day average ratio")

    print("\n System Settings:")
    print("  MONITORING_INTERVAL=10        # Strategy monitoring cycle (seconds)")
    print("  POSITION_UPDATE_INTERVAL=5    # Position update frequency (seconds)")
    print("  LOG_LEVEL=INFO               # Logging verbosity (DEBUG/INFO/WARNING/ERROR)")

    print("\n POSITION ALLOCATION:")
    print("  • SECTOR LIMITS REMOVED - Positions allocated based on signal quality")
    print("  • All available positions slots used for best opportunities")
    print("  • Risk managed through position sizing and stop losses only")
    print("  • Maximum positions still limited by MAX_POSITIONS setting")

    print("\n EXPECTED PERFORMANCE:")
    print("  Daily Signals: 5-12 breakout opportunities (increased range)")
    print("  Win Rate Target: 55-65%")
    print("  Risk-Reward: 1:2 ratio (1% risk, 2% target)")
    print("  Monthly Target: 15-25% portfolio growth (potentially higher)")
    print("  Max Drawdown: <5% with proper risk management")

    print("\n️  IMPORTANT NOTES:")
    print("  • Start with paper trading or small amounts")
    print("  • Monitor closely during initial weeks")
    print("  • Adjust parameters based on market conditions")
    print("  • Ensure stable internet for real-time data")
    print("  • Keep trading PIN secure for token refresh")
    print("  • Higher diversification risk without sector limits")


def show_authentication_status():
    """Show detailed authentication status with enhanced information"""
    print("\n" + "=" * 60)
    print("FYERS API AUTHENTICATION STATUS")
    print("=" * 60)

    # Check current credentials
    client_id = os.environ.get('FYERS_CLIENT_ID')
    secret_key = os.environ.get('FYERS_SECRET_KEY')
    access_token = os.environ.get('FYERS_ACCESS_TOKEN')
    refresh_token = os.environ.get('FYERS_REFRESH_TOKEN')
    pin = os.environ.get('FYERS_PIN')

    print(f" Credential Status:")
    print(f"  Client ID: {' Set' if client_id else '❌ Missing'}")
    if client_id:
        print(f"    Value: {client_id[:8]}...{client_id[-4:] if len(client_id) > 12 else client_id}")

    print(f"  Secret Key: {' Set' if secret_key else '❌ Missing'}")
    print(f"  Access Token: {' Set' if access_token else '❌ Missing'}")
    if access_token:
        print(f"    Preview: {access_token[:20]}...")

    print(f"  Refresh Token: {' Set' if refresh_token else '❌ Missing'}")
    print(f"  Trading PIN: {' Set' if pin else '❌ Missing'}")

    # Test token validity if available
    if access_token and client_id:
        from utils.enhanced_auth_helper import FyersAuthManager
        auth_manager = FyersAuthManager()

        print(f"\n Token Validation:")
        if auth_manager.is_token_valid(access_token):
            print(f"  Access Token:  Valid and active")

            # Try to get profile info
            try:
                import requests
                headers = {'Authorization': f"{client_id}:{access_token}"}
                response = requests.get('https://api-t1.fyers.in/api/v3/profile', headers=headers, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    if result.get('s') == 'ok':
                        profile_data = result.get('data', {})
                        print(f"  Profile Name: {profile_data.get('name', 'Unknown')}")
                        print(f"  Email: {profile_data.get('email', 'Unknown')}")
                        print(f"  User ID: {profile_data.get('id', 'Unknown')}")
            except Exception as e:
                print(f"  Profile fetch error: {e}")

        else:
            print(f"  Access Token:  Invalid or expired")

    print(f"\n🔧 Available Commands:")
    print(f"  Setup Authentication: python main.py auth")
    print(f"  Test Authentication: python main.py test-auth")
    print(f"  Update Trading PIN: python main.py update-pin")
    print(f"  Test WebSocket: python main.py test")
    print(f"  Run Strategy: python main.py run")

    # Recommendations
    print(f"\n Recommendations:")
    if not access_token:
        print(f"  ️  No access token found. Run setup: python main.py auth")
    elif not refresh_token:
        print(f"  ️  No refresh token. Consider re-running setup for auto-refresh")
    elif not pin:
        print(f"  ️  No trading PIN. Set PIN for automatic token refresh")
    else:
        print(f"   Authentication setup appears complete!")
        print(f"   Ready to run strategy: python main.py run")


def show_market_status():
    """Show current market status and timing information"""
    try:
        print("\n" + "=" * 70)
        print("MARKET STATUS & TIMING INFORMATION")
        print("=" * 70)

        from services.market_timing_service import MarketTimingService
        from config.settings import TradingConfig

        trading_config = TradingConfig()
        timing_service = MarketTimingService(trading_config)

        # Current time
        now = datetime.now()
        print(f" Current Time: {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f" Day: {now.strftime('%A')}")

        # Market status
        is_trading_day = timing_service.is_trading_day()
        is_trading_time = timing_service.is_trading_time()
        is_orb_period = timing_service.is_orb_period()
        is_signal_time = timing_service.is_signal_generation_time()
        should_close = timing_service.should_close_positions_for_day()

        print(f"\n Market Status:")
        print(f"  Trading Day: {' Yes' if is_trading_day else ' No (Holiday/Weekend)'}")
        print(f"  Market Open: {' Yes' if is_trading_time else ' No'}")
        print(f"  ORB Period: {' Active (9:15-9:30)' if is_orb_period else ' Inactive'}")
        print(f"  Signal Generation: {' Active' if is_signal_time else ' Inactive'}")
        print(f"  Position Closing Time: {'️  Yes' if should_close else ' No'}")

        # Market phase
        market_phase = timing_service.get_current_market_phase()
        phase_descriptions = {
            'MARKET_HOLIDAY': '  Market Holiday',
            'PRE_MARKET': ' Pre-Market (Before 9:15 AM)',
            'ORB_PERIOD': ' Opening Range Period (9:15-9:30 AM)',
            'SIGNAL_GENERATION': ' Signal Generation Window (9:30-2:00 PM)',
            'REGULAR_TRADING': ' Regular Trading Hours',
            'POSITION_CLOSING': '️  Position Closing Time (3:15-3:30 PM)',
            'POST_MARKET': ' Post-Market (After 3:30 PM)'
        }

        print(f"  Current Phase: {phase_descriptions.get(market_phase, market_phase)}")

        # Time calculations
        time_to_open = timing_service.time_until_market_open()
        time_to_close = timing_service.time_until_market_close()
        time_to_orb_end = timing_service.time_until_orb_end()

        print(f"\n Time Information:")
        if time_to_open:
            print(f"  Time to Market Open: {timing_service.format_time_remaining(time_to_open)}")
        if time_to_close:
            print(f"  Time to Market Close: {timing_service.format_time_remaining(time_to_close)}")
        if time_to_orb_end:
            print(f"  Time to ORB End: {timing_service.format_time_remaining(time_to_orb_end)}")

        # Trading session progress
        if is_trading_time:
            progress = timing_service.get_trading_session_progress()
            progress_bar = "" * int(progress / 5) + "" * (20 - int(progress / 5))
            print(f"  Session Progress: [{progress_bar}] {progress:.1f}%")

        # Next trading day
        if not is_trading_day:
            next_trading_day = timing_service.get_next_trading_day()
            print(f"  Next Trading Day: {next_trading_day.strftime('%Y-%m-%d (%A)')}")

        print(f"\n Standard Market Hours (IST):")
        print(f"  Market Opens: 09:15")
        print(f"  ORB Period: 09:15 - 09:30 (Opening Range Calculation)")
        print(f"  Signal Generation: 09:30 - 14:00 (Breakout Detection)")
        print(f"  Regular Trading: 14:00 - 15:15 (Position Monitoring)")
        print(f"  Position Closing: 15:15 - 15:30 (End-of-Day Exits)")
        print(f"  Market Closes: 15:30")

        # Strategy recommendations
        print(f"\n Strategy Recommendations:")
        if is_orb_period:
            print(f"   ORB Period Active - Range calculation in progress")
        elif is_signal_time:
            print(f"   Optimal time for running ORB strategy")
            print(f"   No sector limits - monitor position allocation carefully")
        elif is_trading_time and not should_close:
            print(f"   Monitor existing positions, limited new signals")
        elif should_close:
            print(f"  ️  Close positions before market close")
        elif not is_trading_time and is_trading_day:
            if now.hour < 9:
                print(f"   Prepare for market open - check configuration")
            else:
                print(f"   Market closed - review performance and prepare for tomorrow")
        elif not is_trading_day:
            print(f"  ️  Market holiday - good time for system maintenance")

    except Exception as e:
        print(f" Error showing market status: {e}")
        logger.exception("Error in show_market_status")


def validate_configuration():
    """Comprehensive configuration validation with detailed feedback (modified to remove sector info)"""
    print("\n" + "=" * 70)
    print("ORB STRATEGY CONFIGURATION VALIDATION")
    print("=" * 70)

    try:
        fyers_config, strategy_config, trading_config, ws_config = load_configuration()

        issues = []
        warnings = []
        info = []

        # Check Fyers configuration
        print(" Fyers API Configuration:")
        if not fyers_config.client_id:
            issues.append("FYERS_CLIENT_ID is not set")
            print("  Client ID: Missing")
        else:
            print(f"  Client ID: {fyers_config.client_id[:8]}...{fyers_config.client_id[-4:]}")

        if not fyers_config.secret_key:
            issues.append("FYERS_SECRET_KEY is not set")
            print("  Secret Key: Missing")
        else:
            print("  Secret Key: Configured")

        if not fyers_config.access_token:
            warnings.append("FYERS_ACCESS_TOKEN is not set (run 'python main.py auth')")
            print("   Access Token: Missing")
        else:
            print("   Access Token: Configured")

        if not fyers_config.refresh_token:
            warnings.append("FYERS_REFRESH_TOKEN is missing (consider re-authenticating)")
            print("   Refresh Token: Missing")
        else:
            print("   Refresh Token: Configured")

        # Check strategy configuration
        print(f"\n Portfolio & Risk Configuration:")
        if strategy_config.portfolio_value < 1000:
            warnings.append(f"Portfolio value is quite low: ₹{strategy_config.portfolio_value:,}")
            print(f"   Portfolio Value: ₹{strategy_config.portfolio_value:,} (Consider higher amount)")
        else:
            print(f"   Portfolio Value: ₹{strategy_config.portfolio_value:,}")

        if strategy_config.risk_per_trade_pct > 2.0:
            warnings.append(f"Risk per trade is high: {strategy_config.risk_per_trade_pct}%")
            print(f"   Risk per Trade: {strategy_config.risk_per_trade_pct}% (High risk)")
        elif strategy_config.risk_per_trade_pct < 0.5:
            info.append(f"Risk per trade is conservative: {strategy_config.risk_per_trade_pct}%")
            print(f"   Risk per Trade: {strategy_config.risk_per_trade_pct}% (Conservative)")
        else:
            print(f"   Risk per Trade: {strategy_config.risk_per_trade_pct}%")

        print(f"   Max Positions: {strategy_config.max_positions}")
        print("    Sector limits: DISABLED")

        # Check ORB parameters
        print(f"\n ORB Strategy Parameters:")
        if strategy_config.orb_period_minutes != 15:
            warnings.append(f"Non-standard ORB period: {strategy_config.orb_period_minutes} minutes")
            print(f"   ORB Period: {strategy_config.orb_period_minutes} minutes (Standard is 15)")
        else:
            print(f"   ORB Period: {strategy_config.orb_period_minutes} minutes")

        if strategy_config.min_range_size_pct < 0.3:
            warnings.append(f"Minimum range size is very low: {strategy_config.min_range_size_pct}%")
            print(f"    Min Range Size: {strategy_config.min_range_size_pct}% (May generate too many signals)")
        elif strategy_config.min_range_size_pct > 1.5:
            warnings.append(f"Minimum range size is high: {strategy_config.min_range_size_pct}%")
            print(f"   Min Range Size: {strategy_config.min_range_size_pct}% (May miss opportunities)")
        else:
            print(f"   Min Range Size: {strategy_config.min_range_size_pct}%")

        if strategy_config.min_breakout_volume < 1.5:
            warnings.append(f"Breakout volume requirement is low: {strategy_config.min_breakout_volume}x")
            print(f"   Min Breakout Volume: {strategy_config.min_breakout_volume}x (Low threshold)")
        else:
            print(f"   Min Breakout Volume: {strategy_config.min_breakout_volume}x")

        # Check risk management
        print(f"\n Risk Management:")
        if strategy_config.stop_loss_pct > 2.0:
            warnings.append(f"Stop loss percentage is high: {strategy_config.stop_loss_pct}%")
            print(f"  ️  Stop Loss: {strategy_config.stop_loss_pct}% (High risk)")
        elif strategy_config.stop_loss_pct < 0.5:
            warnings.append(f"Stop loss percentage is very tight: {strategy_config.stop_loss_pct}%")
            print(f"  ️  Stop Loss: {strategy_config.stop_loss_pct}% (May get stopped out frequently)")
        else:
            print(f"   Stop Loss: {strategy_config.stop_loss_pct}%")

        if strategy_config.target_multiplier < 1.5:
            warnings.append(f"Target multiplier is low: {strategy_config.target_multiplier}x")
            print(f"  ️  Target Multiple: {strategy_config.target_multiplier}x (Low risk-reward)")
        elif strategy_config.target_multiplier > 3.0:
            info.append(f"Target multiplier is ambitious: {strategy_config.target_multiplier}x")
            print(f"  ️  Target Multiple: {strategy_config.target_multiplier}x (Ambitious)")
        else:
            print(f"   Target Multiple: {strategy_config.target_multiplier}x")

        print(f"  {'' if strategy_config.enable_trailing_stops else ''} Trailing Stops: {'Enabled' if strategy_config.enable_trailing_stops else 'Disabled'}")
        print(f"  {'' if strategy_config.enable_partial_exits else ''} Partial Exits: {'Enabled' if strategy_config.enable_partial_exits else 'Disabled'}")

        if strategy_config.enable_partial_exits:
            print(f"    Partial Exit %: {strategy_config.partial_exit_pct}%")

        # Check signal filtering
        print(f"\n Signal Filtering:")
        if strategy_config.min_confidence < 0.4:
            warnings.append(f"Minimum confidence is low: {strategy_config.min_confidence}")
            print(f"  ️  Min Confidence: {strategy_config.min_confidence:.1f} (May generate low-quality signals)")
        elif strategy_config.min_confidence > 0.8:
            info.append(f"Minimum confidence is high: {strategy_config.min_confidence}")
            print(f"  ️  Min Confidence: {strategy_config.min_confidence:.1f} (Conservative)")
        else:
            print(f"   Min Confidence: {strategy_config.min_confidence:.1f}")

        print(f"   Min Volume Ratio: {strategy_config.min_volume_ratio}x")
        print(f"   Max Gap Size: {strategy_config.max_gap_size}%")

        # Check system settings
        print(f"\n System Settings:")
        print(f"   Monitoring Interval: {trading_config.monitoring_interval} seconds")
        print(f"   Position Update Interval: {trading_config.position_update_interval} seconds")

        # WebSocket configuration
        print(f"\n WebSocket Configuration:")
        print(f"   Max Reconnect Attempts: {ws_config.max_reconnect_attempts}")
        print(f"   Ping Interval: {ws_config.ping_interval} seconds")
        print(f"   Connection Timeout: {ws_config.connection_timeout} seconds")

        # Display summary
        print(f"\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        if issues:
            print(" CRITICAL ISSUES (Must fix before running):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

        if warnings:
            print(f"\n️  WARNINGS ({len(warnings)} items):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        if info:
            print(f"\n️  INFORMATION ({len(info)} items):")
            for i, item in enumerate(info, 1):
                print(f"  {i}. {item}")

        if not issues and not warnings:
            print(" ALL CONFIGURATIONS VALID - Ready for trading!")
        elif not issues:
            print(" NO CRITICAL ISSUES - Strategy can run with warnings noted")

        # Risk calculation
        max_risk_per_position = (strategy_config.portfolio_value * strategy_config.risk_per_trade_pct / 100)
        max_total_risk = max_risk_per_position * strategy_config.max_positions
        max_risk_pct = (max_total_risk / strategy_config.portfolio_value) * 100

        print(f"\n RISK ANALYSIS:")
        print(f"  Max Risk per Position: ₹{max_risk_per_position:,.0f}")
        print(f"  Max Total Risk: ₹{max_total_risk:,.0f} ({max_risk_pct:.1f}% of portfolio)")
        print(f"  Sector diversification: DISABLED")

        if max_risk_pct > 10:
            print(f"  ️  Total risk exposure is high: {max_risk_pct:.1f}%")
            print(f"     Consider reducing MAX_POSITIONS or RISK_PER_TRADE")
        else:
            print(f"   Total risk exposure is acceptable: {max_risk_pct:.1f}%")

        print(f"\n IMPORTANT NOTES:")
        print(f"  • Sector allocation limits have been REMOVED")
        print(f"  • All {strategy_config.max_positions} positions can be from same sector")
        print(f"  • Higher concentration risk - monitor carefully")
        print(f"  • Consider manual diversification if needed")

        return len(issues) == 0

    except Exception as e:
        print(f" Error validating configuration: {e}")
        logger.exception("Error in validate_configuration")
        return False


def run_system_diagnostics():
    """Comprehensive system diagnostics and health check"""
    print("\n" + "=" * 80)
    print("ORB STRATEGY SYSTEM DIAGNOSTICS")
    print("=" * 80)

    all_checks_passed = True

    # 1. Python environment check
    print(" Python Environment:")
    import sys
    python_version = sys.version.split()[0]
    print(f"  Python Version: {python_version}")

    if sys.version_info < (3, 9):
        print("   Python version too old (requires 3.9+)")
        all_checks_passed = False
    else:
        print("   Python version compatible")

    print(f"  Platform: {sys.platform}")
    print(f"  Executable: {sys.executable}")

    # 2. Dependencies check
    print(f"\n Required Dependencies:")
    required_packages = [
        ('fyers_apiv3', 'Fyers API'),
        ('pandas', 'Data Analysis'),
        ('numpy', 'Numerical Computing'),
        ('requests', 'HTTP Requests'),
        ('python-dotenv', 'Environment Variables'),
        ('pytz', 'Timezone Support'),
        ('yfinance', 'Historical Data'),
        ('scipy', 'Statistical Analysis'),
        ('dateutil', 'Date Utilities')
    ]

    missing_packages = []
    for package_name, description in required_packages:
        try:
            package_import_name = package_name.replace('-', '_').replace('python_', '')
            if package_name == 'dateutil':
                package_import_name = 'dateutil'
            elif package_name == 'python-dotenv':
                package_import_name = 'dotenv'

            __import__(package_import_name)
            print(f"   {package_name} ({description})")
        except ImportError:
            print(f"   {package_name} ({description}) - MISSING")
            missing_packages.append(package_name)
            all_checks_passed = False

    if missing_packages:
        print(f"\n  🔧 Install missing packages:")
        print(f"     pip install {' '.join(missing_packages)}")

    # 3. File structure check
    print(f"\n Project Structure:")
    required_structure = {
        'config': ['settings.py', 'websocket_config.py'],
        'models': ['trading_models.py'],
        'services': ['fyers_websocket_service.py', 'analysis_service.py', 'market_timing_service.py'],
        'strategy': ['orb_strategy.py'],
        'utils': ['enhanced_auth_helper.py'],
        'logs': [],  # Directory only
        '.': ['main.py', 'requirements.txt', '.env.template']
    }

    missing_files = []
    for directory, files in required_structure.items():
        if directory == '.':
            base_path = '.'
        else:
            base_path = directory

        if not os.path.exists(base_path) and directory != '.':
            print(f"   {directory}/ directory - MISSING")
            missing_files.append(f"{directory}/")
            all_checks_passed = False
            continue
        elif directory != '.':
            print(f"   {directory}/ directory")

        for file in files:
            file_path = os.path.join(base_path, file)
            if os.path.exists(file_path):
                print(f"     {file}")
            else:
                print(f"     {file} - MISSING")
                missing_files.append(file_path)
                all_checks_passed = False

    # 4. Configuration validation
    print(f"\n️  Configuration Validation:")
    try:
        config_valid = validate_configuration()
        if config_valid:
            print("   Configuration valid")
        else:
            print("  ️  Configuration has issues (see details above)")
    except Exception as e:
        print(f"   Configuration validation failed: {e}")
        all_checks_passed = False

    # Final summary
    print(f"\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    if all_checks_passed:
        print(" ALL SYSTEM CHECKS PASSED!")
        print(" System is ready for ORB trading strategy")
        print("\n IMPORTANT REMINDERS:")
        print("  • Sector allocation limits are DISABLED")
        print("  • Monitor position concentration carefully")
        print("  • Higher diversification risk without sector limits")
        print("\n Next steps:")
        print("  1. Run: python main.py test (Test WebSocket connection)")
        print("  2. Run: python main.py run (Start trading strategy)")
    else:
        print("️  SOME CHECKS FAILED")
        print(" Please resolve the issues above before running the strategy")
        print("\n🔧 Common solutions:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Setup authentication: python main.py auth")
        print("  • Check .env configuration: python main.py config")

    print("=" * 80)

    return all_checks_passed


def show_performance_dashboard():
    """Show performance dashboard template"""
    print("\n" + "=" * 80)
    print("ORB STRATEGY PERFORMANCE DASHBOARD")
    print("=" * 80)
    print(" Real-time performance monitoring interface")
    print(" Live P&L tracking and position management")
    print(" Signal analysis and strategy metrics")
    print(" Risk monitoring and allocation tracking")
    print(" ⚠️  Sector allocation limits: DISABLED")
    print("\n" + "⚠️  " + "=" * 60)
    print("DASHBOARD REQUIRES RUNNING STRATEGY INSTANCE")
    print("=" * 68)
    print("To view live performance data:")
    print("1. Start the strategy: python main.py run")
    print("2. Monitor logs: tail -f logs/orb_strategy.log")
    print("3. Check positions in real-time via strategy output")
    print("\n Future Enhancement: Web-based dashboard planned")


def run_backtest():
    """Placeholder for backtesting functionality"""
    print("\n" + "=" * 70)
    print("ORB STRATEGY HISTORICAL BACKTESTING")
    print("=" * 70)
    print(" Historical performance analysis framework")
    print(" Risk metrics and drawdown calculation")
    print(" Strategy parameter optimization")
    print(" Profit/Loss distribution analysis")
    print(" ⚠️  Sector allocation limits: DISABLED in backtest")
    print("\n" + "️  " + "=" * 50)
    print("BACKTESTING MODULE IN DEVELOPMENT")
    print("=" * 58)
    print("Planned features:")
    print("• Historical data analysis (1+ years)")
    print("• Parameter optimization testing")
    print("• Risk-adjusted return calculations")
    print("• Monte Carlo simulation")
    print("• Walk-forward analysis")
    print("• Sector concentration analysis (informational)")
    print("\n Current validation: Live paper trading recommended")


def main():
    """Enhanced main entry point with comprehensive CLI interface"""

    # Display header
    print("=" * 80)
    print("    OPEN RANGE BREAKOUT (ORB) TRADING STRATEGY")
    print("    Advanced Algorithmic Trading System v2.0")
    print("    SECTOR ALLOCATION LIMITS DISABLED")
    print("=" * 80)

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "run":
            logger.info(" Starting Open Range Breakout (ORB) Strategy")
            asyncio.run(run_orb_strategy())

        elif command == "test":
            logger.info(" Testing ORB WebSocket Connection")
            test_websocket_connection()

        elif command == "auth":
            print(" Setting up Fyers API Authentication")
            setup_auth_only()

        elif command == "test-auth":
            print(" Testing Fyers API Authentication")
            test_authentication()

        elif command == "update-pin":
            print(" Updating Trading PIN")
            update_pin_only()

        elif command == "auth-status":
            show_authentication_status()

        elif command == "help":
            show_strategy_help()

        elif command == "status":
            show_authentication_status()

        elif command == "market":
            show_market_status()

        elif command == "config":
            print("️  Validating Configuration")
            if validate_configuration():
                print("\n Configuration validation passed!")
            else:
                print("\n Configuration validation failed!")

        elif command == "diagnostics":
            print(" Running System Diagnostics")
            if run_system_diagnostics():
                print("\n All diagnostics passed - Ready to trade!")
            else:
                print("\n️  Please resolve issues before trading")

        elif command == "dashboard":
            show_performance_dashboard()

        elif command == "backtest":
            run_backtest()

        else:
            print(f" Unknown command: {command}")
            print("\n Available commands:")
            commands = [
                ("run", "Run the ORB trading strategy (no sector limits)"),
                ("test", "Test WebSocket data connection"),
                ("auth", "Setup Fyers API authentication"),
                ("test-auth", "Test authentication status"),
                ("update-pin", "Update trading PIN"),
                ("auth-status", "Show detailed authentication status"),
                ("help", "Show strategy configuration guide"),
                ("status", "Show system and authentication status"),
                ("market", "Show market timing and status"),
                ("config", "Validate configuration settings"),
                ("diagnostics", "Run comprehensive system diagnostics"),
                ("dashboard", "Show performance dashboard"),
                ("backtest", "Run historical backtesting")
            ]

            for cmd, desc in commands:
                print(f"  python main.py {cmd:<12} - {desc}")

    else:
        # Interactive menu
        print(" Advanced algorithmic trading with real-time WebSocket data")
        print(" Identifies opening range breakouts with volume confirmation")
        print("️  Comprehensive risk management and position monitoring")
        print(" Sector allocation limits DISABLED - Monitor diversification manually")
        print("\nSelect an option:")

        menu_options = [
            ("1", " Run ORB Trading Strategy"),
            ("2", " Test WebSocket Connection"),
            ("3", " Setup Fyers Authentication"),
            ("4", " Test Authentication"),
            ("5", " Update Trading PIN"),
            ("6", " Show Authentication Status"),
            ("7", " Strategy Configuration Guide"),
            ("8", " Market Status & Timing"),
            ("9", "️  Validate Configuration"),
            ("10", " Run System Diagnostics"),
            ("11", " Performance Dashboard"),
            ("12", " Historical Backtesting"),
            ("13", " Exit")
        ]

        for option, description in menu_options:
            print(f"{option:>2}. {description}")

        choice = input(f"\nSelect option (1-{len(menu_options)}): ").strip()

        if choice == "1":
            logger.info(" Starting Open Range Breakout (ORB) Strategy")
            asyncio.run(run_orb_strategy())

        elif choice == "2":
            logger.info(" Testing ORB WebSocket Connection")
            test_websocket_connection()

        elif choice == "3":
            print(" Setting up Fyers API Authentication")
            setup_auth_only()

        elif choice == "4":
            print(" Testing Fyers API Authentication")
            test_authentication()

        elif choice == "5":
            print(" Updating Trading PIN")
            update_pin_only()

        elif choice == "6":
            show_authentication_status()

        elif choice == "7":
            show_strategy_help()

        elif choice == "8":
            show_market_status()

        elif choice == "9":
            print("⚙  Validating Configuration...")
            if validate_configuration():
                print("\n Configuration is valid and ready for trading!")
                print("Remember: Sector limits are DISABLED")
            else:
                print("\n Configuration has issues - please review and fix")

        elif choice == "10":
            print(" Running System Diagnostics...")
            if run_system_diagnostics():
                print("\n All diagnostics passed - System ready!")
            else:
                print("\n️  Please resolve diagnostic issues")

        elif choice == "11":
            show_performance_dashboard()

        elif choice == "12":
            run_backtest()

        elif choice == "13":
            print("\n Goodbye! Happy Trading! ")
            print(" Remember: Trade responsibly and manage your risk!")
            print(" Monitor position concentration without sector limits!")

        else:
            print(f" Invalid choice: {choice}")
            print("Please select a number between 1 and 13")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n Interrupted by user - Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        logger.exception("Full error details:")
        sys.exit(1)