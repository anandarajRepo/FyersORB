# services/analysis_service.py

"""
Enhanced Technical Analysis Service for Open Range Breakout Strategy
Provides ORB-specific indicators, volume analysis, and momentum calculations
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from models.trading_models import LiveQuote, OpenRange, ORBSignal
from config.settings import SignalType, Sector
from config.symbols import convert_to_fyers_format

logger = logging.getLogger(__name__)


class ORBTechnicalAnalysisService:
    """Enhanced technical analysis service for ORB strategy"""

    def __init__(self, websocket_service):
        self.websocket_service = websocket_service
        self.volume_cache: Dict[str, List[int]] = {}
        self.price_cache: Dict[str, List[float]] = {}

        # Initialize Fyers client for historical data
        self.fyers_client = None
        self._initialize_fyers_client()

    def _initialize_fyers_client(self):
        """Initialize Fyers client for historical data fetching"""
        try:
            # Get Fyers config from the data service
            if hasattr(self.websocket_service, 'fyers_config'):
                fyers_config = self.websocket_service.fyers_config
                self.fyers_client = fyersModel.FyersModel(
                    client_id=fyers_config.client_id,
                    token=fyers_config.access_token,
                    log_path=""
                )
                logger.info("Fyers client initialized successfully for historical data")
            elif hasattr(self.websocket_service, 'fallback_service') and self.websocket_service.fallback_service:
                # Use fallback service's client
                self.fyers_client = self.websocket_service.fallback_service.fyers
                logger.info("Using fallback service's Fyers client for historical data")
            elif hasattr(self.websocket_service, 'primary_service') and self.websocket_service.primary_service:
                # Try to get config from primary service
                if hasattr(self.websocket_service.primary_service, 'fyers_config'):
                    fyers_config = self.websocket_service.primary_service.fyers_config
                    self.fyers_client = fyersModel.FyersModel(
                        client_id=fyers_config.client_id,
                        token=fyers_config.access_token,
                        log_path=""
                    )
                    logger.info("Fyers client initialized from primary service config")
            else:
                logger.warning("Could not initialize Fyers client - no config available")
        except Exception as e:
            logger.error(f"Error initializing Fyers client: {e}")

    def calculate_breakout_strength(self, symbol: str, breakout_price: float,
                                    opening_range: OpenRange, live_quote: LiveQuote) -> float:
        """
        Calculate breakout strength score (0-100)
        Higher score indicates stronger breakout
        """
        try:
            strength_score = 0.0

            # 1. Range size factor (30% weight)
            # Larger ranges tend to produce more significant breakouts
            range_size_factor = min(opening_range.range_pct / 2.0, 1.0) * 30
            strength_score += range_size_factor

            # 2. Volume confirmation (25% weight)
            volume_ratio = self.calculate_volume_ratio(symbol, live_quote)
            volume_factor = min(volume_ratio / 3.0, 1.0) * 25
            strength_score += volume_factor

            # 3. Breakout momentum (20% weight)
            momentum_score = self.calculate_momentum_score(symbol, breakout_price, opening_range)
            strength_score += momentum_score * 20

            # 4. Price action quality (15% weight)
            price_action_score = self.calculate_price_action_score(symbol, live_quote)
            strength_score += price_action_score * 15

            # 5. Market context (10% weight)
            market_context_score = self.calculate_market_context_score()
            strength_score += market_context_score * 10

            return min(max(strength_score, 0), 100)

        except Exception as e:
            logger.error(f"Error calculating breakout strength for {symbol}: {e}")
            return 0.0

    def calculate_volume_ratio(self, symbol: str, live_quote: LiveQuote) -> float:
        """Calculate current volume vs average volume ratio"""
        try:
            # Get historical average volume
            avg_volume = self.get_average_volume(symbol, period_days=20)

            if avg_volume <= 0:
                return 1.0

            current_volume = live_quote.volume

            # Estimate full day volume based on time elapsed
            now = datetime.now()
            market_hours_elapsed = max((now.hour - 9) + (now.minute - 15) / 60, 0.5)

            if market_hours_elapsed > 0:
                estimated_full_day_volume = current_volume * (6.5 / market_hours_elapsed)
                volume_ratio = estimated_full_day_volume / avg_volume
                return volume_ratio

            return 1.0

        except Exception as e:
            logger.error(f"Error calculating volume ratio for {symbol}: {e}")
            return 1.0

    def calculate_momentum_score(self, symbol: str, breakout_price: float,
                                 opening_range: OpenRange) -> float:
        """Calculate momentum score for breakout (0-1)"""
        try:
            # Get current price
            live_quote = self.websocket_service.get_live_quote(symbol)
            if not live_quote:
                return 0.0

            current_price = live_quote.ltp

            # Calculate distance from breakout level
            if breakout_price == opening_range.high:  # Upside breakout
                if current_price > breakout_price:
                    momentum_distance = (current_price - breakout_price) / breakout_price
                    return min(momentum_distance * 10, 1.0)  # Scale to 0-1
            else:  # Downside breakout
                if current_price < breakout_price:
                    momentum_distance = (breakout_price - current_price) / breakout_price
                    return min(momentum_distance * 10, 1.0)

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating momentum score for {symbol}: {e}")
            return 0.0

    def calculate_price_action_score(self, symbol: str, live_quote: LiveQuote) -> float:
        """Calculate price action quality score (0-1)"""
        try:
            # Get recent price data
            price_history = self.get_recent_prices(symbol, periods=10)

            if len(price_history) < 5:
                return 0.5  # Neutral score if insufficient data

            # Calculate various price action metrics
            score = 0.0

            # 1. Trend consistency
            recent_trend = self.calculate_trend_consistency(price_history)
            score += recent_trend * 0.4

            # 2. Volatility appropriateness (not too high, not too low)
            volatility = float(np.std(price_history)) / float(np.mean(price_history))
            optimal_volatility = 0.01  # 1% daily volatility is optimal
            volatility_score = 1 - abs(volatility - optimal_volatility) / optimal_volatility
            score += max(volatility_score, 0) * 0.3

            # 3. Price level relative to recent range
            price_position = self.calculate_price_position(live_quote.ltp, price_history)
            score += price_position * 0.3

            return min(max(score, 0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating price action score for {symbol}: {e}")
            return 0.5

    def calculate_market_context_score(self) -> float:
        """Calculate overall market context score (0-1)"""
        try:
            # Simple market context - can be enhanced with index data
            # For now, return neutral score
            return 0.5

        except Exception as e:
            logger.error(f"Error calculating market context score: {e}")
            return 0.5

    def get_average_volume(self, symbol: str, period_days: int = 20) -> float:
        """Get average volume for a symbol using Fyers API"""
        try:
            if not self.fyers_client:
                logger.warning(f"Fyers client not initialized, cannot fetch volume for {symbol}")
                return 0.0

            # Convert symbol to Fyers format
            fyers_symbol = convert_to_fyers_format(symbol)
            if not fyers_symbol:
                logger.error(f"Could not convert symbol {symbol} to Fyers format")
                return 0.0

            # Calculate date range (fetch more days to account for weekends/holidays)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 10)

            # Prepare data for Fyers history API
            data = {
                "symbol": fyers_symbol,
                "resolution": "D",  # Daily candles
                "date_format": "1",  # Unix timestamp format
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }

            # Fetch historical data
            response = self.fyers_client.history(data=data)

            if response and response.get('s') == 'ok':
                candles = response.get('candles', [])
                if len(candles) == 0:
                    logger.warning(f"No volume data returned for {symbol}")
                    return 0.0

                # Extract volumes (index 5 in candle data: [timestamp, open, high, low, close, volume])
                volumes = [candle[5] for candle in candles if len(candle) > 5]

                # Take last 'period_days' volumes
                recent_volumes = volumes[-period_days:] if len(volumes) > period_days else volumes

                if len(recent_volumes) > 0:
                    avg_volume = sum(recent_volumes) / len(recent_volumes)
                    logger.debug(f"Average volume for {symbol}: {avg_volume:.0f} (over {len(recent_volumes)} days)")
                    return float(avg_volume)
                else:
                    return 0.0
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                logger.error(f"Failed to fetch volume data for {symbol}: {error_msg}")
                return 0.0

        except Exception as e:
            logger.error(f"Error getting average volume for {symbol}: {e}")
            return 0.0

    def get_recent_prices(self, symbol: str, periods: int = 10) -> List[float]:
        """Get recent price data using Fyers API"""
        try:
            # Try to get from cache first
            if symbol in self.price_cache:
                cached_prices = self.price_cache[symbol][-periods:]
                if len(cached_prices) >= periods:
                    return cached_prices

            if not self.fyers_client:
                logger.warning(f"Fyers client not initialized, cannot fetch prices for {symbol}")
                return []

            # Convert symbol to Fyers format
            fyers_symbol = convert_to_fyers_format(symbol)
            if not fyers_symbol:
                logger.error(f"Could not convert symbol {symbol} to Fyers format")
                return []

            # Calculate date range (5 days of 5-minute candles)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)

            # Prepare data for Fyers history API
            data = {
                "symbol": fyers_symbol,
                "resolution": "5",  # 5-minute candles
                "date_format": "1",  # Unix timestamp format
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }

            # Fetch historical data
            response = self.fyers_client.history(data=data)

            if response and response.get('s') == 'ok':
                candles = response.get('candles', [])
                if len(candles) == 0:
                    logger.warning(f"No price data returned for {symbol}")
                    return []

                # Extract close prices (index 4 in candle data: [timestamp, open, high, low, close, volume])
                prices = [float(candle[4]) for candle in candles if len(candle) > 4]

                # Cache the prices
                self.price_cache[symbol] = prices

                # Return last 'periods' prices
                return prices[-periods:] if len(prices) > periods else prices
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                logger.error(f"Failed to fetch price data for {symbol}: {error_msg}")
                return []

        except Exception as e:
            logger.error(f"Error getting recent prices for {symbol}: {e}")
            return []

    def calculate_trend_consistency(self, prices: List[float]) -> float:
        """Calculate trend consistency score (0-1)"""
        try:
            if len(prices) < 3:
                return 0.5

            # Calculate price changes
            changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

            # Count consistent direction changes
            positive_changes = sum(1 for change in changes if change > 0)
            total_changes = len(changes)

            # Calculate consistency
            if total_changes == 0:
                return 0.5

            consistency = max(positive_changes, total_changes - positive_changes) / total_changes
            return consistency

        except Exception as e:
            logger.error(f"Error calculating trend consistency: {e}")
            return 0.5

    def calculate_price_position(self, current_price: float, price_history: List[float]) -> float:
        """Calculate where current price sits in recent range (0-1)"""
        try:
            if not price_history:
                return 0.5

            min_price = min(price_history)
            max_price = max(price_history)

            if max_price == min_price:
                return 0.5

            position = (current_price - min_price) / (max_price - min_price)
            return min(max(position, 0), 1)

        except Exception as e:
            logger.error(f"Error calculating price position: {e}")
            return 0.5

    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """Calculate RSI for a symbol using Fyers API"""
        try:
            if not self.fyers_client:
                logger.warning(f"Fyers client not initialized, cannot calculate RSI for {symbol}")
                return 50.0  # Neutral RSI

            # Convert symbol to Fyers format
            fyers_symbol = convert_to_fyers_format(symbol)
            if not fyers_symbol:
                logger.error(f"Could not convert symbol {symbol} to Fyers format")
                return 50.0

            # Calculate date range (need more days for RSI calculation)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period + 15)

            # Prepare data for Fyers history API
            data = {
                "symbol": fyers_symbol,
                "resolution": "D",  # Daily candles
                "date_format": "1",  # Unix timestamp format
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }

            # Fetch historical data
            response = self.fyers_client.history(data=data)

            if response and response.get('s') == 'ok':
                candles = response.get('candles', [])
                if len(candles) < period:
                    logger.warning(f"Insufficient data for RSI calculation for {symbol}")
                    return 50.0  # Neutral RSI

                # Extract close prices (index 4 in candle data: [timestamp, open, high, low, close, volume])
                prices = np.array([float(candle[4]) for candle in candles if len(candle) > 4])

                if len(prices) < period:
                    return 50.0

                # Calculate price changes
                deltas = np.diff(prices)

                # Separate gains and losses
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)

                # Calculate average gain and loss
                avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
                avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0

                if avg_loss == 0:
                    return 100.0 if avg_gain > 0 else 50.0

                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

                logger.debug(f"RSI for {symbol}: {rsi:.2f}")
                return float(rsi) if not np.isnan(rsi) else 50.0
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                logger.error(f"Failed to fetch data for RSI calculation for {symbol}: {error_msg}")
                return 50.0

        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return 50.0

    def is_range_significant(self, opening_range: OpenRange, min_range_pct: float = 0.5) -> bool:
        """Check if opening range is significant enough for trading"""
        try:
            # Check minimum range size
            if opening_range.range_pct < min_range_pct:
                return False

            # Check if range is not too large (avoid gap situations)
            if opening_range.range_pct > 5.0:
                return False

            # Check volume during range formation
            if opening_range.volume < 1000:  # Minimum volume threshold
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking range significance: {e}")
            return False

    def calculate_stop_loss_level(self, signal_type: SignalType, breakout_price: float,
                                  opening_range: OpenRange, buffer_pct: float = 1.0) -> float:
        """Calculate appropriate stop loss level"""
        try:
            if signal_type == SignalType.LONG:
                # For long positions, stop loss below range low
                stop_loss = opening_range.low * (1 - buffer_pct / 100)
                return stop_loss
            else:
                # For short positions, stop loss above range high
                stop_loss = opening_range.high * (1 + buffer_pct / 100)
                return stop_loss

        except Exception as e:
            logger.error(f"Error calculating stop loss level: {e}")
            return breakout_price

    def calculate_target_price(self, signal_type: SignalType, entry_price: float,
                               stop_loss: float, target_multiplier: float = 2.0) -> float:
        """Calculate target price based on risk-reward ratio"""
        try:
            risk_amount = abs(entry_price - stop_loss)

            if signal_type == SignalType.LONG:
                target_price = entry_price + (risk_amount * target_multiplier)
            else:
                target_price = entry_price - (risk_amount * target_multiplier)

            return target_price

        except Exception as e:
            logger.error(f"Error calculating target price: {e}")
            return entry_price

    def calculate_trailing_stop(self, signal_type: SignalType, entry_price: float,
                                current_price: float, highest_price: float,
                                lowest_price: float, trailing_pct: float = 0.5) -> float:
        """Calculate trailing stop loss level"""
        try:
            if signal_type == SignalType.LONG:
                # For long positions, trail below highest price
                trailing_stop = highest_price * (1 - trailing_pct / 100)
                # Never move stop loss below entry
                return max(trailing_stop, entry_price)
            else:
                # For short positions, trail above lowest price
                trailing_stop = lowest_price * (1 + trailing_pct / 100)
                # Never move stop loss below entry (keep stop above entry)
                return max(trailing_stop, entry_price)

        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return entry_price

    def evaluate_breakout_quality(self, symbol: str, breakout_price: float,
                                  opening_range: OpenRange, signal_type: SignalType) -> Dict[str, float]:
        """Comprehensive breakout quality evaluation"""
        try:
            live_quote = self.websocket_service.get_live_quote(symbol)
            if not live_quote:
                return {'overall_score': 0.0}

            # Individual component scores
            scores = {}

            # 1. Volume confirmation
            volume_ratio = self.calculate_volume_ratio(symbol, live_quote)
            scores['volume_score'] = min(volume_ratio / 2.0, 1.0)  # Normalize to 0-1

            # 2. Momentum strength
            momentum_score = self.calculate_momentum_score(symbol, breakout_price, opening_range)
            scores['momentum_score'] = momentum_score

            # 3. Range quality
            range_quality = self.calculate_range_quality(opening_range)
            scores['range_quality'] = range_quality

            # 4. Technical indicators
            rsi = self.calculate_rsi(symbol)
            if signal_type == SignalType.LONG:
                rsi_score = (rsi - 30) / 40 if rsi > 30 else 0  # Prefer RSI 30-70 for long
            else:
                rsi_score = (70 - rsi) / 40 if rsi < 70 else 0  # Prefer RSI 30-70 for short
            scores['rsi_score'] = max(min(rsi_score, 1.0), 0.0)

            # 5. Time factor (early breakouts are better)
            time_score = self.calculate_time_factor_score()
            scores['time_score'] = time_score

            # Calculate weighted overall score
            weights = {
                'volume_score': 0.25,
                'momentum_score': 0.25,
                'range_quality': 0.20,
                'rsi_score': 0.15,
                'time_score': 0.15
            }

            overall_score = sum(scores[key] * weights[key] for key in weights if key in scores)
            scores['overall_score'] = overall_score

            logger.info(
                f"Breakout quality scores for {symbol}: {scores}, volume_score: {volume_ratio}, momentum_score: {momentum_score}, range_quality: {range_quality}, rsi_score: {rsi_score}, time_score: {time_score}, overall_score: {overall_score}")

            return scores

        except Exception as e:
            logger.error(f"Error evaluating breakout quality for {symbol}: {e}")
            return {'overall_score': 0.0}

    def calculate_range_quality(self, opening_range: OpenRange) -> float:
        """Calculate quality of the opening range (0-1)"""
        try:
            score = 0.0

            # 1. Range size appropriateness (0.5% - 2% is optimal)
            if 0.5 <= opening_range.range_pct <= 2.0:
                size_score = 1.0
            elif opening_range.range_pct < 0.5:
                size_score = opening_range.range_pct / 0.5
            else:  # > 2%
                size_score = max(0.0, 1 - (opening_range.range_pct - 2.0) / 3.0)

            score += size_score * 0.6

            # 2. Volume during range formation
            if opening_range.volume > 10000:  # Good volume
                volume_score = 1.0
            elif opening_range.volume > 5000:  # Moderate volume
                volume_score = 0.7
            else:  # Low volume
                volume_score = 0.3

            score += volume_score * 0.4

            return min(max(score, 0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating range quality: {e}")
            return 0.5

    def calculate_time_factor_score(self) -> float:
        """Calculate time factor score - earlier breakouts score higher"""
        try:
            now = datetime.now()

            # ORB period ends at 9:30 AM
            orb_end = now.replace(hour=9, minute=30, second=0, microsecond=0)

            # Best breakouts happen within 30 minutes of ORB end
            optimal_window_end = orb_end + timedelta(minutes=30)

            if now <= orb_end:
                return 0.0  # No breakout during ORB period
            elif now <= optimal_window_end:
                # Linear score from 1.0 at ORB end to 0.7 at 30 minutes
                minutes_elapsed = (now - orb_end).total_seconds() / 60
                score = 1.0 - (minutes_elapsed / 30) * 0.3
                return max(score, 0.7)
            else:
                # Declining score after optimal window
                minutes_elapsed = (now - orb_end).total_seconds() / 60
                score = 0.7 * np.exp(-(minutes_elapsed - 30) / 60)
                return max(score, 0.1)

        except Exception as e:
            logger.error(f"Error calculating time factor score: {e}")
            return 0.5

    def get_sector_strength(self, sector: Sector) -> float:
        """Get relative strength of a sector (placeholder implementation)"""
        try:
            # Placeholder - in real implementation, this would analyze sector performance
            sector_scores = {
                Sector.IT: 0.8,
                Sector.BANKING: 0.7,
                Sector.FMCG: 0.6,
                Sector.AUTO: 0.5,
                Sector.PHARMA: 0.7,
                Sector.METALS: 0.4,
                Sector.ENERGY: 0.6,
                Sector.TELECOM: 0.5,
            }

            return sector_scores.get(sector, 0.5)

        except Exception as e:
            logger.error(f"Error getting sector strength for {sector}: {e}")
            return 0.5

    def validate_breakout_signal(self, symbol: str, opening_range: OpenRange,
                                 current_price: float, signal_type: SignalType,
                                 min_confidence: float = 0.6) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate if a breakout signal meets quality criteria
        Returns: (is_valid, confidence_score, detailed_scores)
        """
        try:
            # Check if range is significant enough
            if not self.is_range_significant(opening_range):
                return False, 0.0, {'error': 'Insignificant range'}

            # Determine breakout level
            if signal_type == SignalType.LONG:
                breakout_price = opening_range.high
                if current_price <= breakout_price:
                    return False, 0.0, {'error': 'Price not above range high'}
            else:
                breakout_price = opening_range.low
                if current_price >= breakout_price:
                    return False, 0.0, {'error': 'Price not below range low'}

            # Evaluate breakout quality
            quality_scores = self.evaluate_breakout_quality(symbol, breakout_price, opening_range, signal_type)

            overall_confidence = quality_scores.get('overall_score', 0.0)

            # Check minimum confidence threshold
            is_valid = overall_confidence >= min_confidence

            logger.info(f"is_valid: {is_valid}, overall_confidence: {overall_confidence}, minimum_confidence: {min_confidence}, quality_scores: {quality_scores}")

            return is_valid, overall_confidence, quality_scores

        except Exception as e:
            logger.error(f"Error validating breakout signal for {symbol}: {e}")
            return False, 0.0, {'error': str(e)}

    def calculate_position_size(self, portfolio_value: float, risk_per_trade_pct: float,
                                entry_price: float, stop_loss: float) -> int:
        """Calculate appropriate position size based on risk management"""
        try:
            risk_amount = portfolio_value * (risk_per_trade_pct / 100)
            # price_risk = abs(entry_price - stop_loss) // Old Code logic
            price_risk = abs(entry_price)

            if price_risk <= 0:
                return 0

            quantity = int(risk_amount / price_risk)
            logger.info(f"portfolio_value: {portfolio_value}, risk_per_trade_pct: {risk_per_trade_pct}, risk_amount: {risk_amount}, price_risk: {price_risk}, quantity: {quantity}")
            return max(quantity, 0)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def get_market_volatility_regime(self) -> str:
        """Determine current market volatility regime"""
        try:
            # Placeholder implementation
            # In real scenario, this would analyze VIX or calculate market volatility
            return "NORMAL"  # Options: HIGH, NORMAL, LOW

        except Exception as e:
            logger.error(f"Error determining volatility regime: {e}")
            return "NORMAL"

    def should_allow_new_positions(self, current_positions: int, max_positions: int,
                                   daily_pnl: float, max_daily_loss: float) -> bool:
        """Check if new positions should be allowed based on risk limits"""
        try:
            # Check position count limit
            if current_positions >= max_positions:
                return False

            # Check daily loss limit
            if daily_pnl < -abs(max_daily_loss):
                return False

            # Check market hours (only allow new positions during specific hours)
            now = datetime.now()
            if now.hour >= 15:  # Stop taking new positions after 2 PM
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking position allowance: {e}")
            return False