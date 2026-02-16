# services/momentum_service.py

"""
Momentum Scoring Service for ORB Trading Strategy
Screens stocks based on recent momentum indicators and ranks them
for prioritized order placement during the trading day.

Momentum Score Components:
  1. Price Rate of Change (ROC) - 5, 10, 20 day
  2. RSI positioning (strength without being overbought)
  3. Volume trend (increasing volume confirms momentum)
  4. Moving average alignment (price > SMA5 > SMA10 > SMA20)
  5. Consecutive up/down days streak
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from fyers_apiv3 import fyersModel

from config.settings import FyersConfig
from config.symbols import symbol_manager, convert_to_fyers_format

logger = logging.getLogger(__name__)


@dataclass
class MomentumScore:
    """Momentum score breakdown for a single stock"""
    symbol: str
    composite_score: float = 0.0  # 0-100 overall score

    # Component scores (each 0-100)
    roc_score: float = 0.0         # Price rate of change score
    rsi_score: float = 0.0         # RSI positioning score
    volume_trend_score: float = 0.0  # Volume trend score
    ma_alignment_score: float = 0.0  # Moving average alignment score
    streak_score: float = 0.0       # Consecutive day streak score

    # Raw indicator values
    roc_5d: float = 0.0   # 5-day rate of change %
    roc_10d: float = 0.0  # 10-day rate of change %
    roc_20d: float = 0.0  # 20-day rate of change %
    rsi_14: float = 50.0  # 14-day RSI
    volume_ratio_5d: float = 1.0  # Recent 5-day avg volume / 20-day avg volume
    consecutive_up_days: int = 0
    consecutive_down_days: int = 0
    price_vs_sma20: float = 0.0  # % above/below 20-day SMA

    # Metadata
    last_close: float = 0.0
    avg_daily_volume: float = 0.0
    scored_at: datetime = field(default_factory=datetime.now)
    data_quality: str = "UNKNOWN"  # GOOD, PARTIAL, INSUFFICIENT

    @property
    def is_bullish(self) -> bool:
        """Stock has bullish momentum (composite > 60)"""
        return self.composite_score >= 60.0

    @property
    def is_strong_momentum(self) -> bool:
        """Stock has strong momentum (composite > 75)"""
        return self.composite_score >= 75.0


class MomentumScoringService:
    """
    Screens all stocks in the trading universe for momentum
    and produces a ranked list for the ORB strategy to use.
    """

    # Component weights for composite score
    WEIGHT_ROC = 0.30          # Rate of change (trend strength)
    WEIGHT_RSI = 0.15          # RSI positioning
    WEIGHT_VOLUME_TREND = 0.20 # Volume confirmation
    WEIGHT_MA_ALIGNMENT = 0.25 # Moving average structure
    WEIGHT_STREAK = 0.10       # Consecutive day streak

    def __init__(self, fyers_config: FyersConfig):
        self.fyers_config = fyers_config
        self.fyers_client: Optional[fyersModel.FyersModel] = None
        self._initialize_client()

        # Cache for momentum scores (refreshed once per day)
        self._scores_cache: Dict[str, MomentumScore] = {}
        self._cache_date: Optional[str] = None

    def _initialize_client(self):
        """Initialize Fyers API client for historical data"""
        try:
            self.fyers_client = fyersModel.FyersModel(
                client_id=self.fyers_config.client_id,
                token=self.fyers_config.access_token,
                log_path=""
            )
            logger.info("Momentum service: Fyers client initialized")
        except Exception as e:
            logger.error(f"Momentum service: Failed to initialize Fyers client: {e}")

    def _fetch_daily_candles(self, symbol: str, lookback_days: int = 50) -> Optional[List[List]]:
        """
        Fetch daily candle data for a symbol.
        Returns list of [timestamp, open, high, low, close, volume] or None.
        """
        try:
            if not self.fyers_client:
                logger.warning(f"Fyers client not available for {symbol}")
                return None

            fyers_symbol = convert_to_fyers_format(symbol)
            if not fyers_symbol:
                logger.error(f"Cannot convert symbol {symbol} to Fyers format")
                return None

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 15)  # Extra buffer for weekends

            data = {
                "symbol": fyers_symbol,
                "resolution": "D",
                "date_format": "1",
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }

            response = self.fyers_client.history(data=data)

            if response and response.get('s') == 'ok':
                candles = response.get('candles', [])
                if candles:
                    logger.debug(f"Fetched {len(candles)} daily candles for {symbol}")
                    return candles
                else:
                    logger.warning(f"No candle data returned for {symbol}")
                    return None
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                logger.error(f"Failed to fetch candles for {symbol}: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Error fetching daily candles for {symbol}: {e}")
            return None

    def _calculate_roc(self, closes: np.ndarray, period: int) -> float:
        """Calculate Rate of Change over given period"""
        if len(closes) <= period:
            return 0.0
        return ((closes[-1] - closes[-(period + 1)]) / closes[-(period + 1)]) * 100

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI using Wilder's smoothing method"""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Use Wilder's smoothing (exponential)
        recent_gains = gains[-(period):]
        recent_losses = losses[-(period):]

        avg_gain = np.mean(recent_gains)
        avg_loss = np.mean(recent_losses)

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def _calculate_sma(self, closes: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(closes) < period:
            return 0.0
        return float(np.mean(closes[-period:]))

    def _calculate_consecutive_days(self, closes: np.ndarray) -> Tuple[int, int]:
        """Calculate consecutive up days and consecutive down days from most recent"""
        if len(closes) < 2:
            return 0, 0

        up_days = 0
        down_days = 0

        # Count from most recent backwards
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i - 1]:
                if down_days > 0:
                    break
                up_days += 1
            elif closes[i] < closes[i - 1]:
                if up_days > 0:
                    break
                down_days += 1
            else:
                break

        return up_days, down_days

    def _score_roc(self, roc_5d: float, roc_10d: float, roc_20d: float) -> float:
        """
        Score rate of change: positive ROC across timeframes = higher score.
        Prefer moderate positive ROC (not too extreme which signals overextension).
        """
        score = 0.0

        # 5-day ROC (short-term momentum) - 40% of ROC score
        if roc_5d > 0:
            # Sweet spot: 1-8% gain in 5 days
            if 1.0 <= roc_5d <= 8.0:
                score += 40.0
            elif roc_5d > 8.0:
                # Overextended, reduce score slightly
                score += max(20.0, 40.0 - (roc_5d - 8.0) * 2)
            else:
                score += roc_5d * 40.0  # 0-1% partial credit
        else:
            # Negative ROC: partial credit if not too negative
            score += max(0.0, 10.0 + roc_5d * 2)

        # 10-day ROC (medium-term momentum) - 35% of ROC score
        if roc_10d > 0:
            if 2.0 <= roc_10d <= 15.0:
                score += 35.0
            elif roc_10d > 15.0:
                score += max(15.0, 35.0 - (roc_10d - 15.0) * 1.5)
            else:
                score += roc_10d * 17.5
        else:
            score += max(0.0, 5.0 + roc_10d * 1)

        # 20-day ROC (longer-term trend) - 25% of ROC score
        if roc_20d > 0:
            if 3.0 <= roc_20d <= 20.0:
                score += 25.0
            elif roc_20d > 20.0:
                score += max(10.0, 25.0 - (roc_20d - 20.0))
            else:
                score += roc_20d * 8.33
        else:
            score += max(0.0, 5.0 + roc_20d * 0.5)

        return min(max(score, 0.0), 100.0)

    def _score_rsi(self, rsi: float) -> float:
        """
        Score RSI positioning.
        Ideal zone: 55-70 (strong but not overbought).
        Avoid: <30 (weak) or >80 (overbought).
        """
        if 55 <= rsi <= 70:
            return 100.0
        elif 50 <= rsi < 55:
            return 70.0 + (rsi - 50) * 6  # 70-100
        elif 70 < rsi <= 80:
            return 100.0 - (rsi - 70) * 5  # 100-50
        elif 40 <= rsi < 50:
            return 40.0 + (rsi - 40) * 3  # 40-70
        elif rsi > 80:
            return max(0.0, 50.0 - (rsi - 80) * 5)
        elif 30 <= rsi < 40:
            return 20.0 + (rsi - 30) * 2  # 20-40
        else:  # < 30
            return max(0.0, rsi * 0.67)

    def _score_volume_trend(self, recent_avg: float, longer_avg: float) -> float:
        """
        Score volume trend.
        Increasing volume (recent > longer average) = positive momentum confirmation.
        """
        if longer_avg <= 0:
            return 50.0

        ratio = recent_avg / longer_avg

        if ratio >= 1.5:
            return 100.0  # Strong volume increase
        elif ratio >= 1.2:
            return 80.0 + (ratio - 1.2) * 66.7  # 80-100
        elif ratio >= 1.0:
            return 60.0 + (ratio - 1.0) * 100  # 60-80
        elif ratio >= 0.8:
            return 30.0 + (ratio - 0.8) * 150  # 30-60
        else:
            return max(0.0, ratio * 37.5)

    def _score_ma_alignment(self, price: float, sma5: float, sma10: float, sma20: float) -> float:
        """
        Score moving average alignment.
        Ideal bullish: Price > SMA5 > SMA10 > SMA20 = 100
        """
        if sma5 == 0 or sma10 == 0 or sma20 == 0:
            return 50.0

        score = 0.0

        # Price above SMA5 (25 points)
        if price > sma5:
            score += 25.0

        # Price above SMA20 (25 points)
        if price > sma20:
            score += 25.0

        # SMA5 > SMA10 (short-term trend up) (25 points)
        if sma5 > sma10:
            score += 25.0

        # SMA10 > SMA20 (medium-term trend up) (25 points)
        if sma10 > sma20:
            score += 25.0

        return score

    def _score_streak(self, up_days: int, down_days: int) -> float:
        """
        Score consecutive day streaks.
        Moderate up streaks (2-4 days) are ideal.
        Too long a streak (>5) may mean overextended.
        """
        if up_days >= 2 and up_days <= 4:
            return 100.0
        elif up_days == 1:
            return 60.0
        elif up_days == 5:
            return 70.0
        elif up_days > 5:
            return max(30.0, 70.0 - (up_days - 5) * 10)
        elif down_days == 1:
            return 40.0  # Mild pullback can be okay
        elif down_days == 2:
            return 25.0
        else:
            return max(0.0, 20.0 - down_days * 5)

    def calculate_momentum_score(self, symbol: str, lookback_days: int = 30) -> Optional[MomentumScore]:
        """
        Calculate comprehensive momentum score for a single stock.

        Args:
            symbol: Display symbol (e.g., 'STLTECH')
            lookback_days: Number of trading days to analyze

        Returns:
            MomentumScore object or None if insufficient data
        """
        try:
            candles = self._fetch_daily_candles(symbol, lookback_days)
            if not candles or len(candles) < 10:
                logger.warning(f"Insufficient data for momentum scoring: {symbol} "
                               f"(got {len(candles) if candles else 0} candles)")
                return MomentumScore(
                    symbol=symbol,
                    data_quality="INSUFFICIENT"
                )

            # Extract OHLCV arrays
            closes = np.array([float(c[4]) for c in candles])
            volumes = np.array([float(c[5]) for c in candles])
            highs = np.array([float(c[2]) for c in candles])
            lows = np.array([float(c[3]) for c in candles])

            score = MomentumScore(symbol=symbol)
            score.last_close = float(closes[-1])
            score.avg_daily_volume = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))

            # Determine data quality
            if len(candles) >= 25:
                score.data_quality = "GOOD"
            elif len(candles) >= 15:
                score.data_quality = "PARTIAL"
            else:
                score.data_quality = "INSUFFICIENT"

            # 1. Rate of Change
            score.roc_5d = self._calculate_roc(closes, 5) if len(closes) > 5 else 0.0
            score.roc_10d = self._calculate_roc(closes, 10) if len(closes) > 10 else 0.0
            score.roc_20d = self._calculate_roc(closes, 20) if len(closes) > 20 else 0.0
            score.roc_score = self._score_roc(score.roc_5d, score.roc_10d, score.roc_20d)

            # 2. RSI
            score.rsi_14 = self._calculate_rsi(closes, 14)
            score.rsi_score = self._score_rsi(score.rsi_14)

            # 3. Volume trend
            vol_5d_avg = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else float(np.mean(volumes))
            vol_20d_avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
            score.volume_ratio_5d = vol_5d_avg / vol_20d_avg if vol_20d_avg > 0 else 1.0
            score.volume_trend_score = self._score_volume_trend(vol_5d_avg, vol_20d_avg)

            # 4. Moving average alignment
            sma5 = self._calculate_sma(closes, 5)
            sma10 = self._calculate_sma(closes, 10)
            sma20 = self._calculate_sma(closes, 20) if len(closes) >= 20 else self._calculate_sma(closes, len(closes))
            score.ma_alignment_score = self._score_ma_alignment(score.last_close, sma5, sma10, sma20)
            score.price_vs_sma20 = ((score.last_close - sma20) / sma20 * 100) if sma20 > 0 else 0.0

            # 5. Consecutive day streak
            up_days, down_days = self._calculate_consecutive_days(closes)
            score.consecutive_up_days = up_days
            score.consecutive_down_days = down_days
            score.streak_score = self._score_streak(up_days, down_days)

            # Composite score (weighted average)
            score.composite_score = (
                score.roc_score * self.WEIGHT_ROC +
                score.rsi_score * self.WEIGHT_RSI +
                score.volume_trend_score * self.WEIGHT_VOLUME_TREND +
                score.ma_alignment_score * self.WEIGHT_MA_ALIGNMENT +
                score.streak_score * self.WEIGHT_STREAK
            )

            score.scored_at = datetime.now()

            logger.info(f"Momentum score for {symbol}: {score.composite_score:.1f}/100 "
                        f"(ROC:{score.roc_score:.0f} RSI:{score.rsi_score:.0f} "
                        f"Vol:{score.volume_trend_score:.0f} MA:{score.ma_alignment_score:.0f} "
                        f"Streak:{score.streak_score:.0f})")

            return score

        except Exception as e:
            logger.error(f"Error calculating momentum score for {symbol}: {e}")
            return MomentumScore(symbol=symbol, data_quality="INSUFFICIENT")

    def screen_all_symbols(self, min_score: float = 50.0,
                           top_n: int = 15,
                           lookback_days: int = 30) -> List[MomentumScore]:
        """
        Screen all symbols in the trading universe for momentum.

        Args:
            min_score: Minimum composite score to qualify (0-100)
            top_n: Return top N stocks by momentum score
            lookback_days: Days of history to analyze

        Returns:
            List of MomentumScore objects, sorted by composite_score descending
        """
        logger.info(f"Starting momentum screening for {symbol_manager.get_trading_universe_size()} symbols "
                     f"(min_score={min_score}, top_n={top_n})")

        all_scores: List[MomentumScore] = []
        symbols = symbol_manager.get_all_symbols()
        success_count = 0
        fail_count = 0

        for symbol in symbols:
            score = self.calculate_momentum_score(symbol, lookback_days)
            if score and score.data_quality != "INSUFFICIENT":
                all_scores.append(score)
                success_count += 1
            else:
                fail_count += 1
                logger.debug(f"Skipping {symbol}: insufficient data for momentum scoring")

        # Sort by composite score descending
        all_scores.sort(key=lambda s: s.composite_score, reverse=True)

        # Filter by minimum score
        qualified = [s for s in all_scores if s.composite_score >= min_score]

        # Take top N
        top_stocks = qualified[:top_n]

        logger.info(f"Momentum screening complete: "
                     f"{success_count} scored, {fail_count} failed, "
                     f"{len(qualified)} qualified (>={min_score}), "
                     f"returning top {len(top_stocks)}")

        # Log the top stocks
        for i, s in enumerate(top_stocks, 1):
            logger.info(f"  #{i} {s.symbol}: {s.composite_score:.1f}/100 "
                        f"(ROC5d:{s.roc_5d:+.1f}% RSI:{s.rsi_14:.0f} "
                        f"VolRatio:{s.volume_ratio_5d:.2f} Close:Rs.{s.last_close:.2f})")

        # Update cache
        today = datetime.now().strftime("%Y-%m-%d")
        self._cache_date = today
        self._scores_cache = {s.symbol: s for s in all_scores}

        return top_stocks

    def get_cached_scores(self) -> Dict[str, MomentumScore]:
        """Return cached momentum scores (from today's screening)"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cache_date == today:
            return self._scores_cache
        return {}

    def get_momentum_filtered_symbols(self, min_score: float = 50.0,
                                       top_n: int = 15,
                                       lookback_days: int = 30) -> List[str]:
        """
        Get list of symbol names that pass momentum filtering.
        Uses cache if available for today, otherwise runs fresh screening.

        Args:
            min_score: Minimum composite momentum score
            top_n: Maximum number of stocks to return
            lookback_days: Historical lookback period

        Returns:
            List of display symbol names sorted by momentum score
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Use cache if available for today
        if self._cache_date == today and self._scores_cache:
            qualified = [s for s in self._scores_cache.values()
                         if s.composite_score >= min_score and s.data_quality != "INSUFFICIENT"]
            qualified.sort(key=lambda s: s.composite_score, reverse=True)
            return [s.symbol for s in qualified[:top_n]]

        # Run fresh screening
        top_stocks = self.screen_all_symbols(min_score, top_n, lookback_days)
        return [s.symbol for s in top_stocks]

    def get_symbol_momentum(self, symbol: str) -> Optional[MomentumScore]:
        """Get momentum score for a specific symbol from cache"""
        return self._scores_cache.get(symbol)

    def get_screening_summary(self) -> Dict:
        """Get a summary of the most recent momentum screening"""
        if not self._scores_cache:
            return {"status": "no_screening_done", "symbols_scored": 0}

        scores = list(self._scores_cache.values())
        valid_scores = [s for s in scores if s.data_quality != "INSUFFICIENT"]

        if not valid_scores:
            return {"status": "no_valid_scores", "symbols_scored": 0}

        composites = [s.composite_score for s in valid_scores]

        bullish = [s for s in valid_scores if s.is_bullish]
        strong = [s for s in valid_scores if s.is_strong_momentum]

        return {
            "status": "completed",
            "screened_date": self._cache_date,
            "symbols_scored": len(valid_scores),
            "avg_momentum_score": float(np.mean(composites)),
            "median_momentum_score": float(np.median(composites)),
            "max_momentum_score": float(np.max(composites)),
            "min_momentum_score": float(np.min(composites)),
            "bullish_count": len(bullish),
            "strong_momentum_count": len(strong),
            "top_5": [
                {"symbol": s.symbol, "score": s.composite_score, "roc_5d": s.roc_5d}
                for s in sorted(valid_scores, key=lambda x: x.composite_score, reverse=True)[:5]
            ],
            "bottom_5": [
                {"symbol": s.symbol, "score": s.composite_score, "roc_5d": s.roc_5d}
                for s in sorted(valid_scores, key=lambda x: x.composite_score)[:5]
            ]
        }

    def print_momentum_report(self):
        """Print a formatted momentum screening report to console"""
        if not self._scores_cache:
            print("\nNo momentum screening data available. Run screening first.")
            return

        scores = sorted(self._scores_cache.values(),
                        key=lambda s: s.composite_score, reverse=True)
        valid_scores = [s for s in scores if s.data_quality != "INSUFFICIENT"]

        print("\n" + "=" * 95)
        print("MOMENTUM SCREENING REPORT")
        print(f"Date: {self._cache_date}  |  Symbols Scored: {len(valid_scores)}")
        print("=" * 95)

        print(f"\n{'#':<3} {'Symbol':<12} {'Score':>6} {'ROC5d':>7} {'ROC10d':>8} "
              f"{'RSI':>5} {'VolRatio':>9} {'MA Align':>9} {'Streak':>8} {'Close':>10}")
        print("-" * 95)

        for i, s in enumerate(valid_scores, 1):
            streak_str = f"+{s.consecutive_up_days}d" if s.consecutive_up_days > 0 else f"-{s.consecutive_down_days}d"
            momentum_tag = ""
            if s.is_strong_momentum:
                momentum_tag = " [STRONG]"
            elif s.is_bullish:
                momentum_tag = " [BULLISH]"

            print(f"{i:<3} {s.symbol:<12} {s.composite_score:>5.1f}  "
                  f"{s.roc_5d:>+6.1f}% {s.roc_10d:>+7.1f}% "
                  f"{s.rsi_14:>5.0f} {s.volume_ratio_5d:>8.2f}x "
                  f"{s.ma_alignment_score:>8.0f} {streak_str:>8} "
                  f"{s.last_close:>9.2f}{momentum_tag}")

        # Summary stats
        composites = [s.composite_score for s in valid_scores]
        bullish_count = sum(1 for s in valid_scores if s.is_bullish)
        strong_count = sum(1 for s in valid_scores if s.is_strong_momentum)

        print("-" * 95)
        print(f"\nSummary: Avg Score: {np.mean(composites):.1f} | "
              f"Bullish (>=60): {bullish_count} | "
              f"Strong (>=75): {strong_count} | "
              f"Total: {len(valid_scores)}")
        print("=" * 95)
