# services/trend_direction_service.py

"""
Trend Direction Service for ORB Trading Strategy

Analyzes multi-day trend direction for individual stocks and the Nifty 50 index.
Used to filter ORB signals so orders are placed only in the direction of the
prevailing trend.

Trend Analysis Components:
  1. EMA Alignment       - EMA9 vs EMA21 vs EMA50 structure
  2. Swing Structure     - Higher highs / lower lows pattern
  3. Price Slope         - % change over lookback period
  4. ADX / DI            - Average Directional Index for trend strength

Composite Decision (weighted voting):
  - EMA Alignment  : 35%
  - Swing Structure: 30%
  - Price Slope    : 20%
  - ADX / DI       : 15%
"""

import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from fyers_apiv3 import fyersModel

from config.settings import FyersConfig, SignalType
from config.symbols import convert_to_fyers_format

logger = logging.getLogger(__name__)

# Nifty 50 index symbol in Fyers API format
NIFTY50_FYERS_SYMBOL = "NSE:NIFTY50-INDEX"


class TrendDirection(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class TrendAnalysis:
    """Trend analysis result for a single symbol"""
    symbol: str
    trend: TrendDirection
    strength: float = 0.0   # 0-100 (higher = stronger trend)

    # Sub-signals
    ema_signal: TrendDirection = TrendDirection.SIDEWAYS
    swing_signal: TrendDirection = TrendDirection.SIDEWAYS
    slope_signal: TrendDirection = TrendDirection.SIDEWAYS
    adx_signal: TrendDirection = TrendDirection.SIDEWAYS

    # Raw indicator values
    ema9: float = 0.0
    ema21: float = 0.0
    ema50: float = 0.0
    price_slope: float = 0.0    # % change over lookback period
    higher_highs: bool = False
    lower_lows: bool = False
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0

    last_close: float = 0.0
    analyzed_at: datetime = field(default_factory=datetime.now)
    data_quality: str = "UNKNOWN"

    @property
    def is_uptrend(self) -> bool:
        return self.trend == TrendDirection.UPTREND

    @property
    def is_downtrend(self) -> bool:
        return self.trend == TrendDirection.DOWNTREND

    @property
    def is_sideways(self) -> bool:
        return self.trend == TrendDirection.SIDEWAYS

    @property
    def is_strong_trend(self) -> bool:
        """Trend strength >= 60 means a clear directional move"""
        return self.strength >= 60.0


class TrendDirectionService:
    """
    Determines multi-day trend direction for stocks and Nifty 50.
    Results are cached once per day and reused across signal evaluations.
    """

    def __init__(self, fyers_config: FyersConfig):
        self.fyers_config = fyers_config
        self.fyers_client: Optional[fyersModel.FyersModel] = None
        self._initialize_client()

        # Per-day cache
        self._symbol_trend_cache: Dict[str, TrendAnalysis] = {}
        self._nifty_trend: Optional[TrendAnalysis] = None
        self._cache_date: Optional[str] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_client(self):
        try:
            self.fyers_client = fyersModel.FyersModel(
                client_id=self.fyers_config.client_id,
                token=self.fyers_config.access_token,
                log_path=""
            )
            logger.info("Trend direction service: Fyers client initialized")
        except Exception as e:
            logger.error(f"Trend direction service: Failed to initialize Fyers client: {e}")

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_daily_candles(self, fyers_symbol: str, lookback_days: int = 60) -> Optional[List]:
        """Fetch daily OHLCV candles from Fyers API."""
        try:
            if not self.fyers_client:
                logger.warning(f"Fyers client not available for {fyers_symbol}")
                return None

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 20)  # Buffer for weekends

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
                    logger.debug(f"Fetched {len(candles)} daily candles for {fyers_symbol}")
                    return candles
                logger.warning(f"No candle data returned for {fyers_symbol}")
                return None
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                logger.error(f"Failed to fetch candles for {fyers_symbol}: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Error fetching candles for {fyers_symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Indicator calculations
    # ------------------------------------------------------------------

    def _calculate_ema(self, closes: np.ndarray, period: int) -> float:
        """Calculate EMA for the full array; return last value."""
        if len(closes) < 2:
            return float(closes[-1]) if len(closes) > 0 else 0.0
        if len(closes) < period:
            period = len(closes)

        multiplier = 2.0 / (period + 1)
        ema = float(np.mean(closes[:period]))
        for price in closes[period:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return ema

    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
        """
        Calculate simplified ADX, +DI, -DI over the last `period` bars.
        Returns (adx, plus_di, minus_di).
        """
        if len(closes) < period + 2:
            return 25.0, 50.0, 50.0

        tr_list, plus_dm_list, minus_dm_list = [], [], []

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hpc = abs(highs[i] - closes[i - 1])
            lpc = abs(lows[i] - closes[i - 1])
            tr_list.append(max(hl, hpc, lpc))

            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm_list.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dm_list.append(down_move if down_move > up_move and down_move > 0 else 0.0)

        tr_sum = float(np.sum(tr_list[-period:]))
        if tr_sum == 0:
            return 0.0, 50.0, 50.0

        plus_di = (float(np.sum(plus_dm_list[-period:])) / tr_sum) * 100
        minus_di = (float(np.sum(minus_dm_list[-period:])) / tr_sum) * 100

        di_sum = plus_di + minus_di
        adx = (abs(plus_di - minus_di) / di_sum * 100) if di_sum > 0 else 0.0

        return adx, plus_di, minus_di

    def _analyze_swing_structure(self, highs: np.ndarray, lows: np.ndarray,
                                  lookback: int = 5) -> Tuple[bool, bool]:
        """
        Detect higher-highs and lower-lows in the most recent `lookback` candles.
        Returns (higher_highs, lower_lows).
        """
        n = min(lookback, len(highs))
        if n < 2:
            return False, False

        recent_highs = highs[-n:]
        recent_lows = lows[-n:]
        comparisons = n - 1

        hh_count = sum(1 for i in range(1, n) if recent_highs[i] > recent_highs[i - 1])
        ll_count = sum(1 for i in range(1, n) if recent_lows[i] < recent_lows[i - 1])

        threshold = comparisons * 0.6   # 60% of candles must confirm

        return hh_count >= threshold, ll_count >= threshold

    def _calculate_price_slope(self, closes: np.ndarray, lookback: int) -> float:
        """% price change from `lookback` bars ago to the last bar."""
        lookback = min(lookback, len(closes) - 1)
        if lookback <= 0:
            return 0.0
        start = closes[-(lookback + 1)]
        end = closes[-1]
        return ((end - start) / start) * 100 if start > 0 else 0.0

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_trend(self, symbol: str, fyers_symbol: str,
                      lookback_days: int = 10) -> TrendAnalysis:
        """
        Perform multi-day trend analysis for a single symbol.

        Args:
            symbol:       Display name (e.g. 'RELIANCE', 'NIFTY50')
            fyers_symbol: Fyers API format (e.g. 'NSE:RELIANCE-EQ')
            lookback_days: Number of recent trading days used for swing / slope

        Returns:
            TrendAnalysis with trend direction and detailed breakdown
        """
        try:
            # Fetch enough history for EMAs (50-period needs ~70+ calendar days)
            candles = self._fetch_daily_candles(fyers_symbol, lookback_days=max(lookback_days * 3, 70))

            if not candles or len(candles) < 5:
                logger.warning(f"Insufficient candles for trend analysis: {symbol}")
                return TrendAnalysis(symbol=symbol, trend=TrendDirection.SIDEWAYS,
                                     strength=0.0, data_quality="INSUFFICIENT")

            closes = np.array([float(c[4]) for c in candles])
            highs  = np.array([float(c[2]) for c in candles])
            lows   = np.array([float(c[3]) for c in candles])

            result = TrendAnalysis(symbol=symbol, trend=TrendDirection.SIDEWAYS, strength=0.0)
            result.last_close = float(closes[-1])
            result.data_quality = "GOOD" if len(candles) >= 30 else "PARTIAL"

            # ── 1. EMA Alignment ────────────────────────────────────────
            result.ema9  = self._calculate_ema(closes, 9)
            result.ema21 = self._calculate_ema(closes, 21)
            result.ema50 = self._calculate_ema(closes, min(50, len(closes)))

            price = float(closes[-1])
            ema_bullish = price > result.ema9 and result.ema9 > result.ema21
            ema_bearish = price < result.ema9 and result.ema9 < result.ema21

            if ema_bullish:
                result.ema_signal = TrendDirection.UPTREND
            elif ema_bearish:
                result.ema_signal = TrendDirection.DOWNTREND
            else:
                result.ema_signal = TrendDirection.SIDEWAYS

            # ── 2. Swing Structure (Higher Highs / Lower Lows) ──────────
            n = min(lookback_days, len(highs))
            result.higher_highs, result.lower_lows = self._analyze_swing_structure(
                highs[-n:], lows[-n:], lookback=n
            )

            if result.higher_highs and not result.lower_lows:
                result.swing_signal = TrendDirection.UPTREND
            elif result.lower_lows and not result.higher_highs:
                result.swing_signal = TrendDirection.DOWNTREND
            elif result.higher_highs and result.lower_lows:
                # Expansion / volatility – lean on EMA direction
                result.swing_signal = result.ema_signal
            else:
                result.swing_signal = TrendDirection.SIDEWAYS

            # ── 3. Price Slope ───────────────────────────────────────────
            result.price_slope = self._calculate_price_slope(closes, lookback_days)

            if result.price_slope > 1.5:
                result.slope_signal = TrendDirection.UPTREND
            elif result.price_slope < -1.5:
                result.slope_signal = TrendDirection.DOWNTREND
            else:
                result.slope_signal = TrendDirection.SIDEWAYS

            # ── 4. ADX / DI ─────────────────────────────────────────────
            result.adx, result.plus_di, result.minus_di = self._calculate_adx(
                highs, lows, closes, period=14
            )

            if result.plus_di > result.minus_di and result.adx > 20:
                result.adx_signal = TrendDirection.UPTREND
            elif result.minus_di > result.plus_di and result.adx > 20:
                result.adx_signal = TrendDirection.DOWNTREND
            else:
                result.adx_signal = TrendDirection.SIDEWAYS

            # ── 5. Composite Weighted Voting ─────────────────────────────
            # Weights: EMA=35%, Swing=30%, Slope=20%, ADX=15%
            signal_weights = [
                (result.ema_signal,   0.35),
                (result.swing_signal, 0.30),
                (result.slope_signal, 0.20),
                (result.adx_signal,   0.15),
            ]

            up_score = sum(w for sig, w in signal_weights if sig == TrendDirection.UPTREND)
            down_score = sum(w for sig, w in signal_weights if sig == TrendDirection.DOWNTREND)

            # Need at least 55% weighted agreement for a directional call
            if up_score >= 0.55:
                result.trend = TrendDirection.UPTREND
                result.strength = min(100.0, up_score * 100 + result.adx * 0.3)
            elif down_score >= 0.55:
                result.trend = TrendDirection.DOWNTREND
                result.strength = min(100.0, down_score * 100 + result.adx * 0.3)
            else:
                result.trend = TrendDirection.SIDEWAYS
                result.strength = min(100.0, max(up_score, down_score) * 100)

            result.analyzed_at = datetime.now()

            logger.info(
                f"Trend [{symbol}]: {result.trend.value}  "
                f"strength={result.strength:.1f}  slope={result.price_slope:+.2f}%  "
                f"EMA={result.ema_signal.value}  Swing={result.swing_signal.value}  "
                f"ADX={result.adx:.1f}(+DI={result.plus_di:.1f} -DI={result.minus_di:.1f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error analyzing trend for {symbol}: {e}")
            return TrendAnalysis(symbol=symbol, trend=TrendDirection.SIDEWAYS,
                                 strength=0.0, data_quality="ERROR")

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def analyze_nifty_trend(self, lookback_days: int = 10) -> TrendAnalysis:
        """Analyze and cache Nifty 50 index trend."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cache_date == today and self._nifty_trend is not None:
            return self._nifty_trend

        logger.info("Analyzing Nifty 50 trend direction...")
        trend = self.analyze_trend("NIFTY50", NIFTY50_FYERS_SYMBOL, lookback_days)
        self._nifty_trend = trend
        return trend

    def analyze_symbol_trend(self, symbol: str, lookback_days: int = 10) -> TrendAnalysis:
        """Analyze and cache trend for a single stock symbol."""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{symbol}_{lookback_days}"

        if self._cache_date == today and cache_key in self._symbol_trend_cache:
            return self._symbol_trend_cache[cache_key]

        fyers_symbol = convert_to_fyers_format(symbol)
        if not fyers_symbol:
            logger.error(f"Cannot convert symbol {symbol} to Fyers format")
            return TrendAnalysis(symbol=symbol, trend=TrendDirection.SIDEWAYS,
                                 strength=0.0, data_quality="INVALID_SYMBOL")

        trend = self.analyze_trend(symbol, fyers_symbol, lookback_days)
        self._symbol_trend_cache[cache_key] = trend
        return trend

    def analyze_all_symbols(self, symbols: List[str],
                             lookback_days: int = 10) -> Dict[str, TrendAnalysis]:
        """
        Analyze trend for all provided symbols plus Nifty 50.

        Returns:
            Dict mapping symbol -> TrendAnalysis (key 'NIFTY50' for the index)
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Reset cache on a new day
        if self._cache_date != today:
            self._symbol_trend_cache.clear()
            self._nifty_trend = None
            self._cache_date = today

        results: Dict[str, TrendAnalysis] = {}

        # Nifty first (market context)
        nifty = self.analyze_nifty_trend(lookback_days)
        results["NIFTY50"] = nifty
        logger.info(f"Nifty 50 Trend: {nifty.trend.value} "
                    f"(strength={nifty.strength:.1f}, slope={nifty.price_slope:+.2f}%)")

        for symbol in symbols:
            results[symbol] = self.analyze_symbol_trend(symbol, lookback_days)

        return results

    # ------------------------------------------------------------------
    # Signal alignment check
    # ------------------------------------------------------------------

    def is_signal_aligned_with_trend(
        self,
        signal_type: SignalType,
        stock_trend: TrendAnalysis,
        nifty_trend: TrendAnalysis,
        filter_mode: str = "STRICT"
    ) -> Tuple[bool, str]:
        """
        Decide whether a trade signal is aligned with prevailing trends.

        Filter modes:
          STRICT  – Reject if stock or Nifty trend is opposite to signal direction.
          LENIENT – Reject only if BOTH stock and Nifty trend oppose the signal.

        Returns:
            (is_aligned, reason_message)
        """
        stock_dir = stock_trend.trend
        nifty_dir = nifty_trend.trend

        if signal_type == SignalType.LONG:
            stock_against = stock_dir == TrendDirection.DOWNTREND
            nifty_against = nifty_dir == TrendDirection.DOWNTREND

            if filter_mode == "STRICT":
                # Reject if stock alone is in downtrend
                if stock_against:
                    return False, (
                        f"LONG rejected: stock {stock_trend.symbol} in DOWNTREND "
                        f"(slope={stock_trend.price_slope:+.2f}%)"
                    )
                # Reject if Nifty is in downtrend AND stock is not clearly up
                if nifty_against and stock_dir != TrendDirection.UPTREND:
                    return False, (
                        f"LONG rejected: Nifty in DOWNTREND with stock only {stock_dir.value}"
                    )
            else:  # LENIENT
                if stock_against and nifty_against:
                    return False, (
                        f"LONG rejected: both stock ({stock_dir.value}) "
                        f"and Nifty ({nifty_dir.value}) in DOWNTREND"
                    )

            return True, (
                f"LONG aligned: stock={stock_dir.value}, Nifty={nifty_dir.value}"
            )

        elif signal_type == SignalType.SHORT:
            stock_against = stock_dir == TrendDirection.UPTREND
            nifty_against = nifty_dir == TrendDirection.UPTREND

            if filter_mode == "STRICT":
                # Reject if stock alone is in uptrend
                if stock_against:
                    return False, (
                        f"SHORT rejected: stock {stock_trend.symbol} in UPTREND "
                        f"(slope={stock_trend.price_slope:+.2f}%)"
                    )
                # Reject if Nifty is in uptrend AND stock is not clearly down
                if nifty_against and stock_dir != TrendDirection.DOWNTREND:
                    return False, (
                        f"SHORT rejected: Nifty in UPTREND with stock only {stock_dir.value}"
                    )
            else:  # LENIENT
                if stock_against and nifty_against:
                    return False, (
                        f"SHORT rejected: both stock ({stock_dir.value}) "
                        f"and Nifty ({nifty_dir.value}) in UPTREND"
                    )

            return True, (
                f"SHORT aligned: stock={stock_dir.value}, Nifty={nifty_dir.value}"
            )

        # Unknown signal type – allow through
        return True, "Unknown signal type – passing through"

    # ------------------------------------------------------------------
    # Cache accessors
    # ------------------------------------------------------------------

    def get_nifty_trend(self) -> Optional[TrendAnalysis]:
        """Return cached Nifty trend (None if not yet analyzed today)."""
        return self._nifty_trend

    def get_symbol_trend(self, symbol: str,
                          lookback_days: int = 10) -> Optional[TrendAnalysis]:
        """Return cached trend for a symbol (None if not yet analyzed today)."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cache_date != today:
            return None
        return self._symbol_trend_cache.get(f"{symbol}_{lookback_days}")

    def get_trend_summary(self) -> Dict:
        """Return a dict summarising today's cached trend analyses."""
        if not self._symbol_trend_cache:
            return {"status": "no_analysis_done"}

        analyses = list(self._symbol_trend_cache.values())
        up_count   = sum(1 for a in analyses if a.trend == TrendDirection.UPTREND)
        down_count = sum(1 for a in analyses if a.trend == TrendDirection.DOWNTREND)
        side_count = sum(1 for a in analyses if a.trend == TrendDirection.SIDEWAYS)

        return {
            "status": "completed",
            "analysis_date": self._cache_date,
            "symbols_analyzed": len(analyses),
            "nifty_trend": self._nifty_trend.trend.value if self._nifty_trend else "N/A",
            "nifty_strength": self._nifty_trend.strength if self._nifty_trend else 0.0,
            "nifty_slope": self._nifty_trend.price_slope if self._nifty_trend else 0.0,
            "uptrend_count": up_count,
            "downtrend_count": down_count,
            "sideways_count": side_count,
            "uptrend_symbols": [a.symbol for a in analyses
                                if a.trend == TrendDirection.UPTREND],
            "downtrend_symbols": [a.symbol for a in analyses
                                  if a.trend == TrendDirection.DOWNTREND],
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_trend_report(self, trend_analyses: Dict[str, 'TrendAnalysis']):
        """Print a formatted trend direction report to the console."""
        print("\n" + "=" * 110)
        print("TREND DIRECTION ANALYSIS REPORT")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
              f"Symbols: {len(trend_analyses)}")
        print("=" * 110)

        # Nifty index first
        if "NIFTY50" in trend_analyses:
            n = trend_analyses["NIFTY50"]
            print(f"\n  NIFTY 50: {n.trend.value:<12}  "
                  f"Strength: {n.strength:5.1f}  "
                  f"Slope: {n.price_slope:+.2f}%  "
                  f"EMA: {n.ema_signal.value:<10}  "
                  f"ADX: {n.adx:.1f} (+DI={n.plus_di:.1f} -DI={n.minus_di:.1f})")

        stock_analyses = {k: v for k, v in trend_analyses.items() if k != "NIFTY50"}
        up_c   = sum(1 for v in stock_analyses.values() if v.trend == TrendDirection.UPTREND)
        down_c = sum(1 for v in stock_analyses.values() if v.trend == TrendDirection.DOWNTREND)
        side_c = sum(1 for v in stock_analyses.values() if v.trend == TrendDirection.SIDEWAYS)

        print(f"\n{'Symbol':<13} {'Trend':<12} {'Strength':>9} "
              f"{'Slope%':>8} {'EMA':>10} {'Swing':>10} "
              f"{'ADX':>6} {'Close':>10}")
        print("-" * 110)

        for sym, a in sorted(stock_analyses.items(),
                             key=lambda x: x[1].strength, reverse=True):
            print(f"{sym:<13} {a.trend.value:<12} {a.strength:>8.1f} "
                  f"{a.price_slope:>+7.2f}% {a.ema_signal.value:>10} "
                  f"{a.swing_signal.value:>10} {a.adx:>6.1f} {a.last_close:>10.2f}")

        print("-" * 110)
        print(f"\nSummary: UPTREND={up_c} | DOWNTREND={down_c} | SIDEWAYS={side_c}")
        print("=" * 110)
