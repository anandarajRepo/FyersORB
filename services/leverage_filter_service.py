# services/leverage_filter_service.py

"""
Leverage Filter Service for ORB Trading Strategy

After momentum screening, filters stocks based on available intraday
leverage/margin from the broker. Only stocks with >= 5x leverage
(i.e., margin required <= 20% of full position value) are retained
for intraday ORB trading.

How leverage is calculated:
    leverage = current_price / margin_required_per_share (for MIS/INTRADAY orders)

A stock qualifies if its leverage >= min_leverage (default: 5x).

This filter runs once per day after momentum screening and caches
results to avoid repeated API calls.
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from fyers_apiv3 import fyersModel

from config.settings import FyersConfig
from config.symbols import convert_to_fyers_format

logger = logging.getLogger(__name__)


@dataclass
class LeverageInfo:
    """Leverage/margin information for a single stock"""
    symbol: str
    current_price: float
    margin_required: float          # Margin required per share for MIS order
    leverage: float                 # Effective leverage = price / margin_required
    qualifies: bool                 # True if leverage >= min_leverage
    checked_at: datetime = field(default_factory=datetime.now)


class LeverageFilterService:
    """
    Checks intraday leverage/margin for momentum-screened stocks via the
    Fyers generate_margin API and filters out stocks below the minimum
    required leverage.

    For 5x leverage: margin_required <= 20% of stock price.
    """

    def __init__(self, fyers_config: FyersConfig, min_leverage: float = 5.0):
        self.fyers_config = fyers_config
        self.min_leverage = min_leverage
        self.fyers_client: Optional[fyersModel.FyersModel] = None

        # Daily cache: symbol -> LeverageInfo
        self._leverage_cache: Dict[str, LeverageInfo] = {}
        self._cache_date: Optional[str] = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize Fyers API client"""
        try:
            self.fyers_client = fyersModel.FyersModel(
                client_id=self.fyers_config.client_id,
                token=self.fyers_config.access_token,
                log_path=""
            )
            logger.info("Leverage filter service: Fyers client initialized")
        except Exception as e:
            logger.error(f"Leverage filter service: Failed to initialize Fyers client: {e}")

    def _check_symbol_leverage(self, symbol: str, current_price: float) -> Optional[LeverageInfo]:
        """
        Query Fyers generate_margin API for a single symbol to determine
        the intraday (MIS) margin requirement and compute effective leverage.

        Args:
            symbol: Display symbol name (e.g., 'STLTECH')
            current_price: Last close price used to compute leverage ratio

        Returns:
            LeverageInfo object, or None if the API call fails
        """
        try:
            if not self.fyers_client:
                logger.warning(f"Fyers client unavailable for leverage check: {symbol}")
                return None

            fyers_symbol = convert_to_fyers_format(symbol)
            if not fyers_symbol:
                logger.error(f"Cannot convert symbol {symbol} to Fyers format")
                return None

            if current_price <= 0:
                logger.warning(f"Invalid price for leverage check: {symbol} price={current_price}")
                return None

            # Request margin for 1 share (INTRADAY product type)
            # The fyers_apiv3 SDK does not expose a generate_margin / span_margin
            # method, so we call the REST endpoint directly.
            auth_header = "{}:{}".format(
                self.fyers_config.client_id, self.fyers_config.access_token
            )
            order = {
                "symbol": fyers_symbol,
                "qty": 1,
                "side": 1,          # Buy side
                "type": 2,          # Market order
                "productType": "INTRADAY",
                "limitPrice": 0,
                "stopPrice": 0,
            }
            # Fyers API v3: span_margin expects orders wrapped in a "data" array
            payload = {"data": [order]}
            raw = requests.post(
                f"{self.fyers_config.base_url}/span_margin",
                json=payload,
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            if not raw.content:
                response = {}
            else:
                try:
                    response = raw.json()
                except json.JSONDecodeError:
                    # Fyers sometimes returns a non-JSON prefix (e.g. "true") before
                    # the actual JSON payload.  Locate the first { or [ and parse from there.
                    text = raw.text or ""
                    logger.debug(
                        f"Non-JSON prefix in span_margin response for {symbol}: {text[:120]!r}"
                    )
                    start = next((i for i, c in enumerate(text) if c in "{["), None)
                    if start is not None:
                        try:
                            response = json.loads(text[start:])
                        except json.JSONDecodeError as inner:
                            logger.error(
                                f"Could not parse span_margin response for {symbol} "
                                f"(raw={text[:200]!r}): {inner}"
                            )
                            return None
                    else:
                        logger.error(
                            f"No JSON object found in span_margin response for {symbol}: "
                            f"{text[:200]!r}"
                        )
                        return None

            if not response or response.get('s') != 'ok':
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                logger.warning(f"Margin API failed for {symbol}: {error_msg}")
                return None

            # data is an array (one entry per submitted order)
            data_list = response.get('data', [])
            margin_data = data_list[0] if data_list else {}
            required_margin = float(margin_data.get('required_margin', 0))

            if required_margin <= 0:
                logger.warning(f"Zero or negative margin returned for {symbol}: {required_margin}")
                return None

            # Effective leverage: full single-share value divided by margin required
            leverage = current_price / required_margin
            qualifies = leverage >= self.min_leverage

            info = LeverageInfo(
                symbol=symbol,
                current_price=current_price,
                margin_required=required_margin,
                leverage=leverage,
                qualifies=qualifies
            )

            status = "PASS" if qualifies else f"FAIL (<{self.min_leverage:.0f}x)"
            logger.info(
                f"Leverage check {symbol}: price=Rs.{current_price:.2f} "
                f"margin=Rs.{required_margin:.2f} leverage={leverage:.1f}x [{status}]"
            )

            return info

        except Exception as e:
            logger.error(f"Error checking leverage for {symbol}: {e}")
            return None

    def filter_by_leverage(self, momentum_scores: list) -> List[str]:
        """
        Filter momentum-screened stocks by minimum intraday leverage.

        Called immediately after momentum screening. Uses the last_close
        price from each MomentumScore object to compute leverage.

        Args:
            momentum_scores: List of MomentumScore objects (from momentum_service)
                             Each must have .symbol and .last_close attributes.

        Returns:
            List of symbol names that have >= min_leverage intraday margin.
            Stocks whose margin check fails (API error) are included by default
            (fail-open) to avoid accidentally excluding valid stocks.
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Reset cache on a new trading day
        if self._cache_date != today:
            self._leverage_cache = {}
            self._cache_date = today

        logger.info("=" * 60)
        logger.info(f"APPLYING LEVERAGE FILTER (min {self.min_leverage:.0f}x intraday margin)")
        logger.info(f"Checking {len(momentum_scores)} momentum-screened stocks")
        logger.info("=" * 60)

        qualified_symbols: List[str] = []
        failed_symbols: List[Tuple[str, float]] = []
        skipped_symbols: List[str] = []       # API failures → fail-open

        for score in momentum_scores:
            symbol = score.symbol
            current_price = score.last_close

            # Serve from cache if already checked today
            if symbol in self._leverage_cache:
                info = self._leverage_cache[symbol]
                if info.qualifies:
                    qualified_symbols.append(symbol)
                else:
                    failed_symbols.append((symbol, info.leverage))
                continue

            # Fresh API check
            info = self._check_symbol_leverage(symbol, current_price)

            if info is None:
                # API check failed — include the symbol (fail-open)
                logger.warning(
                    f"Could not verify leverage for {symbol} (API error) — including by default"
                )
                skipped_symbols.append(symbol)
                qualified_symbols.append(symbol)
                continue

            # Cache and categorise
            self._leverage_cache[symbol] = info

            if info.qualifies:
                qualified_symbols.append(symbol)
            else:
                failed_symbols.append((symbol, info.leverage))

        # Summary log
        if failed_symbols:
            removed_str = ", ".join(f"{s}({l:.1f}x)" for s, l in failed_symbols)
            logger.info(
                f"Leverage filter removed {len(failed_symbols)} stock(s) "
                f"below {self.min_leverage:.0f}x: {removed_str}"
            )

        if skipped_symbols:
            logger.warning(
                f"Leverage check skipped (API error) for {len(skipped_symbols)} stock(s): "
                f"{', '.join(skipped_symbols)} — included by default"
            )

        logger.info(
            f"Leverage filter result: {len(qualified_symbols)} stock(s) qualify "
            f"for {self.min_leverage:.0f}x intraday margin"
        )
        logger.info("=" * 60)

        return qualified_symbols

    def get_leverage_report(self) -> Dict:
        """
        Return a summary dict of leverage checks performed today.
        Useful for logging / debugging.
        """
        if not self._leverage_cache:
            return {"status": "no_data", "symbols_checked": 0}

        all_info = list(self._leverage_cache.values())
        qualified = [i for i in all_info if i.qualifies]
        failed = [i for i in all_info if not i.qualifies]

        return {
            "status": "completed",
            "checked_date": self._cache_date,
            "min_leverage_required": self.min_leverage,
            "symbols_checked": len(all_info),
            "qualified_count": len(qualified),
            "failed_count": len(failed),
            "qualified": [
                {
                    "symbol": i.symbol,
                    "leverage": round(i.leverage, 2),
                    "price": i.current_price,
                    "margin_required": i.margin_required
                }
                for i in sorted(qualified, key=lambda x: x.leverage, reverse=True)
            ],
            "failed": [
                {
                    "symbol": i.symbol,
                    "leverage": round(i.leverage, 2),
                    "price": i.current_price,
                    "margin_required": i.margin_required
                }
                for i in sorted(failed, key=lambda x: x.leverage, reverse=True)
            ]
        }

    def print_leverage_report(self):
        """Print a formatted leverage filter report to console"""
        if not self._leverage_cache:
            print("\nNo leverage check data available. Run filter_by_leverage() first.")
            return

        all_info = sorted(
            self._leverage_cache.values(), key=lambda i: i.leverage, reverse=True
        )

        print("\n" + "=" * 70)
        print("LEVERAGE FILTER REPORT")
        print(f"Date: {self._cache_date}  |  Min Required: {self.min_leverage:.0f}x")
        print("=" * 70)
        print(f"\n{'Symbol':<12} {'Price':>10} {'Margin Req':>12} {'Leverage':>10} {'Status':>8}")
        print("-" * 70)

        for i in all_info:
            status = "PASS" if i.qualifies else "FAIL"
            print(
                f"{i.symbol:<12} {i.current_price:>9.2f}  "
                f"{i.margin_required:>11.2f}  "
                f"{i.leverage:>9.1f}x  {status:>6}"
            )

        qualified_count = sum(1 for i in all_info if i.qualifies)
        print("-" * 70)
        print(
            f"\nResult: {qualified_count}/{len(all_info)} stocks qualify "
            f"for {self.min_leverage:.0f}x intraday margin"
        )
        print("=" * 70)
