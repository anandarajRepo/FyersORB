# services/market_timing_service.py

"""
Enhanced Market Timing Service for ORB Strategy
Handles market hours, ORB periods, and trading windows
"""

from datetime import datetime, time
import pytz
from typing import Tuple, Optional
from config.settings import TradingConfig

# Indian Standard Time
IST = pytz.timezone('Asia/Kolkata')


class MarketTimingService:
    """Enhanced market timing service for ORB strategy"""

    def __init__(self, config: TradingConfig):
        self.config = config

        # Define market holidays (can be expanded)
        self.market_holidays = {
            # 2025 holidays - update as needed
            datetime(2025, 1, 26).date(),  # Republic Day
            datetime(2025, 3, 14).date(),  # Holi
            datetime(2025, 4, 14).date(),  # Ram Navami
            datetime(2025, 4, 18).date(),  # Good Friday
            datetime(2025, 8, 15).date(),  # Independence Day
            datetime(2025, 10, 2).date(),  # Gandhi Jayanti
            datetime(2025, 10, 31).date(),  # Diwali
            # Add more holidays as needed
        }

    def is_trading_day(self, date_to_check: Optional[datetime] = None) -> bool:
        """Check if given date is a trading day"""
        if date_to_check is None:
            date_to_check = datetime.now(IST)

        # Check if it's a weekend (Saturday=5, Sunday=6)
        if date_to_check.weekday() >= 5:
            return False

        # Check if it's a market holiday
        if date_to_check.date() in self.market_holidays:
            return False

        return True

    def is_trading_time(self) -> bool:
        """Check if within trading hours on a trading day"""
        now = datetime.now(IST)

        # First check if it's a trading day
        if not self.is_trading_day(now):
            return False

        market_start = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.market_start_minute,
            second=0,
            microsecond=0
        )

        market_end = now.replace(
            hour=self.config.market_end_hour,
            minute=self.config.market_end_minute,
            second=0,
            microsecond=0
        )

        return market_start <= now <= market_end

    def is_orb_period(self) -> bool:
        """Check if we're currently in the ORB period (9:15 - 9:30 AM)"""
        now = datetime.now(IST)

        if not self.is_trading_day(now):
            return False

        orb_start = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.market_start_minute,
            second=0,
            microsecond=0
        )

        orb_end = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.orb_end_minute,
            second=0,
            microsecond=0
        )

        return orb_start <= now <= orb_end

    def is_signal_generation_time(self) -> bool:
        """Check if within signal generation window (after ORB until 2 PM)"""
        now = datetime.now(IST)

        if not self.is_trading_day(now):
            return False

        # Must be after ORB period
        orb_end = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.orb_end_minute,
            second=0,
            microsecond=0
        )

        # Must be before signal generation cutoff
        signal_cutoff = now.replace(
            hour=self.config.signal_generation_end_hour,
            minute=self.config.signal_generation_end_minute,
            second=0,
            microsecond=0
        )

        return orb_end < now <= signal_cutoff

    def get_orb_period_bounds(self) -> Tuple[datetime, datetime]:
        """Get ORB period start and end times for today"""
        now = datetime.now(IST)

        orb_start = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.market_start_minute,
            second=0,
            microsecond=0
        )

        orb_end = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.orb_end_minute,
            second=0,
            microsecond=0
        )

        return orb_start, orb_end

    def get_trading_session_bounds(self) -> Tuple[datetime, datetime]:
        """Get trading session start and end times for today"""
        now = datetime.now(IST)

        session_start = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.market_start_minute,
            second=0,
            microsecond=0
        )

        session_end = now.replace(
            hour=self.config.market_end_hour,
            minute=self.config.market_end_minute,
            second=0,
            microsecond=0
        )

        return session_start, session_end

    def time_until_market_open(self) -> Optional[int]:
        """Get seconds until market opens (None if market is open or closed for the day)"""
        now = datetime.now(IST)

        if not self.is_trading_day(now):
            # Find next trading day
            next_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            while not self.is_trading_day(next_day):
                next_day = next_day.replace(day=next_day.day + 1)

            market_open = next_day.replace(
                hour=self.config.market_start_hour,
                minute=self.config.market_start_minute
            )

            return int((market_open - now).total_seconds())

        # Check if market is already open
        if self.is_trading_time():
            return None

        # Market opens today but not yet open
        market_open = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.market_start_minute,
            second=0,
            microsecond=0
        )

        if now < market_open:
            return int((market_open - now).total_seconds())

        # Market closed for the day
        return None

    def time_until_orb_end(self) -> Optional[int]:
        """Get seconds until ORB period ends (None if not in ORB period)"""
        if not self.is_orb_period():
            return None

        now = datetime.now(IST)
        orb_end = now.replace(
            hour=self.config.market_start_hour,
            minute=self.config.orb_end_minute,
            second=0,
            microsecond=0
        )

        return int((orb_end - now).total_seconds())

    def time_until_market_close(self) -> Optional[int]:
        """Get seconds until market closes (None if market is closed)"""
        if not self.is_trading_time():
            return None

        now = datetime.now(IST)
        market_close = now.replace(
            hour=self.config.market_end_hour,
            minute=self.config.market_end_minute,
            second=0,
            microsecond=0
        )

        return int((market_close - now).total_seconds())

    def should_close_positions_for_day(self) -> bool:
        """Check if positions should be closed for end of day"""
        now = datetime.now(IST)

        if not self.is_trading_time():
            return False

        # Close positions 15 minutes before market close
        close_time = now.replace(
            hour=self.config.market_end_hour,
            minute=self.config.market_end_minute - 15,
            second=0,
            microsecond=0
        )

        return now >= close_time

    def get_current_market_phase(self) -> str:
        """Get current market phase"""
        if not self.is_trading_day():
            return "MARKET_HOLIDAY"

        if not self.is_trading_time():
            now = datetime.now(IST)
            market_start = now.replace(
                hour=self.config.market_start_hour,
                minute=self.config.market_start_minute
            )

            if now < market_start:
                return "PRE_MARKET"
            else:
                return "POST_MARKET"

        if self.is_orb_period():
            return "ORB_PERIOD"

        if self.is_signal_generation_time():
            return "SIGNAL_GENERATION"

        if self.should_close_positions_for_day():
            return "POSITION_CLOSING"

        return "REGULAR_TRADING"

    def get_trading_session_progress(self) -> float:
        """Get trading session progress as percentage (0-100)"""
        if not self.is_trading_time():
            return 0.0

        now = datetime.now(IST)
        session_start, session_end = self.get_trading_session_bounds()

        total_duration = (session_end - session_start).total_seconds()
        elapsed_duration = (now - session_start).total_seconds()

        progress = (elapsed_duration / total_duration) * 100
        return min(max(progress, 0), 100)

    def add_market_holiday(self, holiday_date: datetime):
        """Add a market holiday"""
        self.market_holidays.add(holiday_date.date())

    def remove_market_holiday(self, holiday_date: datetime):
        """Remove a market holiday"""
        self.market_holidays.discard(holiday_date.date())

    def get_next_trading_day(self, from_date: Optional[datetime] = None) -> datetime:
        """Get the next trading day"""
        if from_date is None:
            from_date = datetime.now(IST)

        next_day = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = next_day.replace(day=next_day.day + 1)

        while not self.is_trading_day(next_day):
            next_day = next_day.replace(day=next_day.day + 1)

        return next_day

    def format_time_remaining(self, seconds: int) -> str:
        """Format seconds into human readable time"""
        if seconds <= 0:
            return "0 seconds"

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if remaining_seconds > 0 and hours == 0:
            parts.append(f"{remaining_seconds} second{'s' if remaining_seconds != 1 else ''}")

        return " ".join(parts)


# Example usage and testing
if __name__ == "__main__":
    from config.settings import TradingConfig

    config = TradingConfig()
    timing_service = MarketTimingService(config)

    print("Market Timing Service Test")
    print("=" * 40)
    print(f"Is trading day: {timing_service.is_trading_day()}")
    print(f"Is trading time: {timing_service.is_trading_time()}")
    print(f"Is ORB period: {timing_service.is_orb_period()}")
    print(f"Is signal generation time: {timing_service.is_signal_generation_time()}")
    print(f"Current market phase: {timing_service.get_current_market_phase()}")
    print(f"Trading session progress: {timing_service.get_trading_session_progress():.1f}%")

    # Time remaining examples
    time_to_open = timing_service.time_until_market_open()
    if time_to_open:
        print(f"Time until market open: {timing_service.format_time_remaining(time_to_open)}")

    time_to_close = timing_service.time_until_market_close()
    if time_to_close:
        print(f"Time until market close: {timing_service.format_time_remaining(time_to_close)}")

    time_to_orb_end = timing_service.time_until_orb_end()
    if time_to_orb_end:
        print(f"Time until ORB end: {timing_service.format_time_remaining(time_to_orb_end)}")