# config/symbols.py

"""
Centralized Symbol Configuration for ORB Trading Strategy
Single source of truth for all tradable symbols and their Fyers format
"""

from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum


class SymbolCategory(Enum):
    """Symbol categories for organizational purposes (no allocation limits)"""
    LARGE_CAP = "LARGE_CAP"
    MID_CAP = "MID_CAP"
    SMALL_CAP = "SMALL_CAP"
    SECTORAL = "SECTORAL"
    INDEX = "INDEX"


@dataclass
class SymbolInfo:
    """Complete symbol information"""
    display_symbol: str  # Human readable symbol (e.g., "RELIANCE")
    fyers_symbol: str  # Fyers WebSocket format (e.g., "NSE:RELIANCE-EQ")
    company_name: str  # Full company name
    category: SymbolCategory  # Symbol category
    lot_size: int = 1  # Lot size for trading
    tick_size: float = 0.05  # Minimum price movement


class ORBSymbolManager:
    """Centralized symbol management for ORB strategy"""

    def __init__(self):
        # Single source of truth for all symbols
        self._symbols: Dict[str, SymbolInfo] = {

            # Large Cap Stocks - High Volume, Good for ORB
            "RELIANCE": SymbolInfo(
                display_symbol="RELIANCE",
                fyers_symbol="NSE:RELIANCE-EQ",
                company_name="Reliance Industries Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "TCS": SymbolInfo(
                display_symbol="TCS",
                fyers_symbol="NSE:TCS-EQ",
                company_name="Tata Consultancy Services Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "HDFCBANK": SymbolInfo(
                display_symbol="HDFCBANK",
                fyers_symbol="NSE:HDFCBANK-EQ",
                company_name="HDFC Bank Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "INFY": SymbolInfo(
                display_symbol="INFY",
                fyers_symbol="NSE:INFY-EQ",
                company_name="Infosys Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "ICICIBANK": SymbolInfo(
                display_symbol="ICICIBANK",
                fyers_symbol="NSE:ICICIBANK-EQ",
                company_name="ICICI Bank Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "HINDUNILVR": SymbolInfo(
                display_symbol="HINDUNILVR",
                fyers_symbol="NSE:HINDUNILVR-EQ",
                company_name="Hindustan Unilever Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "ITC": SymbolInfo(
                display_symbol="ITC",
                fyers_symbol="NSE:ITC-EQ",
                company_name="ITC Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "SBIN": SymbolInfo(
                display_symbol="SBIN",
                fyers_symbol="NSE:SBIN-EQ",
                company_name="State Bank of India",
                category=SymbolCategory.LARGE_CAP
            ),
            "BHARTIARTL": SymbolInfo(
                display_symbol="BHARTIARTL",
                fyers_symbol="NSE:BHARTIARTL-EQ",
                company_name="Bharti Airtel Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "WIPRO": SymbolInfo(
                display_symbol="WIPRO",
                fyers_symbol="NSE:WIPRO-EQ",
                company_name="Wipro Limited",
                category=SymbolCategory.LARGE_CAP
            ),

            # More Large Cap Stocks
            "MARUTI": SymbolInfo(
                display_symbol="MARUTI",
                fyers_symbol="NSE:MARUTI-EQ",
                company_name="Maruti Suzuki India Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "ASIANPAINT": SymbolInfo(
                display_symbol="ASIANPAINT",
                fyers_symbol="NSE:ASIANPAINT-EQ",
                company_name="Asian Paints Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "NESTLEIND": SymbolInfo(
                display_symbol="NESTLEIND",
                fyers_symbol="NSE:NESTLEIND-EQ",
                company_name="Nestle India Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "KOTAKBANK": SymbolInfo(
                display_symbol="KOTAKBANK",
                fyers_symbol="NSE:KOTAKBANK-EQ",
                company_name="Kotak Mahindra Bank Limited",
                category=SymbolCategory.LARGE_CAP
            ),
            "AXISBANK": SymbolInfo(
                display_symbol="AXISBANK",
                fyers_symbol="NSE:AXISBANK-EQ",
                company_name="Axis Bank Limited",
                category=SymbolCategory.LARGE_CAP
            ),

            # Mid Cap Stocks with Good ORB Potential
            "HCLTECH": SymbolInfo(
                display_symbol="HCLTECH",
                fyers_symbol="NSE:HCLTECH-EQ",
                company_name="HCL Technologies Limited",
                category=SymbolCategory.MID_CAP
            ),
            "TECHM": SymbolInfo(
                display_symbol="TECHM",
                fyers_symbol="NSE:TECHM-EQ",
                company_name="Tech Mahindra Limited",
                category=SymbolCategory.MID_CAP
            ),
            "TATAMOTORS": SymbolInfo(
                display_symbol="TATAMOTORS",
                fyers_symbol="NSE:TATAMOTORS-EQ",
                company_name="Tata Motors Limited",
                category=SymbolCategory.MID_CAP
            ),
            "SUNPHARMA": SymbolInfo(
                display_symbol="SUNPHARMA",
                fyers_symbol="NSE:SUNPHARMA-EQ",
                company_name="Sun Pharmaceutical Industries Limited",
                category=SymbolCategory.MID_CAP
            ),
            "DRREDDY": SymbolInfo(
                display_symbol="DRREDDY",
                fyers_symbol="NSE:DRREDDY-EQ",
                company_name="Dr. Reddy's Laboratories Limited",
                category=SymbolCategory.MID_CAP
            ),
            "TATASTEEL": SymbolInfo(
                display_symbol="TATASTEEL",
                fyers_symbol="NSE:TATASTEEL-EQ",
                company_name="Tata Steel Limited",
                category=SymbolCategory.MID_CAP
            ),
            "JSWSTEEL": SymbolInfo(
                display_symbol="JSWSTEEL",
                fyers_symbol="NSE:JSWSTEEL-EQ",
                company_name="JSW Steel Limited",
                category=SymbolCategory.MID_CAP
            ),
            "ONGC": SymbolInfo(
                display_symbol="ONGC",
                fyers_symbol="NSE:ONGC-EQ",
                company_name="Oil and Natural Gas Corporation Limited",
                category=SymbolCategory.MID_CAP
            ),
            "NTPC": SymbolInfo(
                display_symbol="NTPC",
                fyers_symbol="NSE:NTPC-EQ",
                company_name="NTPC Limited",
                category=SymbolCategory.MID_CAP
            ),
            "POWERGRID": SymbolInfo(
                display_symbol="POWERGRID",
                fyers_symbol="NSE:POWERGRID-EQ",
                company_name="Power Grid Corporation of India Limited",
                category=SymbolCategory.MID_CAP
            ),

            # Additional High-Volume Stocks
            "BAJAJ-AUTO": SymbolInfo(
                display_symbol="BAJAJ-AUTO",
                fyers_symbol="NSE:BAJAJ-AUTO-EQ",
                company_name="Bajaj Auto Limited",
                category=SymbolCategory.MID_CAP
            ),
            "M&M": SymbolInfo(
                display_symbol="M&M",
                fyers_symbol="NSE:M&M-EQ",
                company_name="Mahindra & Mahindra Limited",
                category=SymbolCategory.MID_CAP
            ),
            "HEROMOTOCO": SymbolInfo(
                display_symbol="HEROMOTOCO",
                fyers_symbol="NSE:HEROMOTOCO-EQ",
                company_name="Hero MotoCorp Limited",
                category=SymbolCategory.MID_CAP
            ),
            "BRITANNIA": SymbolInfo(
                display_symbol="BRITANNIA",
                fyers_symbol="NSE:BRITANNIA-EQ",
                company_name="Britannia Industries Limited",
                category=SymbolCategory.MID_CAP
            ),
            "DABUR": SymbolInfo(
                display_symbol="DABUR",
                fyers_symbol="NSE:DABUR-EQ",
                company_name="Dabur India Limited",
                category=SymbolCategory.MID_CAP
            ),
            "MARICO": SymbolInfo(
                display_symbol="MARICO",
                fyers_symbol="NSE:MARICO-EQ",
                company_name="Marico Limited",
                category=SymbolCategory.MID_CAP
            ),
            "TATACONSUM": SymbolInfo(
                display_symbol="TATACONSUM",
                fyers_symbol="NSE:TATACONSUM-EQ",
                company_name="Tata Consumer Products Limited",
                category=SymbolCategory.MID_CAP
            ),
            "COALINDIA": SymbolInfo(
                display_symbol="COALINDIA",
                fyers_symbol="NSE:COALINDIA-EQ",
                company_name="Coal India Limited",
                category=SymbolCategory.MID_CAP
            ),
            "IOC": SymbolInfo(
                display_symbol="IOC",
                fyers_symbol="NSE:IOC-EQ",
                company_name="Indian Oil Corporation Limited",
                category=SymbolCategory.MID_CAP
            ),
            "BPCL": SymbolInfo(
                display_symbol="BPCL",
                fyers_symbol="NSE:BPCL-EQ",
                company_name="Bharat Petroleum Corporation Limited",
                category=SymbolCategory.MID_CAP
            ),
        }

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get complete symbol information"""
        return self._symbols.get(symbol.upper())

    def get_fyers_symbol(self, symbol: str) -> str:
        """Get Fyers format symbol"""
        info = self.get_symbol_info(symbol)
        return info.fyers_symbol if info else None

    def get_display_symbol(self, fyers_symbol: str) -> str:
        """Get display symbol from Fyers format"""
        for info in self._symbols.values():
            if info.fyers_symbol == fyers_symbol:
                return info.display_symbol
        return None

    def get_all_symbols(self) -> List[str]:
        """Get all available symbols"""
        return list(self._symbols.keys())

    def get_all_fyers_symbols(self) -> List[str]:
        """Get all symbols in Fyers format"""
        return [info.fyers_symbol for info in self._symbols.values()]

    def get_symbols_by_category(self, category: SymbolCategory) -> List[str]:
        """Get symbols by category"""
        return [symbol for symbol, info in self._symbols.items()
                if info.category == category]

    def get_fyers_symbols_by_category(self, category: SymbolCategory) -> List[str]:
        """Get Fyers format symbols by category"""
        return [info.fyers_symbol for info in self._symbols.values()
                if info.category == category]

    def create_symbol_mappings(self) -> tuple:
        """Create forward and reverse mapping dictionaries"""
        forward_mapping = {info.display_symbol: info.fyers_symbol
                           for info in self._symbols.values()}
        reverse_mapping = {info.fyers_symbol: info.display_symbol
                           for info in self._symbols.values()}
        return forward_mapping, reverse_mapping

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported"""
        return symbol.upper() in self._symbols

    def get_trading_universe_size(self) -> int:
        """Get total number of tradable symbols"""
        return len(self._symbols)

    def get_category_distribution(self) -> Dict[SymbolCategory, int]:
        """Get distribution of symbols by category"""
        distribution = {}
        for info in self._symbols.values():
            distribution[info.category] = distribution.get(info.category, 0) + 1
        return distribution

    def export_for_websocket(self) -> Dict[str, str]:
        """Export symbols in format suitable for WebSocket subscription"""
        return {info.display_symbol: info.fyers_symbol
                for info in self._symbols.values()}

    def get_symbol_summary(self) -> Dict:
        """Get comprehensive summary of symbol universe"""
        return {
            'total_symbols': len(self._symbols),
            'category_distribution': self.get_category_distribution(),
            'sample_symbols': {
                'large_cap': self.get_symbols_by_category(SymbolCategory.LARGE_CAP)[:5],
                'mid_cap': self.get_symbols_by_category(SymbolCategory.MID_CAP)[:5]
            }
        }


# Global symbol manager instance
symbol_manager = ORBSymbolManager()


# Convenience functions for backward compatibility and easy access
def get_orb_symbols() -> List[str]:
    """Get all ORB trading symbols (display format)"""
    return symbol_manager.get_all_symbols()


def get_orb_fyers_symbols() -> List[str]:
    """Get all ORB symbols in Fyers WebSocket format"""
    return symbol_manager.get_all_fyers_symbols()


def convert_to_fyers_format(symbol: str) -> str:
    """Convert display symbol to Fyers format"""
    return symbol_manager.get_fyers_symbol(symbol)


def convert_from_fyers_format(fyers_symbol: str) -> str:
    """Convert Fyers format to display symbol"""
    return symbol_manager.get_display_symbol(fyers_symbol)


def validate_orb_symbol(symbol: str) -> bool:
    """Validate if symbol is supported for ORB trading"""
    return symbol_manager.validate_symbol(symbol)


def get_symbol_mappings() -> tuple:
    """Get symbol mappings for WebSocket services"""
    return symbol_manager.create_symbol_mappings()


# Example usage and testing
if __name__ == "__main__":
    print("ORB Symbol Manager Test")
    print("=" * 50)

    # Test symbol manager
    print(f"Total symbols: {symbol_manager.get_trading_universe_size()}")
    print(f"Category distribution: {symbol_manager.get_category_distribution()}")

    # Test specific symbol
    test_symbol = "RELIANCE"
    info = symbol_manager.get_symbol_info(test_symbol)
    if info:
        print(f"\nTesting {test_symbol}:")
        print(f"  Display: {info.display_symbol}")
        print(f"  Fyers: {info.fyers_symbol}")
        print(f"  Company: {info.company_name}")
        print(f"  Category: {info.category.value}")

    # Test mappings
    forward_map, reverse_map = symbol_manager.create_symbol_mappings()
    print(f"\nMapping test:")
    print(f"  RELIANCE -> {forward_map.get('RELIANCE')}")
    print(f"  NSE:TCS-EQ -> {reverse_map.get('NSE:TCS-EQ')}")

    # Test categories
    large_cap_symbols = symbol_manager.get_symbols_by_category(SymbolCategory.LARGE_CAP)
    print(f"\nLarge cap symbols ({len(large_cap_symbols)}): {large_cap_symbols[:10]}...")

    mid_cap_symbols = symbol_manager.get_symbols_by_category(SymbolCategory.MID_CAP)
    print(f"Mid cap symbols ({len(mid_cap_symbols)}): {mid_cap_symbols[:10]}...")

    # Test convenience functions
    print(f"\nConvenience functions:")
    print(f"  All symbols count: {len(get_orb_symbols())}")
    print(f"  All Fyers symbols count: {len(get_orb_fyers_symbols())}")
    print(f"  RELIANCE -> {convert_to_fyers_format('RELIANCE')}")
    print(f"  NSE:INFY-EQ -> {convert_from_fyers_format('NSE:INFY-EQ')}")
    print(f"  Valid symbol 'TCS': {validate_orb_symbol('TCS')}")
    print(f"  Valid symbol 'INVALID': {validate_orb_symbol('INVALID')}")