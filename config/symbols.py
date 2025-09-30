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

            # IPO Stocks
            "EUROPRATIK": SymbolInfo(
                display_symbol="EUROPRATIK",
                fyers_symbol="NSE:EUROPRATIK-EQ",
                company_name="Euro Pratik Limited",
                category=SymbolCategory.SMALL_CAP
            ),
            "SHRINGARMS": SymbolInfo(
                display_symbol="SHRINGARMS",
                fyers_symbol="NSE:SHRINGARMS-EQ",
                company_name="Shringar House of Mangalsutra Ltd",
                category=SymbolCategory.SMALL_CAP
            ),
            "URBANCO": SymbolInfo(
                display_symbol="URBANCO",
                fyers_symbol="NSE:URBANCO-EQ",
                company_name="Urban Company Ltd",
                category=SymbolCategory.SMALL_CAP
            ),
            "AMANTA": SymbolInfo(
                display_symbol="AMANTA",
                fyers_symbol="NSE:AMANTA-EQ",
                company_name="Amanta Limited",
                category=SymbolCategory.SMALL_CAP
            ),
            "VIKRAMSOLR": SymbolInfo(
                display_symbol="VIKRAMSOLR",
                fyers_symbol="NSE:VIKRAMSOLR-EQ",
                company_name="Vikram Solar Limited",
                category=SymbolCategory.SMALL_CAP
            ),
            "SHREEJISPG": SymbolInfo(
                display_symbol="SHREEJISPG",
                fyers_symbol="NSE:SHREEJISPG-EQ",
                company_name="Shreeji Shipping Global Limited",
                category=SymbolCategory.SMALL_CAP
            ),
            "PATELRMART": SymbolInfo(
                display_symbol="PATELRMART",
                fyers_symbol="NSE:PATELRMART-EQ",
                company_name="Patel Retail Limited",
                category=SymbolCategory.SMALL_CAP
            ),
            "REGAAL": SymbolInfo(
                display_symbol="REGAAL",
                fyers_symbol="NSE:REGAAL-EQ",
                company_name="Regaal Resources",
                category=SymbolCategory.SMALL_CAP
            ),
            "HILINFRA": SymbolInfo(
                display_symbol="HILINFRA",
                fyers_symbol="NSE:HILINFRA-EQ",
                company_name="Highway Infrastructure",
                category=SymbolCategory.SMALL_CAP
            ),
            "SAATVIKGL": SymbolInfo(
                display_symbol="SAATVIKGL",
                fyers_symbol="NSE:SAATVIKGL-EQ",
                company_name="Saatvik Green Energy",
                category=SymbolCategory.SMALL_CAP
            ),


            # Favourite Stocks
            "STLNETWORK": SymbolInfo(
                display_symbol="STLNETWORK",
                fyers_symbol="NSE:STLNETWORK-EQ",
                company_name="STL Network Limited",
                category=SymbolCategory.SMALL_CAP
            ),
            "STLTECH": SymbolInfo(
                display_symbol="STLTECH",
                fyers_symbol="NSE:STLTECH-EQ",
                company_name="STL Technology Limited",
                category=SymbolCategory.SMALL_CAP
            ),
            "SKYGOLD": SymbolInfo(
                display_symbol="SKYGOLD",
                fyers_symbol="NSE:SKYGOLD-EQ",
                company_name="Sky Gold Limited",
                category=SymbolCategory.SMALL_CAP
            ),
            "AXISCADES": SymbolInfo(
                display_symbol="AXISCADES",
                fyers_symbol="NSE:AXISCADES-EQ",
                company_name="Axiscades Technologies Ltd",
                category=SymbolCategory.SMALL_CAP
            ),
            "SATTRIX": SymbolInfo(
                display_symbol="SATTRIX",
                fyers_symbol="BSE:SATTRIX-MT",
                company_name="Satrix Information Security Ltd",
                category=SymbolCategory.SMALL_CAP
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