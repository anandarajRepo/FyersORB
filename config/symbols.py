# config/symbols.py

"""
Simplified Symbol Configuration for ORB Trading Strategy
Single source of truth for symbol to Fyers format mapping
"""

from typing import Dict, List, Tuple


class ORBSymbolManager:
    """Simplified symbol manager for ORB strategy - just symbol mappings"""

    def __init__(self):
        # Simple mapping: symbol -> Fyers WebSocket format
        self._symbol_mappings: Dict[str, str] = {
            # IPO Stocks
            "EUROPRATIK": "NSE:EUROPRATIK-EQ",
            "SHRINGARMS": "NSE:SHRINGARMS-EQ",
            "URBANCO": "NSE:URBANCO-EQ",
            "AMANTA": "NSE:AMANTA-EQ",
            "VIKRAMSOLR": "NSE:VIKRAMSOLR-EQ",
            "SHREEJISPG": "NSE:SHREEJISPG-EQ",
            "PATELRMART": "NSE:PATELRMART-EQ",
            "REGAAL": "NSE:REGAAL-EQ",
            "HILINFRA": "NSE:HILINFRA-EQ",
            "SAATVIKGL": "NSE:SAATVIKGL-EQ",
            "ATLANTAELE": "NSE:ATLANTAELE-EQ",
            "STYL": "NSE:STYL-EQ",
            "SOLARWORLD": "NSE:SOLARWORLD-EQ",
            "TRUALT": "NSE:TRUALT-EQ",
            "ADVANCE": "NSE:ADVANCE-EQ",
            "LGEINDIA": "NSE:LGEINDIA-EQ",
            "RUBICON": "NSE:RUBICON-EQ",
            "MIDWESTLTD": "NSE:MIDWESTLTD-EQ",
            "ORKLAINDIA": "NSE:ORKLAINDIA-EQ",
            "LENSKART": "NSE:LENSKART-EQ",
            "GROWW": "NSE:GROWW-EQ",
            "SUDEEPPHRM": "NSE:SUDEEPPHRM-EQ",
            "EXCELSOFT": "NSE:EXCELSOFT-EQ",
            "CAPILLARY": "NSE:CAPILLARY-B",
            "TENNIND": "NSE:TENNIND-EQ",

            # Favourite Stocks
            "STLNETWORK": "NSE:STLNETWORK-EQ",
            "STLTECH": "NSE:STLTECH-EQ",
            "SKYGOLD": "NSE:SKYGOLD-EQ",
            "SATTRIX": "BSE:SATTRIX-MT",
        }

        # Create reverse mapping for quick lookups
        self._reverse_mappings = {v: k for k, v in self._symbol_mappings.items()}

    def get_fyers_symbol(self, symbol: str) -> str:
        """Get Fyers format symbol"""
        return self._symbol_mappings.get(symbol.upper())

    def get_display_symbol(self, fyers_symbol: str) -> str:
        """Get display symbol from Fyers format"""
        return self._reverse_mappings.get(fyers_symbol)

    def get_all_symbols(self) -> List[str]:
        """Get all available symbols"""
        return list(self._symbol_mappings.keys())

    def get_all_fyers_symbols(self) -> List[str]:
        """Get all symbols in Fyers format"""
        return list(self._symbol_mappings.values())

    def create_symbol_mappings(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Create forward and reverse mapping dictionaries"""
        return self._symbol_mappings.copy(), self._reverse_mappings.copy()

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported"""
        return symbol.upper() in self._symbol_mappings

    def get_trading_universe_size(self) -> int:
        """Get total number of tradable symbols"""
        return len(self._symbol_mappings)

    def export_for_websocket(self) -> Dict[str, str]:
        """Export symbols in format suitable for WebSocket subscription"""
        return self._symbol_mappings.copy()

    def get_symbol_summary(self) -> Dict:
        """Get summary of symbol universe"""
        return {
            'total_symbols': len(self._symbol_mappings),
            'symbols': self.get_all_symbols(),
            'fyers_symbols': self.get_all_fyers_symbols()
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


def get_symbol_mappings() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Get symbol mappings for WebSocket services"""
    return symbol_manager.create_symbol_mappings()


# Example usage and testing
if __name__ == "__main__":
    print("ORB Symbol Manager Test")
    print("=" * 50)

    # Test symbol manager
    print(f"Total symbols: {symbol_manager.get_trading_universe_size()}")

    # Test specific symbol
    test_symbol = "EUROPRATIK"
    fyers_format = convert_to_fyers_format(test_symbol)
    print(f"\nTesting {test_symbol}:")
    print(f"  Display: {test_symbol}")
    print(f"  Fyers: {fyers_format}")

    # Test reverse conversion
    display_format = convert_from_fyers_format(fyers_format)
    print(f"  Reverse: {display_format}")

    # Test mappings
    forward_map, reverse_map = get_symbol_mappings()
    print(f"\nMapping test:")
    print(f"  {test_symbol} -> {forward_map.get(test_symbol)}")
    print(f"  {fyers_format} -> {reverse_map.get(fyers_format)}")

    # Test validation
    print(f"\nValidation test:")
    print(f"  Valid symbol '{test_symbol}': {validate_orb_symbol(test_symbol)}")
    print(f"  Invalid symbol 'INVALID': {validate_orb_symbol('INVALID')}")

    # Test convenience functions
    print(f"\nConvenience functions:")
    print(f"  All symbols count: {len(get_orb_symbols())}")
    print(f"  All Fyers symbols count: {len(get_orb_fyers_symbols())}")

    # Display all symbols
    print(f"\nAll symbols:")
    for symbol in get_orb_symbols():
        print(f"  {symbol} -> {convert_to_fyers_format(symbol)}")