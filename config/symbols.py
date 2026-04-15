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
            # Upstream Energy — Clear Winners
            "ONGC": "NSE:ONGC-EQ",
            "OIL": "NSE:OIL-EQ",
            "GAIL": "NSE:GAIL-EQ",

            # Renewables — Structural Beneficiaries
            "ADANIGREEN": "NSE:ADANIGREEN-EQ",
            "TATAPOWER": "NSE:TATAPOWER-EQ",
            # "NTPCGREEN": "NSE:NTPCGREEN-EQ",
            # "SJVN": "NSE:SJVN-EQ",
            "CESC": "NSE:CESC-EQ",

            # City Gas / LNG Distribution
            "Indraprastha Gas": "NSE:IGL-EQ",
            "Mahanagar Gas": "NSE:MGL-EQ",
            "GUJGASLTD": "NSE:GUJGASLTD-EQ",
            "PETRONET": "NSE:PETRONET-EQ",

            # Defence
            "Hindustan Aeronautics": "NSE:HAL-EQ",
            "Bharat Electronics": "NSE:BEL-EQ",
            "Mazagon Dock Shipbuilders": "NSE:MAZDOCK-EQ",
            "Data Patterns (India)": "NSE:DATAPATTNS-EQ",

            # Sugar - Ethanol
            "EID Parry (India)": "NSE:EIDPARRY-EQ",
            "Balrampur Chini Mills": "NSE:BALRAMCHIN-EQ",
            # "Shree Renuka Sugars": "NSE:RENUKA-EQ",
            "Triveni Engineering & Industries": "NSE:TRIVENI-EQ",
            # "Bajaj Hindusthan Sugar": "NSE:BAJAJHIND-EQ",

            # Pharmaceuticals
            "Sun Pharmaceutical Industries": "NSE:SUNPHARMA-EQ",
            "Divis Laboratories": "NSE:DIVISLAB-EQ",
            "Cipla": "NSE:CIPLA-EQ",

            # Petroluem (Oil Marketing Companies)
            "Indian Oil Corporation": "NSE:IOC-EQ",
            "Bharat Petroleum Corporation": "NSE:BPCL-EQ",
            "Hindustan Petroleum Corporation": "NSE:HINDPETRO-EQ",

            # Airlines
            "InterGlobe Aviation": "NSE:INDIGO-EQ",

            # Paints
            "Asian Paints": "NSE:ASIANPAINT-EQ",
            "Berger Paints India": "NSE:BERGEPAINT-EQ",
            "Kansai Nerolac Paints": "NSE:KANSAINER-EQ",
            "Indigo Paints": "NSE:INDIGOPNTS-EQ",

            # Tyres
            "CEAT": "NSE:CEATLTD-EQ",
            "MRF": "NSE:MRF-EQ",
            "Apollo Tyres": "NSE:APOLLOTYRE-EQ",
            "JK Tyre & Industries": "NSE:JKTYRE-EQ",
            "Balkrishna Industries": "NSE:BALKRISIND-EQ",

            # Autos(Nifty Auto)
            "Maruti Suzuki India": "NSE:MARUTI-EQ",
            "Mahindra & Mahindra": "NSE:M&M-EQ",
            "Bajaj Auto": "NSE:BAJAJ-AUTO-EQ",
            "Eicher Motors": "NSE:EICHERMOT-EQ",
            "TVS Motor Company": "NSE:TVSMOTOR-EQ",

            # Jewellery Retail
            "TITAN": "NSE:TITAN-EQ",
            "KALYANKJIL": "NSE:KALYANKJIL-EQ",
            "PCJEWELLER": "NSE:PCJEWELLER-EQ",
            "PNGBL": "NSE:PNGBL-EQ",
            "THANGAMAYL": "NSE:THANGAMAYL-EQ",
            "SENCO": "NSE:SENCO-EQ",
            "RJIL": "NSE:RJIL-EQ",
            "GOLDIAM": "NSE:GOLDIAM-EQ",
            "DIVHJL": "NSE:DIVHJL-EQ",  # Divine Hira Jewellers
            "ZODIACJL": "NSE:ZODIACJL-EQ",  # Zodiac‑JRD‑MKJ (listed jeweller)
            "NARBADAG": "NSE:NARBADAG-EQ",  # Narbada Gems & Jewellery
            "MOKSH": "NSE:MOKSH-EQ",  # Moksh Ornaments
            "SWARN": "NSE:SWARN-EQ", # Swarnsarita Jewels India

            # Jewellery Retail
            "MUTHOOTFIN": "NSE:MUTHOOTFIN-EQ",      # Muthoot Finance – large gold‑loan NBFC
            "MANAPPURAM": "NSE:MANAPPURAM-EQ",      # Manappuram Finance – big gold‑loan NBFC
            "KARURFIN": "NSE:KARURFIN-EQ",        # Karur Vysya Bank – has gold‑loan books
            "KTKBANK": "NSE:KTKBANK-EQ",          # KTK Bank – active in gold‑loan segment

            # IPO Stocks
            "VIKRAMSOLR": "NSE:VIKRAMSOLR-EQ",
            "ATLANTAELE": "NSE:ATLANTAELE-EQ",
            "SOLARWORLD": "NSE:SOLARWORLD-EQ",
            "RUBICON": "NSE:RUBICON-EQ",
            "MIDWESTLTD": "NSE:MIDWESTLTD-EQ",

            # Favourite Stocks
            "STLTECH": "NSE:STLTECH-EQ",
            "SKYGOLD": "NSE:SKYGOLD-EQ",
            "AXISCADES": "NSE:AXISCADES-EQ",
        }

        # Create reverse mapping for quick lookups
        self._reverse_mappings = {v: k for k, v in self._symbol_mappings.items()}

        # Case-insensitive lookup index (uppercase key -> fyers symbol)
        self._upper_mappings: Dict[str, str] = {
            k.upper(): v for k, v in self._symbol_mappings.items()
        }

    def get_fyers_symbol(self, symbol: str) -> str:
        """Get Fyers format symbol (case-insensitive)"""
        return self._upper_mappings.get(symbol.upper())

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
        """Check if symbol is supported (case-insensitive)"""
        return symbol.upper() in self._upper_mappings

    def add_symbol(self, symbol: str, fyers_symbol: str) -> bool:
        """
        Dynamically add a symbol to the trading universe (e.g. from Moneycontrol).
        Returns True if the symbol was newly added, False if it already existed.
        """
        symbol = symbol.upper()
        if symbol in self._upper_mappings:
            return False
        self._symbol_mappings[symbol] = fyers_symbol
        self._upper_mappings[symbol] = fyers_symbol
        self._reverse_mappings[fyers_symbol] = symbol
        return True

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