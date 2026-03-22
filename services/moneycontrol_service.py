# services/moneycontrol_service.py

"""
Moneycontrol 'Stocks to Watch Today' Service

Fetches and parses the daily 'Stocks to watch today' article from Moneycontrol
to extract NSE stock symbols. These symbols are added to the trading universe
before momentum scoring and leverage filtering, so they get screened the same
way as all other candidates.
"""

import logging
import re
import requests
from typing import List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Tag page listing 'stocks-to-watch' articles
MONEYCONTROL_TAG_URL = "https://www.moneycontrol.com/news/tags/stocks-to-watch.html"

REQUEST_TIMEOUT = 15  # seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# Common words that look like NSE symbols but aren't
_EXCLUDED_WORDS = {
    "THE", "AND", "FOR", "NSE", "BSE", "SEBI", "RBI", "ETF", "IPO", "NFO",
    "LTP", "MIS", "SIP", "STP", "SWP", "NAV", "AUM", "FII", "DII", "FPI",
    "GDP", "CPI", "WPI", "EMI", "NPA", "ROE", "EPS", "PE", "PB", "EV",
    "US", "UK", "IT", "AI", "PE",
}

# Patterns to extract NSE ticker symbols from article text.
# Ordered from most specific to least specific.
_NSE_PATTERNS = [
    r'NSE[:\s]+([A-Z][A-Z0-9&]{1,14})-EQ',      # NSE:SYMBOL-EQ
    r'\(NSE[:\s]*([A-Z][A-Z0-9&]{1,14})\)',       # (NSE: SYMBOL) or (NSE:SYMBOL)
    r'NSE[:\s]+([A-Z][A-Z0-9&]{1,14})\b',         # NSE: SYMBOL
    r'\bNSE[:/]([A-Z][A-Z0-9&]{1,14})\b',         # NSE/SYMBOL or NSE:SYMBOL
]


class MoneycontrolWatchlistService:
    """
    Fetches the 'Stocks to watch today' article from Moneycontrol and extracts
    NSE equity symbols from it.

    Usage:
        service = MoneycontrolWatchlistService()
        stocks = service.fetch_stocks_to_watch()
        # returns [("RELIANCE", "NSE:RELIANCE-EQ"), ("TCS", "NSE:TCS-EQ"), ...]
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    def fetch_stocks_to_watch(self) -> List[Tuple[str, str]]:
        """
        Find today's 'Stocks to watch today' article and return all NSE symbols
        mentioned in it as (symbol, fyers_format) tuples.

        Returns empty list on any failure so callers can treat it as optional.
        """
        try:
            article_url = self._find_todays_article_url()
            if not article_url:
                logger.warning("MoneyControl: Could not find today's 'Stocks to watch' article")
                return []

            logger.info(f"MoneyControl: Fetching article → {article_url}")
            symbols = self._extract_symbols_from_article(article_url)

            if symbols:
                names = [s[0] for s in symbols]
                logger.info(f"MoneyControl: Found {len(symbols)} symbol(s): {names}")
            else:
                logger.warning("MoneyControl: No NSE symbols found in article")

            return symbols

        except Exception as e:
            logger.error(f"MoneyControl: Unexpected error while fetching watchlist: {e}")
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_todays_article_url(self) -> Optional[str]:
        """
        Scrape the tag listing page and return the URL of the first article
        that matches today's date. Falls back to the most recent article if
        no exact date match is found.
        """
        try:
            resp = self._session.get(MONEYCONTROL_TAG_URL, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"MoneyControl: Failed to load tag page: {e}")
            return None

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("MoneyControl: 'beautifulsoup4' not installed. Run: pip install beautifulsoup4")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        today = datetime.now()

        # Today's date in several formats that may appear in article URLs or listing text
        today_path = today.strftime("%Y/%m/%d")               # 2026/03/20
        today_month = today.strftime("%Y/%m")                  # 2026/03
        today_url_text = today.strftime("%B-%d-%Y").lower()    # march-20-2026
        today_date_formats = [
            today.strftime("%B %d, %Y"),    # March 20, 2026
            today.strftime("%B %-d, %Y"),   # March 20, 2026 (no zero-pad)
            today.strftime("%d %B %Y"),     # 20 March 2026
            today.strftime("%b %d, %Y"),    # Mar 20, 2026
            today.strftime("%d %b %Y"),     # 20 Mar 2026
            today.strftime("%Y-%m-%d"),     # 2026-03-20
        ]

        def _is_article_link(href: str) -> bool:
            """True for stocks-to-watch article links (not the tag page itself)."""
            path = href.split("?")[0]
            if "tags/stocks-to-watch" in path.lower():
                return False
            return "stocks-to-watch" in path.lower() and "/news/" in path.lower()

        # Build ordered list of candidate (anchor, absolute_url) pairs as they
        # appear in the listing (top = most recent).
        candidates = []
        seen_urls = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not _is_article_link(href):
                continue
            url = self._absolute_url(href)
            if url and url not in seen_urls:
                seen_urls.add(url)
                candidates.append((a, url))

        logger.debug(f"MoneyControl: Found {len(candidates)} candidate article link(s) on listing page")

        # Pass 1: URL contains today's full date path (e.g. 2026/03/20)
        for a, url in candidates:
            if today_path in url or today_url_text in url.lower():
                logger.debug(f"MoneyControl: Matched today's article via URL date: {url}")
                return url

        # Pass 2: look for today's date text in the nearest enclosing container
        for a, url in candidates:
            parent = a.parent
            for _ in range(5):
                if parent is None:
                    break
                parent_text = parent.get_text(" ", strip=True)
                for fmt in today_date_formats:
                    if fmt.lower() in parent_text.lower():
                        logger.debug(f"MoneyControl: Matched today's article via listing date text: {url}")
                        return url
                parent = parent.parent

        # Pass 3: first candidate whose URL contains the current month path
        for a, url in candidates:
            if today_month in url:
                logger.debug(f"MoneyControl: Falling back to first article from this month: {url}")
                return url

        # Pass 4: just take the first (most recent) candidate
        if candidates:
            _, url = candidates[0]
            logger.debug(f"MoneyControl: Falling back to most recent article: {url}")
            return url

        return None

    def _extract_symbols_from_article(self, url: str) -> List[Tuple[str, str]]:
        """Fetch article HTML and extract all NSE equity symbols."""
        try:
            resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"MoneyControl: Failed to fetch article {url}: {e}")
            return []

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Try to narrow down to the article body to reduce noise.
        # Moneycontrol uses several class names across layouts.
        article_div = (
            soup.find("div", class_=re.compile(r"article[-_]?(body|content|text|detail)", re.I))
            or soup.find("div", class_=re.compile(r"(content_wrapper|arti[-_]?content|story[-_]?content|article-page)", re.I))
            or soup.find("article")
            or soup.find("div", id=re.compile(r"article|content|story|main[-_]?content", re.I))
        )
        text = article_div.get_text(" ", strip=True) if article_div else soup.get_text(" ", strip=True)

        symbols_found: set = set()

        # Pattern-based extraction from article text
        for pattern in _NSE_PATTERNS:
            for match in re.finditer(pattern, text):
                symbol = match.group(1).strip().upper()
                if self._is_valid_nse_symbol(symbol):
                    symbols_found.add(symbol)

        # Also extract symbols from embedded Moneycontrol stock-quote links, e.g.
        # href="/india/stockpricequote/.../SYMBOL" or stockPageName=SYMBOL in href
        search_scope = article_div if article_div else soup
        for a in search_scope.find_all("a", href=True):
            href = a["href"]
            # /india/stockpricequote/<sector>/<name>/<SYMBOL>
            m = re.search(r'/stockpricequote/[^/]+/[^/]+/([A-Z][A-Z0-9&]{1,14})(?:\b|$)', href)
            if m:
                symbol = m.group(1).upper()
                if self._is_valid_nse_symbol(symbol):
                    symbols_found.add(symbol)
            # data-* attributes sometimes carry the NSE code
            for attr_val in a.attrs.values():
                if isinstance(attr_val, str):
                    m2 = re.search(r'\bNSE[:/]([A-Z][A-Z0-9&]{1,14})\b', attr_val)
                    if m2:
                        symbol = m2.group(1).upper()
                        if self._is_valid_nse_symbol(symbol):
                            symbols_found.add(symbol)

        return [(sym, f"NSE:{sym}-EQ") for sym in sorted(symbols_found)]

    def _is_valid_nse_symbol(self, symbol: str) -> bool:
        """Return True if the string looks like a plausible NSE equity ticker."""
        if not symbol or len(symbol) < 2 or len(symbol) > 15:
            return False
        if not re.match(r'^[A-Z][A-Z0-9&]{1,14}$', symbol):
            return False
        return symbol not in _EXCLUDED_WORDS

    @staticmethod
    def _absolute_url(href: str) -> str:
        if href.startswith("https://") or href.startswith("http://"):
            return href
        # Reject non-HTTP scheme URIs (whatsapp://, mailto:, tel:, etc.)
        if "://" in href:
            return ""
        return "https://www.moneycontrol.com" + href
