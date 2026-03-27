# FyersORB — Project Prompt

## Project Overview

FyersORB is a fully automated intraday algorithmic trading system that implements the **Open Range Breakout (ORB)** strategy for the Indian stock market (NSE). It integrates with the Fyers broker platform via their v3 API, executing trades automatically based on technical breakouts detected during the market opening period.

---

## Core Strategy

### Open Range Breakout (ORB)

1. **Opening Range Detection** (9:15–9:30 AM IST, first 15 minutes)
   - Track the high and low prices during the opening range window
   - Measure range size, volume, and volatility

2. **Breakout Signal Generation**
   - Detect price breaks above (bullish) or below (bearish) the opening range
   - Confirm with volume spike (≥3x average volume by default)
   - Score breakout strength (0–100) based on range size, volume, and momentum

3. **Entry & Exit Logic**
   - Enter at breakout price with a limit order slightly above/below range boundary
   - Set stop-loss as a percentage below/above entry
   - Set profit target as a multiple of risk (configurable risk-reward ratio)
   - Trailing stop to lock in profits; partial exit at first target (50% of position)

4. **Crossover Timing Validation**
   - Skip the first crossover; trade only from the 2nd crossover onwards
   - Enforce a 30-second delay between consecutive crossover signals

---

## Architecture

```
FyersORB/
├── main.py                          # Entry point, CLI, async event loop
├── config/
│   ├── settings.py                  # FyersConfig, ORBStrategyConfig, TradingConfig dataclasses
│   ├── symbols.py                   # Centralized symbol management & Fyers format mapping
│   └── websocket_config.py          # WebSocket connection profiles
├── models/
│   └── trading_models.py            # Domain objects: Position, ORBSignal, LiveQuote, OpenRange, TradeResult, StrategyMetrics
├── services/
│   ├── fyers_websocket_service.py   # Hybrid WebSocket / REST data feed with reconnection
│   ├── analysis_service.py          # ORB detection, breakout strength scoring, FVG analysis
│   ├── momentum_service.py          # Momentum scoring & stock screening (ROC, RSI, volume, MAs)
│   ├── leverage_filter_service.py   # Intraday leverage validation via Fyers span_margin API
│   ├── market_timing_service.py     # Market hours, ORB windows, trading day checks
│   ├── trend_direction_service.py   # Multi-day trend analysis (EMA, ADX, swing structure)
│   └── moneycontrol_service.py      # Daily "Stocks to watch" scraping from Moneycontrol
├── strategy/
│   ├── orb_strategy.py              # Main state machine: signal → entry → management → exit
│   └── order_manager.py             # Fyers API order placement, tracking, modifications
└── utils/
    └── enhanced_auth_helper.py      # OAuth authentication & token refresh
```

**Tech Stack**: Python 3.11+, asyncio, fyers-apiv3, pandas, numpy, scipy, requests, beautifulsoup4, pytz, python-dotenv, psutil, colorlog

---

## Multi-Factor Signal Filtering

Every breakout signal passes through three independent filters before an order is placed:

| Filter | Purpose | Default |
|---|---|---|
| **Momentum Filter** | Score stocks on 5/10/20-day ROC, RSI, volume trend, MA alignment; select top N | Enabled, top 15 |
| **Leverage Filter** | Validate ≥5x intraday margin availability via Fyers broker API | Enabled |
| **Trend Direction Filter** | Ensure trade aligns with prevailing trend (EMA 9/21/50, ADX, swing structure) on both the stock and Nifty 50 | Enabled |

Stocks also sourced from Moneycontrol's daily "Stocks to watch" article to augment the trading universe.

---

## Risk Management

- **Position sizing**: Risk-per-trade allocation (default 30% of portfolio value)
- **Max concurrent positions**: 3
- **Max daily loss limit**: 2% of portfolio
- **Trailing stop-loss**: Adjusts as position becomes profitable
- **Partial exits**: Sell 50% at first target; hold remainder for extended run
- **Fair Value Gap (FVG)** detection: Optional — blocks trades, reduces confidence, or enforces gap alignment

---

## Configuration

All parameters are driven by environment variables (`.env` file):

```env
# Fyers API credentials
FYERS_CLIENT_ID=
FYERS_SECRET_KEY=
FYERS_ACCESS_TOKEN=
FYERS_REFRESH_TOKEN=

# Portfolio & risk
PORTFOLIO_VALUE=30000
RISK_PER_TRADE=0.30
MAX_POSITIONS=3

# ORB strategy
ORB_PERIOD_MINUTES=15
MIN_BREAKOUT_VOLUME=3.0
STOP_LOSS_PCT=0.005
TARGET_MULTIPLIER=2.0
TRAILING_STOP_PCT=0.003

# Filters
ENABLE_MOMENTUM_FILTER=true
ENABLE_LEVERAGE_FILTER=true
ENABLE_TREND_FILTER=true
MOMENTUM_TOP_N=15
MOMENTUM_LOOKBACK_DAYS=20
MIN_INTRADAY_LEVERAGE=5.0

# Logging
LOG_LEVEL=INFO
```

---

## Operations

### Running the System

```bash
# Authenticate with Fyers (first time or token refresh)
python main.py auth

# Start the ORB strategy
python main.py run

# Test WebSocket data connection
python main.py test-websocket
```

### Shell Scripts

```bash
./run_orb.sh     # Activates venv, truncates logs, starts strategy
./stop_orb.sh    # Gracefully stops the process; force-kills if needed
```

### Cron Schedule

```cron
55 8 * * 1-5   /path/to/run_orb.sh    # Start at 8:55 AM IST (weekdays)
30 15 * * 1-5  /path/to/stop_orb.sh   # Stop at 3:30 PM IST (weekdays)
```

Logs are written to `logs/orb_strategy.log` and stdout.

---

## Key Data Models

| Model | Description |
|---|---|
| `OpenRange` | High, low, volume, and time window of the ORB period |
| `ORBSignal` | Direction, breakout price, strength score (0–100), confidence |
| `Position` | Entry price, quantity, stop-loss, target, P&L, status |
| `TradeResult` | Entry/exit prices, duration, P&L, exit reason |
| `StrategyMetrics` | Win rate, cumulative P&L, max drawdown, Sharpe ratio |
| `LiveQuote` | Real-time bid/ask/LTP/volume from WebSocket or REST |

---

## Trading Universe

~150+ NSE stocks across sectors: Energy, Renewables, Defense, Pharma, IT, Auto, Paints, Tyres, and more. Symbol mapping is managed in `config/symbols.py` and supports case-insensitive lookup and Fyers-format conversion (e.g., `NSE:RELIANCE-EQ`).

---

## Development Guidelines

1. **Do not modify live trading logic** without thorough backtesting — incorrect stop-loss or order logic leads to real financial loss.
2. **All configuration** must flow through environment variables and the config dataclasses in `config/settings.py`. Never hardcode credentials or thresholds.
3. **Services are stateless** — strategy state lives in `orb_strategy.py`. Keep services focused on their single responsibility.
4. **Async throughout** — use `asyncio` consistently. Avoid blocking calls in the async event loop; use `run_in_executor` for synchronous I/O.
5. **Error handling** — network failures (API, WebSocket) must be retried with exponential backoff. Log all exceptions with context.
6. **Logging** — use the project's `colorlog`-based logger. Never use `print()` in production code.
7. **Testing** — validate any changes to signal generation, filters, or order logic with paper trading (`DRY_RUN=true`) before live deployment.
