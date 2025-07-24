# MEXC Trading Bot Configuration
# Replace with your actual MEXC API credentials

# API Configuration
MEXC_API_KEY = "your_mexc_api_key_here"
MEXC_SECRET_KEY = "your_mexc_secret_key_here"

# Trading Parameters
MAX_POSITIONS = 5                    # Maximum concurrent positions
RISK_PER_TRADE = 0.02               # 2% risk per trade
MAX_DAILY_LOSS = 0.10               # 10% maximum daily loss
MIN_PROFIT_TARGET = 0.005           # 0.5% minimum profit target

# Fee Structure (MEXC fees vary by region)
MAKER_FEE = 0.0001                  # 0.01% (or 0% in some regions)
TAKER_FEE = 0.0004                  # 0.04%

# Strategy Configuration
ENABLE_MACD_SCALPING = True         # High-return scalping strategy
ENABLE_MOMENTUM = True              # Momentum trading
ENABLE_RANGE_TRADING = True         # Range/sideways trading
ENABLE_RSI_DIVERGENCE = True        # RSI divergence signals
ENABLE_VOLUME_BREAKOUT = True       # Volume-based breakouts

# Strategy Weights (total should equal 1.0)
STRATEGY_WEIGHTS = {
    'macd_scalping': 0.3,           # 30% - Highest weight for proven strategy
    'momentum': 0.25,               # 25%
    'range_trading': 0.2,           # 20%
    'rsi_divergence': 0.15,         # 15%
    'volume_breakout': 0.1          # 10%
}

# Trading Pairs (high liquidity recommended)
TRADING_PAIRS = [
    'BTCUSDT',      # Bitcoin - highest liquidity
    'ETHUSDT',      # Ethereum
    'BNBUSDT',      # Binance Coin
    'ADAUSDT',      # Cardano
    'SOLUSDT',      # Solana
    'XRPUSDT',      # Ripple
    'DOTUSDT',      # Polkadot
    'AVAXUSDT',     # Avalanche
    'MATICUSDT',    # Polygon
    'LINKUSDT'      # Chainlink
]

# Technical Analysis Settings
MACD_FAST = 3                       # Fast EMA for scalping MACD
MACD_SLOW = 10                      # Slow EMA for scalping MACD
MACD_SIGNAL = 16                    # Signal line smoothing
RSI_PERIOD = 14                     # RSI calculation period
STOCH_K = 14                        # Stochastic %K period
STOCH_D = 3                         # Stochastic %D period

# Risk Management
STOP_LOSS_PCT = 0.005               # 0.5% stop loss
TAKE_PROFIT_RATIO = 3.0             # 3:1 reward:risk ratio
TRAILING_STOP = True                # Enable trailing stops
TRAILING_STOP_PCT = 0.003           # 0.3% trailing stop

# Database Settings
DATABASE_FILE = 'mexc_bot_data.db'
LOG_FILE = 'mexc_bot.log'

# Performance Monitoring
STATS_UPDATE_INTERVAL = 300         # 5 minutes
POSITION_CHECK_INTERVAL = 5         # 5 seconds
WEBSOCKET_RECONNECT_DELAY = 10      # 10 seconds

# Safety Features
ENABLE_PAPER_TRADING = False        # Set to True for testing
REQUIRE_CONFIRMATION = False        # Set to True for manual confirmation
MAX_SLIPPAGE = 0.001               # 0.1% maximum slippage tolerance 