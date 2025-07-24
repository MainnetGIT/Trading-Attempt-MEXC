#!/usr/bin/env python3
"""
MEXC Trading Bot Setup Script
Installs dependencies and helps configure the bot
"""

import os
import sys
import subprocess
import platform

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Install TA-Lib (requires different approach on different systems)
    system = platform.system().lower()
    
    if system == "linux":
        print("Installing TA-Lib for Linux...")
        os.system("sudo apt-get update")
        os.system("sudo apt-get install build-essential wget")
        os.system("wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz")
        os.system("tar -xzf ta-lib-0.4.0-src.tar.gz")
        os.system("cd ta-lib && ./configure --prefix=/usr && make && sudo make install")
        
    elif system == "darwin":  # macOS
        print("Installing TA-Lib for macOS...")
        os.system("brew install ta-lib")
        
    elif system == "windows":
        print("For Windows, please download TA-Lib from:")
        print("https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("Install the appropriate .whl file for your Python version")
        input("Press Enter after installing TA-Lib...")
    
    # Install Python packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("All packages installed successfully!")

def configure_api():
    """Help user configure API credentials"""
    print("\n" + "="*50)
    print("MEXC API CONFIGURATION")
    print("="*50)
    print("Please follow these steps to get your MEXC API credentials:")
    print("1. Log in to your MEXC account")
    print("2. Go to Account → API Management")
    print("3. Create a new API key")
    print("4. Enable 'Spot Trading' permission")
    print("5. Copy your API key and Secret key")
    print("\nIMPORTANT SECURITY NOTES:")
    print("- Never share your API credentials")
    print("- Use IP whitelist restriction")
    print("- Start with small amounts for testing")
    print("- Never enable withdrawal permissions")
    
    print("\n" + "-"*50)
    api_key = input("Enter your MEXC API Key: ").strip()
    secret_key = input("Enter your MEXC Secret Key: ").strip()
    
    if not api_key or not secret_key:
        print("API credentials cannot be empty!")
        return False
    
    # Update config file
    config_content = f'''# MEXC Trading Bot Configuration
# Auto-generated configuration

# API Configuration
MEXC_API_KEY = "{api_key}"
MEXC_SECRET_KEY = "{secret_key}"

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
STRATEGY_WEIGHTS = {{
    'macd_scalping': 0.3,           # 30% - Highest weight for proven strategy
    'momentum': 0.25,               # 25%
    'range_trading': 0.2,           # 20%
    'rsi_divergence': 0.15,         # 15%
    'volume_breakout': 0.1          # 10%
}}

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
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("\nConfiguration saved to config.py")
    return True

def run_tests():
    """Run basic tests to verify setup"""
    print("\n" + "="*50)
    print("RUNNING SETUP TESTS")
    print("="*50)
    
    try:
        # Test imports
        print("Testing imports...")
        import requests
        import websocket
        import pandas as pd
        import numpy as np
        import talib
        print("✓ All imports successful")
        
        # Test API connection
        print("Testing API connection...")
        import config
        from mexc_trading_bot import MEXCTrader, MEXCConfig
        
        bot_config = MEXCConfig()
        bot_config.API_KEY = config.MEXC_API_KEY
        bot_config.SECRET_KEY = config.MEXC_SECRET_KEY
        
        trader = MEXCTrader(bot_config)
        account_info = trader.get_account_info()
        
        if account_info:
            print("✓ API connection successful")
            print(f"Account type: {account_info.get('accountType', 'Unknown')}")
        else:
            print("✗ API connection failed")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    print("\n✓ All tests passed! Bot is ready to run.")
    return True

def main():
    """Main setup function"""
    print("MEXC Trading Bot Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)
    
    # Configure API
    if not configure_api():
        print("Setup incomplete. Please run setup again.")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("Setup tests failed. Please check your configuration.")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("To start the bot, run: python mexc_trading_bot.py")
    print("\nIMPORTANT REMINDERS:")
    print("- Start with small amounts")
    print("- Monitor the bot regularly")
    print("- Check logs for any issues")
    print("- Never risk more than you can afford to lose")
    print("\nHappy trading! 🚀")

if __name__ == "__main__":
    main() 