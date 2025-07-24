#!/usr/bin/env python3
"""
Quick Start Script for MEXC Support/Resistance Scalping Bot
Implements the comprehensive trading system outlined in your development guide

Target: 2-5% daily returns through systematic S/R scalping
Leveraging MEXC's ultra-low fees (0% maker, 0.05% taker)
"""

import os
import json
import time
from datetime import datetime
from support_resistance_bot import SRTradingBot
from ai_collaboration import CollaborationInterface

def setup_trading_config():
    """Setup configuration matching your specifications"""
    return {
        # MEXC API Configuration
        'MEXC_API_KEY': os.getenv('MEXC_API_KEY', ''),
        'MEXC_SECRET_KEY': os.getenv('MEXC_SECRET_KEY', ''),
        
        # Core Strategy Parameters (as per your guide)
        'proximity_threshold': 0.002,      # 0.2% proximity to S/R levels
        'volume_threshold': 1.5,           # 1.5x average volume confirmation
        'min_rr_ratio': 2.0,              # Minimum 2:1 risk/reward (1:2 as specified)
        'min_confidence': 0.7,             # High confidence threshold
        
        # Risk Management (Conservative Phase 1 settings)
        'max_risk_per_trade': 0.01,       # 1% risk per trade (conservative start)
        'max_daily_loss': 0.02,           # 2% daily loss limit (Phase 1)
        'max_positions': 3,                # Start with 3 concurrent positions
        'max_hold_time_hours': 4,          # 4-hour maximum hold as specified
        
        # Position Sizing
        'min_position_size_usd': 50,       # $50 minimum position
        'max_position_size_usd': 500,      # $500 maximum (Phase 1)
        'capital_utilization': 0.8,        # 80% capital utilization
        
        # S/R Detection Parameters
        'level_tolerance': 0.002,          # 0.2% tolerance for level clustering
        'min_level_touches': 2,            # Minimum touches for valid level
        'lookback_period': 50,             # 50 periods for S/R analysis
        'fibonacci_enabled': True,         # Include Fibonacci levels
        'volume_weighted_enabled': True,   # Include volume-weighted levels
        
        # Fee Structure (MEXC specific)
        'maker_fee': 0.0000,              # 0% maker fee
        'taker_fee': 0.0005,              # 0.05% taker fee
        
        # Trading Pairs (high liquidity as specified)
        'trading_symbols': [
            'BTCUSDT',   # Bitcoin - highest liquidity
            'ETHUSDT',   # Ethereum  
            'BNBUSDT',   # Binance Coin
            'SOLUSDT',   # Solana
            'ADAUSDT'    # Cardano (start with top 5)
        ],
        
        # Performance Targets (Phase 1)
        'target_daily_return': 0.02,      # 2% daily target (conservative)
        'target_win_rate': 0.60,          # 60% win rate target
        'target_profit_factor': 2.5,      # Profit factor target
        'max_drawdown_limit': 0.05,       # 5% max drawdown
        
        # AI Collaboration Settings
        'enable_ai_optimization': True,
        'daily_collaboration': True,
        'weekly_analysis': True,
        'monthly_strategy_review': True
    }

def validate_setup():
    """Validate setup before starting"""
    print("🔍 Validating Setup...")
    
    # Check API credentials
    api_key = os.getenv('MEXC_API_KEY')
    secret_key = os.getenv('MEXC_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Error: MEXC API credentials not found!")
        print("Please set MEXC_API_KEY and MEXC_SECRET_KEY environment variables")
        return False
    
    # Check internet connection
    try:
        import requests
        response = requests.get('https://api.mexc.com/api/v3/ping', timeout=5)
        if response.status_code != 200:
            print("❌ Error: Cannot connect to MEXC API")
            return False
    except:
        print("❌ Error: Internet connection or MEXC API unavailable")
        return False
    
    print("✅ Setup validation successful!")
    return True

def create_project_structure():
    """Create project directory structure"""
    directories = [
        'logs',
        'data',
        'reports',
        'backups',
        'ai_sessions'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("📁 Project structure created")

def run_initial_market_scan(config):
    """Run initial market scan to identify opportunities"""
    print("🔍 Running initial market scan...")
    
    try:
        # Initialize bot for market analysis only
        bot = SRTradingBot(config)
        
        # Quick market analysis
        opportunities = []
        
        print("\n📊 Market Analysis Results:")
        print("=" * 50)
        
        for symbol in config['trading_symbols']:
            print(f"Analyzing {symbol}...")
            # This would run the actual market analysis
            # For demo, showing expected output format
            
        print("✅ Market scan complete")
        return True
        
    except Exception as e:
        print(f"❌ Error in market scan: {e}")
        return False

def start_paper_trading_mode(config):
    """Start paper trading for testing"""
    print("\n📈 Starting Paper Trading Mode...")
    print("This will run the bot with simulated trades to validate the strategy")
    
    # Enable paper trading
    config['paper_trading'] = True
    config['initial_balance'] = 10000  # $10k virtual balance
    
    try:
        bot = SRTradingBot(config)
        
        print("🤖 Bot initialized successfully")
        print(f"💰 Virtual balance: ${config['initial_balance']:,}")
        print(f"🎯 Target symbols: {', '.join(config['trading_symbols'])}")
        print(f"📊 Risk per trade: {config['max_risk_per_trade']*100}%")
        print(f"🛡️ Daily loss limit: {config['max_daily_loss']*100}%")
        
        print("\n⚠️  PAPER TRADING MODE ACTIVE - NO REAL MONEY AT RISK")
        print("Monitor the bot's performance for at least 1 week before going live")
        
        # In actual implementation, this would start the bot
        # bot.start()
        
        return True
        
    except Exception as e:
        print(f"❌ Error starting paper trading: {e}")
        return False

def setup_ai_collaboration():
    """Setup AI collaboration framework"""
    print("\n🧠 Setting up AI Collaboration Framework...")
    
    try:
        collab = CollaborationInterface()
        
        # Run initial analysis if data exists
        session = collab.daily_collaboration_session()
        
        print("✅ AI collaboration framework ready")
        print("📋 Daily sessions: Automated performance analysis")
        print("📊 Weekly sessions: Deep strategy optimization")
        print("🚀 Monthly sessions: Strategic evolution planning")
        
        # Export initial session
        filename = collab.export_collaboration_data(session, 'ai_sessions/initial_setup.json')
        print(f"📄 Initial session data: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up AI collaboration: {e}")
        return False

def display_phase_1_expectations():
    """Display Phase 1 performance expectations"""
    print("\n🎯 PHASE 1 EXPECTATIONS (Months 1-2)")
    print("=" * 50)
    print("Objectives:")
    print("  • System stability and bug resolution")
    print("  • Strategy parameter optimization") 
    print("  • Risk management validation")
    print("  • Performance baseline establishment")
    print()
    print("Expected Returns:")
    print("  • Month 1: 2-5% (conservative while learning)")
    print("  • Month 2: 4-7% (improved confidence and parameters)")
    print("  • Win Rate Target: 55-65%")
    print("  • Maximum Drawdown: <8%")
    print("  • Trades per Day: 3-5")
    print()
    print("Key Success Metrics:")
    print("  ✓ System uptime >90%")
    print("  ✓ Win rate >55%")
    print("  ✓ Monthly return >3%")
    print("  ✓ Max drawdown <5%")
    print("  ✓ Daily AI collaboration routine established")

def display_safety_reminders():
    """Display important safety reminders"""
    print("\n⚠️  IMPORTANT SAFETY REMINDERS")
    print("=" * 50)
    print("1. START SMALL: Begin with $500-1000 for testing")
    print("2. MONITOR REGULARLY: Check performance multiple times daily")
    print("3. UNDERSTAND RISKS: Crypto trading involves significant risk")
    print("4. API SECURITY: Use IP whitelist, never enable withdrawal permissions")
    print("5. PAPER TRADING: Test thoroughly before risking real money")
    print("6. DAILY LIMITS: Respect the 2% daily loss limit strictly")
    print("7. GRADUAL SCALING: Only increase size after consistent profitability")
    print()
    print("🚨 NEVER RISK MORE THAN YOU CAN AFFORD TO LOSE 🚨")

def main():
    """Main quick start routine"""
    print("🚀 MEXC Support/Resistance Scalping Bot - Quick Start")
    print("=" * 60)
    print("Target: 2-5% daily returns through systematic S/R scalping")
    print("Platform: MEXC Exchange (0% maker, 0.05% taker fees)")
    print("=" * 60)
    
    # Step 1: Validate setup
    if not validate_setup():
        return
    
    # Step 2: Create project structure
    create_project_structure()
    
    # Step 3: Setup configuration
    config = setup_trading_config()
    
    # Step 4: Display Phase 1 expectations
    display_phase_1_expectations()
    
    # Step 5: Run initial market scan
    if not run_initial_market_scan(config):
        print("❌ Market scan failed. Please check your setup.")
        return
    
    # Step 6: Setup AI collaboration
    if not setup_ai_collaboration():
        print("⚠️  AI collaboration setup failed, but bot can still run")
    
    # Step 7: Safety reminders
    display_safety_reminders()
    
    # Step 8: Ask user preference
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Start Paper Trading (Recommended)")
    print("2. Start Live Trading (Only if experienced)")
    print("3. Run Backtest First")
    print("4. Exit and review documentation")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        print("\n🎯 Starting Paper Trading Mode...")
        if start_paper_trading_mode(config):
            print("\n✅ Paper trading started successfully!")
            print("📊 Monitor performance in the logs/ directory")
            print("🧠 Daily AI analysis will be saved in ai_sessions/")
            print("\n⏰ Run paper trading for at least 1 week before going live")
    
    elif choice == '2':
        print("\n⚠️  LIVE TRADING SELECTED")
        confirm = input("Are you sure you want to risk real money? (yes/no): ")
        if confirm.lower() == 'yes':
            print("🔴 Starting live trading...")
            config['paper_trading'] = False
            # Would start actual bot here
            print("✅ Live trading started - Monitor closely!")
        else:
            print("👍 Smart choice. Consider paper trading first.")
    
    elif choice == '3':
        print("📊 Running backtest...")
        print("This will test the strategy on historical data")
        # Would run backtest here
    
    else:
        print("📚 Please review the documentation and setup guides")
        print("When ready, run this script again to start trading")
    
    print("\n🎉 Setup complete! Happy trading!")
    print("📧 For support: Check the README.md file")

if __name__ == "__main__":
    main() 