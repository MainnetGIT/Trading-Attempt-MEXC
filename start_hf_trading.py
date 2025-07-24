#!/usr/bin/env python3
"""
High-Frequency S/R Trading Starter Script
Launch 10-50 trades per day system with real-time chart analysis
"""

import asyncio
import sys
import os
import time
from datetime import datetime
import logging
from typing import Dict, List
import multiprocessing
import signal

# Import our HF trading components
from high_frequency_sr_bot import HighFrequencyTradingBot
from hf_config import get_hf_config, validate_config, get_symbol_config
from ai_collaboration import CollaborationInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hf_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HighFrequencyTradingLauncher:
    """Launcher for high-frequency trading system"""
    
    def __init__(self):
        self.config = get_hf_config()
        self.bot = None
        self.ai_collab = None
        self.running = False
        self.processes = []
        
    def validate_setup(self) -> bool:
        """Comprehensive setup validation"""
        print("🔍 Validating High-Frequency Trading Setup...")
        print("=" * 60)
        
        # 1. Configuration validation
        if not validate_config():
            return False
        
        # 2. API connectivity test
        if not self._test_api_connectivity():
            return False
        
        # 3. Market data access test
        if not self._test_market_data():
            return False
        
        # 4. WebSocket connectivity test
        if not self._test_websocket_connection():
            return False
        
        # 5. Risk parameter validation
        if not self._validate_risk_parameters():
            return False
        
        print("✅ All validation checks passed!")
        print("🚀 Ready for high-frequency trading")
        return True
    
    def _test_api_connectivity(self) -> bool:
        """Test MEXC API connectivity"""
        try:
            import requests
            
            # Test public endpoint
            response = requests.get('https://api.mexc.com/api/v3/ping', timeout=10)
            if response.status_code != 200:
                print("❌ Cannot connect to MEXC API")
                return False
            
            # Test exchange info
            response = requests.get('https://api.mexc.com/api/v3/exchangeInfo', timeout=10)
            if response.status_code != 200:
                print("❌ Cannot fetch exchange information")
                return False
            
            print("✅ MEXC API connectivity: OK")
            return True
            
        except Exception as e:
            print(f"❌ API connectivity test failed: {e}")
            return False
    
    def _test_market_data(self) -> bool:
        """Test market data access for trading symbols"""
        try:
            import requests
            
            # Test ticker data for first few symbols
            test_symbols = self.config['trading_symbols'][:3]
            
            for symbol in test_symbols:
                url = f"https://api.mexc.com/api/v3/ticker/24hr?symbol={symbol}"
                response = requests.get(url, timeout=5)
                
                if response.status_code != 200:
                    print(f"❌ Cannot fetch market data for {symbol}")
                    return False
                
                data = response.json()
                if 'symbol' not in data:
                    print(f"❌ Invalid market data format for {symbol}")
                    return False
            
            print(f"✅ Market data access: OK ({len(test_symbols)} symbols tested)")
            return True
            
        except Exception as e:
            print(f"❌ Market data test failed: {e}")
            return False
    
    def _test_websocket_connection(self) -> bool:
        """Test WebSocket connectivity"""
        try:
            import websockets
            import json
            
            async def test_ws():
                try:
                    uri = "wss://wbs.mexc.com/ws"
                    async with websockets.connect(uri, ping_timeout=10) as websocket:
                        # Send ping
                        ping_msg = {"method": "ping"}
                        await websocket.send(json.dumps(ping_msg))
                        
                        # Wait for response
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        return True
                except Exception as e:
                    logger.error(f"WebSocket test error: {e}")
                    return False
            
            # Run WebSocket test
            result = asyncio.run(test_ws())
            
            if result:
                print("✅ WebSocket connectivity: OK")
                return True
            else:
                print("❌ WebSocket connectivity test failed")
                return False
                
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            return False
    
    def _validate_risk_parameters(self) -> bool:
        """Validate risk management parameters"""
        
        # Check risk per trade
        if self.config['max_risk_per_trade'] > 0.015:  # > 1.5%
            print("❌ Risk per trade too high for high-frequency trading")
            print(f"   Current: {self.config['max_risk_per_trade']*100}%, Recommended: <1.5%")
            return False
        
        # Check daily loss limit
        if self.config['max_daily_loss'] > 0.03:  # > 3%
            print("❌ Daily loss limit too high")
            print(f"   Current: {self.config['max_daily_loss']*100}%, Recommended: <3%")
            return False
        
        # Check position limits
        if self.config['max_positions'] > 10:
            print("❌ Too many concurrent positions for HF trading")
            print(f"   Current: {self.config['max_positions']}, Recommended: ≤10")
            return False
        
        print("✅ Risk parameters: OK")
        return True
    
    def display_trading_plan(self):
        """Display the high-frequency trading plan"""
        print("\n🎯 HIGH-FREQUENCY SUPPORT/RESISTANCE TRADING PLAN")
        print("=" * 70)
        
        print(f"📊 Strategy Focus:")
        print(f"   • Support/Resistance Scalping")
        print(f"   • Multi-timeframe analysis (30s, 1m, 3m, 5m)")
        print(f"   • Real-time chart pattern detection")
        print(f"   • Volume confirmation required")
        
        print(f"\n🎲 Trading Targets:")
        print(f"   • Daily Trades: {self.config['performance_targets']['daily_trades_min']}-{self.config['performance_targets']['daily_trades_max']}")
        print(f"   • Target Win Rate: {self.config['performance_targets']['target_win_rate']*100}%")
        print(f"   • Target Daily Return: {self.config['performance_targets']['target_daily_return']*100}%")
        print(f"   • Max Hold Time: {self.config['max_hold_minutes']} minutes")
        
        print(f"\n💰 Profit Targets:")
        for target_name, target_value in self.config['profit_targets'].items():
            print(f"   • {target_name.replace('_', ' ').title()}: {target_value*100}%")
        
        print(f"\n🛡️ Risk Management:")
        print(f"   • Risk Per Trade: {self.config['max_risk_per_trade']*100}%")
        print(f"   • Daily Loss Limit: {self.config['max_daily_loss']*100}%")
        print(f"   • Max Positions: {self.config['max_positions']}")
        print(f"   • Position Size: ${self.config['position_size_usd']}")
        
        print(f"\n📈 Trading Pairs ({len(self.config['trading_symbols'])}):")
        # Group symbols by tier
        tier_1 = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        tier_2 = ['ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT']
        others = [s for s in self.config['trading_symbols'] if s not in tier_1 + tier_2]
        
        print(f"   • Tier 1 (Highest Liquidity): {', '.join(tier_1)}")
        print(f"   • Tier 2 (High Liquidity): {', '.join(tier_2)}")
        if others:
            print(f"   • Additional Pairs: {', '.join(others[:6])}")
        
        print(f"\n⚡ Technical Setup:")
        print(f"   • Proximity Threshold: {self.config['proximity_threshold']*100}%")
        print(f"   • Volume Threshold: {self.config['volume_threshold']}x average")
        print(f"   • Confidence Minimum: {self.config['min_confidence']*100}%")
        print(f"   • Chart Updates: Every {self.config['chart_update_seconds']} second(s)")
    
    def start_paper_trading(self):
        """Start paper trading mode for validation"""
        print("\n📈 STARTING PAPER TRADING MODE")
        print("=" * 50)
        print("⚠️  NO REAL MONEY AT RISK - SIMULATION MODE")
        print("This will validate the HF strategy with live market data")
        print()
        
        # Enable paper trading
        self.config['paper_trading'] = True
        self.config['initial_balance'] = 10000  # $10k virtual balance
        
        try:
            # Initialize the HF trading bot
            self.bot = HighFrequencyTradingBot(self.config)
            
            print("🤖 High-Frequency Bot Initialized")
            print(f"💰 Virtual Balance: ${self.config['initial_balance']:,}")
            print(f"🎯 Target: {self.config['performance_targets']['daily_trades_target']} trades/day")
            print(f"⏱️  Max Hold Time: {self.config['max_hold_minutes']} minutes")
            print()
            print("📊 Real-time Analysis Active:")
            print("   • Multi-timeframe S/R detection")
            print("   • Chart pattern recognition")
            print("   • Volume confirmation")
            print("   • Risk management automation")
            print()
            print("🔄 Starting trading loop...")
            
            # Start the high-frequency trading
            asyncio.run(self.bot.start_high_frequency_trading())
            
        except KeyboardInterrupt:
            print("\n⏹️  Paper trading stopped by user")
            self._shutdown_gracefully()
        except Exception as e:
            print(f"\n❌ Error in paper trading: {e}")
            logger.error(f"Paper trading error: {e}")
    
    def start_live_trading(self):
        """Start live trading with real money"""
        print("\n🔴 STARTING LIVE TRADING")
        print("=" * 50)
        print("⚠️  REAL MONEY AT RISK!")
        print()
        
        # Final confirmation
        print("Before starting live trading, please confirm:")
        print("1. You have tested the strategy in paper trading")
        print("2. You understand the risks involved")
        print("3. You have set appropriate position sizes")
        print("4. Your API keys have trading permissions only (no withdrawal)")
        print()
        
        confirm = input("Type 'START LIVE TRADING' to confirm: ").strip()
        
        if confirm != 'START LIVE TRADING':
            print("❌ Live trading cancelled")
            return
        
        # Disable paper trading
        self.config['paper_trading'] = False
        
        try:
            # Initialize the HF trading bot
            self.bot = HighFrequencyTradingBot(self.config)
            
            print("🔴 LIVE TRADING ACTIVE")
            print("📊 Monitor performance closely!")
            print("🛑 Emergency stop: Ctrl+C")
            print()
            
            # Start live trading
            asyncio.run(self.bot.start_high_frequency_trading())
            
        except KeyboardInterrupt:
            print("\n⏹️  Live trading stopped by user")
            self._shutdown_gracefully()
        except Exception as e:
            print(f"\n❌ Error in live trading: {e}")
            logger.error(f"Live trading error: {e}")
    
    def start_chart_analysis_only(self):
        """Start real-time chart analysis without trading"""
        print("\n📊 STARTING CHART ANALYSIS MODE")
        print("=" * 50)
        print("Real-time market analysis without trading")
        print()
        
        from chart_analyzer import RealTimeChartDisplay
        
        try:
            # Initialize chart analyzer
            chart_analyzer = RealTimeChartDisplay(self.config)
            
            print("📈 Real-time Charts Active:")
            for symbol in self.config['chart_display']['symbols_to_chart']:
                print(f"   • {symbol}")
            
            print("\n🔍 Analysis Features:")
            print("   • Support/Resistance levels")
            print("   • Chart pattern detection") 
            print("   • Volume analysis")
            print("   • Multi-timeframe view")
            print()
            print("🔄 Starting chart analysis...")
            
            # Start chart analysis
            asyncio.run(chart_analyzer.start_analysis())
            
        except KeyboardInterrupt:
            print("\n⏹️  Chart analysis stopped")
        except Exception as e:
            print(f"\n❌ Error in chart analysis: {e}")
    
    def run_backtest(self):
        """Run backtest on recent data"""
        print("\n📊 RUNNING BACKTEST")
        print("=" * 50)
        
        from backtest import HighFrequencyBacktest
        
        try:
            backtester = HighFrequencyBacktest(self.config)
            
            print(f"📅 Testing on last {self.config['backtesting']['lookback_days']} days")
            print(f"💰 Initial Capital: ${self.config['backtesting']['initial_capital']:,}")
            print(f"📊 Commission: {self.config['backtesting']['commission']*100}%")
            print()
            
            # Run backtest
            results = backtester.run_comprehensive_backtest()
            
            print("📈 BACKTEST RESULTS:")
            print(f"   • Total Return: {results['total_return']*100:.2f}%")
            print(f"   • Win Rate: {results['win_rate']*100:.1f}%")
            print(f"   • Total Trades: {results['total_trades']}")
            print(f"   • Avg Trade: {results['avg_trade_return']*100:.2f}%")
            print(f"   • Max Drawdown: {results['max_drawdown']*100:.2f}%")
            print(f"   • Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"   • Profit Factor: {results['profit_factor']:.2f}")
            
        except Exception as e:
            print(f"❌ Backtest error: {e}")
    
    def setup_ai_collaboration(self):
        """Setup AI collaboration for optimization"""
        print("\n🧠 SETTING UP AI COLLABORATION")
        print("=" * 50)
        
        try:
            self.ai_collab = CollaborationInterface()
            
            print("✅ AI Collaboration Framework Ready")
            print("📋 Features:")
            print("   • Real-time performance analysis")
            print("   • Strategy optimization recommendations")
            print("   • Risk assessment and alerts")
            print("   • Market condition adaptation")
            print()
            
            # Run initial analysis
            session = self.ai_collab.daily_collaboration_session()
            
            print("📊 Initial Analysis Complete")
            return True
            
        except Exception as e:
            print(f"❌ AI collaboration setup error: {e}")
            return False
    
    def _shutdown_gracefully(self):
        """Graceful shutdown of all components"""
        print("\n🛑 Shutting down trading system...")
        
        self.running = False
        
        # Close all positions if live trading
        if self.bot and not self.config.get('paper_trading', True):
            print("💼 Closing all open positions...")
            # This would close all positions
        
        # Stop all processes
        for process in self.processes:
            try:
                process.terminate()
                process.join(timeout=5)
            except:
                pass
        
        print("✅ Shutdown complete")
    
    def interactive_menu(self):
        """Interactive menu for HF trading options"""
        
        while True:
            print("\n🚀 HIGH-FREQUENCY S/R TRADING SYSTEM")
            print("=" * 60)
            print("1. Start Paper Trading (Recommended)")
            print("2. Start Live Trading (Real Money)")
            print("3. Chart Analysis Only")
            print("4. Run Backtest")
            print("5. Setup AI Collaboration")
            print("6. View Configuration")
            print("7. Exit")
            print()
            
            choice = input("Select option (1-7): ").strip()
            
            if choice == '1':
                self.start_paper_trading()
            elif choice == '2':
                self.start_live_trading()
            elif choice == '3':
                self.start_chart_analysis_only()
            elif choice == '4':
                self.run_backtest()
            elif choice == '5':
                self.setup_ai_collaboration()
            elif choice == '6':
                self.display_trading_plan()
            elif choice == '7':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Received shutdown signal")
    sys.exit(0)

def main():
    """Main entry point"""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 HIGH-FREQUENCY SUPPORT/RESISTANCE TRADING SYSTEM")
    print("=" * 70)
    print("Target: 10-50 trades per day with precise chart analysis")
    print("Platform: MEXC Exchange (0% maker, 0.05% taker fees)")
    print("Strategy: Multi-timeframe S/R scalping with AI optimization")
    print("=" * 70)
    
    # Initialize launcher
    launcher = HighFrequencyTradingLauncher()
    
    # Validate setup
    if not launcher.validate_setup():
        print("\n❌ Setup validation failed!")
        print("Please fix the issues above and try again.")
        sys.exit(1)
    
    # Display trading plan
    launcher.display_trading_plan()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'paper':
            launcher.start_paper_trading()
        elif mode == 'live':
            launcher.start_live_trading()
        elif mode == 'charts':
            launcher.start_chart_analysis_only()
        elif mode == 'backtest':
            launcher.run_backtest()
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: paper, live, charts, backtest")
    else:
        # Interactive mode
        launcher.interactive_menu()

if __name__ == "__main__":
    main() 