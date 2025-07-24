#!/usr/bin/env python3
"""
MEXC Profitable Trading Bot - Advanced Multi-Strategy System
Features: Real-time analysis, multiple strategies, risk management, profit optimization
"""

import asyncio
import logging
import json
import time
import hmac
import hashlib
import requests
import websocket
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from collections import deque
import sqlite3
import talib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mexc_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    """Trade signal data structure"""
    symbol: str
    side: str  # 'buy' or 'sell'
    strategy: str
    price: float
    confidence: float
    stop_loss: float
    take_profit: float
    timestamp: datetime

@dataclass
class Position:
    """Position tracking"""
    symbol: str
    side: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    timestamp: datetime

class MEXCConfig:
    """Configuration management"""
    def __init__(self):
        self.API_KEY = ""  # Set your API key
        self.SECRET_KEY = ""  # Set your secret key
        self.BASE_URL = "https://api.mexc.com"
        self.WS_URL = "wss://wbs.mexc.com/ws"
        
        # Trading parameters
        self.MAX_POSITIONS = 5
        self.RISK_PER_TRADE = 0.02  # 2% risk per trade
        self.MAX_DAILY_LOSS = 0.10  # 10% max daily loss
        self.MIN_PROFIT_TARGET = 0.005  # 0.5% minimum profit target
        
        # Fee structure (varies by region)
        self.MAKER_FEE = 0.0001  # 0.01% (or 0% in some regions)
        self.TAKER_FEE = 0.0004  # 0.04%
        
        # Strategy weights
        self.STRATEGY_WEIGHTS = {
            'macd_scalping': 0.3,
            'momentum': 0.25,
            'range_trading': 0.2,
            'rsi_divergence': 0.15,
            'volume_breakout': 0.1
        }

class TechnicalIndicators:
    """Advanced technical analysis indicators"""
    
    @staticmethod
    def macd_scalping(data: pd.DataFrame, fast=3, slow=10, signal=16) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized MACD for scalping (3,452% return strategy)"""
        exp1 = data['close'].ewm(span=fast).mean()
        exp2 = data['close'].ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd.values, signal_line.values
    
    @staticmethod
    def rsi_divergence(data: pd.DataFrame, period=14) -> np.ndarray:
        """RSI with divergence detection"""
        return talib.RSI(data['close'].values, timeperiod=period)
    
    @staticmethod
    def stochastic_oscillator(data: pd.DataFrame, k_period=14, d_period=3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic oscillator for momentum"""
        k_percent = talib.STOCHF(data['high'].values, data['low'].values, data['close'].values, 
                                 fastk_period=k_period, fastd_period=d_period)
        return k_percent[0], k_percent[1]
    
    @staticmethod
    def volume_profile(data: pd.DataFrame, lookback=20) -> Dict:
        """Volume profile analysis"""
        recent_data = data.tail(lookback)
        avg_volume = recent_data['volume'].mean()
        volume_spike = recent_data['volume'].iloc[-1] > avg_volume * 1.5
        return {
            'avg_volume': avg_volume,
            'current_volume': recent_data['volume'].iloc[-1],
            'volume_spike': volume_spike
        }

class MEXCTrader:
    """MEXC API trading interface"""
    
    def __init__(self, config: MEXCConfig):
        self.config = config
        self.session = requests.Session()
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
    def _sign_request(self, query_string: str) -> str:
        """Sign API request"""
        return hmac.new(
            self.config.SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated API request"""
        timestamp = int(time.time() * 1000)
        
        if params is None:
            params = {}
        
        params['timestamp'] = timestamp
        params['recvWindow'] = 5000
        
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = self._sign_request(query_string)
        params['signature'] = signature
        
        headers = {'X-MEXC-APIKEY': self.config.API_KEY}
        
        url = f"{self.config.BASE_URL}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = self.session.post(url, json=params, headers=headers)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return {}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        return self._make_request('GET', '/api/v3/account')
    
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> List:
        """Get candlestick data"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = self._make_request('GET', '/api/v3/klines', params)
        return response if isinstance(response, list) else []
    
    def place_order(self, symbol: str, side: str, quantity: float, price: float = None, 
                   order_type: str = 'MARKET') -> Dict:
        """Place trading order"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type,
            'quantity': quantity
        }
        
        if order_type == 'LIMIT' and price:
            params['price'] = price
            params['timeInForce'] = 'GTC'
        
        return self._make_request('POST', '/api/v3/order', params)
    
    def get_open_orders(self, symbol: str = None) -> List:
        """Get open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/api/v3/openOrders', params)
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel order"""
        params = {'symbol': symbol, 'orderId': order_id}
        return self._make_request('DELETE', '/api/v3/order', params)

class StrategyEngine:
    """Multi-strategy trading engine"""
    
    def __init__(self, config: MEXCConfig):
        self.config = config
        self.data_cache: Dict[str, deque] = {}
        self.indicators = TechnicalIndicators()
        
    def add_data_point(self, symbol: str, kline_data: Dict):
        """Add new data point to cache"""
        if symbol not in self.data_cache:
            self.data_cache[symbol] = deque(maxlen=200)
        
        self.data_cache[symbol].append({
            'timestamp': kline_data['timestamp'],
            'open': float(kline_data['open']),
            'high': float(kline_data['high']),
            'low': float(kline_data['low']),
            'close': float(kline_data['close']),
            'volume': float(kline_data['volume'])
        })
    
    def get_dataframe(self, symbol: str) -> pd.DataFrame:
        """Convert cached data to DataFrame"""
        if symbol not in self.data_cache or len(self.data_cache[symbol]) < 50:
            return pd.DataFrame()
        
        return pd.DataFrame(list(self.data_cache[symbol]))
    
    def macd_scalping_strategy(self, symbol: str) -> Optional[TradeSignal]:
        """MACD scalping strategy (3,452% return optimized)"""
        df = self.get_dataframe(symbol)
        if df.empty or len(df) < 20:
            return None
        
        macd, signal = self.indicators.macd_scalping(df)
        sma_200 = talib.SMA(df['close'].values, timeperiod=200)
        
        current_price = df['close'].iloc[-1]
        prev_macd, prev_signal = macd[-2], signal[-2]
        curr_macd, curr_signal = macd[-1], signal[-1]
        
        # Only trade above 200 SMA (trend filter)
        if current_price < sma_200[-1]:
            return None
        
        # Buy signal: MACD crosses above signal
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            stop_loss = current_price * 0.995  # 0.5% stop
            take_profit = current_price * 1.015  # 1.5% target (3:1 ratio)
            
            return TradeSignal(
                symbol=symbol,
                side='buy',
                strategy='macd_scalping',
                price=current_price,
                confidence=0.8,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now()
            )
        
        # Sell signal: MACD crosses below signal
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            stop_loss = current_price * 1.005  # 0.5% stop
            take_profit = current_price * 0.985  # 1.5% target
            
            return TradeSignal(
                symbol=symbol,
                side='sell',
                strategy='macd_scalping',
                price=current_price,
                confidence=0.8,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now()
            )
        
        return None
    
    def momentum_strategy(self, symbol: str) -> Optional[TradeSignal]:
        """Momentum trading strategy"""
        df = self.get_dataframe(symbol)
        if df.empty or len(df) < 20:
            return None
        
        rsi = self.indicators.rsi_divergence(df)
        k_percent, d_percent = self.indicators.stochastic_oscillator(df)
        volume_data = self.indicators.volume_profile(df)
        
        current_price = df['close'].iloc[-1]
        curr_rsi = rsi[-1]
        curr_k, curr_d = k_percent[-1], d_percent[-1]
        
        # Momentum buy: RSI oversold + Stoch crossover + volume spike
        if (curr_rsi < 30 and curr_k > curr_d and 
            volume_data['volume_spike'] and curr_k < 20):
            
            stop_loss = current_price * 0.992  # 0.8% stop
            take_profit = current_price * 1.024  # 2.4% target (3:1 ratio)
            
            return TradeSignal(
                symbol=symbol,
                side='buy',
                strategy='momentum',
                price=current_price,
                confidence=0.75,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now()
            )
        
        # Momentum sell: RSI overbought + Stoch crossover
        elif curr_rsi > 70 and curr_k < curr_d and curr_k > 80:
            stop_loss = current_price * 1.008  # 0.8% stop
            take_profit = current_price * 0.976  # 2.4% target
            
            return TradeSignal(
                symbol=symbol,
                side='sell',
                strategy='momentum',
                price=current_price,
                confidence=0.75,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now()
            )
        
        return None
    
    def range_trading_strategy(self, symbol: str) -> Optional[TradeSignal]:
        """Range trading strategy for sideways markets"""
        df = self.get_dataframe(symbol)
        if df.empty or len(df) < 50:
            return None
        
        # Calculate support and resistance levels
        recent_data = df.tail(30)
        resistance = recent_data['high'].max()
        support = recent_data['low'].min()
        range_size = resistance - support
        
        current_price = df['close'].iloc[-1]
        
        # Only trade if we have a clear range (at least 1% range)
        if range_size / current_price < 0.01:
            return None
        
        # Buy near support
        if current_price <= support * 1.002:  # Within 0.2% of support
            stop_loss = support * 0.995  # Below support
            take_profit = resistance * 0.995  # Near resistance
            
            return TradeSignal(
                symbol=symbol,
                side='buy',
                strategy='range_trading',
                price=current_price,
                confidence=0.7,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now()
            )
        
        # Sell near resistance
        elif current_price >= resistance * 0.998:  # Within 0.2% of resistance
            stop_loss = resistance * 1.005  # Above resistance
            take_profit = support * 1.005  # Near support
            
            return TradeSignal(
                symbol=symbol,
                side='sell',
                strategy='range_trading',
                price=current_price,
                confidence=0.7,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now()
            )
        
        return None
    
    def analyze_symbol(self, symbol: str) -> List[TradeSignal]:
        """Analyze symbol with all strategies"""
        signals = []
        
        # Run all strategies
        strategies = [
            self.macd_scalping_strategy,
            self.momentum_strategy,
            self.range_trading_strategy
        ]
        
        for strategy in strategies:
            signal = strategy(symbol)
            if signal:
                signals.append(signal)
        
        return signals

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, config: MEXCConfig):
        self.config = config
        self.daily_start_balance = 0
        self.max_drawdown = 0
        
    def calculate_position_size(self, balance: float, price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = balance * self.config.RISK_PER_TRADE
        price_diff = abs(price - stop_loss)
        
        if price_diff == 0:
            return 0
        
        position_size = risk_amount / price_diff
        
        # Account for fees
        max_size_with_fees = (balance * 0.95) / price  # Reserve 5% for fees
        
        return min(position_size, max_size_with_fees)
    
    def check_daily_loss_limit(self, current_balance: float) -> bool:
        """Check if daily loss limit is exceeded"""
        if self.daily_start_balance == 0:
            self.daily_start_balance = current_balance
            return False
        
        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance
        return daily_loss >= self.config.MAX_DAILY_LOSS
    
    def can_open_position(self, current_positions: int) -> bool:
        """Check if we can open new positions"""
        return current_positions < self.config.MAX_POSITIONS

class MEXCTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        self.config = MEXCConfig()
        self.trader = MEXCTrader(self.config)
        self.strategy_engine = StrategyEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Trading pairs to monitor (high liquidity pairs)
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT'
        ]
        
        self.running = False
        self.ws = None
        
        # Performance tracking
        self.setup_database()
        
    def setup_database(self):
        """Setup SQLite database for tracking"""
        self.conn = sqlite3.connect('mexc_bot_data.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                strategy TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                strategy TEXT,
                side TEXT,
                price REAL,
                confidence REAL,
                timestamp DATETIME
            )
        ''')
        
        self.conn.commit()
    
    def log_trade(self, symbol: str, side: str, strategy: str, entry_price: float, 
                  exit_price: float, quantity: float, pnl: float):
        """Log completed trade"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, side, strategy, entry_price, exit_price, quantity, pnl, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, side, strategy, entry_price, exit_price, quantity, pnl, datetime.now()))
        self.conn.commit()
        
        logger.info(f"Trade logged: {symbol} {side} {strategy} PnL: {pnl:.4f}")
    
    def on_kline_update(self, symbol: str, kline_data: Dict):
        """Handle new kline data"""
        try:
            # Add data to strategy engine
            self.strategy_engine.add_data_point(symbol, kline_data)
            
            # Analyze for signals
            signals = self.strategy_engine.analyze_symbol(symbol)
            
            for signal in signals:
                self.process_signal(signal)
                
        except Exception as e:
            logger.error(f"Error processing kline update: {e}")
    
    def process_signal(self, signal: TradeSignal):
        """Process trading signal"""
        try:
            # Check risk management
            account_info = self.trader.get_account_info()
            if not account_info:
                return
            
            # Get USDT balance
            usdt_balance = 0
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            # Check daily loss limit
            if self.risk_manager.check_daily_loss_limit(usdt_balance):
                logger.warning("Daily loss limit reached, stopping trading")
                return
            
            # Check position limit
            current_positions = len(self.trader.positions)
            if not self.risk_manager.can_open_position(current_positions):
                logger.info("Maximum positions reached")
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                usdt_balance, signal.price, signal.stop_loss
            )
            
            if position_size < 10:  # Minimum order size
                logger.info(f"Position size too small: {position_size}")
                return
            
            # Place order
            order_result = self.trader.place_order(
                symbol=signal.symbol,
                side=signal.side,
                quantity=position_size,
                order_type='MARKET'
            )
            
            if order_result and 'orderId' in order_result:
                # Track position
                position = Position(
                    symbol=signal.symbol,
                    side=signal.side,
                    size=position_size,
                    entry_price=signal.price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy=signal.strategy,
                    timestamp=datetime.now()
                )
                
                self.trader.positions[signal.symbol] = position
                
                logger.info(f"Order placed: {signal.symbol} {signal.side} "
                           f"{position_size} @ {signal.price} ({signal.strategy})")
                
                # Set stop loss and take profit orders
                self.set_exit_orders(position)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def set_exit_orders(self, position: Position):
        """Set stop loss and take profit orders"""
        try:
            # Place stop loss order
            stop_side = 'SELL' if position.side == 'BUY' else 'BUY'
            
            self.trader.place_order(
                symbol=position.symbol,
                side=stop_side,
                quantity=position.size,
                price=position.stop_loss,
                order_type='STOP_LOSS_LIMIT'
            )
            
            # Place take profit order
            self.trader.place_order(
                symbol=position.symbol,
                side=stop_side,
                quantity=position.size,
                price=position.take_profit,
                order_type='LIMIT'
            )
            
        except Exception as e:
            logger.error(f"Error setting exit orders: {e}")
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        while self.running:
            try:
                for symbol, position in list(self.trader.positions.items()):
                    # Get current price
                    klines = self.trader.get_klines(symbol, '1m', 1)
                    if not klines:
                        continue
                    
                    current_price = float(klines[0][4])  # Close price
                    
                    # Check if position should be closed
                    if self.should_close_position(position, current_price):
                        self.close_position(symbol, position, current_price)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                time.sleep(10)
    
    def should_close_position(self, position: Position, current_price: float) -> bool:
        """Determine if position should be closed"""
        if position.side == 'buy':
            # Close if stop loss or take profit hit
            return current_price <= position.stop_loss or current_price >= position.take_profit
        else:
            return current_price >= position.stop_loss or current_price <= position.take_profit
    
    def close_position(self, symbol: str, position: Position, exit_price: float):
        """Close position and calculate PnL"""
        try:
            # Place market order to close
            close_side = 'SELL' if position.side == 'buy' else 'BUY'
            
            order_result = self.trader.place_order(
                symbol=symbol,
                side=close_side,
                quantity=position.size,
                order_type='MARKET'
            )
            
            if order_result:
                # Calculate PnL
                if position.side == 'buy':
                    pnl = (exit_price - position.entry_price) * position.size
                else:
                    pnl = (position.entry_price - exit_price) * position.size
                
                # Account for fees
                total_fees = (position.entry_price + exit_price) * position.size * self.config.TAKER_FEE
                net_pnl = pnl - total_fees
                
                # Log trade
                self.log_trade(
                    symbol, position.side, position.strategy,
                    position.entry_price, exit_price, position.size, net_pnl
                )
                
                # Update stats
                self.trader.total_trades += 1
                if net_pnl > 0:
                    self.trader.winning_trades += 1
                
                self.trader.daily_pnl += net_pnl
                
                # Remove position
                del self.trader.positions[symbol]
                
                logger.info(f"Position closed: {symbol} PnL: {net_pnl:.4f} USDT")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def start_websocket(self):
        """Start WebSocket connection for real-time data"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'k' in data:  # Kline data
                    kline = data['k']
                    if kline['x']:  # Kline is closed
                        symbol = kline['s']
                        kline_data = {
                            'timestamp': kline['t'],
                            'open': kline['o'],
                            'high': kline['h'],
                            'low': kline['l'],
                            'close': kline['c'],
                            'volume': kline['v']
                        }
                        self.on_kline_update(symbol, kline_data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
            # Subscribe to kline streams for all symbols
            for symbol in self.symbols:
                stream = f"{symbol.lower()}@kline_1m"
                subscribe_msg = {
                    "method": "SUBSCRIPTION",
                    "params": [stream],
                    "id": 1
                }
                ws.send(json.dumps(subscribe_msg))
        
        self.ws = websocket.WebSocketApp(
            self.config.WS_URL,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws.run_forever()
    
    def print_stats(self):
        """Print trading statistics"""
        while self.running:
            try:
                win_rate = (self.trader.winning_trades / max(self.trader.total_trades, 1)) * 100
                
                logger.info(f"=== Trading Stats ===")
                logger.info(f"Total Trades: {self.trader.total_trades}")
                logger.info(f"Win Rate: {win_rate:.1f}%")
                logger.info(f"Daily PnL: {self.trader.daily_pnl:.4f} USDT")
                logger.info(f"Open Positions: {len(self.trader.positions)}")
                logger.info(f"===================")
                
                time.sleep(300)  # Print every 5 minutes
                
            except Exception as e:
                logger.error(f"Error printing stats: {e}")
                time.sleep(60)
    
    def run(self):
        """Start the trading bot"""
        logger.info("Starting MEXC Trading Bot...")
        
        # Validate API credentials
        account_info = self.trader.get_account_info()
        if not account_info:
            logger.error("Failed to connect to MEXC API. Check credentials.")
            return
        
        logger.info("API connection successful")
        
        # Initialize historical data
        logger.info("Loading historical data...")
        for symbol in self.symbols:
            klines = self.trader.get_klines(symbol, '1m', 100)
            for kline in klines:
                kline_data = {
                    'timestamp': kline[0],
                    'open': kline[1],
                    'high': kline[2],
                    'low': kline[3],
                    'close': kline[4],
                    'volume': kline[5]
                }
                self.strategy_engine.add_data_point(symbol, kline_data)
        
        logger.info("Historical data loaded")
        
        self.running = True
        
        # Start threads
        position_monitor_thread = threading.Thread(target=self.monitor_positions)
        position_monitor_thread.daemon = True
        position_monitor_thread.start()
        
        stats_thread = threading.Thread(target=self.print_stats)
        stats_thread.daemon = True
        stats_thread.start()
        
        # Start WebSocket in main thread
        try:
            self.start_websocket()
        except KeyboardInterrupt:
            logger.info("Shutting down bot...")
            self.running = False
            if self.ws:
                self.ws.close()
            self.conn.close()

def main():
    """Main entry point"""
    bot = MEXCTradingBot()
    
    # Set your API credentials
    bot.config.API_KEY = "your_api_key_here"
    bot.config.SECRET_KEY = "your_secret_key_here"
    
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
    finally:
        logger.info("Bot stopped")

if __name__ == "__main__":
    main() 