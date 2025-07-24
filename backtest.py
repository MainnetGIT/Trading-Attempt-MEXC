#!/usr/bin/env python3
"""
MEXC Trading Bot Backtesting Framework
Test strategies on historical data before live trading
"""

import pandas as pd
import numpy as np
import sqlite3
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json

from mexc_trading_bot import TechnicalIndicators, StrategyEngine, MEXCConfig

class BacktestEngine:
    """Backtesting engine for strategy validation"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trades = []
        self.indicators = TechnicalIndicators()
        self.config = MEXCConfig()
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        
    def get_historical_data(self, symbol: str, interval: str = '1m', 
                           days: int = 30) -> pd.DataFrame:
        """Get historical kline data from MEXC"""
        try:
            # Calculate timestamps
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            # MEXC API endpoint for historical data
            url = "https://api.mexc.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if isinstance(data, list) and data:
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                    'taker_buy_quote_volume', 'ignore'
                ])
                
                # Convert data types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                
                return df[['open', 'high', 'low', 'close', 'volume']].copy()
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
        
        return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if len(df) < 50:
            return df
        
        # MACD Scalping
        macd, signal = self.indicators.macd_scalping(df)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_crossover'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_crossunder'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # RSI
        df['rsi'] = self.indicators.rsi_divergence(df)
        
        # Stochastic
        k_percent, d_percent = self.indicators.stochastic_oscillator(df)
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        # Moving averages
        df['sma_200'] = df['close'].rolling(200).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * 1.5
        
        return df
    
    def macd_scalping_backtest(self, df: pd.DataFrame) -> List[Dict]:
        """Backtest MACD scalping strategy"""
        signals = []
        
        for i in range(200, len(df)):  # Start after indicators are calculated
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Only trade above 200 SMA
            if pd.isna(current['sma_200']) or current['close'] < current['sma_200']:
                continue
            
            # Buy signal: MACD crosses above signal
            if (prev['macd'] <= prev['macd_signal'] and 
                current['macd'] > current['macd_signal']):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'buy',
                    'strategy': 'macd_scalping',
                    'price': current['close'],
                    'stop_loss': current['close'] * 0.995,
                    'take_profit': current['close'] * 1.015,
                    'confidence': 0.8
                })
            
            # Sell signal: MACD crosses below signal  
            elif (prev['macd'] >= prev['macd_signal'] and 
                  current['macd'] < current['macd_signal']):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'sell',
                    'strategy': 'macd_scalping', 
                    'price': current['close'],
                    'stop_loss': current['close'] * 1.005,
                    'take_profit': current['close'] * 0.985,
                    'confidence': 0.8
                })
        
        return signals
    
    def momentum_backtest(self, df: pd.DataFrame) -> List[Dict]:
        """Backtest momentum strategy"""
        signals = []
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Buy signal: RSI oversold + Stoch crossover + volume spike
            if (current['rsi'] < 30 and 
                current['stoch_k'] > current['stoch_d'] and
                prev['stoch_k'] <= prev['stoch_d'] and
                current['volume_spike'] and
                current['stoch_k'] < 20):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'buy',
                    'strategy': 'momentum',
                    'price': current['close'],
                    'stop_loss': current['close'] * 0.992,
                    'take_profit': current['close'] * 1.024,
                    'confidence': 0.75
                })
            
            # Sell signal: RSI overbought + Stoch crossover
            elif (current['rsi'] > 70 and
                  current['stoch_k'] < current['stoch_d'] and
                  prev['stoch_k'] >= prev['stoch_d'] and
                  current['stoch_k'] > 80):
                
                signals.append({
                    'timestamp': current.name,
                    'type': 'sell',
                    'strategy': 'momentum',
                    'price': current['close'],
                    'stop_loss': current['close'] * 1.008,
                    'take_profit': current['close'] * 0.976,
                    'confidence': 0.75
                })
        
        return signals
    
    def range_trading_backtest(self, df: pd.DataFrame) -> List[Dict]:
        """Backtest range trading strategy"""
        signals = []
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            
            # Get recent price range
            recent_data = df.iloc[max(0, i-30):i]
            if len(recent_data) < 20:
                continue
                
            resistance = recent_data['high'].max()
            support = recent_data['low'].min()
            range_size = resistance - support
            
            # Only trade if we have a clear range
            if range_size / current['close'] < 0.01:
                continue
            
            # Buy near support
            if current['close'] <= support * 1.002:
                signals.append({
                    'timestamp': current.name,
                    'type': 'buy',
                    'strategy': 'range_trading',
                    'price': current['close'],
                    'stop_loss': support * 0.995,
                    'take_profit': resistance * 0.995,
                    'confidence': 0.7
                })
            
            # Sell near resistance
            elif current['close'] >= resistance * 0.998:
                signals.append({
                    'timestamp': current.name,
                    'type': 'sell', 
                    'strategy': 'range_trading',
                    'price': current['close'],
                    'stop_loss': resistance * 1.005,
                    'take_profit': support * 1.005,
                    'confidence': 0.7
                })
        
        return signals
    
    def simulate_trades(self, df: pd.DataFrame, signals: List[Dict]) -> Dict:
        """Simulate trading based on signals"""
        self.balance = self.initial_balance
        self.positions = []
        self.trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        
        # Sort signals by timestamp
        signals.sort(key=lambda x: x['timestamp'])
        
        for signal in signals:
            # Skip if we already have too many positions
            if len(self.positions) >= 3:  # Limit for backtesting
                continue
            
            # Calculate position size (2% risk)
            risk_amount = self.balance * 0.02
            price_diff = abs(signal['price'] - signal['stop_loss'])
            
            if price_diff == 0:
                continue
                
            position_size = risk_amount / price_diff
            position_value = position_size * signal['price']
            
            # Check if we have enough balance
            if position_value > self.balance * 0.8:  # Reserve 20% for fees
                continue
            
            # Create position
            position = {
                'entry_time': signal['timestamp'],
                'strategy': signal['strategy'],
                'type': signal['type'],
                'size': position_size,
                'entry_price': signal['price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'value': position_value
            }
            
            self.positions.append(position)
            self.balance -= position_value
        
        # Process position exits
        for i, row in df.iterrows():
            positions_to_remove = []
            
            for pos_idx, position in enumerate(self.positions):
                if i <= position['entry_time']:
                    continue
                
                current_price = row['close']
                exit_price = None
                exit_reason = None
                
                # Check exit conditions
                if position['type'] == 'buy':
                    if current_price <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_price >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'take_profit'
                else:  # sell
                    if current_price >= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_price <= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'take_profit'
                
                # Close position if exit condition met
                if exit_price:
                    # Calculate PnL
                    if position['type'] == 'buy':
                        pnl = (exit_price - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['size']
                    
                    # Account for fees (0.04% taker fee both ways)
                    entry_fee = position['value'] * 0.0004
                    exit_fee = exit_price * position['size'] * 0.0004
                    net_pnl = pnl - entry_fee - exit_fee
                    
                    # Update balance
                    self.balance += exit_price * position['size']
                    self.total_pnl += net_pnl
                    
                    # Track trade
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': i,
                        'strategy': position['strategy'],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': net_pnl,
                        'exit_reason': exit_reason
                    })
                    
                    self.total_trades += 1
                    if net_pnl > 0:
                        self.winning_trades += 1
                    
                    positions_to_remove.append(pos_idx)
            
            # Remove closed positions
            for idx in reversed(positions_to_remove):
                self.positions.pop(idx)
        
        # Calculate final metrics
        final_balance = self.balance + sum(pos['value'] for pos in self.positions)
        total_return = (final_balance / self.initial_balance - 1) * 100
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        # Calculate buy and hold return
        buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'trades': self.trades,
            'signals_generated': len(signals)
        }
    
    def run_backtest(self, symbol: str, days: int = 30) -> Dict:
        """Run complete backtest for a symbol"""
        print(f"Running backtest for {symbol} ({days} days)...")
        
        # Get historical data
        df = self.get_historical_data(symbol, '5m', days)
        if df.empty:
            print(f"No data available for {symbol}")
            return {}
        
        print(f"Loaded {len(df)} data points")
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Generate signals from all strategies
        all_signals = []
        
        # MACD Scalping
        macd_signals = self.macd_scalping_backtest(df)
        all_signals.extend(macd_signals)
        
        # Momentum
        momentum_signals = self.momentum_backtest(df)
        all_signals.extend(momentum_signals)
        
        # Range Trading
        range_signals = self.range_trading_backtest(df)
        all_signals.extend(range_signals)
        
        print(f"Generated {len(all_signals)} signals:")
        print(f"  MACD: {len(macd_signals)}")
        print(f"  Momentum: {len(momentum_signals)}")
        print(f"  Range: {len(range_signals)}")
        
        # Simulate trading
        results = self.simulate_trades(df, all_signals)
        results['symbol'] = symbol
        results['days'] = days
        results['data_points'] = len(df)
        
        return results
    
    def print_results(self, results: Dict):
        """Print backtest results"""
        if not results:
            return
        
        print("\n" + "="*50)
        print(f"BACKTEST RESULTS - {results['symbol']}")
        print("="*50)
        print(f"Period: {results['days']} days ({results['data_points']} data points)")
        print(f"Initial Balance: ${results['initial_balance']:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
        print(f"Alpha: {results['total_return'] - results['buy_hold_return']:.2f}%")
        print()
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Total PnL: ${results['total_pnl']:,.2f}")
        print(f"Signals Generated: {results['signals_generated']}")
        
        if results['trades']:
            profits = [trade['pnl'] for trade in results['trades'] if trade['pnl'] > 0]
            losses = [trade['pnl'] for trade in results['trades'] if trade['pnl'] < 0]
            
            if profits:
                avg_profit = sum(profits) / len(profits)
                print(f"Average Profit: ${avg_profit:.2f}")
            
            if losses:
                avg_loss = sum(losses) / len(losses)
                print(f"Average Loss: ${avg_loss:.2f}")
        
        print("="*50)
    
    def save_results(self, results: Dict, filename: str = "backtest_results.json"):
        """Save backtest results to file"""
        # Convert datetime objects to strings for JSON serialization
        results_copy = results.copy()
        if 'trades' in results_copy:
            for trade in results_copy['trades']:
                if 'entry_time' in trade:
                    trade['entry_time'] = str(trade['entry_time'])
                if 'exit_time' in trade:
                    trade['exit_time'] = str(trade['exit_time'])
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")

def main():
    """Run backtesting for multiple symbols"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    days = 7  # Test with 1 week of data
    
    backtest = BacktestEngine(initial_balance=10000)
    
    all_results = {}
    
    for symbol in symbols:
        try:
            results = backtest.run_backtest(symbol, days)
            if results:
                backtest.print_results(results)
                all_results[symbol] = results
                
                # Save individual results
                backtest.save_results(results, f"backtest_{symbol}_{days}d.json")
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")
    
    # Summary
    if all_results:
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        total_return_sum = 0
        buy_hold_sum = 0
        win_rate_sum = 0
        
        for symbol, results in all_results.items():
            print(f"{symbol:10s}: {results['total_return']:6.1f}% "
                  f"(vs {results['buy_hold_return']:6.1f}% B&H) "
                  f"- {results['win_rate']:4.1f}% win rate")
            
            total_return_sum += results['total_return']
            buy_hold_sum += results['buy_hold_return']
            win_rate_sum += results['win_rate']
        
        avg_return = total_return_sum / len(all_results)
        avg_buy_hold = buy_hold_sum / len(all_results)
        avg_win_rate = win_rate_sum / len(all_results)
        
        print("-" * 50)
        print(f"{'AVERAGE':10s}: {avg_return:6.1f}% "
              f"(vs {avg_buy_hold:6.1f}% B&H) "
              f"- {avg_win_rate:4.1f}% win rate")
        print(f"Alpha: {avg_return - avg_buy_hold:+.1f}%")

if __name__ == "__main__":
    main() 