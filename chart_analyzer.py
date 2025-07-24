#!/usr/bin/env python3
"""
Real-Time Chart Analyzer for High-Frequency S/R Trading
Provides precise chart analysis with multiple timeframes for 10-50 trades per day
"""

import asyncio
import websockets
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import threading
import queue
from collections import deque
import logging
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from dataclasses import dataclass
import requests

# Configure styling
plt.style.use('dark_background')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

@dataclass
class ChartLevel:
    """Chart level for visualization"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float
    timeframe: str
    color: str
    style: str  # 'solid', 'dashed', 'dotted'

@dataclass
class ChartPattern:
    """Chart pattern for visualization"""
    pattern_type: str
    points: List[Tuple[float, float]]  # (x, y) coordinates
    confidence: float
    color: str
    label: str

class RealTimeChartDisplay:
    """Real-time chart display with precise analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.chart_symbols = config['chart_display']['symbols_to_chart']
        self.update_frequency = config['chart_display']['update_frequency_seconds']
        
        # Data storage
        self.price_data = {}
        self.chart_levels = {}
        self.chart_patterns = {}
        self.signal_queue = queue.Queue()
        
        # Chart setup
        self.figures = {}
        self.axes = {}
        self.running = False
        
        # Timeframes to display
        self.timeframes = ['1m', '5m']
        
        # Initialize data containers
        for symbol in self.chart_symbols:
            self.price_data[symbol] = {
                '1m': deque(maxlen=200),   # 200 minutes
                '5m': deque(maxlen=100)    # 500 minutes
            }
            self.chart_levels[symbol] = []
            self.chart_patterns[symbol] = []
            
        # Colors for different elements
        self.colors = {
            'support': '#00ff00',      # Green
            'resistance': '#ff0000',   # Red
            'pattern': '#ffff00',      # Yellow
            'volume': '#8080ff',       # Light blue
            'price': '#ffffff',        # White
            'ma_fast': '#ff8040',      # Orange
            'ma_slow': '#4080ff'       # Blue
        }
        
    async def start_analysis(self):
        """Start real-time chart analysis"""
        logger.info("Starting real-time chart analysis...")
        self.running = True
        
        # Create tasks
        tasks = []
        
        # WebSocket data feeds
        for symbol in self.chart_symbols:
            task = asyncio.create_task(self._websocket_feed(symbol))
            tasks.append(task)
        
        # Chart update task
        chart_task = asyncio.create_task(self._update_charts())
        tasks.append(chart_task)
        
        # Analysis task
        analysis_task = asyncio.create_task(self._analyze_charts())
        tasks.append(analysis_task)
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Chart analysis stopped by user")
        finally:
            self.running = False
    
    async def _websocket_feed(self, symbol: str):
        """WebSocket feed for real-time price data"""
        url = "wss://wbs.mexc.com/ws"
        
        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [
                f"spot@public.kline.v3.api@{symbol}@Min1",
                f"spot@public.kline.v3.api@{symbol}@Min5",
                f"spot@public.deals.v3.api@{symbol}"
            ]
        }
        
        while self.running:
            try:
                async with websockets.connect(url) as websocket:
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        data = json.loads(message)
                        await self._process_market_data(symbol, data)
                        
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _process_market_data(self, symbol: str, data: Dict):
        """Process incoming market data"""
        try:
            if 'c' in data and 'd' in data:  # Kline data
                # Extract kline information
                kline_data = data['d']
                
                # Determine timeframe
                if 'Min1' in data.get('c', ''):
                    timeframe = '1m'
                elif 'Min5' in data.get('c', ''):
                    timeframe = '5m'
                else:
                    return
                
                # Create OHLCV data point
                ohlcv = {
                    'timestamp': datetime.fromtimestamp(kline_data['t'] / 1000),
                    'open': float(kline_data['o']),
                    'high': float(kline_data['h']),
                    'low': float(kline_data['l']),
                    'close': float(kline_data['c']),
                    'volume': float(kline_data['v'])
                }
                
                # Update data
                data_queue = self.price_data[symbol][timeframe]
                
                # Check if this is an update to existing candle or new candle
                if len(data_queue) > 0:
                    last_candle = data_queue[-1]
                    if abs((ohlcv['timestamp'] - last_candle['timestamp']).total_seconds()) < 30:
                        # Update existing candle
                        data_queue[-1] = ohlcv
                    else:
                        # New candle
                        data_queue.append(ohlcv)
                else:
                    # First candle
                    data_queue.append(ohlcv)
                
                # Trigger analysis
                await self._update_levels_and_patterns(symbol, timeframe)
                
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {e}")
    
    async def _update_levels_and_patterns(self, symbol: str, timeframe: str):
        """Update support/resistance levels and patterns"""
        try:
            data_queue = self.price_data[symbol][timeframe]
            if len(data_queue) < 20:
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(list(data_queue))
            
            # Find support/resistance levels
            levels = self._find_chart_levels(df, timeframe)
            
            # Find chart patterns
            patterns = self._find_chart_patterns(df, timeframe)
            
            # Update storage
            if symbol not in self.chart_levels:
                self.chart_levels[symbol] = []
            if symbol not in self.chart_patterns:
                self.chart_patterns[symbol] = []
            
            # Filter levels by timeframe and update
            existing_levels = [l for l in self.chart_levels[symbol] if l.timeframe != timeframe]
            self.chart_levels[symbol] = existing_levels + levels
            
            # Update patterns
            existing_patterns = [p for p in self.chart_patterns[symbol] if timeframe not in p.label]
            self.chart_patterns[symbol] = existing_patterns + patterns
            
        except Exception as e:
            logger.error(f"Error updating levels and patterns for {symbol}: {e}")
    
    def _find_chart_levels(self, df: pd.DataFrame, timeframe: str) -> List[ChartLevel]:
        """Find support and resistance levels for charting"""
        levels = []
        
        if len(df) < 10:
            return levels
        
        # Method 1: Local extrema
        window = 3 if timeframe == '1m' else 5
        
        # Find local highs (resistance)
        for i in range(window, len(df) - window):
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                
                levels.append(ChartLevel(
                    price=df['high'].iloc[i],
                    level_type='resistance',
                    strength=self._calculate_level_strength(df, df['high'].iloc[i]),
                    timeframe=timeframe,
                    color=self.colors['resistance'],
                    style='solid' if timeframe == '5m' else 'dashed'
                ))
        
        # Find local lows (support)
        for i in range(window, len(df) - window):
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                
                levels.append(ChartLevel(
                    price=df['low'].iloc[i],
                    level_type='support',
                    strength=self._calculate_level_strength(df, df['low'].iloc[i]),
                    timeframe=timeframe,
                    color=self.colors['support'],
                    style='solid' if timeframe == '5m' else 'dashed'
                ))
        
        # Method 2: Volume-weighted levels
        volume_levels = self._find_volume_weighted_levels(df, timeframe)
        levels.extend(volume_levels)
        
        # Filter by strength and return top levels
        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[:8]  # Top 8 levels per timeframe
    
    def _calculate_level_strength(self, df: pd.DataFrame, price: float) -> float:
        """Calculate strength of support/resistance level"""
        tolerance = price * 0.002  # 0.2% tolerance
        
        # Count touches
        touches = 0
        total_volume = 0
        
        for _, row in df.iterrows():
            if row['low'] <= price + tolerance and row['high'] >= price - tolerance:
                touches += 1
                total_volume += row['volume']
        
        # Base strength
        strength = min(touches / 3.0, 1.0)
        
        # Volume bonus
        avg_volume = df['volume'].mean()
        if avg_volume > 0:
            volume_factor = (total_volume / touches) / avg_volume if touches > 0 else 0
            strength += min(volume_factor / 2.0, 0.3)
        
        return min(strength, 1.0)
    
    def _find_volume_weighted_levels(self, df: pd.DataFrame, timeframe: str) -> List[ChartLevel]:
        """Find volume-weighted support/resistance levels"""
        levels = []
        
        if len(df) < 10:
            return levels
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        num_bins = 15
        
        price_bins = np.linspace(price_min, price_max, num_bins)
        volume_profile = np.zeros(num_bins - 1)
        
        # Calculate volume at each price level
        for i, (_, row) in enumerate(df.iterrows()):
            # Distribute volume across price range of candle
            candle_range = row['high'] - row['low']
            if candle_range > 0:
                for j in range(len(price_bins) - 1):
                    bin_low = price_bins[j]
                    bin_high = price_bins[j + 1]
                    
                    # Check overlap with candle
                    overlap_low = max(bin_low, row['low'])
                    overlap_high = min(bin_high, row['high'])
                    
                    if overlap_high > overlap_low:
                        overlap_ratio = (overlap_high - overlap_low) / candle_range
                        volume_profile[j] += row['volume'] * overlap_ratio
        
        # Find high-volume areas
        avg_volume = np.mean(volume_profile)
        for i, volume in enumerate(volume_profile):
            if volume > avg_volume * 1.5:  # Above average volume
                price = (price_bins[i] + price_bins[i + 1]) / 2
                current_price = df['close'].iloc[-1]
                
                level_type = 'support' if price < current_price else 'resistance'
                
                levels.append(ChartLevel(
                    price=price,
                    level_type=level_type,
                    strength=min(volume / (avg_volume * 2), 1.0),
                    timeframe=f"{timeframe}_volume",
                    color=self.colors[level_type],
                    style='dotted'
                ))
        
        return levels
    
    def _find_chart_patterns(self, df: pd.DataFrame, timeframe: str) -> List[ChartPattern]:
        """Find chart patterns for visualization"""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        # Pattern 1: Double tops/bottoms
        double_patterns = self._find_double_patterns(df, timeframe)
        patterns.extend(double_patterns)
        
        # Pattern 2: Trend lines
        trend_patterns = self._find_trend_lines(df, timeframe)
        patterns.extend(trend_patterns)
        
        # Pattern 3: Triangles
        triangle_patterns = self._find_triangles(df, timeframe)
        patterns.extend(triangle_patterns)
        
        return patterns
    
    def _find_double_patterns(self, df: pd.DataFrame, timeframe: str) -> List[ChartPattern]:
        """Find double top/bottom patterns"""
        patterns = []
        
        # Find potential double bottoms
        lows = []
        for i in range(2, len(df) - 2):
            if df['low'].iloc[i] <= df['low'].iloc[i-1] and df['low'].iloc[i] <= df['low'].iloc[i+1]:
                lows.append((i, df['low'].iloc[i]))
        
        # Check for double bottoms
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                idx1, price1 = lows[i]
                idx2, price2 = lows[j]
                
                if abs(price1 - price2) / price1 <= 0.02:  # Within 2%
                    patterns.append(ChartPattern(
                        pattern_type='double_bottom',
                        points=[(idx1, price1), (idx2, price2)],
                        confidence=0.7,
                        color=self.colors['pattern'],
                        label=f'Double Bottom ({timeframe})'
                    ))
        
        return patterns
    
    def _find_trend_lines(self, df: pd.DataFrame, timeframe: str) -> List[ChartPattern]:
        """Find trend lines"""
        patterns = []
        
        if len(df) < 10:
            return patterns
        
        # Find swing highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(df) - 2):
            # Swing high
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i+1]):
                highs.append((i, df['high'].iloc[i]))
            
            # Swing low
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i+1]):
                lows.append((i, df['low'].iloc[i]))
        
        # Create trend lines from highs (resistance trend lines)
        if len(highs) >= 2:
            for i in range(len(highs) - 1):
                patterns.append(ChartPattern(
                    pattern_type='resistance_trendline',
                    points=[highs[i], highs[i+1]],
                    confidence=0.6,
                    color=self.colors['resistance'],
                    label=f'Resistance Line ({timeframe})'
                ))
        
        # Create trend lines from lows (support trend lines)
        if len(lows) >= 2:
            for i in range(len(lows) - 1):
                patterns.append(ChartPattern(
                    pattern_type='support_trendline',
                    points=[lows[i], lows[i+1]],
                    confidence=0.6,
                    color=self.colors['support'],
                    label=f'Support Line ({timeframe})'
                ))
        
        return patterns[:4]  # Limit to 4 trend lines per timeframe
    
    def _find_triangles(self, df: pd.DataFrame, timeframe: str) -> List[ChartPattern]:
        """Find triangle patterns"""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        recent_data = df.tail(15)
        
        # Simple ascending triangle detection
        resistance_level = recent_data['high'].max()
        resistance_touches = len(recent_data[recent_data['high'] >= resistance_level * 0.995])
        
        if resistance_touches >= 2:
            # Check if lows are generally rising
            lows = recent_data['low'].values
            if len(lows) >= 3 and lows[-1] > lows[0]:
                patterns.append(ChartPattern(
                    pattern_type='ascending_triangle',
                    points=[(0, lows[0]), (len(lows)-1, lows[-1]), 
                           (0, resistance_level), (len(lows)-1, resistance_level)],
                    confidence=0.6,
                    color=self.colors['pattern'],
                    label=f'Ascending Triangle ({timeframe})'
                ))
        
        return patterns
    
    async def _update_charts(self):
        """Update chart displays"""
        logger.info("Starting chart update loop...")
        
        # Setup matplotlib for real-time display
        plt.ion()
        
        # Create subplots for each symbol
        for symbol in self.chart_symbols:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{symbol} - Real-Time Analysis', fontsize=16, color='white')
            
            self.figures[symbol] = fig
            self.axes[symbol] = {
                '1m_price': axes[0, 0],
                '1m_volume': axes[1, 0],
                '5m_price': axes[0, 1],
                '5m_volume': axes[1, 1]
            }
            
            # Style the axes
            for ax in axes.flat:
                ax.set_facecolor('black')
                ax.grid(True, alpha=0.3)
                ax.tick_params(colors='white')
        
        while self.running:
            try:
                # Update each symbol's charts
                for symbol in self.chart_symbols:
                    await self._update_symbol_chart(symbol)
                
                # Refresh displays
                for fig in self.figures.values():
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
                await asyncio.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"Error updating charts: {e}")
                await asyncio.sleep(5)
    
    async def _update_symbol_chart(self, symbol: str):
        """Update charts for a specific symbol"""
        try:
            if symbol not in self.figures:
                return
            
            axes = self.axes[symbol]
            
            # Update 1-minute chart
            await self._plot_timeframe_data(symbol, '1m', axes['1m_price'], axes['1m_volume'])
            
            # Update 5-minute chart
            await self._plot_timeframe_data(symbol, '5m', axes['5m_price'], axes['5m_volume'])
            
        except Exception as e:
            logger.error(f"Error updating chart for {symbol}: {e}")
    
    async def _plot_timeframe_data(self, symbol: str, timeframe: str, price_ax, volume_ax):
        """Plot data for specific timeframe"""
        try:
            data_queue = self.price_data[symbol][timeframe]
            if len(data_queue) < 2:
                return
            
            # Clear axes
            price_ax.clear()
            volume_ax.clear()
            
            # Convert to DataFrame
            df = pd.DataFrame(list(data_queue))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Plot candlesticks (simplified as OHLC line)
            price_ax.plot(df['timestamp'], df['close'], color=self.colors['price'], linewidth=1)
            
            # Plot moving averages
            if len(df) >= 9:
                df['ema9'] = talib.EMA(df['close'].values, timeperiod=9)
                price_ax.plot(df['timestamp'], df['ema9'], color=self.colors['ma_fast'], linewidth=1, alpha=0.7)
            
            if len(df) >= 21:
                df['ema21'] = talib.EMA(df['close'].values, timeperiod=21)
                price_ax.plot(df['timestamp'], df['ema21'], color=self.colors['ma_slow'], linewidth=1, alpha=0.7)
            
            # Plot support/resistance levels
            levels = [l for l in self.chart_levels.get(symbol, []) if timeframe in l.timeframe]
            for level in levels:
                price_ax.axhline(y=level.price, color=level.color, linestyle=level.style, 
                               alpha=0.7, linewidth=1)
                
                # Add level label
                price_ax.text(df['timestamp'].iloc[-1], level.price, 
                            f'{level.level_type.title()}: {level.price:.6f}',
                            color=level.color, fontsize=8, ha='left')
            
            # Plot patterns
            patterns = [p for p in self.chart_patterns.get(symbol, []) if timeframe in p.label]
            for pattern in patterns:
                if len(pattern.points) >= 2:
                    x_coords = [df['timestamp'].iloc[int(p[0])] if p[0] < len(df) else df['timestamp'].iloc[-1] 
                               for p in pattern.points]
                    y_coords = [p[1] for p in pattern.points]
                    price_ax.plot(x_coords, y_coords, color=pattern.color, linestyle='--', alpha=0.6)
            
            # Plot volume
            volume_ax.bar(df['timestamp'], df['volume'], color=self.colors['volume'], alpha=0.6, width=0.8)
            
            # Format axes
            price_ax.set_title(f'{symbol} - {timeframe.upper()} Price', color='white', fontsize=10)
            price_ax.set_ylabel('Price (USDT)', color='white')
            price_ax.tick_params(colors='white')
            
            volume_ax.set_title(f'{symbol} - {timeframe.upper()} Volume', color='white', fontsize=10)
            volume_ax.set_ylabel('Volume', color='white')
            volume_ax.set_xlabel('Time', color='white')
            volume_ax.tick_params(colors='white')
            
            # Format x-axis
            if len(df) > 0:
                price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                volume_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                
                # Rotate labels
                price_ax.tick_params(axis='x', rotation=45)
                volume_ax.tick_params(axis='x', rotation=45)
            
            # Set axis colors
            price_ax.set_facecolor('black')
            volume_ax.set_facecolor('black')
            price_ax.grid(True, alpha=0.3, color='gray')
            volume_ax.grid(True, alpha=0.3, color='gray')
            
        except Exception as e:
            logger.error(f"Error plotting {timeframe} data for {symbol}: {e}")
    
    async def _analyze_charts(self):
        """Continuous chart analysis for trading signals"""
        logger.info("Starting chart analysis loop...")
        
        while self.running:
            try:
                for symbol in self.chart_symbols:
                    await self._analyze_symbol_charts(symbol)
                
                await asyncio.sleep(5)  # Analyze every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in chart analysis: {e}")
                await asyncio.sleep(10)
    
    async def _analyze_symbol_charts(self, symbol: str):
        """Analyze charts for a specific symbol"""
        try:
            # Get current price
            data_1m = list(self.price_data[symbol]['1m'])
            if len(data_1m) < 5:
                return
            
            current_price = data_1m[-1]['close']
            
            # Check for signals near S/R levels
            levels = self.chart_levels.get(symbol, [])
            for level in levels:
                distance = abs(current_price - level.price) / current_price
                
                if distance <= 0.002:  # Within 0.2%
                    signal = {
                        'symbol': symbol,
                        'type': 'LEVEL_APPROACH',
                        'level_type': level.level_type,
                        'level_price': level.price,
                        'current_price': current_price,
                        'distance': distance,
                        'strength': level.strength,
                        'timeframe': level.timeframe,
                        'timestamp': datetime.now()
                    }
                    
                    # Add to signal queue for processing
                    self.signal_queue.put(signal)
                    
                    logger.info(f"Signal: {symbol} approaching {level.level_type} at {level.price:.6f} "
                               f"(current: {current_price:.6f}, distance: {distance*100:.2f}%)")
            
            # Check patterns for breakouts
            patterns = self.chart_patterns.get(symbol, [])
            for pattern in patterns:
                if pattern.pattern_type == 'ascending_triangle':
                    # Check for breakout
                    if len(pattern.points) >= 2:
                        resistance_level = max(p[1] for p in pattern.points)
                        if current_price > resistance_level * 1.003:  # 0.3% breakout
                            signal = {
                                'symbol': symbol,
                                'type': 'PATTERN_BREAKOUT',
                                'pattern_type': pattern.pattern_type,
                                'breakout_level': resistance_level,
                                'current_price': current_price,
                                'confidence': pattern.confidence,
                                'timestamp': datetime.now()
                            }
                            
                            self.signal_queue.put(signal)
                            
                            logger.info(f"Pattern Breakout: {symbol} {pattern.pattern_type} "
                                       f"above {resistance_level:.6f}")
            
        except Exception as e:
            logger.error(f"Error analyzing charts for {symbol}: {e}")
    
    def get_signals(self) -> List[Dict]:
        """Get pending signals from analysis"""
        signals = []
        
        while not self.signal_queue.empty():
            try:
                signal = self.signal_queue.get_nowait()
                signals.append(signal)
            except queue.Empty:
                break
        
        return signals
    
    def save_chart_screenshot(self, symbol: str, filename: str = None):
        """Save chart screenshot"""
        try:
            if symbol in self.figures:
                if not filename:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'chart_{symbol}_{timestamp}.png'
                
                self.figures[symbol].savefig(filename, 
                                           facecolor='black', 
                                           edgecolor='none',
                                           dpi=150,
                                           bbox_inches='tight')
                logger.info(f"Chart screenshot saved: {filename}")
                return filename
        except Exception as e:
            logger.error(f"Error saving chart screenshot: {e}")
        
        return None

async def main():
    """Test the chart analyzer"""
    # Test configuration
    config = {
        'chart_display': {
            'symbols_to_chart': ['BTCUSDT', 'ETHUSDT'],
            'update_frequency_seconds': 5,
            'timeframes_displayed': ['1m', '5m'],
            'indicators_overlay': ['EMA9', 'EMA21', 'SR_LEVELS'],
            'pattern_overlay': True,
            'volume_display': True,
            'save_screenshots': True
        }
    }
    
    # Create chart analyzer
    chart_analyzer = RealTimeChartDisplay(config)
    
    try:
        await chart_analyzer.start_analysis()
    except KeyboardInterrupt:
        print("Chart analysis stopped")

if __name__ == "__main__":
    asyncio.run(main()) 