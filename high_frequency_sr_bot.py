#!/usr/bin/env python3
"""
High-Frequency MEXC Support/Resistance Scalping Bot
Designed for 10-50 trades per day with real-time chart analysis

Key Features:
- 1-minute and 30-second timeframe analysis
- Real-time WebSocket price feeds
- Aggressive S/R detection for more opportunities
- Quick profit targets (0.5-1.5%)
- Fast position management (5-30 minute holds)
- Multiple timeframe confirmation
- Enhanced chart pattern recognition
"""

import asyncio
import websockets
import json
import logging
import time
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import deque
import requests
import hmac
import hashlib
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hf_sr_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HighFrequencySignal:
    """High-frequency trading signal with precise timing"""
    symbol: str
    signal_type: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    urgency: str  # 'IMMEDIATE', 'HIGH', 'MEDIUM'
    timeframe: str  # '1m', '30s'
    level_strength: float
    volume_confirmation: float
    chart_pattern: str
    expected_hold_minutes: int
    risk_reward_ratio: float
    timestamp: datetime

@dataclass
class MicroSRLevel:
    """Micro support/resistance level for scalping"""
    price: float
    level_type: str
    strength: float
    touch_count: int
    volume_at_level: float
    recent_bounce: bool
    timeframe_confirmed: List[str]  # Which timeframes confirm this level
    age_minutes: float
    next_target: Optional[float]

class RealTimeChartAnalyzer:
    """Real-time chart analysis for high-frequency trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.price_data = {}  # Store real-time price data
        self.micro_levels = {}  # Store micro S/R levels
        self.chart_patterns = {}  # Store detected patterns
        
        # High-frequency parameters
        self.min_profit_target = 0.005  # 0.5% minimum profit
        self.max_profit_target = 0.015  # 1.5% maximum profit
        self.proximity_threshold = 0.001  # 0.1% proximity (tighter)
        self.volume_spike_threshold = 1.3  # 1.3x volume (more sensitive)
        
        # Timeframes for analysis
        self.timeframes = ['30s', '1m', '3m', '5m']
        
        # Initialize data containers
        for symbol in config.get('trading_symbols', []):
            self.price_data[symbol] = {
                '30s': deque(maxlen=240),  # 2 hours of 30s data
                '1m': deque(maxlen=120),   # 2 hours of 1m data
                '3m': deque(maxlen=40),    # 2 hours of 3m data
                '5m': deque(maxlen=24)     # 2 hours of 5m data
            }
            self.micro_levels[symbol] = []
            self.chart_patterns[symbol] = []
    
    def update_price_data(self, symbol: str, price_update: Dict):
        """Update real-time price data across all timeframes"""
        timestamp = datetime.now()
        
        # Create OHLCV data point
        price_point = {
            'timestamp': timestamp,
            'open': price_update.get('open', price_update['price']),
            'high': price_update.get('high', price_update['price']),
            'low': price_update.get('low', price_update['price']),
            'close': price_update['price'],
            'volume': price_update.get('volume', 0)
        }
        
        # Update each timeframe based on its interval
        self._update_timeframe_data(symbol, '30s', price_point, 30)
        self._update_timeframe_data(symbol, '1m', price_point, 60)
        self._update_timeframe_data(symbol, '3m', price_point, 180)
        self._update_timeframe_data(symbol, '5m', price_point, 300)
        
        # Trigger analysis after data update
        self._analyze_micro_levels(symbol)
        self._detect_chart_patterns(symbol)
    
    def _update_timeframe_data(self, symbol: str, timeframe: str, price_point: Dict, interval_seconds: int):
        """Update specific timeframe data with aggregation"""
        data_queue = self.price_data[symbol][timeframe]
        
        current_time = price_point['timestamp']
        
        # Check if we need to create a new candle or update existing
        if len(data_queue) == 0:
            # First data point
            data_queue.append(price_point.copy())
        else:
            last_candle = data_queue[-1]
            time_diff = (current_time - last_candle['timestamp']).total_seconds()
            
            if time_diff >= interval_seconds:
                # Start new candle
                data_queue.append(price_point.copy())
            else:
                # Update current candle
                last_candle['high'] = max(last_candle['high'], price_point['high'])
                last_candle['low'] = min(last_candle['low'], price_point['low'])
                last_candle['close'] = price_point['close']
                last_candle['volume'] += price_point['volume']
    
    def _analyze_micro_levels(self, symbol: str):
        """Analyze and update micro support/resistance levels"""
        micro_levels = []
        
        # Analyze each timeframe for S/R levels
        for timeframe in ['1m', '3m', '5m']:
            data_queue = self.price_data[symbol][timeframe]
            if len(data_queue) < 20:
                continue
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(list(data_queue))
            
            # Find micro levels using multiple methods
            levels = self._find_micro_levels(df, timeframe)
            micro_levels.extend(levels)
        
        # Cluster and validate levels
        self.micro_levels[symbol] = self._cluster_micro_levels(micro_levels)
    
    def _find_micro_levels(self, df: pd.DataFrame, timeframe: str) -> List[MicroSRLevel]:
        """Find micro support/resistance levels in specific timeframe"""
        levels = []
        
        if len(df) < 10:
            return levels
        
        # Method 1: Local extrema with smaller windows
        window = 3 if timeframe == '1m' else 5
        
        # Find local highs (resistance)
        highs = df['high'].rolling(window=window, center=True).max()
        resistance_points = df[df['high'] == highs]['high'].values
        
        # Find local lows (support)
        lows = df['low'].rolling(window=window, center=True).min()
        support_points = df[df['low'] == lows]['low'].values
        
        # Create MicroSRLevel objects
        for price in resistance_points:
            if not np.isnan(price):
                levels.append(MicroSRLevel(
                    price=float(price),
                    level_type='resistance',
                    strength=self._calculate_micro_strength(df, price, 'resistance'),
                    touch_count=self._count_touches(df, price),
                    volume_at_level=self._get_volume_at_price(df, price),
                    recent_bounce=self._check_recent_bounce(df, price, 'resistance'),
                    timeframe_confirmed=[timeframe],
                    age_minutes=self._calculate_age_minutes(df, price),
                    next_target=self._find_next_target(df, price, 'resistance')
                ))
        
        for price in support_points:
            if not np.isnan(price):
                levels.append(MicroSRLevel(
                    price=float(price),
                    level_type='support',
                    strength=self._calculate_micro_strength(df, price, 'support'),
                    touch_count=self._count_touches(df, price),
                    volume_at_level=self._get_volume_at_price(df, price),
                    recent_bounce=self._check_recent_bounce(df, price, 'support'),
                    timeframe_confirmed=[timeframe],
                    age_minutes=self._calculate_age_minutes(df, price),
                    next_target=self._find_next_target(df, price, 'support')
                ))
        
        # Method 2: Volume profile levels
        volume_levels = self._find_volume_profile_levels(df, timeframe)
        levels.extend(volume_levels)
        
        return levels
    
    def _calculate_micro_strength(self, df: pd.DataFrame, price: float, level_type: str) -> float:
        """Calculate strength of micro S/R level"""
        tolerance = price * 0.001  # 0.1% tolerance
        
        # Count how many times price approached this level
        if level_type == 'resistance':
            touches = len(df[(df['high'] >= price - tolerance) & (df['high'] <= price + tolerance)])
        else:
            touches = len(df[(df['low'] >= price - tolerance) & (df['low'] <= price + tolerance)])
        
        # Base strength on touches and volume
        base_strength = min(touches / 3.0, 1.0)  # Normalize to max 1.0
        
        # Volume confirmation
        volume_factor = self._get_volume_at_price(df, price) / df['volume'].mean()
        volume_bonus = min(volume_factor / 2.0, 0.3)  # Max 0.3 bonus
        
        return min(base_strength + volume_bonus, 1.0)
    
    def _count_touches(self, df: pd.DataFrame, price: float) -> int:
        """Count how many times price touched this level"""
        tolerance = price * 0.001
        return len(df[(df['low'] <= price + tolerance) & (df['high'] >= price - tolerance)])
    
    def _get_volume_at_price(self, df: pd.DataFrame, price: float) -> float:
        """Get average volume when price was near this level"""
        tolerance = price * 0.002  # 0.2% tolerance
        near_price = df[(df['low'] <= price + tolerance) & (df['high'] >= price - tolerance)]
        return near_price['volume'].mean() if len(near_price) > 0 else 0
    
    def _check_recent_bounce(self, df: pd.DataFrame, price: float, level_type: str) -> bool:
        """Check if there was a recent bounce from this level"""
        if len(df) < 5:
            return False
        
        recent_data = df.tail(5)
        tolerance = price * 0.001
        
        if level_type == 'resistance':
            # Check if price recently hit resistance and bounced down
            hit_resistance = any(recent_data['high'] >= price - tolerance)
            bounced_down = recent_data['close'].iloc[-1] < price - tolerance
            return hit_resistance and bounced_down
        else:
            # Check if price recently hit support and bounced up
            hit_support = any(recent_data['low'] <= price + tolerance)
            bounced_up = recent_data['close'].iloc[-1] > price + tolerance
            return hit_support and bounced_up
    
    def _calculate_age_minutes(self, df: pd.DataFrame, price: float) -> float:
        """Calculate how long ago this level was formed"""
        if len(df) == 0:
            return 999  # Very old
        
        return (datetime.now() - df['timestamp'].iloc[-1]).total_seconds() / 60
    
    def _find_next_target(self, df: pd.DataFrame, price: float, level_type: str) -> Optional[float]:
        """Find the next logical target for this level"""
        if level_type == 'resistance':
            # Find next resistance above
            higher_highs = df[df['high'] > price * 1.003]['high']  # At least 0.3% higher
            return float(higher_highs.min()) if len(higher_highs) > 0 else None
        else:
            # Find next support below
            lower_lows = df[df['low'] < price * 0.997]['low']  # At least 0.3% lower
            return float(lower_lows.max()) if len(lower_lows) > 0 else None
    
    def _find_volume_profile_levels(self, df: pd.DataFrame, timeframe: str) -> List[MicroSRLevel]:
        """Find support/resistance levels based on volume profile"""
        levels = []
        
        if len(df) < 10:
            return levels
        
        # Create price-volume profile
        price_range = np.linspace(df['low'].min(), df['high'].max(), 20)
        volume_profile = []
        
        for price in price_range:
            tolerance = (df['high'].max() - df['low'].min()) * 0.05  # 5% of range
            volume_at_price = df[
                (df['low'] <= price + tolerance) & 
                (df['high'] >= price - tolerance)
            ]['volume'].sum()
            volume_profile.append(volume_at_price)
        
        # Find high-volume price levels
        avg_volume = np.mean(volume_profile)
        for i, (price, volume) in enumerate(zip(price_range, volume_profile)):
            if volume > avg_volume * 1.5:  # High volume area
                # Determine if support or resistance based on recent price action
                current_price = df['close'].iloc[-1]
                level_type = 'support' if price < current_price else 'resistance'
                
                levels.append(MicroSRLevel(
                    price=float(price),
                    level_type=level_type,
                    strength=min(volume / (avg_volume * 2), 1.0),
                    touch_count=1,
                    volume_at_level=volume,
                    recent_bounce=False,
                    timeframe_confirmed=[timeframe],
                    age_minutes=5.0,  # Assume recent
                    next_target=None
                ))
        
        return levels
    
    def _cluster_micro_levels(self, levels: List[MicroSRLevel]) -> List[MicroSRLevel]:
        """Cluster nearby micro levels to avoid redundancy"""
        if not levels:
            return []
        
        clustered = []
        used_indices = set()
        
        for i, level in enumerate(levels):
            if i in used_indices:
                continue
            
            # Find nearby levels to cluster
            cluster_levels = [level]
            used_indices.add(i)
            
            for j, other_level in enumerate(levels[i+1:], i+1):
                if j in used_indices:
                    continue
                
                price_diff = abs(level.price - other_level.price) / level.price
                if price_diff <= 0.002:  # Within 0.2%
                    cluster_levels.append(other_level)
                    used_indices.add(j)
            
            # Merge cluster into strongest level
            if len(cluster_levels) > 1:
                strongest = max(cluster_levels, key=lambda x: x.strength)
                strongest.touch_count = sum(l.touch_count for l in cluster_levels)
                strongest.volume_at_level = sum(l.volume_at_level for l in cluster_levels)
                
                # Combine timeframe confirmations
                all_timeframes = []
                for l in cluster_levels:
                    all_timeframes.extend(l.timeframe_confirmed)
                strongest.timeframe_confirmed = list(set(all_timeframes))
                
                clustered.append(strongest)
            else:
                clustered.append(level)
        
        # Sort by strength and recency
        clustered.sort(key=lambda x: (x.strength, -x.age_minutes), reverse=True)
        return clustered[:10]  # Keep top 10 levels per symbol
    
    def _detect_chart_patterns(self, symbol: str):
        """Detect chart patterns for additional confirmation"""
        patterns = []
        
        # Analyze 1-minute data for patterns
        data_1m = list(self.price_data[symbol]['1m'])
        if len(data_1m) < 10:
            return
        
        df = pd.DataFrame(data_1m)
        
        # Pattern 1: Double Bottom/Top
        double_patterns = self._detect_double_patterns(df)
        patterns.extend(double_patterns)
        
        # Pattern 2: Ascending/Descending Triangle
        triangle_patterns = self._detect_triangle_patterns(df)
        patterns.extend(triangle_patterns)
        
        # Pattern 3: Break and Retest
        retest_patterns = self._detect_retest_patterns(df)
        patterns.extend(retest_patterns)
        
        self.chart_patterns[symbol] = patterns
    
    def _detect_double_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double bottom/top patterns"""
        patterns = []
        
        if len(df) < 10:
            return patterns
        
        # Simple double bottom detection
        lows = df['low'].rolling(window=3, center=True).min()
        low_points = df[df['low'] == lows]['low'].values
        
        if len(low_points) >= 2:
            last_two_lows = low_points[-2:]
            if abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] <= 0.005:  # Within 0.5%
                patterns.append({
                    'type': 'double_bottom',
                    'support_level': float(np.mean(last_two_lows)),
                    'confidence': 0.7,
                    'target': float(np.mean(last_two_lows) * 1.01)  # 1% target
                })
        
        # Simple double top detection
        highs = df['high'].rolling(window=3, center=True).max()
        high_points = df[df['high'] == highs]['high'].values
        
        if len(high_points) >= 2:
            last_two_highs = high_points[-2:]
            if abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] <= 0.005:  # Within 0.5%
                patterns.append({
                    'type': 'double_top',
                    'resistance_level': float(np.mean(last_two_highs)),
                    'confidence': 0.7,
                    'target': float(np.mean(last_two_highs) * 0.99)  # 1% target
                })
        
        return patterns
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect triangle patterns"""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        recent_data = df.tail(15)
        
        # Simple ascending triangle (horizontal resistance, rising support)
        resistance_level = recent_data['high'].max()
        resistance_touches = len(recent_data[recent_data['high'] >= resistance_level * 0.999])
        
        if resistance_touches >= 2:
            # Check if lows are generally rising
            lows = recent_data['low'].values
            if len(lows) >= 3 and lows[-1] > lows[0]:
                patterns.append({
                    'type': 'ascending_triangle',
                    'resistance_level': float(resistance_level),
                    'confidence': 0.6,
                    'breakout_target': float(resistance_level * 1.008)  # 0.8% above resistance
                })
        
        return patterns
    
    def _detect_retest_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect break and retest patterns"""
        patterns = []
        
        if len(df) < 10:
            return patterns
        
        recent_data = df.tail(10)
        
        # Look for recent breakouts followed by retests
        for i in range(2, len(recent_data) - 2):
            current_candle = recent_data.iloc[i]
            previous_resistance = recent_data.iloc[:i]['high'].max()
            
            # Check if current candle broke above resistance
            if current_candle['close'] > previous_resistance * 1.003:  # 0.3% breakout
                # Check if subsequent candles retested the breakout level
                post_break = recent_data.iloc[i+1:]
                retest_occurred = any(post_break['low'] <= previous_resistance * 1.001)
                
                if retest_occurred and post_break['close'].iloc[-1] > previous_resistance:
                    patterns.append({
                        'type': 'bullish_retest',
                        'breakout_level': float(previous_resistance),
                        'confidence': 0.8,
                        'target': float(previous_resistance * 1.015)  # 1.5% target
                    })
        
        return patterns

class HighFrequencySignalGenerator:
    """Generate high-frequency trading signals"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.chart_analyzer = RealTimeChartAnalyzer(config)
        
        # High-frequency parameters
        self.min_confidence = 0.6  # Lower threshold for more signals
        self.volume_confirmation = 1.2  # 1.2x volume (more sensitive)
        self.profit_targets = {
            'quick': 0.005,    # 0.5% quick scalp
            'normal': 0.01,    # 1% normal target
            'extended': 0.015  # 1.5% extended target
        }
        
    def generate_signals(self, symbol: str, price_update: Dict) -> List[HighFrequencySignal]:
        """Generate trading signals from real-time price updates"""
        signals = []
        
        # Update chart analysis
        self.chart_analyzer.update_price_data(symbol, price_update)
        
        current_price = price_update['price']
        micro_levels = self.chart_analyzer.micro_levels.get(symbol, [])
        chart_patterns = self.chart_analyzer.chart_patterns.get(symbol, [])
        
        # Generate signals from micro S/R levels
        sr_signals = self._generate_sr_signals(symbol, current_price, micro_levels)
        signals.extend(sr_signals)
        
        # Generate signals from chart patterns
        pattern_signals = self._generate_pattern_signals(symbol, current_price, chart_patterns)
        signals.extend(pattern_signals)
        
        # Sort by confidence and urgency
        signals.sort(key=lambda x: (x.urgency == 'IMMEDIATE', x.confidence), reverse=True)
        
        return signals[:3]  # Return top 3 signals
    
    def _generate_sr_signals(self, symbol: str, current_price: float, 
                           micro_levels: List[MicroSRLevel]) -> List[HighFrequencySignal]:
        """Generate signals from support/resistance levels"""
        signals = []
        
        for level in micro_levels:
            # Calculate distance to level
            distance = abs(current_price - level.price) / current_price
            
            # Check if price is near level
            if distance <= self.proximity_threshold:
                signal = self._create_sr_signal(symbol, current_price, level)
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _create_sr_signal(self, symbol: str, current_price: float, 
                         level: MicroSRLevel) -> Optional[HighFrequencySignal]:
        """Create trading signal from S/R level"""
        
        # Determine signal direction
        if level.level_type == 'support' and current_price <= level.price * 1.002:
            signal_type = 'BUY'
            entry_price = current_price
            stop_loss = level.price * 0.995  # 0.5% below support
            
            # Dynamic take profit based on level strength
            if level.next_target and level.next_target > current_price:
                take_profit = min(level.next_target * 0.998, current_price * 1.015)
            else:
                take_profit = current_price * (1 + self.profit_targets['normal'])
                
        elif level.level_type == 'resistance' and current_price >= level.price * 0.998:
            signal_type = 'SELL'
            entry_price = current_price
            stop_loss = level.price * 1.005  # 0.5% above resistance
            
            # Dynamic take profit
            if level.next_target and level.next_target < current_price:
                take_profit = max(level.next_target * 1.002, current_price * 0.985)
            else:
                take_profit = current_price * (1 - self.profit_targets['normal'])
        else:
            return None
        
        # Calculate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Minimum R:R filter
        if rr_ratio < 1.5:  # Lower R:R for high frequency
            return None
        
        # Calculate confidence
        confidence = self._calculate_signal_confidence(level, current_price)
        
        if confidence < self.min_confidence:
            return None
        
        # Determine urgency based on level strength and recent bounce
        if level.recent_bounce and level.strength > 0.8:
            urgency = 'IMMEDIATE'
        elif level.strength > 0.6:
            urgency = 'HIGH'
        else:
            urgency = 'MEDIUM'
        
        # Expected hold time based on timeframes confirmed
        if '30s' in level.timeframe_confirmed or '1m' in level.timeframe_confirmed:
            expected_hold = 10  # 10 minutes for micro scalps
        else:
            expected_hold = 25  # 25 minutes for regular scalps
        
        return HighFrequencySignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            urgency=urgency,
            timeframe='1m',
            level_strength=level.strength,
            volume_confirmation=level.volume_at_level,
            chart_pattern='support_resistance',
            expected_hold_minutes=expected_hold,
            risk_reward_ratio=rr_ratio,
            timestamp=datetime.now()
        )
    
    def _calculate_signal_confidence(self, level: MicroSRLevel, current_price: float) -> float:
        """Calculate confidence score for signal"""
        confidence = level.strength
        
        # Recent bounce bonus
        if level.recent_bounce:
            confidence += 0.15
        
        # Multiple timeframe confirmation bonus
        timeframe_bonus = len(level.timeframe_confirmed) * 0.05
        confidence += min(timeframe_bonus, 0.15)
        
        # Volume confirmation bonus
        if level.volume_at_level > 0:
            confidence += 0.1
        
        # Proximity bonus (closer to level = higher confidence)
        distance = abs(current_price - level.price) / current_price
        proximity_bonus = max(0, (self.proximity_threshold - distance) / self.proximity_threshold * 0.1)
        confidence += proximity_bonus
        
        return min(confidence, 1.0)
    
    def _generate_pattern_signals(self, symbol: str, current_price: float, 
                                chart_patterns: List[Dict]) -> List[HighFrequencySignal]:
        """Generate signals from chart patterns"""
        signals = []
        
        for pattern in chart_patterns:
            signal = self._create_pattern_signal(symbol, current_price, pattern)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _create_pattern_signal(self, symbol: str, current_price: float, 
                             pattern: Dict) -> Optional[HighFrequencySignal]:
        """Create signal from chart pattern"""
        
        pattern_type = pattern['type']
        
        if pattern_type == 'double_bottom':
            if current_price <= pattern['support_level'] * 1.002:
                return HighFrequencySignal(
                    symbol=symbol,
                    signal_type='BUY',
                    entry_price=current_price,
                    stop_loss=pattern['support_level'] * 0.995,
                    take_profit=pattern['target'],
                    confidence=pattern['confidence'],
                    urgency='HIGH',
                    timeframe='1m',
                    level_strength=0.7,
                    volume_confirmation=1.0,
                    chart_pattern='double_bottom',
                    expected_hold_minutes=20,
                    risk_reward_ratio=2.0,
                    timestamp=datetime.now()
                )
        
        elif pattern_type == 'bullish_retest':
            if current_price >= pattern['breakout_level'] * 0.999:
                return HighFrequencySignal(
                    symbol=symbol,
                    signal_type='BUY',
                    entry_price=current_price,
                    stop_loss=pattern['breakout_level'] * 0.997,
                    take_profit=pattern['target'],
                    confidence=pattern['confidence'],
                    urgency='IMMEDIATE',
                    timeframe='1m',
                    level_strength=0.8,
                    volume_confirmation=1.2,
                    chart_pattern='bullish_retest',
                    expected_hold_minutes=15,
                    risk_reward_ratio=2.5,
                    timestamp=datetime.now()
                )
        
        return None

class HighFrequencyTradingBot:
    """Main high-frequency trading bot"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.signal_generator = HighFrequencySignalGenerator(config)
        
        # High-frequency specific settings
        self.max_positions = 8  # More positions for HF trading
        self.position_size_usd = 200  # Smaller positions for quick scalps
        self.max_hold_minutes = 30  # Maximum 30 minute holds
        
        # WebSocket connections
        self.websocket_connections = {}
        self.price_feeds = {}
        
        # Position tracking
        self.active_positions = {}
        self.daily_trades = 0
        self.target_daily_trades = 25  # Target 25 trades per day
        
        # Real-time chart display
        self.chart_enabled = config.get('enable_charts', True)
        self.chart_symbols = config.get('chart_symbols', ['BTCUSDT', 'ETHUSDT'])
        
    async def start_high_frequency_trading(self):
        """Start the high-frequency trading system"""
        logger.info("Starting High-Frequency Support/Resistance Trading...")
        
        # Start WebSocket connections for all symbols
        await self._start_websocket_feeds()
        
        # Start chart analysis if enabled
        if self.chart_enabled:
            await self._start_chart_analysis()
        
        # Start main trading loop
        await self._hf_trading_loop()
    
    async def _start_websocket_feeds(self):
        """Start real-time WebSocket price feeds"""
        logger.info("Starting WebSocket price feeds...")
        
        for symbol in self.config['trading_symbols']:
            try:
                # Start individual WebSocket for each symbol
                asyncio.create_task(self._websocket_price_feed(symbol))
                logger.info(f"WebSocket feed started for {symbol}")
            except Exception as e:
                logger.error(f"Failed to start WebSocket for {symbol}: {e}")
    
    async def _websocket_price_feed(self, symbol: str):
        """Individual WebSocket connection for real-time prices"""
        url = f"wss://wbs.mexc.com/ws"
        
        # Subscribe to ticker and trade streams
        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [
                f"spot@public.miniTicker.v3.api@{symbol}",
                f"spot@public.deals.v3.api@{symbol}"
            ]
        }
        
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    # Send subscription
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    async for message in websocket:
                        data = json.loads(message)
                        await self._process_price_update(symbol, data)
                        
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)  # Reconnect after 5 seconds
    
    async def _process_price_update(self, symbol: str, data: Dict):
        """Process incoming price updates and generate signals"""
        try:
            # Extract price information
            if 'c' in data:  # Close price from ticker
                price_update = {
                    'price': float(data['c']),
                    'volume': float(data.get('v', 0)),
                    'timestamp': datetime.now()
                }
            elif 'p' in data:  # Price from trade data
                price_update = {
                    'price': float(data['p']),
                    'volume': float(data.get('v', 0)),
                    'timestamp': datetime.now()
                }
            else:
                return
            
            # Store latest price
            self.price_feeds[symbol] = price_update
            
            # Generate signals
            signals = self.signal_generator.generate_signals(symbol, price_update)
            
            # Process signals
            for signal in signals:
                await self._process_signal(signal)
                
        except Exception as e:
            logger.error(f"Error processing price update for {symbol}: {e}")
    
    async def _process_signal(self, signal: HighFrequencySignal):
        """Process and potentially execute trading signal"""
        try:
            # Check if we can take more positions
            if len(self.active_positions) >= self.max_positions:
                return
            
            # Check daily trade limit
            if self.daily_trades >= 50:  # Max 50 trades per day
                return
            
            # Risk validation
            if not self._validate_hf_signal(signal):
                return
            
            # Execute signal
            success = await self._execute_hf_signal(signal)
            
            if success:
                logger.info(f"HF Signal executed: {signal.symbol} {signal.signal_type} "
                          f"@ {signal.entry_price:.6f} (confidence: {signal.confidence:.2f}, "
                          f"urgency: {signal.urgency})")
                self.daily_trades += 1
                
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _validate_hf_signal(self, signal: HighFrequencySignal) -> bool:
        """Validate high-frequency signal"""
        
        # Confidence threshold
        if signal.confidence < 0.6:
            return False
        
        # Risk/reward minimum
        if signal.risk_reward_ratio < 1.5:
            return False
        
        # Check if we already have a position in this symbol
        symbol_positions = [p for p in self.active_positions.values() 
                          if p['symbol'] == signal.symbol]
        if len(symbol_positions) >= 2:  # Max 2 positions per symbol
            return False
        
        # Urgency filter (prioritize immediate and high urgency)
        if signal.urgency not in ['IMMEDIATE', 'HIGH'] and len(self.active_positions) > 4:
            return False
        
        return True
    
    async def _execute_hf_signal(self, signal: HighFrequencySignal) -> bool:
        """Execute high-frequency trading signal"""
        try:
            # Calculate position size
            position_size = self._calculate_hf_position_size(signal)
            
            if position_size == 0:
                return False
            
            # Simulate order execution (replace with actual MEXC API calls)
            order_result = await self._place_hf_order(signal, position_size)
            
            if order_result:
                # Store position
                position_id = f"hf_{signal.symbol}_{int(time.time())}"
                self.active_positions[position_id] = {
                    'symbol': signal.symbol,
                    'side': signal.signal_type,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'position_size': position_size,
                    'entry_time': datetime.now(),
                    'expected_hold_minutes': signal.expected_hold_minutes,
                    'chart_pattern': signal.chart_pattern,
                    'confidence': signal.confidence
                }
                
                return True
                
        except Exception as e:
            logger.error(f"Error executing HF signal: {e}")
        
        return False
    
    def _calculate_hf_position_size(self, signal: HighFrequencySignal) -> float:
        """Calculate position size for high-frequency trading"""
        
        # Base position size
        base_size_usd = self.position_size_usd
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (signal.confidence * 0.5)  # 0.5x to 1.0x
        
        # Adjust based on urgency
        urgency_multiplier = {
            'IMMEDIATE': 1.2,
            'HIGH': 1.0,
            'MEDIUM': 0.8
        }.get(signal.urgency, 0.8)
        
        # Adjust based on expected hold time (shorter = smaller size)
        hold_multiplier = min(signal.expected_hold_minutes / 20, 1.0)
        
        final_size_usd = base_size_usd * confidence_multiplier * urgency_multiplier * hold_multiplier
        
        # Convert to quantity
        position_quantity = final_size_usd / signal.entry_price
        
        return position_quantity
    
    async def _place_hf_order(self, signal: HighFrequencySignal, quantity: float) -> bool:
        """Place high-frequency order (placeholder for MEXC API integration)"""
        
        # This would be replaced with actual MEXC API calls
        logger.info(f"Placing HF order: {signal.symbol} {signal.signal_type} "
                   f"qty: {quantity:.6f} @ {signal.entry_price:.6f}")
        
        # Simulate successful order
        return True
    
    async def _hf_trading_loop(self):
        """Main high-frequency trading loop"""
        logger.info("Starting HF trading loop...")
        
        while True:
            try:
                # Manage existing positions
                await self._manage_hf_positions()
                
                # Update performance metrics
                self._update_hf_metrics()
                
                # Short sleep for high frequency
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in HF trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _manage_hf_positions(self):
        """Manage high-frequency positions"""
        positions_to_close = []
        
        for position_id, position in self.active_positions.items():
            try:
                # Get current price
                current_price = self._get_current_price(position['symbol'])
                if not current_price:
                    continue
                
                # Check exit conditions
                should_close, reason = self._check_hf_exit_conditions(position, current_price)
                
                if should_close:
                    positions_to_close.append((position_id, reason, current_price))
                    
            except Exception as e:
                logger.error(f"Error managing position {position_id}: {e}")
        
        # Close positions
        for position_id, reason, exit_price in positions_to_close:
            await self._close_hf_position(position_id, reason, exit_price)
    
    def _check_hf_exit_conditions(self, position: Dict, current_price: float) -> Tuple[bool, str]:
        """Check exit conditions for high-frequency positions"""
        
        # Stop loss check
        if position['side'] == 'BUY' and current_price <= position['stop_loss']:
            return True, 'STOP_LOSS'
        elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
            return True, 'STOP_LOSS'
        
        # Take profit check
        if position['side'] == 'BUY' and current_price >= position['take_profit']:
            return True, 'TAKE_PROFIT'
        elif position['side'] == 'SELL' and current_price <= position['take_profit']:
            return True, 'TAKE_PROFIT'
        
        # Time-based exit (more aggressive for HF)
        position_age = datetime.now() - position['entry_time']
        if position_age.total_seconds() / 60 > position['expected_hold_minutes']:
            return True, 'TIME_LIMIT'
        
        # Quick profit taking for micro scalps
        if position['chart_pattern'] == 'support_resistance':
            profit_threshold = 0.003  # 0.3% quick profit
            if position['side'] == 'BUY' and current_price >= position['entry_price'] * (1 + profit_threshold):
                return True, 'QUICK_PROFIT'
            elif position['side'] == 'SELL' and current_price <= position['entry_price'] * (1 - profit_threshold):
                return True, 'QUICK_PROFIT'
        
        return False, ''
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from price feeds"""
        price_data = self.price_feeds.get(symbol)
        return price_data['price'] if price_data else None
    
    async def _close_hf_position(self, position_id: str, reason: str, exit_price: float):
        """Close high-frequency position"""
        try:
            position = self.active_positions[position_id]
            
            # Calculate P&L
            pnl = self._calculate_hf_pnl(position, exit_price)
            
            # Log trade
            logger.info(f"HF Position closed: {position['symbol']} {reason} "
                       f"P&L: {pnl:.4f} (Hold time: "
                       f"{(datetime.now() - position['entry_time']).total_seconds()/60:.1f}m)")
            
            # Remove position
            del self.active_positions[position_id]
            
        except Exception as e:
            logger.error(f"Error closing HF position {position_id}: {e}")
    
    def _calculate_hf_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate P&L for high-frequency position"""
        entry_price = position['entry_price']
        size = position['position_size']
        
        if position['side'] == 'BUY':
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size
        
        # MEXC fees (optimized for high frequency)
        entry_fee = entry_price * size * 0.0000  # 0% maker fee
        exit_fee = exit_price * size * 0.0005   # 0.05% taker fee
        
        net_pnl = gross_pnl - entry_fee - exit_fee
        return net_pnl
    
    def _update_hf_metrics(self):
        """Update high-frequency trading metrics"""
        # Real-time performance tracking
        active_count = len(self.active_positions)
        
        # Log metrics every 100 trades
        if self.daily_trades > 0 and self.daily_trades % 10 == 0:
            logger.info(f"HF Metrics: {self.daily_trades} trades today, "
                       f"{active_count} active positions")
    
    async def _start_chart_analysis(self):
        """Start real-time chart analysis and display"""
        if not self.chart_enabled:
            return
        
        logger.info("Starting real-time chart analysis...")
        
        # This would start a real-time chart display
        # For now, just log that charts are available
        for symbol in self.chart_symbols:
            logger.info(f"Real-time chart analysis active for {symbol}")

def main():
    """Main entry point for high-frequency trading"""
    
    # High-frequency configuration
    config = {
        'trading_symbols': [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT'
        ],
        'enable_charts': True,
        'chart_symbols': ['BTCUSDT', 'ETHUSDT'],
        'MEXC_API_KEY': 'your_api_key',
        'MEXC_SECRET_KEY': 'your_secret_key'
    }
    
    # Create and start bot
    bot = HighFrequencyTradingBot(config)
    
    try:
        asyncio.run(bot.start_high_frequency_trading())
    except KeyboardInterrupt:
        logger.info("High-frequency trading stopped by user")
    except Exception as e:
        logger.error(f"HF trading error: {e}")

if __name__ == "__main__":
    main() 