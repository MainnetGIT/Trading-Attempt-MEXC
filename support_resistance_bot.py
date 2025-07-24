#!/usr/bin/env python3
"""
MEXC Support/Resistance Scalping Bot
Implementation of the comprehensive trading system outlined in the development guide
Focus: 2-5% daily returns through systematic S/R scalping with AI optimization
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
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import threading
from dataclasses import dataclass
from collections import deque
import sqlite3
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sr_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SupportResistanceLevel:
    """Support/Resistance level data structure"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-1 confidence score
    touches: int     # Number of times price touched this level
    volume_confirmation: float  # Volume at this level
    last_touch: datetime
    age_score: float  # How recent/relevant the level is

@dataclass
class SRSignal:
    """Support/Resistance trading signal"""
    symbol: str
    signal_type: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    level: SupportResistanceLevel
    risk_reward_ratio: float
    position_size: float
    timestamp: datetime
    market_conditions: Dict

class SupportResistanceAnalyzer:
    """Advanced Support/Resistance detection and analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_touches = 2
        self.level_tolerance = 0.002  # 0.2% tolerance for level clustering
        self.min_strength = 0.6  # Minimum confidence for valid levels
        
    def calculate_support_resistance_levels(self, df: pd.DataFrame, 
                                          lookback_period: int = 50) -> Dict[str, List[SupportResistanceLevel]]:
        """
        Advanced S/R calculation with multiple validation methods
        """
        if len(df) < lookback_period:
            return {'support': [], 'resistance': []}
        
        # Method 1: Local extrema detection
        highs, lows = self._find_local_extrema(df, lookback_period)
        
        # Method 2: Volume-weighted levels
        volume_levels = self._calculate_volume_weighted_levels(df)
        
        # Method 3: Fibonacci retracements
        fib_levels = self._calculate_fibonacci_levels(df)
        
        # Method 4: Historical test strength
        all_levels = highs + lows + volume_levels + fib_levels
        validated_levels = self._validate_level_strength(all_levels, df)
        
        # Cluster and rank levels
        support_levels, resistance_levels = self._cluster_and_rank_levels(
            validated_levels, df['close'].iloc[-1]
        )
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def _find_local_extrema(self, df: pd.DataFrame, lookback: int) -> Tuple[List[float], List[float]]:
        """Find local highs and lows using scipy.signal.find_peaks"""
        recent_data = df.tail(lookback)
        
        # Find peaks (resistance levels)
        high_peaks, _ = find_peaks(recent_data['high'].values, 
                                   distance=5,  # Minimum 5 periods between peaks
                                   prominence=recent_data['high'].std() * 0.5)
        
        # Find valleys (support levels) by inverting the data
        low_peaks, _ = find_peaks(-recent_data['low'].values,
                                  distance=5,
                                  prominence=recent_data['low'].std() * 0.5)
        
        highs = recent_data['high'].iloc[high_peaks].tolist()
        lows = recent_data['low'].iloc[low_peaks].tolist()
        
        return highs, lows
    
    def _calculate_volume_weighted_levels(self, df: pd.DataFrame) -> List[float]:
        """Calculate volume-weighted price levels"""
        # Use VWAP clusters as potential S/R levels
        recent_data = df.tail(100)  # Last 100 periods
        
        # Calculate volume-weighted prices
        vwap_data = []
        for i in range(len(recent_data)):
            row = recent_data.iloc[i]
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            vw_price = typical_price * row['volume']
            vwap_data.append(vw_price)
        
        # Find volume clusters using DBSCAN
        if len(vwap_data) > 10:
            X = np.array(vwap_data).reshape(-1, 1)
            clustering = DBSCAN(eps=np.std(vwap_data) * 0.1, min_samples=3).fit(X)
            
            # Get cluster centers as potential levels
            levels = []
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Ignore noise points
                    cluster_points = [vwap_data[i] for i, label in enumerate(clustering.labels_) if label == cluster_id]
                    levels.append(np.mean(cluster_points))
            
            return levels
        
        return []
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> List[float]:
        """Calculate Fibonacci retracement levels"""
        recent_data = df.tail(100)
        if len(recent_data) < 20:
            return []
        
        # Find swing high and low
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        diff = swing_high - swing_low
        
        # Key Fibonacci levels
        fib_levels = [
            swing_high - diff * 0.236,  # 23.6% retracement
            swing_high - diff * 0.382,  # 38.2% retracement
            swing_high - diff * 0.500,  # 50% retracement
            swing_high - diff * 0.618,  # 61.8% retracement
            swing_high - diff * 0.786   # 78.6% retracement
        ]
        
        return [level for level in fib_levels if swing_low <= level <= swing_high]
    
    def _validate_level_strength(self, levels: List[float], df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Validate and score level strength based on historical interactions"""
        validated_levels = []
        current_time = datetime.now()
        
        for price_level in levels:
            touches = 0
            total_volume = 0
            last_touch = None
            level_type = None
            
            # Check historical interactions with this level
            for i, row in df.iterrows():
                # Check if price touched this level (within tolerance)
                tolerance = price_level * self.level_tolerance
                
                if abs(row['low'] - price_level) <= tolerance:
                    touches += 1
                    total_volume += row['volume']
                    last_touch = i
                    level_type = 'support'
                elif abs(row['high'] - price_level) <= tolerance:
                    touches += 1
                    total_volume += row['volume']
                    last_touch = i
                    level_type = 'resistance'
            
            # Only include levels with minimum touches
            if touches >= self.min_touches and level_type:
                # Calculate strength score
                strength = min(touches / 5.0, 1.0)  # Normalize to 0-1
                
                # Volume confirmation score
                avg_volume = df['volume'].mean()
                volume_score = min(total_volume / (touches * avg_volume), 2.0) / 2.0
                
                # Age score (more recent = higher score)
                if last_touch:
                    days_ago = len(df) - last_touch
                    age_score = max(0, 1 - (days_ago / 100))  # Decay over 100 periods
                else:
                    age_score = 0
                
                # Combined confidence score
                confidence = (strength * 0.4 + volume_score * 0.3 + age_score * 0.3)
                
                if confidence >= self.min_strength:
                    sr_level = SupportResistanceLevel(
                        price=price_level,
                        level_type=level_type,
                        strength=confidence,
                        touches=touches,
                        volume_confirmation=total_volume,
                        last_touch=current_time if last_touch else current_time,
                        age_score=age_score
                    )
                    validated_levels.append(sr_level)
        
        return validated_levels
    
    def _cluster_and_rank_levels(self, levels: List[SupportResistanceLevel], 
                                current_price: float) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Cluster nearby levels and rank by strength"""
        if not levels:
            return [], []
        
        # Separate support and resistance
        support_levels = [l for l in levels if l.level_type == 'support' and l.price < current_price]
        resistance_levels = [l for l in levels if l.level_type == 'resistance' and l.price > current_price]
        
        # Cluster nearby levels
        support_levels = self._cluster_nearby_levels(support_levels)
        resistance_levels = self._cluster_nearby_levels(resistance_levels)
        
        # Sort by strength and proximity to current price
        support_levels.sort(key=lambda x: (x.strength, -abs(current_price - x.price)), reverse=True)
        resistance_levels.sort(key=lambda x: (x.strength, -abs(current_price - x.price)), reverse=True)
        
        # Return top levels only
        return support_levels[:5], resistance_levels[:5]
    
    def _cluster_nearby_levels(self, levels: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
        """Merge levels that are very close to each other"""
        if len(levels) <= 1:
            return levels
        
        clustered = []
        used_indices = set()
        
        for i, level in enumerate(levels):
            if i in used_indices:
                continue
            
            # Find nearby levels
            cluster_levels = [level]
            used_indices.add(i)
            
            for j, other_level in enumerate(levels[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if abs(level.price - other_level.price) / level.price <= self.level_tolerance:
                    cluster_levels.append(other_level)
                    used_indices.add(j)
            
            # Merge cluster into single level
            if len(cluster_levels) > 1:
                avg_price = np.mean([l.price for l in cluster_levels])
                total_strength = np.mean([l.strength for l in cluster_levels])
                total_touches = sum([l.touches for l in cluster_levels])
                total_volume = sum([l.volume_confirmation for l in cluster_levels])
                
                merged_level = SupportResistanceLevel(
                    price=avg_price,
                    level_type=level.level_type,
                    strength=total_strength,
                    touches=total_touches,
                    volume_confirmation=total_volume,
                    last_touch=max([l.last_touch for l in cluster_levels]),
                    age_score=np.mean([l.age_score for l in cluster_levels])
                )
                clustered.append(merged_level)
            else:
                clustered.append(level)
        
        return clustered

class SRSignalGenerator:
    """Generate trading signals based on support/resistance analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_rr_ratio = 2.0  # Minimum 2:1 risk/reward
        self.proximity_threshold = 0.005  # Within 0.5% of level
        self.volume_threshold = 1.5  # 1.5x average volume
        
    def generate_signals(self, symbol: str, df: pd.DataFrame, 
                        sr_levels: Dict[str, List[SupportResistanceLevel]]) -> List[SRSignal]:
        """Generate trading signals based on current price action and S/R levels"""
        signals = []
        
        if len(df) < 20:
            return signals
        
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        # Analyze current market conditions
        market_conditions = self._analyze_market_conditions(df)
        
        # Check for long signals near support
        for support in sr_levels['support']:
            signal = self._check_long_signal(
                symbol, current_price, current_volume, avg_volume,
                support, sr_levels['resistance'], market_conditions
            )
            if signal:
                signals.append(signal)
        
        # Check for short signals near resistance
        for resistance in sr_levels['resistance']:
            signal = self._check_short_signal(
                symbol, current_price, current_volume, avg_volume,
                resistance, sr_levels['support'], market_conditions
            )
            if signal:
                signals.append(signal)
        
        # Sort by confidence and return best signals
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals[:3]  # Max 3 signals per symbol
    
    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Analyze current market conditions"""
        if len(df) < 20:
            return {}
        
        recent_data = df.tail(20)
        
        # Volatility analysis
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(288)  # Annualized for 5-min data
        
        # Volume analysis
        avg_volume = recent_data['volume'].mean()
        current_volume = recent_data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Trend analysis
        sma_short = recent_data['close'].rolling(5).mean().iloc[-1]
        sma_long = recent_data['close'].rolling(10).mean().iloc[-1]
        
        if sma_short > sma_long * 1.002:
            trend = 'BULLISH'
        elif sma_short < sma_long * 0.998:
            trend = 'BEARISH'
        else:
            trend = 'SIDEWAYS'
        
        # Range analysis
        high_20 = recent_data['high'].max()
        low_20 = recent_data['low'].min()
        range_size = (high_20 - low_20) / recent_data['close'].iloc[-1]
        
        return {
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'trend': trend,
            'range_size': range_size,
            'is_high_volume': volume_ratio > self.volume_threshold
        }
    
    def _check_long_signal(self, symbol: str, current_price: float, current_volume: float,
                          avg_volume: float, support: SupportResistanceLevel,
                          resistance_levels: List[SupportResistanceLevel],
                          market_conditions: Dict) -> Optional[SRSignal]:
        """Check for valid long signal near support"""
        
        # Distance from support level
        distance_to_support = (current_price - support.price) / current_price
        
        # Must be close to support but not below it
        if not (0 <= distance_to_support <= self.proximity_threshold):
            return None
        
        # Volume confirmation
        volume_confirmed = current_volume > avg_volume * self.volume_threshold
        
        # Find nearest resistance for take profit
        valid_resistance = [r for r in resistance_levels if r.price > current_price * 1.01]
        if not valid_resistance:
            return None
        
        nearest_resistance = min(valid_resistance, key=lambda x: abs(x.price - current_price))
        
        # Calculate trade parameters
        entry_price = current_price
        stop_loss = support.price * 0.992  # 0.8% below support
        take_profit = min(nearest_resistance.price * 0.998, current_price * 1.025)  # Max 2.5% target
        
        # Risk/reward validation
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < self.min_rr_ratio:
            return None
        
        # Calculate confidence score
        confidence = self._calculate_signal_confidence(
            support, distance_to_support, volume_confirmed, 
            rr_ratio, market_conditions, 'LONG'
        )
        
        if confidence < 0.7:  # Minimum confidence threshold
            return None
        
        return SRSignal(
            symbol=symbol,
            signal_type='BUY',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            level=support,
            risk_reward_ratio=rr_ratio,
            position_size=0,  # Will be calculated by risk manager
            timestamp=datetime.now(),
            market_conditions=market_conditions
        )
    
    def _check_short_signal(self, symbol: str, current_price: float, current_volume: float,
                           avg_volume: float, resistance: SupportResistanceLevel,
                           support_levels: List[SupportResistanceLevel],
                           market_conditions: Dict) -> Optional[SRSignal]:
        """Check for valid short signal near resistance"""
        
        # Distance from resistance level
        distance_to_resistance = (resistance.price - current_price) / current_price
        
        # Must be close to resistance but not above it
        if not (0 <= distance_to_resistance <= self.proximity_threshold):
            return None
        
        # Volume confirmation
        volume_confirmed = current_volume > avg_volume * self.volume_threshold
        
        # Find nearest support for take profit
        valid_support = [s for s in support_levels if s.price < current_price * 0.99]
        if not valid_support:
            return None
        
        nearest_support = max(valid_support, key=lambda x: x.price)
        
        # Calculate trade parameters
        entry_price = current_price
        stop_loss = resistance.price * 1.008  # 0.8% above resistance
        take_profit = max(nearest_support.price * 1.002, current_price * 0.975)  # Max 2.5% target
        
        # Risk/reward validation
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < self.min_rr_ratio:
            return None
        
        # Calculate confidence score
        confidence = self._calculate_signal_confidence(
            resistance, distance_to_resistance, volume_confirmed,
            rr_ratio, market_conditions, 'SHORT'
        )
        
        if confidence < 0.7:  # Minimum confidence threshold
            return None
        
        return SRSignal(
            symbol=symbol,
            signal_type='SELL',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            level=resistance,
            risk_reward_ratio=rr_ratio,
            position_size=0,  # Will be calculated by risk manager
            timestamp=datetime.now(),
            market_conditions=market_conditions
        )
    
    def _calculate_signal_confidence(self, level: SupportResistanceLevel, 
                                   distance: float, volume_confirmed: bool,
                                   rr_ratio: float, market_conditions: Dict,
                                   signal_type: str) -> float:
        """Calculate overall signal confidence score"""
        
        # Base confidence from level strength
        confidence = level.strength
        
        # Proximity bonus (closer = better)
        proximity_score = max(0, 1 - (distance / self.proximity_threshold))
        confidence += proximity_score * 0.1
        
        # Volume confirmation bonus
        if volume_confirmed:
            confidence += 0.1
        
        # Risk/reward bonus
        rr_bonus = min((rr_ratio - self.min_rr_ratio) * 0.1, 0.1)
        confidence += rr_bonus
        
        # Market condition adjustments
        trend = market_conditions.get('trend', 'SIDEWAYS')
        if signal_type == 'BUY' and trend == 'BULLISH':
            confidence += 0.05
        elif signal_type == 'SELL' and trend == 'BEARISH':
            confidence += 0.05
        elif trend == 'SIDEWAYS':  # Ideal for S/R trading
            confidence += 0.1
        
        # Volatility adjustment (moderate volatility is best)
        volatility = market_conditions.get('volatility', 0.5)
        if 0.3 <= volatility <= 0.8:  # Sweet spot for S/R trading
            confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 1.0

class SRRiskManager:
    """Risk management specifically for S/R trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_risk_per_trade = 0.015  # 1.5% per trade
        self.max_daily_loss = 0.03  # 3% daily loss limit
        self.max_positions = 5
        self.max_correlation = 0.7
        
    def calculate_position_size(self, account_balance: float, signal: SRSignal) -> float:
        """Calculate optimal position size based on risk management"""
        
        # Risk amount for this trade
        risk_amount = account_balance * self.max_risk_per_trade
        
        # Price risk
        price_risk = abs(signal.entry_price - signal.stop_loss)
        
        if price_risk == 0:
            return 0
        
        # Position size based on risk
        position_size = risk_amount / price_risk
        
        # Apply additional constraints
        max_position_value = account_balance * 0.15  # Max 15% per position
        max_size_by_value = max_position_value / signal.entry_price
        
        # MEXC minimum order size check
        min_order_size = 50 / signal.entry_price  # $50 minimum
        
        final_size = min(position_size, max_size_by_value)
        
        return max(final_size, min_order_size) if final_size >= min_order_size else 0
    
    def validate_signal(self, signal: SRSignal, current_positions: List[Dict],
                       account_balance: float, daily_pnl: float) -> Tuple[bool, str]:
        """Comprehensive signal validation"""
        
        # Daily loss limit check
        if daily_pnl < 0 and abs(daily_pnl) >= account_balance * self.max_daily_loss:
            return False, "Daily loss limit reached"
        
        # Maximum positions check
        if len(current_positions) >= self.max_positions:
            return False, "Maximum positions limit reached"
        
        # Position sizing validation
        position_size = self.calculate_position_size(account_balance, signal)
        if position_size == 0:
            return False, "Position size too small or invalid"
        
        # Correlation check (avoid too many correlated positions)
        if self._check_correlation_risk(signal, current_positions):
            return False, "High correlation risk with existing positions"
        
        # Risk/reward validation
        if signal.risk_reward_ratio < 2.0:
            return False, "Risk/reward ratio too low"
        
        # Confidence threshold
        if signal.confidence < 0.7:
            return False, "Signal confidence too low"
        
        return True, "Signal approved"
    
    def _check_correlation_risk(self, signal: SRSignal, current_positions: List[Dict]) -> bool:
        """Check if new position would create excessive correlation"""
        
        # Simple correlation check based on symbols
        crypto_correlations = {
            'BTC': ['BTC'],
            'ETH': ['ETH'],
            'BNB': ['BNB', 'BSC'],
            'ADA': ['ADA'],
            'SOL': ['SOL'],
            'XRP': ['XRP'],
            'DOT': ['DOT'],
            'AVAX': ['AVAX'],
            'MATIC': ['MATIC', 'POLYGON'],
            'LINK': ['LINK']
        }
        
        signal_crypto = None
        for crypto, symbols in crypto_correlations.items():
            if any(symbol in signal.symbol for symbol in symbols):
                signal_crypto = crypto
                break
        
        if not signal_crypto:
            return False  # Unknown crypto, allow trade
        
        # Count existing positions in same crypto family
        correlated_positions = 0
        for position in current_positions:
            for crypto, symbols in crypto_correlations.items():
                if crypto == signal_crypto and any(symbol in position['symbol'] for symbol in symbols):
                    correlated_positions += 1
        
        # Allow maximum 2 positions in same crypto family
        return correlated_positions >= 2

class SRTradingBot:
    """Main Support/Resistance Trading Bot"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sr_analyzer = SupportResistanceAnalyzer(config)
        self.signal_generator = SRSignalGenerator(config)
        self.risk_manager = SRRiskManager(config)
        
        # MEXC API credentials
        self.api_key = config.get('MEXC_API_KEY', '')
        self.secret_key = config.get('MEXC_SECRET_KEY', '')
        self.base_url = 'https://api.mexc.com'
        
        # Trading state
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.running = False
        
        # Data storage
        self.market_data = {}
        self.sr_levels = {}
        
        # Performance tracking
        self.setup_database()
        
        # Trading pairs (high liquidity pairs as specified)
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT'
        ]
    
    def setup_database(self):
        """Setup database for tracking trades and performance"""
        self.conn = sqlite3.connect('sr_trading_data.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                signal_type TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL,
                pnl REAL,
                pnl_percentage REAL,
                confidence REAL,
                risk_reward_ratio REAL,
                exit_reason TEXT,
                level_type TEXT,
                level_strength REAL,
                market_conditions TEXT,
                entry_time DATETIME,
                exit_time DATETIME,
                hold_duration_minutes INTEGER
            )
        ''')
        
        # Support/Resistance levels table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sr_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                price REAL,
                level_type TEXT,
                strength REAL,
                touches INTEGER,
                volume_confirmation REAL,
                identified_time DATETIME
            )
        ''')
        
        # Daily performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                win_rate REAL,
                daily_pnl REAL,
                daily_return_pct REAL,
                max_drawdown REAL,
                avg_hold_time_minutes REAL,
                best_trade REAL,
                worst_trade REAL
            )
        ''')
        
        self.conn.commit()
    
    async def trading_loop(self):
        """Main trading loop implementing the S/R strategy"""
        logger.info("Starting Support/Resistance trading loop...")
        
        while self.running:
            try:
                # 1. Update market data for all symbols
                await self._update_market_data()
                
                # 2. Analyze S/R levels for each symbol
                await self._analyze_sr_levels()
                
                # 3. Generate and evaluate signals
                signals = await self._generate_signals()
                
                # 4. Execute approved trades
                await self._execute_signals(signals)
                
                # 5. Manage existing positions
                await self._manage_positions()
                
                # 6. Update performance metrics
                self._update_performance_metrics()
                
                # 7. Check risk limits
                if self._check_risk_limits():
                    logger.warning("Risk limits exceeded, pausing trading")
                    break
                
                # Wait before next cycle (30 seconds for 5-minute strategy)
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _update_market_data(self):
        """Update market data for all trading symbols"""
        for symbol in self.symbols:
            try:
                # Get recent kline data (5-minute intervals)
                klines = await self._get_klines(symbol, '5m', 200)
                if klines:
                    df = self._klines_to_dataframe(klines)
                    self.market_data[symbol] = df
                    
            except Exception as e:
                logger.error(f"Error updating market data for {symbol}: {e}")
    
    async def _analyze_sr_levels(self):
        """Analyze support/resistance levels for all symbols"""
        for symbol, df in self.market_data.items():
            try:
                levels = self.sr_analyzer.calculate_support_resistance_levels(df)
                self.sr_levels[symbol] = levels
                
                # Log significant levels
                logger.info(f"{symbol}: {len(levels['support'])} support, {len(levels['resistance'])} resistance levels")
                
            except Exception as e:
                logger.error(f"Error analyzing S/R levels for {symbol}: {e}")
    
    async def _generate_signals(self) -> List[SRSignal]:
        """Generate trading signals for all symbols"""
        all_signals = []
        
        for symbol in self.symbols:
            if symbol in self.market_data and symbol in self.sr_levels:
                try:
                    df = self.market_data[symbol]
                    levels = self.sr_levels[symbol]
                    
                    signals = self.signal_generator.generate_signals(symbol, df, levels)
                    all_signals.extend(signals)
                    
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol}: {e}")
        
        # Sort by confidence and return top signals
        all_signals.sort(key=lambda x: x.confidence, reverse=True)
        return all_signals[:10]  # Top 10 signals
    
    async def _execute_signals(self, signals: List[SRSignal]):
        """Execute approved trading signals"""
        account_balance = await self._get_account_balance()
        
        for signal in signals:
            try:
                # Validate signal
                is_valid, reason = self.risk_manager.validate_signal(
                    signal, list(self.active_positions.values()), 
                    account_balance, self.daily_pnl
                )
                
                if not is_valid:
                    logger.info(f"Signal rejected for {signal.symbol}: {reason}")
                    continue
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(account_balance, signal)
                signal.position_size = position_size
                
                # Execute trade
                success = await self._place_order(signal)
                if success:
                    logger.info(f"Trade executed: {signal.symbol} {signal.signal_type} "
                              f"@ {signal.entry_price} (confidence: {signal.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    async def _manage_positions(self):
        """Manage existing positions"""
        positions_to_close = []
        
        for position_id, position in self.active_positions.items():
            try:
                # Get current price
                current_price = await self._get_current_price(position['symbol'])
                if not current_price:
                    continue
                
                # Check exit conditions
                exit_signal = self._check_exit_conditions(position, current_price)
                
                if exit_signal:
                    positions_to_close.append((position_id, exit_signal))
                
            except Exception as e:
                logger.error(f"Error managing position {position_id}: {e}")
        
        # Close positions that need to be closed
        for position_id, exit_signal in positions_to_close:
            await self._close_position(position_id, exit_signal)
    
    def _check_exit_conditions(self, position: Dict, current_price: float) -> Optional[Dict]:
        """Check if position should be closed"""
        
        # Stop loss check
        if position['side'] == 'BUY' and current_price <= position['stop_loss']:
            return {'reason': 'STOP_LOSS', 'price': current_price}
        elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
            return {'reason': 'STOP_LOSS', 'price': current_price}
        
        # Take profit check
        if position['side'] == 'BUY' and current_price >= position['take_profit']:
            return {'reason': 'TAKE_PROFIT', 'price': current_price}
        elif position['side'] == 'SELL' and current_price <= position['take_profit']:
            return {'reason': 'TAKE_PROFIT', 'price': current_price}
        
        # Time-based exit (max 4 hours as specified)
        position_age = datetime.now() - position['entry_time']
        if position_age > timedelta(hours=4):
            return {'reason': 'TIME_LIMIT', 'price': current_price}
        
        # Support/resistance break check
        if self._detect_level_break(position, current_price):
            return {'reason': 'LEVEL_BREAK', 'price': current_price}
        
        return None
    
    def _detect_level_break(self, position: Dict, current_price: float) -> bool:
        """Detect if support/resistance level has been broken"""
        level_price = position.get('level_price', 0)
        level_type = position.get('level_type', '')
        
        if not level_price:
            return False
        
        # Check for significant break (more than 0.5% beyond level)
        break_threshold = 0.005
        
        if level_type == 'support' and current_price < level_price * (1 - break_threshold):
            return True
        elif level_type == 'resistance' and current_price > level_price * (1 + break_threshold):
            return True
        
        return False
    
    async def _place_order(self, signal: SRSignal) -> bool:
        """Place order on MEXC"""
        try:
            # Prepare order parameters
            side = 'BUY' if signal.signal_type == 'BUY' else 'SELL'
            
            # Place market order for immediate execution
            order_params = {
                'symbol': signal.symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': signal.position_size
            }
            
            # Make API request
            result = await self._make_api_request('POST', '/api/v3/order', order_params)
            
            if result and 'orderId' in result:
                # Store position information
                self.active_positions[result['orderId']] = {
                    'symbol': signal.symbol,
                    'side': side,
                    'entry_price': signal.entry_price,
                    'position_size': signal.position_size,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'level_price': signal.level.price,
                    'level_type': signal.level.level_type,
                    'confidence': signal.confidence,
                    'risk_reward_ratio': signal.risk_reward_ratio,
                    'entry_time': datetime.now(),
                    'market_conditions': signal.market_conditions
                }
                
                self.daily_trades += 1
                return True
        
        except Exception as e:
            logger.error(f"Error placing order: {e}")
        
        return False
    
    async def _close_position(self, position_id: str, exit_signal: Dict):
        """Close position and record trade"""
        try:
            position = self.active_positions[position_id]
            
            # Place closing order
            side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            
            order_params = {
                'symbol': position['symbol'],
                'side': side,
                'type': 'MARKET',
                'quantity': position['position_size']
            }
            
            result = await self._make_api_request('POST', '/api/v3/order', order_params)
            
            if result:
                # Calculate P&L
                exit_price = exit_signal['price']
                pnl = self._calculate_pnl(position, exit_price)
                
                # Record trade
                await self._record_trade(position, exit_price, exit_signal['reason'], pnl)
                
                # Update daily P&L
                self.daily_pnl += pnl
                
                # Remove from active positions
                del self.active_positions[position_id]
                
                logger.info(f"Position closed: {position['symbol']} P&L: {pnl:.4f}")
        
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate position P&L including fees"""
        entry_price = position['entry_price']
        size = position['position_size']
        
        if position['side'] == 'BUY':
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size
        
        # MEXC fees (0% maker, 0.05% taker as specified)
        entry_fee = entry_price * size * 0.0005  # Market order = taker
        exit_fee = exit_price * size * 0.0005    # Market order = taker
        
        net_pnl = gross_pnl - entry_fee - exit_fee
        return net_pnl
    
    async def _record_trade(self, position: Dict, exit_price: float, exit_reason: str, pnl: float):
        """Record completed trade in database"""
        try:
            cursor = self.conn.cursor()
            
            entry_time = position['entry_time']
            exit_time = datetime.now()
            hold_duration = int((exit_time - entry_time).total_seconds() / 60)  # minutes
            
            pnl_percentage = (pnl / (position['entry_price'] * position['position_size'])) * 100
            
            cursor.execute('''
                INSERT INTO trades (
                    symbol, signal_type, entry_price, exit_price, stop_loss, take_profit,
                    position_size, pnl, pnl_percentage, confidence, risk_reward_ratio,
                    exit_reason, level_type, level_strength, market_conditions,
                    entry_time, exit_time, hold_duration_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position['symbol'], position['side'], position['entry_price'], exit_price,
                position['stop_loss'], position['take_profit'], position['position_size'],
                pnl, pnl_percentage, position['confidence'], position['risk_reward_ratio'],
                exit_reason, position['level_type'], 0.0,  # level_strength placeholder
                json.dumps(position['market_conditions']), entry_time, exit_time, hold_duration
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    # API and utility methods
    async def _make_api_request(self, method: str, endpoint: str, params: Dict) -> Dict:
        """Make authenticated API request to MEXC"""
        try:
            timestamp = int(time.time() * 1000)
            params['timestamp'] = timestamp
            params['recvWindow'] = 5000
            
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            headers = {'X-MEXC-APIKEY': self.api_key}
            url = f"{self.base_url}{endpoint}"
            
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=params, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {}
        
        except Exception as e:
            logger.error(f"API request error: {e}")
            return {}
    
    async def _get_klines(self, symbol: str, interval: str, limit: int) -> List:
        """Get kline data from MEXC"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        # This is a public endpoint, no auth required
        try:
            url = f"{self.base_url}/api/v3/klines"
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get klines for {symbol}: {response.status_code}")
                return []
        
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []
    
    def _klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """Convert kline data to pandas DataFrame"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Convert data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')
    
    async def _get_account_balance(self) -> float:
        """Get USDT account balance"""
        try:
            result = await self._make_api_request('GET', '/api/v3/account', {})
            
            if result and 'balances' in result:
                for balance in result['balances']:
                    if balance['asset'] == 'USDT':
                        return float(balance['free'])
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # Use public ticker endpoint
            url = f"{self.base_url}/api/v3/ticker/price"
            response = requests.get(url, params={'symbol': symbol})
            
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _update_performance_metrics(self):
        """Update real-time performance metrics"""
        # This would update dashboard metrics, send alerts, etc.
        pass
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits have been exceeded"""
        # Daily loss limit
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= 1000 * self.risk_manager.max_daily_loss:
            return True
        
        # Maximum positions
        if len(self.active_positions) >= self.risk_manager.max_positions:
            return True
        
        return False
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Support/Resistance Trading Bot...")
        self.running = True
        
        try:
            asyncio.run(self.trading_loop())
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
        finally:
            self.running = False
            self.conn.close()

def main():
    """Main entry point"""
    # Configuration
    config = {
        'MEXC_API_KEY': 'your_api_key_here',
        'MEXC_SECRET_KEY': 'your_secret_key_here'
    }
    
    # Create and start bot
    bot = SRTradingBot(config)
    bot.start()

if __name__ == "__main__":
    main() 