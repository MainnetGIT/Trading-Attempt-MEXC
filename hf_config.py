#!/usr/bin/env python3
"""
High-Frequency Support/Resistance Configuration
Optimized for 10-50 trades per day with precise chart analysis
"""

import os
from typing import Dict, List

class HighFrequencyConfig:
    """Configuration for high-frequency S/R scalping"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self) -> Dict:
        """Load high-frequency trading configuration"""
        
        return {
            # MEXC API Configuration
            'MEXC_API_KEY': os.getenv('MEXC_API_KEY', ''),
            'MEXC_SECRET_KEY': os.getenv('MEXC_SECRET_KEY', ''),
            'MEXC_BASE_URL': 'https://api.mexc.com',
            'MEXC_WS_URL': 'wss://wbs.mexc.com/ws',
            
            # High-Frequency Strategy Parameters
            'proximity_threshold': 0.001,      # 0.1% proximity (very tight)
            'volume_threshold': 1.2,           # 1.2x volume confirmation (sensitive)
            'min_confidence': 0.6,             # Lower threshold for more signals
            'min_rr_ratio': 1.5,              # Lower R:R for HF trading
            
            # Profit Targets (Aggressive Scalping)
            'profit_targets': {
                'micro_scalp': 0.003,          # 0.3% micro scalp
                'quick_scalp': 0.005,          # 0.5% quick scalp
                'normal_scalp': 0.01,          # 1.0% normal scalp
                'extended_scalp': 0.015        # 1.5% extended scalp
            },
            
            # Position Management (High Frequency)
            'max_positions': 8,                # More concurrent positions
            'position_size_usd': 150,          # Smaller positions for quick scalps
            'max_hold_minutes': 30,            # Maximum 30-minute holds
            'quick_exit_threshold': 0.003,     # 0.3% quick profit taking
            
            # Risk Management (Aggressive but Safe)
            'max_risk_per_trade': 0.008,       # 0.8% risk per trade
            'max_daily_loss': 0.025,           # 2.5% daily loss limit
            'max_correlation_positions': 2,     # Max 2 correlated positions
            'emergency_stop_loss': 0.005,      # 0.5% emergency stop
            
            # Timeframe Analysis (Multi-timeframe)
            'primary_timeframe': '1m',         # Primary analysis timeframe
            'confirmation_timeframes': ['30s', '3m', '5m'],
            'chart_update_seconds': 1,         # Update charts every second
            'data_retention_hours': 2,         # Keep 2 hours of data
            
            # Trading Pairs (High Liquidity + More Opportunities)
            'trading_symbols': [
                # Tier 1: Highest liquidity (>$100M daily)
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
                
                # Tier 2: High liquidity (>$50M daily)
                'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT',
                
                # Tier 3: Good liquidity (>$25M daily)
                'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                
                # Tier 4: Additional opportunities (>$15M daily)
                'UNIUSDT', 'LTCUSDT', 'ATOMUSDT',
                
                # Tier 5: More opportunities (>$10M daily)
                'FTMUSDT', 'NEARUSDT', 'ICPUSDT'
            ],
            
            # S/R Detection Parameters (High Sensitivity)
            'sr_detection': {
                'lookback_periods': {
                    '30s': 60,    # 30 minutes of 30s data
                    '1m': 60,     # 1 hour of 1m data
                    '3m': 40,     # 2 hours of 3m data
                    '5m': 24      # 2 hours of 5m data
                },
                'level_tolerance': 0.001,       # 0.1% clustering tolerance
                'min_touches': 2,               # Minimum level touches
                'volume_confirmation': True,    # Require volume confirmation
                'fibonacci_levels': True,       # Include Fibonacci levels
                'pivot_points': True,           # Include pivot points
                'vwap_levels': True,           # Include VWAP levels
                'support_resistance_strength': 0.5  # Minimum level strength
            },
            
            # Chart Pattern Detection
            'pattern_detection': {
                'double_tops_bottoms': True,
                'triangles': True,
                'break_retests': True,
                'flag_pennants': True,
                'head_shoulders': False,        # Too slow for HF
                'wedges': True,
                'channels': True
            },
            
            # Volume Analysis
            'volume_analysis': {
                'spike_threshold': 1.5,         # 1.5x average volume
                'volume_ma_period': 20,         # 20-period volume MA
                'volume_breakout_confirm': True,
                'unusual_volume_alert': 2.0     # 2x volume alert
            },
            
            # Technical Indicators (Fast Settings)
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2,
                'ema_periods': [9, 21, 50]      # Fast EMAs for HF
            },
            
            # WebSocket Configuration
            'websocket': {
                'ping_interval': 20,            # Ping every 20 seconds
                'ping_timeout': 10,             # 10 second timeout
                'close_timeout': 10,            # 10 second close timeout
                'reconnect_delay': 5,           # 5 second reconnect delay
                'max_reconnect_attempts': 10    # Maximum reconnection attempts
            },
            
            # Performance Targets (High Frequency)
            'performance_targets': {
                'daily_trades_min': 10,         # Minimum 10 trades/day
                'daily_trades_target': 25,      # Target 25 trades/day
                'daily_trades_max': 50,         # Maximum 50 trades/day
                'target_win_rate': 0.65,        # 65% win rate target
                'target_daily_return': 0.02,    # 2% daily return target
                'max_daily_drawdown': 0.015,    # 1.5% max daily drawdown
                'target_profit_factor': 2.0     # 2.0 profit factor target
            },
            
            # Order Execution (Fast Execution)
            'execution': {
                'order_type': 'MARKET',         # Market orders for speed
                'slippage_tolerance': 0.001,    # 0.1% slippage tolerance
                'execution_timeout': 5,         # 5 second execution timeout
                'retry_attempts': 3,            # Retry failed orders 3 times
                'partial_fill_timeout': 10      # 10 second partial fill timeout
            },
            
            # Risk Controls (Automated)
            'risk_controls': {
                'position_correlation_check': True,
                'volatility_filter': True,
                'news_event_filter': False,     # Manual for HF
                'market_hours_only': False,     # Trade 24/7
                'min_spread_check': True,
                'max_spread_bps': 20,          # 20 bps maximum spread
                'circuit_breaker': True,
                'max_consecutive_losses': 5    # Stop after 5 consecutive losses
            },
            
            # Monitoring & Alerts
            'monitoring': {
                'real_time_dashboard': True,
                'performance_alerts': True,
                'risk_alerts': True,
                'trade_notifications': True,
                'daily_summary': True,
                'log_level': 'INFO',
                'max_log_size_mb': 100
            },
            
            # Chart Display (Real-time Analysis)
            'chart_display': {
                'enabled': True,
                'symbols_to_chart': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                'update_frequency_seconds': 5,
                'timeframes_displayed': ['1m', '5m'],
                'indicators_overlay': ['EMA9', 'EMA21', 'SR_LEVELS'],
                'pattern_overlay': True,
                'volume_display': True,
                'save_screenshots': True
            },
            
            # Backtesting (Validation)
            'backtesting': {
                'enabled': True,
                'lookback_days': 7,            # Test on last 7 days
                'commission': 0.0005,          # 0.05% MEXC taker fee
                'slippage': 0.0001,           # 0.01% slippage assumption
                'initial_capital': 10000,      # $10k test capital
                'position_sizing': 'fixed'     # Fixed position sizing
            },
            
            # Database Configuration
            'database': {
                'db_path': 'hf_trading_data.db',
                'backup_frequency_hours': 6,
                'max_db_size_mb': 500,
                'archive_old_data': True,
                'keep_data_days': 30
            },
            
            # API Rate Limits (MEXC Specific)
            'api_limits': {
                'requests_per_second': 10,      # MEXC limit
                'orders_per_second': 5,         # Order rate limit
                'weight_per_minute': 6000,      # API weight limit
                'burst_allowance': 20           # Burst request allowance
            }
        }
    
    def get_symbol_specific_config(self, symbol: str) -> Dict:
        """Get symbol-specific configuration"""
        
        # Tier-based configuration
        tier_configs = {
            # Tier 1: Most liquid pairs - aggressive settings
            'tier_1': {
                'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                'position_size_multiplier': 1.2,
                'confidence_threshold': 0.55,
                'proximity_threshold': 0.0008,
                'max_positions_per_symbol': 2
            },
            
            # Tier 2: High liquid pairs - standard settings  
            'tier_2': {
                'symbols': ['ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT'],
                'position_size_multiplier': 1.0,
                'confidence_threshold': 0.6,
                'proximity_threshold': 0.001,
                'max_positions_per_symbol': 2
            },
            
            # Tier 3: Good liquid pairs - conservative settings
            'tier_3': {
                'symbols': ['AVAXUSDT', 'MATICUSDT', 'LINKUSDT'],
                'position_size_multiplier': 0.8,
                'confidence_threshold': 0.65,
                'proximity_threshold': 0.0012,
                'max_positions_per_symbol': 1
            },
            
            # Tier 4: Additional pairs - very conservative
            'tier_4': {
                'symbols': ['UNIUSDT', 'LTCUSDT', 'ATOMUSDT', 'FTMUSDT', 'NEARUSDT', 'ICPUSDT'],
                'position_size_multiplier': 0.6,
                'confidence_threshold': 0.7,
                'proximity_threshold': 0.0015,
                'max_positions_per_symbol': 1
            }
        }
        
        # Find which tier the symbol belongs to
        for tier_name, tier_config in tier_configs.items():
            if symbol in tier_config['symbols']:
                return tier_config
        
        # Default configuration for unlisted symbols
        return tier_configs['tier_4']
    
    def get_market_session_config(self) -> Dict:
        """Get configuration based on market session"""
        from datetime import datetime
        
        current_hour = datetime.utcnow().hour
        
        # Market session configurations
        sessions = {
            'asian_session': {
                'hours': list(range(0, 8)),
                'volume_threshold_multiplier': 1.1,
                'confidence_bonus': 0.05,
                'max_positions_multiplier': 0.8
            },
            'london_session': {
                'hours': list(range(8, 16)),
                'volume_threshold_multiplier': 1.0,
                'confidence_bonus': 0.0,
                'max_positions_multiplier': 1.0
            },
            'ny_session': {
                'hours': list(range(16, 24)),
                'volume_threshold_multiplier': 0.9,
                'confidence_bonus': 0.0,
                'max_positions_multiplier': 1.2
            }
        }
        
        # Determine current session
        for session_name, session_config in sessions.items():
            if current_hour in session_config['hours']:
                return session_config
        
        return sessions['asian_session']  # Default
    
    def get_volatility_adjusted_config(self, symbol: str, volatility: float) -> Dict:
        """Get volatility-adjusted configuration"""
        
        adjustments = {}
        
        if volatility < 0.01:  # Low volatility (< 1%)
            adjustments = {
                'proximity_threshold_multiplier': 0.8,
                'profit_target_multiplier': 0.8,
                'position_size_multiplier': 1.2,
                'confidence_threshold_adjustment': -0.05
            }
        elif volatility > 0.03:  # High volatility (> 3%)
            adjustments = {
                'proximity_threshold_multiplier': 1.3,
                'profit_target_multiplier': 1.2,
                'position_size_multiplier': 0.7,
                'confidence_threshold_adjustment': 0.1
            }
        else:  # Normal volatility
            adjustments = {
                'proximity_threshold_multiplier': 1.0,
                'profit_target_multiplier': 1.0,
                'position_size_multiplier': 1.0,
                'confidence_threshold_adjustment': 0.0
            }
        
        return adjustments

# Global configuration instance
HF_CONFIG = HighFrequencyConfig().load_config()

def get_hf_config() -> Dict:
    """Get high-frequency trading configuration"""
    return HF_CONFIG

def update_config(updates: Dict):
    """Update configuration with new values"""
    global HF_CONFIG
    HF_CONFIG.update(updates)

def get_symbol_config(symbol: str) -> Dict:
    """Get symbol-specific configuration"""
    config_instance = HighFrequencyConfig()
    return config_instance.get_symbol_specific_config(symbol)

def get_session_config() -> Dict:
    """Get current market session configuration"""
    config_instance = HighFrequencyConfig()
    return config_instance.get_market_session_config()

def validate_config() -> bool:
    """Validate configuration settings"""
    
    required_keys = [
        'MEXC_API_KEY',
        'trading_symbols',
        'proximity_threshold',
        'volume_threshold',
        'max_positions',
        'profit_targets'
    ]
    
    for key in required_keys:
        if key not in HF_CONFIG or not HF_CONFIG[key]:
            print(f"❌ Missing required configuration: {key}")
            return False
    
    # Validate API credentials
    if not HF_CONFIG['MEXC_API_KEY'] or not HF_CONFIG['MEXC_SECRET_KEY']:
        print("❌ MEXC API credentials not configured")
        return False
    
    # Validate symbol list
    if len(HF_CONFIG['trading_symbols']) < 5:
        print("❌ Need at least 5 trading symbols for diversification")
        return False
    
    # Validate risk parameters
    if HF_CONFIG['max_risk_per_trade'] > 0.02:  # >2%
        print("❌ Risk per trade too high for HF trading")
        return False
    
    print("✅ Configuration validation passed")
    return True

if __name__ == "__main__":
    # Test configuration
    config = get_hf_config()
    
    print("🔧 High-Frequency Trading Configuration")
    print("=" * 50)
    print(f"Trading Symbols: {len(config['trading_symbols'])}")
    print(f"Max Positions: {config['max_positions']}")
    print(f"Proximity Threshold: {config['proximity_threshold']*100}%")
    print(f"Daily Trade Target: {config['performance_targets']['daily_trades_target']}")
    print(f"Max Hold Time: {config['max_hold_minutes']} minutes")
    
    # Validate configuration
    validate_config() 