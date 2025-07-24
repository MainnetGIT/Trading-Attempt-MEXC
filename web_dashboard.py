#!/usr/bin/env python3
"""
MEXC Trading Bot Web Dashboard
Real-time web interface with charts, performance metrics, and monitoring
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3
import json
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime, timedelta
import threading
import time
import asyncio
import logging
from typing import Dict, List
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mexc_trading_bot_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

logger = logging.getLogger(__name__)

class TradingDashboard:
    """Real-time trading dashboard"""
    
    def __init__(self):
        self.db_path = 'mexc_bot_data.db'
        self.last_update = datetime.now()
        self.performance_data = {}
        self.active_positions = []
        self.recent_signals = []
        
        # Start background data collection
        self.start_data_collection()
    
    def start_data_collection(self):
        """Start background thread for data collection"""
        def collect_data():
            while True:
                try:
                    self.update_dashboard_data()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Dashboard data collection error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=collect_data, daemon=True)
        thread.start()
    
    def update_dashboard_data(self):
        """Update all dashboard data"""
        try:
            # Update performance metrics
            self.performance_data = self.get_performance_metrics()
            
            # Update positions
            self.active_positions = self.get_active_positions()
            
            # Update recent signals
            self.recent_signals = self.get_recent_signals()
            
            # Emit updates to connected clients
            socketio.emit('dashboard_update', {
                'performance': self.performance_data,
                'positions': self.active_positions,
                'signals': self.recent_signals,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Today's performance
            today = datetime.now().date()
            today_trades = pd.read_sql_query(
                "SELECT * FROM trades WHERE DATE(timestamp) = ?",
                conn, params=[today]
            )
            
            # This week's performance
            week_ago = datetime.now() - timedelta(days=7)
            week_trades = pd.read_sql_query(
                "SELECT * FROM trades WHERE timestamp >= ?",
                conn, params=[week_ago]
            )
            
            # Calculate metrics
            today_pnl = today_trades['pnl'].sum() if not today_trades.empty else 0
            week_pnl = week_trades['pnl'].sum() if not week_trades.empty else 0
            
            today_trades_count = len(today_trades)
            week_trades_count = len(week_trades)
            
            # Win rate calculation
            winning_trades_today = len(today_trades[today_trades['pnl'] > 0]) if not today_trades.empty else 0
            win_rate_today = (winning_trades_today / today_trades_count * 100) if today_trades_count > 0 else 0
            
            winning_trades_week = len(week_trades[week_trades['pnl'] > 0]) if not week_trades.empty else 0
            win_rate_week = (winning_trades_week / week_trades_count * 100) if week_trades_count > 0 else 0
            
            conn.close()
            
            return {
                'today_pnl': round(today_pnl, 2),
                'week_pnl': round(week_pnl, 2),
                'today_trades': today_trades_count,
                'week_trades': week_trades_count,
                'win_rate_today': round(win_rate_today, 1),
                'win_rate_week': round(win_rate_week, 1),
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {
                'today_pnl': 0,
                'week_pnl': 0,
                'today_trades': 0,
                'week_trades': 0,
                'win_rate_today': 0,
                'win_rate_week': 0,
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
    
    def get_active_positions(self) -> List[Dict]:
        """Get current active positions"""
        # This would normally come from the trading bot's position tracker
        # For now, return mock data
        return [
            {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'size': 0.005,
                'entry_price': 45250.50,
                'current_price': 45380.25,
                'pnl': 0.65,
                'pnl_pct': 0.29,
                'duration': '12:34'
            },
            {
                'symbol': 'ETHUSDT',
                'side': 'SELL',
                'size': 0.1,
                'entry_price': 2845.75,
                'current_price': 2838.90,
                'pnl': 0.68,
                'pnl_pct': 0.24,
                'duration': '08:22'
            }
        ]
    
    def get_recent_signals(self) -> List[Dict]:
        """Get recent trading signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get last 10 signals
            signals = pd.read_sql_query(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10",
                conn
            )
            
            conn.close()
            
            return signals.to_dict('records') if not signals.empty else []
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    def get_chart_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Dict:
        """Get chart data for plotting"""
        # This would normally fetch from your data source
        # For now, return mock data structure
        import numpy as np
        
        # Generate mock OHLCV data
        times = pd.date_range(
            start=datetime.now() - timedelta(hours=limit),
            end=datetime.now(),
            freq='1H'
        )
        
        # Mock price data
        base_price = 45000 if symbol == 'BTCUSDT' else 2800
        price_data = base_price + np.cumsum(np.random.randn(len(times)) * 50)
        
        ohlc_data = []
        for i, (time, price) in enumerate(zip(times, price_data)):
            high = price + np.random.uniform(0, 100)
            low = price - np.random.uniform(0, 100)
            open_price = price_data[i-1] if i > 0 else price
            
            ohlc_data.append({
                'time': time.isoformat(),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': np.random.uniform(100, 1000)
            })
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': ohlc_data
        }

# Create dashboard instance
dashboard = TradingDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance data"""
    return jsonify(dashboard.performance_data)

@app.route('/api/positions')
def api_positions():
    """API endpoint for positions data"""
    return jsonify(dashboard.active_positions)

@app.route('/api/signals')
def api_signals():
    """API endpoint for signals data"""
    return jsonify(dashboard.recent_signals)

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """API endpoint for chart data"""
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', 100))
    
    chart_data = dashboard.get_chart_data(symbol, timeframe, limit)
    return jsonify(chart_data)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected to MEXC Trading Bot Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Handle manual update request"""
    dashboard.update_dashboard_data()

def create_app():
    """Create Flask app with templates directory"""
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    return app

if __name__ == '__main__':
    print("🚀 Starting MEXC Trading Bot Web Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔄 Real-time updates every 5 seconds")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 