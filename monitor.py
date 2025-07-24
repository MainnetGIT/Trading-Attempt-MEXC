#!/usr/bin/env python3
"""
MEXC Trading Bot Monitoring System
Real-time monitoring, alerting, and performance tracking
"""

import sqlite3
import time
import smtplib
import json
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

class BotMonitor:
    """Trading bot monitoring and alerting system"""
    
    def __init__(self, db_path: str = 'mexc_bot_data.db'):
        self.db_path = db_path
        self.alerts_sent = set()
        
        # Monitoring thresholds
        self.max_daily_loss = 0.10  # 10%
        self.max_drawdown = 0.15    # 15%
        self.min_win_rate = 0.30    # 30%
        self.max_open_positions = 5
        
        # Email settings (configure if needed)
        self.email_enabled = False
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_user = ""
        self.email_pass = ""
        self.alert_email = ""
        
        # Telegram settings (configure if needed)
        self.telegram_enabled = False
        self.telegram_token = ""
        self.telegram_chat_id = ""
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_recent_trades(self, hours: int = 24) -> List[Dict]:
        """Get trades from the last N hours"""
        conn = self.get_connection()
        
        since = datetime.now() - timedelta(hours=hours)
        
        query = """
        SELECT * FROM trades 
        WHERE timestamp > ? 
        ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=[since])
        conn.close()
        
        return df.to_dict('records') if not df.empty else []
    
    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_win': 0,
                'max_loss': 0
            }
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        profits = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        profit_factor = abs(sum(profits) / sum(losses)) if losses else float('inf')
        
        max_win = max(profits) if profits else 0
        max_loss = min(losses) if losses else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss
        }
    
    def check_daily_performance(self) -> Dict:
        """Check today's trading performance"""
        today_trades = self.get_recent_trades(24)
        metrics = self.calculate_performance_metrics(today_trades)
        
        # Check for alerts
        alerts = []
        
        # Daily loss alert
        if metrics['total_pnl'] < 0:
            loss_pct = abs(metrics['total_pnl']) / 10000  # Assuming 10k balance
            if loss_pct > self.max_daily_loss:
                alerts.append(f"Daily loss limit exceeded: {loss_pct:.1%}")
        
        # Win rate alert
        if metrics['total_trades'] >= 10 and metrics['win_rate'] < self.min_win_rate:
            alerts.append(f"Low win rate: {metrics['win_rate']:.1%}")
        
        return {
            'metrics': metrics,
            'alerts': alerts
        }
    
    def check_system_health(self) -> Dict:
        """Check overall system health"""
        alerts = []
        
        try:
            # Check if bot is running (recent activity)
            recent_trades = self.get_recent_trades(1)  # Last hour
            
            # Check log file for errors
            try:
                with open('mexc_bot.log', 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-100:]  # Last 100 lines
                    
                    error_count = sum(1 for line in recent_lines if 'ERROR' in line)
                    if error_count > 5:
                        alerts.append(f"High error count in logs: {error_count}")
            
            except FileNotFoundError:
                alerts.append("Log file not found - bot may not be running")
            
            # Check database integrity
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            if not tables:
                alerts.append("Database tables missing")
            
            conn.close()
            
        except Exception as e:
            alerts.append(f"System health check failed: {e}")
        
        return {
            'status': 'healthy' if not alerts else 'warning',
            'alerts': alerts
        }
    
    def send_email_alert(self, subject: str, message: str):
        """Send email alert"""
        if not self.email_enabled or not self.alert_email:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.alert_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_pass)
            
            text = msg.as_string()
            server.sendmail(self.email_user, self.alert_email, text)
            server.quit()
            
            print(f"Email alert sent: {subject}")
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
    
    def send_telegram_alert(self, message: str):
        """Send Telegram alert"""
        if not self.telegram_enabled or not self.telegram_token:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                print("Telegram alert sent")
            else:
                print(f"Failed to send Telegram alert: {response.status_code}")
                
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")
    
    def send_alert(self, alert_type: str, message: str):
        """Send alert via configured channels"""
        alert_key = f"{alert_type}_{hash(message)}"
        
        # Avoid duplicate alerts
        if alert_key in self.alerts_sent:
            return
        
        self.alerts_sent.add(alert_key)
        
        subject = f"MEXC Bot Alert: {alert_type}"
        
        # Send via email
        if self.email_enabled:
            self.send_email_alert(subject, message)
        
        # Send via Telegram
        if self.telegram_enabled:
            self.send_telegram_alert(f"<b>{subject}</b>\n\n{message}")
        
        print(f"Alert sent: {alert_type} - {message}")
    
    def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        today_trades = self.get_recent_trades(24)
        metrics = self.calculate_performance_metrics(today_trades)
        
        report = f"""
MEXC Trading Bot - Daily Report
Date: {datetime.now().strftime('%Y-%m-%d')}

Performance Metrics:
- Total Trades: {metrics['total_trades']}
- Winning Trades: {metrics['winning_trades']}
- Win Rate: {metrics['win_rate']:.1%}
- Total PnL: ${metrics['total_pnl']:.2f}
- Average Profit: ${metrics['avg_profit']:.2f}
- Average Loss: ${metrics['avg_loss']:.2f}
- Profit Factor: {metrics['profit_factor']:.2f}
- Best Trade: ${metrics['max_win']:.2f}
- Worst Trade: ${metrics['max_loss']:.2f}

Strategy Breakdown:
"""
        
        # Add strategy breakdown
        if today_trades:
            strategies = {}
            for trade in today_trades:
                strategy = trade['strategy']
                if strategy not in strategies:
                    strategies[strategy] = {'trades': 0, 'pnl': 0}
                strategies[strategy]['trades'] += 1
                strategies[strategy]['pnl'] += trade['pnl']
            
            for strategy, data in strategies.items():
                report += f"- {strategy}: {data['trades']} trades, ${data['pnl']:.2f} PnL\n"
        
        return report
    
    def create_performance_chart(self, days: int = 7):
        """Create performance chart"""
        trades = self.get_recent_trades(days * 24)
        
        if not trades:
            print("No trades data for chart")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate cumulative PnL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Create chart
        plt.figure(figsize=(12, 8))
        
        # Cumulative PnL
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['cumulative_pnl'], linewidth=2)
        plt.title('Cumulative PnL')
        plt.ylabel('PnL ($)')
        plt.grid(True, alpha=0.3)
        
        # Trade distribution
        plt.subplot(2, 1, 2)
        profits = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] < 0]['pnl']
        
        plt.hist([profits, losses], bins=20, label=['Profits', 'Losses'], 
                color=['green', 'red'], alpha=0.7)
        plt.title('Trade Distribution')
        plt.xlabel('PnL ($)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'performance_chart_{days}d.png', dpi=300, bbox_inches='tight')
        print(f"Chart saved to performance_chart_{days}d.png")
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        print(f"\n=== Monitoring Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        # Check daily performance
        daily_check = self.check_daily_performance()
        metrics = daily_check['metrics']
        
        print(f"Today's Performance:")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  PnL: ${metrics['total_pnl']:.2f}")
        
        # Send alerts if needed
        for alert in daily_check['alerts']:
            self.send_alert("Performance Warning", alert)
        
        # Check system health
        health_check = self.check_system_health()
        print(f"System Status: {health_check['status']}")
        
        for alert in health_check['alerts']:
            self.send_alert("System Alert", alert)
        
        print("=== Monitoring Cycle Complete ===")
    
    def run_continuous_monitoring(self, interval_minutes: int = 30):
        """Run continuous monitoring"""
        print(f"Starting continuous monitoring (every {interval_minutes} minutes)")
        
        try:
            while True:
                self.run_monitoring_cycle()
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Monitoring error: {e}")
            self.send_alert("System Error", f"Monitoring stopped due to error: {e}")

def main():
    """Main monitoring function"""
    monitor = BotMonitor()
    
    # Configure alerts (uncomment and set up as needed)
    # monitor.email_enabled = True
    # monitor.email_user = "your_email@gmail.com"
    # monitor.email_pass = "your_app_password"
    # monitor.alert_email = "alerts@yourdomain.com"
    
    # monitor.telegram_enabled = True
    # monitor.telegram_token = "your_bot_token"
    # monitor.telegram_chat_id = "your_chat_id"
    
    print("MEXC Trading Bot Monitor")
    print("Options:")
    print("1. Run single monitoring cycle")
    print("2. Generate daily report")
    print("3. Create performance chart")
    print("4. Start continuous monitoring")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        monitor.run_monitoring_cycle()
    
    elif choice == '2':
        report = monitor.generate_daily_report()
        print(report)
        
        # Save report
        with open(f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
            f.write(report)
        print("Report saved to file")
    
    elif choice == '3':
        days = int(input("Number of days for chart (default 7): ") or "7")
        monitor.create_performance_chart(days)
    
    elif choice == '4':
        interval = int(input("Monitoring interval in minutes (default 30): ") or "30")
        monitor.run_continuous_monitoring(interval)
    
    else:
        print("Invalid option")

if __name__ == "__main__":
    main() 