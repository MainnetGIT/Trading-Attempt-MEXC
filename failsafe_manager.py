#!/usr/bin/env python3
"""
Failsafe Manager - Critical System Protection
Handles emergency situations, system failures, and position safety
"""

import asyncio
import logging
import json
import time
import threading
import queue
import os
import signal
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class SystemStatus(Enum):
    """System status enumeration"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"
    SHUTDOWN = "SHUTDOWN"

@dataclass
class FailsafeEvent:
    """Failsafe event data"""
    event_id: str
    event_type: str
    alert_level: AlertLevel
    description: str
    context: Dict
    timestamp: datetime
    handled: bool = False
    resolution: str = ""

@dataclass
class EmergencyPosition:
    """Emergency position data for failsafe closure"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    position_id: str
    priority: int = 1  # 1=highest, 5=lowest

class FailsafeManager:
    """Comprehensive failsafe and emergency management system"""
    
    def __init__(self, config: Dict, execution_engine):
        self.config = config
        self.execution_engine = execution_engine
        
        # System monitoring
        self.system_status = SystemStatus.HEALTHY
        self.last_heartbeat = datetime.now()
        self.heartbeat_interval = 30  # seconds
        self.max_heartbeat_delay = 90  # seconds
        
        # Emergency state
        self.emergency_mode = False
        self.emergency_triggered_at = None
        self.emergency_reason = ""
        
        # Position tracking for emergency closure
        self.emergency_positions = []
        self.position_lock = threading.Lock()
        
        # Failsafe event queue
        self.event_queue = queue.Queue()
        self.event_handlers = {}
        
        # Backup and recovery
        self.backup_directory = Path("failsafe_backups")
        self.backup_directory.mkdir(exist_ok=True)
        
        # System resource monitoring
        self.resource_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'disk_percent': 95.0,
            'network_errors': 10
        }
        
        # Network connectivity monitoring
        self.connectivity_failures = 0
        self.max_connectivity_failures = 5
        
        # Database for persistence
        self.failsafe_db = self._setup_failsafe_database()
        
        # Register signal handlers
        self._setup_signal_handlers()
        
        # Register default event handlers
        self._register_default_handlers()
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        logger.info("Failsafe Manager initialized")
    
    def _setup_failsafe_database(self) -> sqlite3.Connection:
        """Setup failsafe database for persistence"""
        db_path = self.backup_directory / "failsafe.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failsafe_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                event_type TEXT,
                alert_level TEXT,
                description TEXT,
                context TEXT,
                timestamp DATETIME,
                handled BOOLEAN,
                resolution TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergency_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT UNIQUE,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                entry_price REAL,
                current_price REAL,
                unrealized_pnl REAL,
                entry_time DATETIME,
                emergency_timestamp DATETIME,
                closed BOOLEAN DEFAULT FALSE,
                closure_method TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_type TEXT,
                timestamp DATETIME,
                data TEXT,
                checksum TEXT
            )
        ''')
        
        conn.commit()
        return conn
    
    def _setup_signal_handlers(self):
        """Setup system signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.critical(f"Received signal {signum} - initiating emergency shutdown")
            asyncio.create_task(self.emergency_shutdown(f"System signal {signum}"))
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)   # Hangup
    
    def _register_default_handlers(self):
        """Register default event handlers"""
        self.register_handler('network_failure', self._handle_network_failure)
        self.register_handler('api_failure', self._handle_api_failure)
        self.register_handler('execution_failure', self._handle_execution_failure)
        self.register_handler('position_timeout', self._handle_position_timeout)
        self.register_handler('system_resource_critical', self._handle_resource_critical)
        self.register_handler('data_corruption', self._handle_data_corruption)
        self.register_handler('emergency_stop', self._handle_emergency_stop)
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads"""
        # Event processing thread
        threading.Thread(target=self._process_events, daemon=True).start()
        
        # System health monitoring
        threading.Thread(target=self._monitor_system_health, daemon=True).start()
        
        # Network connectivity monitoring
        threading.Thread(target=self._monitor_connectivity, daemon=True).start()
        
        # Position safety monitoring
        threading.Thread(target=self._monitor_position_safety, daemon=True).start()
        
        # Heartbeat monitoring
        threading.Thread(target=self._monitor_heartbeat, daemon=True).start()
        
        # Backup scheduling
        threading.Thread(target=self._backup_scheduler, daemon=True).start()
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type] = handler
        logger.debug(f"Registered handler for event type: {event_type}")
    
    def trigger_event(self, event_type: str, alert_level: AlertLevel, 
                     description: str, context: Dict = None):
        """Trigger a failsafe event"""
        event = FailsafeEvent(
            event_id=f"{event_type}_{int(time.time())}",
            event_type=event_type,
            alert_level=alert_level,
            description=description,
            context=context or {},
            timestamp=datetime.now()
        )
        
        # Log event
        logger.log(
            self._alert_level_to_log_level(alert_level),
            f"Failsafe Event: {event_type} - {description}"
        )
        
        # Store in database
        self._store_event(event)
        
        # Add to processing queue
        self.event_queue.put(event)
        
        # Immediate critical/emergency handling
        if alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            asyncio.create_task(self._handle_critical_event(event))
    
    def add_emergency_position(self, position: EmergencyPosition):
        """Add position to emergency closure list"""
        with self.position_lock:
            # Check if position already exists
            existing = next((p for p in self.emergency_positions if p.position_id == position.position_id), None)
            if existing:
                # Update existing position
                existing.current_price = position.current_price
                existing.unrealized_pnl = position.unrealized_pnl
            else:
                # Add new position
                self.emergency_positions.append(position)
                
                # Store in database
                self._store_emergency_position(position)
        
        logger.info(f"Added emergency position: {position.symbol} {position.side} {position.quantity}")
    
    def remove_emergency_position(self, position_id: str):
        """Remove position from emergency list"""
        with self.position_lock:
            self.emergency_positions = [p for p in self.emergency_positions if p.position_id != position_id]
        
        # Update database
        cursor = self.failsafe_db.cursor()
        cursor.execute(
            "UPDATE emergency_positions SET closed = TRUE WHERE position_id = ?",
            (position_id,)
        )
        self.failsafe_db.commit()
        
        logger.info(f"Removed emergency position: {position_id}")
    
    async def emergency_shutdown(self, reason: str):
        """Emergency system shutdown with position protection"""
        logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")
        
        self.emergency_mode = True
        self.emergency_triggered_at = datetime.now()
        self.emergency_reason = reason
        self.system_status = SystemStatus.EMERGENCY
        
        try:
            # 1. Immediate position closure
            await self._emergency_close_all_positions()
            
            # 2. Cancel all pending orders
            await self._cancel_all_pending_orders()
            
            # 3. Create system snapshot
            await self._create_emergency_snapshot()
            
            # 4. Send emergency notifications
            await self._send_emergency_notifications(reason)
            
            # 5. Safe system state preservation
            self._preserve_system_state()
            
            logger.critical("Emergency shutdown completed")
            
        except Exception as e:
            logger.critical(f"Emergency shutdown failed: {e}")
            # Last resort - kill process
            os._exit(1)
    
    async def _emergency_close_all_positions(self):
        """Emergency closure of all positions"""
        logger.critical("Closing all positions in emergency mode")
        
        with self.position_lock:
            positions = self.emergency_positions.copy()
        
        if not positions:
            logger.info("No emergency positions to close")
            return
        
        # Sort by priority (highest first)
        positions.sort(key=lambda x: x.priority)
        
        # Close positions with maximum redundancy
        closure_tasks = []
        for position in positions:
            task = asyncio.create_task(self._emergency_close_single_position(position))
            closure_tasks.append(task)
        
        # Wait for all closures with timeout
        try:
            await asyncio.wait_for(asyncio.gather(*closure_tasks, return_exceptions=True), timeout=30)
        except asyncio.TimeoutError:
            logger.critical("Emergency position closure timed out")
    
    async def _emergency_close_single_position(self, position: EmergencyPosition):
        """Emergency closure of single position with multiple methods"""
        logger.critical(f"Emergency closing: {position.symbol} {position.side} {position.quantity}")
        
        # Method 1: Standard emergency close
        try:
            response = await self.execution_engine.emergency_close_position(
                position.symbol, position.side, position.quantity
            )
            if response.success:
                self.remove_emergency_position(position.position_id)
                return
        except Exception as e:
            logger.error(f"Standard emergency close failed: {e}")
        
        # Method 2: Market order with aggressive pricing
        try:
            await self._aggressive_market_close(position)
            return
        except Exception as e:
            logger.error(f"Aggressive market close failed: {e}")
        
        # Method 3: Partial closure
        try:
            await self._partial_emergency_close(position)
            return
        except Exception as e:
            logger.error(f"Partial emergency close failed: {e}")
        
        # Method 4: Manual intervention flag
        self._flag_for_manual_intervention(position)
    
    async def _aggressive_market_close(self, position: EmergencyPosition):
        """Aggressive market order to close position"""
        # Get current market price
        current_price = await self._get_emergency_price(position.symbol)
        if not current_price:
            raise Exception("Cannot get current price")
        
        # Place aggressive limit order
        opposite_side = 'SELL' if position.side == 'BUY' else 'BUY'
        
        # Use 5% worse than market price for immediate fill
        if opposite_side == 'BUY':
            price = current_price * 1.05
        else:
            price = current_price * 0.95
        
        from execution_engine import OrderRequest
        emergency_order = OrderRequest(
            symbol=position.symbol,
            side=opposite_side,
            order_type='LIMIT',
            quantity=position.quantity,
            price=price,
            time_in_force='IOC',
            priority='CRITICAL',
            emergency_order=True
        )
        
        response = await self.execution_engine.execute_order(emergency_order)
        if response.success:
            self.remove_emergency_position(position.position_id)
            logger.info(f"Aggressive close successful: {position.position_id}")
        else:
            raise Exception(f"Aggressive close failed: {response.error_message}")
    
    def _flag_for_manual_intervention(self, position: EmergencyPosition):
        """Flag position for manual intervention"""
        logger.critical(f"MANUAL INTERVENTION REQUIRED: {position.symbol} {position.side} {position.quantity}")
        
        # Create emergency file
        intervention_data = {
            'position': asdict(position),
            'emergency_reason': self.emergency_reason,
            'timestamp': datetime.now().isoformat(),
            'instructions': f"Manually close {position.quantity} {position.symbol} {position.side} position"
        }
        
        filename = f"EMERGENCY_MANUAL_INTERVENTION_{position.position_id}.json"
        filepath = self.backup_directory / filename
        
        with open(filepath, 'w') as f:
            json.dump(intervention_data, f, indent=2)
        
        logger.critical(f"Manual intervention file created: {filepath}")
    
    def _process_events(self):
        """Background event processing"""
        while True:
            try:
                if not self.event_queue.empty():
                    event = self.event_queue.get()
                    self._handle_event(event)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                time.sleep(5)
    
    def _handle_event(self, event: FailsafeEvent):
        """Handle individual failsafe event"""
        try:
            handler = self.event_handlers.get(event.event_type)
            if handler:
                result = handler(event)
                
                # Update event as handled
                event.handled = True
                event.resolution = str(result) if result else "Handled by default handler"
                
                # Update database
                self._update_event_status(event)
            else:
                logger.warning(f"No handler for event type: {event.event_type}")
                
        except Exception as e:
            logger.error(f"Error handling event {event.event_id}: {e}")
    
    async def _handle_critical_event(self, event: FailsafeEvent):
        """Immediate handling of critical events"""
        if event.alert_level == AlertLevel.EMERGENCY:
            await self.emergency_shutdown(f"Emergency event: {event.description}")
        elif event.alert_level == AlertLevel.CRITICAL:
            # Trigger high-priority protective measures
            if event.event_type == 'position_timeout':
                await self._handle_position_timeout_critical(event)
            elif event.event_type == 'api_failure':
                await self._handle_api_failure_critical(event)
    
    # Event Handlers
    def _handle_network_failure(self, event: FailsafeEvent) -> str:
        """Handle network connectivity failures"""
        self.connectivity_failures += 1
        
        if self.connectivity_failures >= self.max_connectivity_failures:
            logger.critical("Maximum network failures reached - triggering emergency mode")
            asyncio.create_task(self.emergency_shutdown("Network connectivity failure"))
            return "Emergency shutdown triggered"
        
        # Try backup connections
        logger.warning(f"Network failure {self.connectivity_failures}/{self.max_connectivity_failures}")
        return "Network failure logged"
    
    def _handle_api_failure(self, event: FailsafeEvent) -> str:
        """Handle API execution failures"""
        # Check if this is part of a pattern
        recent_failures = self._count_recent_events('api_failure', minutes=5)
        
        if recent_failures >= 5:
            logger.critical("Multiple API failures detected")
            self.system_status = SystemStatus.CRITICAL
            return "System marked as critical"
        
        return "API failure logged"
    
    def _handle_execution_failure(self, event: FailsafeEvent) -> str:
        """Handle order execution failures"""
        # Check for position that might need emergency closure
        position_id = event.context.get('position_id')
        if position_id:
            logger.warning(f"Execution failure for position {position_id}")
            # Could trigger position monitoring here
        
        return "Execution failure logged"
    
    def _handle_position_timeout(self, event: FailsafeEvent) -> str:
        """Handle position timeout events"""
        position_id = event.context.get('position_id')
        if position_id:
            logger.warning(f"Position timeout: {position_id}")
            # Flag position for emergency closure
            
        return "Position timeout handled"
    
    def _handle_resource_critical(self, event: FailsafeEvent) -> str:
        """Handle critical system resource usage"""
        resource_type = event.context.get('resource_type')
        usage = event.context.get('usage')
        
        logger.critical(f"Critical resource usage: {resource_type} at {usage}%")
        
        # Reduce system load
        if resource_type == 'memory':
            # Could trigger memory cleanup
            pass
        elif resource_type == 'cpu':
            # Could reduce trading frequency
            pass
        
        return f"Resource critical handled: {resource_type}"
    
    def _handle_data_corruption(self, event: FailsafeEvent) -> str:
        """Handle data corruption events"""
        logger.critical("Data corruption detected - creating backup")
        
        # Create immediate backup
        asyncio.create_task(self._create_emergency_snapshot())
        
        return "Data corruption backup created"
    
    def _handle_emergency_stop(self, event: FailsafeEvent) -> str:
        """Handle emergency stop events"""
        reason = event.context.get('reason', 'Manual emergency stop')
        asyncio.create_task(self.emergency_shutdown(reason))
        return "Emergency shutdown initiated"
    
    # Monitoring Functions
    def _monitor_system_health(self):
        """Monitor overall system health"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > self.resource_thresholds['cpu_percent']:
                    self.trigger_event(
                        'system_resource_critical',
                        AlertLevel.CRITICAL,
                        f"High CPU usage: {cpu_percent}%",
                        {'resource_type': 'cpu', 'usage': cpu_percent}
                    )
                
                # Memory usage
                memory = psutil.virtual_memory()
                if memory.percent > self.resource_thresholds['memory_percent']:
                    self.trigger_event(
                        'system_resource_critical',
                        AlertLevel.CRITICAL,
                        f"High memory usage: {memory.percent}%",
                        {'resource_type': 'memory', 'usage': memory.percent}
                    )
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                if disk_percent > self.resource_thresholds['disk_percent']:
                    self.trigger_event(
                        'system_resource_critical',
                        AlertLevel.ERROR,
                        f"High disk usage: {disk_percent:.1f}%",
                        {'resource_type': 'disk', 'usage': disk_percent}
                    )
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"System health monitoring error: {e}")
                time.sleep(60)
    
    def _monitor_connectivity(self):
        """Monitor network connectivity"""
        while True:
            try:
                # Test basic connectivity
                import requests
                
                try:
                    response = requests.get('https://api.mexc.com/api/v3/ping', timeout=5)
                    if response.status_code == 200:
                        self.connectivity_failures = 0  # Reset on success
                    else:
                        self._handle_connectivity_failure()
                except requests.RequestException:
                    self._handle_connectivity_failure()
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Connectivity monitoring error: {e}")
                time.sleep(60)
    
    def _handle_connectivity_failure(self):
        """Handle connectivity failure"""
        self.connectivity_failures += 1
        self.trigger_event(
            'network_failure',
            AlertLevel.ERROR if self.connectivity_failures < 3 else AlertLevel.CRITICAL,
            f"Network connectivity failure #{self.connectivity_failures}",
            {'failure_count': self.connectivity_failures}
        )
    
    def _monitor_position_safety(self):
        """Monitor position safety and timeouts"""
        while True:
            try:
                current_time = datetime.now()
                
                with self.position_lock:
                    for position in self.emergency_positions:
                        # Check position age
                        position_age = current_time - position.entry_time
                        if position_age > timedelta(hours=6):  # 6 hour limit
                            self.trigger_event(
                                'position_timeout',
                                AlertLevel.WARNING,
                                f"Position {position.position_id} aged {position_age}",
                                {'position_id': position.position_id, 'age_hours': position_age.total_seconds() / 3600}
                            )
                        
                        # Check unrealized loss
                        if position.unrealized_pnl < -1000:  # $1000 loss
                            self.trigger_event(
                                'large_unrealized_loss',
                                AlertLevel.ERROR,
                                f"Large loss on {position.position_id}: ${position.unrealized_pnl:.2f}",
                                {'position_id': position.position_id, 'unrealized_pnl': position.unrealized_pnl}
                            )
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Position safety monitoring error: {e}")
                time.sleep(120)
    
    def _monitor_heartbeat(self):
        """Monitor system heartbeat"""
        while True:
            try:
                current_time = datetime.now()
                time_since_heartbeat = (current_time - self.last_heartbeat).total_seconds()
                
                if time_since_heartbeat > self.max_heartbeat_delay:
                    self.trigger_event(
                        'heartbeat_failure',
                        AlertLevel.CRITICAL,
                        f"Heartbeat failure: {time_since_heartbeat:.0f}s since last beat",
                        {'seconds_since_heartbeat': time_since_heartbeat}
                    )
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat monitoring error: {e}")
                time.sleep(60)
    
    def heartbeat(self):
        """Record system heartbeat"""
        self.last_heartbeat = datetime.now()
    
    # Utility Methods
    def _store_event(self, event: FailsafeEvent):
        """Store event in database"""
        try:
            cursor = self.failsafe_db.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO failsafe_events
                (event_id, event_type, alert_level, description, context, timestamp, handled, resolution)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type,
                event.alert_level.value,
                event.description,
                json.dumps(event.context),
                event.timestamp,
                event.handled,
                event.resolution
            ))
            self.failsafe_db.commit()
        except Exception as e:
            logger.error(f"Error storing event: {e}")
    
    def _alert_level_to_log_level(self, alert_level: AlertLevel) -> int:
        """Convert alert level to logging level"""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
            AlertLevel.EMERGENCY: logging.CRITICAL
        }
        return mapping.get(alert_level, logging.INFO)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'status': self.system_status.value,
            'emergency_mode': self.emergency_mode,
            'emergency_reason': self.emergency_reason,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'connectivity_failures': self.connectivity_failures,
            'emergency_positions_count': len(self.emergency_positions),
            'recent_events': self._get_recent_events(24)  # Last 24 hours
        }

# Global failsafe instance
_failsafe_manager = None

def get_failsafe_manager() -> FailsafeManager:
    """Get global failsafe manager instance"""
    return _failsafe_manager

def initialize_failsafe(config: Dict, execution_engine) -> FailsafeManager:
    """Initialize global failsafe manager"""
    global _failsafe_manager
    _failsafe_manager = FailsafeManager(config, execution_engine)
    return _failsafe_manager 