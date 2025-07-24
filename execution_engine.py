#!/usr/bin/env python3
"""
Robust Execution Engine with Comprehensive Retries and Redundancies
Designed for high-frequency trading where execution failures are critical
"""

import asyncio
import time
import logging
import json
import hmac
import hashlib
import requests
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import random
from collections import deque
import threading
import queue

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"

class ExecutionResult(Enum):
    """Execution result enumeration"""
    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    FAIL = "FAIL"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"

@dataclass
class OrderRequest:
    """Order request with retry metadata"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET' or 'LIMIT'
    quantity: float
    price: Optional[float] = None
    time_in_force: str = 'GTC'
    client_order_id: str = ""
    
    # Retry metadata
    attempt_count: int = 0
    max_retries: int = 5
    last_attempt: Optional[datetime] = None
    timeout_seconds: int = 10
    priority: str = "NORMAL"  # CRITICAL, HIGH, NORMAL, LOW
    
    # Redundancy options
    use_backup_endpoint: bool = False
    require_confirmation: bool = True
    emergency_order: bool = False

@dataclass
class ExecutionResponse:
    """Execution response with detailed status"""
    success: bool
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    error_message: str = ""
    response_time_ms: int = 0
    retry_count: int = 0
    execution_path: str = "primary"
    timestamp: datetime = None

class CircuitBreaker:
    """Circuit breaker pattern for API failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"
        
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "CLOSED":
            return True
            
        if self.state == "OPEN":
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.timeout_seconds:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker moved to HALF_OPEN")
                    return True
            return False
            
        if self.state == "HALF_OPEN":
            return True
            
        return False

class RetryPolicy:
    """Configurable retry policy with exponential backoff"""
    
    def __init__(self, 
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 max_delay: float = 30.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def should_retry(self, attempt: int, error_type: str) -> bool:
        """Determine if retry should be attempted"""
        if attempt >= self.max_retries:
            return False
            
        # Retry on specific error types
        retryable_errors = [
            "timeout", "connection_error", "rate_limit", 
            "server_error", "network_error", "partial_fill_timeout"
        ]
        
        return error_type.lower() in retryable_errors
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter ±25%
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            
        return max(delay, 0.1)  # Minimum 100ms delay

class RobustExecutionEngine:
    """Robust execution engine with comprehensive redundancies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('MEXC_API_KEY', '')
        self.secret_key = config.get('MEXC_SECRET_KEY', '')
        
        # Primary and backup endpoints
        self.primary_base_url = 'https://api.mexc.com'
        self.backup_base_urls = [
            'https://api.mexc.com',  # Same endpoint but separate session
            # Add additional backup endpoints if available
        ]
        
        # Circuit breakers for each endpoint
        self.circuit_breakers = {
            'primary': CircuitBreaker(failure_threshold=5, timeout_seconds=60),
            'backup_1': CircuitBreaker(failure_threshold=5, timeout_seconds=60),
            'backup_2': CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        }
        
        # Retry policies by operation type
        self.retry_policies = {
            'order_placement': RetryPolicy(max_retries=3, base_delay=0.5, max_delay=5.0),
            'order_cancellation': RetryPolicy(max_retries=5, base_delay=0.2, max_delay=3.0),
            'position_query': RetryPolicy(max_retries=3, base_delay=1.0, max_delay=10.0),
            'emergency_close': RetryPolicy(max_retries=10, base_delay=0.1, max_delay=2.0)
        }
        
        # Order tracking
        self.pending_orders = {}
        self.order_status_cache = {}
        self.failed_orders_queue = queue.Queue()
        
        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(
            requests_per_second=config.get('api_limits', {}).get('orders_per_second', 5)
        )
        
        # HTTP sessions for connection pooling
        self.sessions = {}
        self.backup_sessions = {}
        
        # Monitoring
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'retry_orders': 0,
            'avg_response_time': 0.0,
            'circuit_breaker_trips': 0
        }
        
        # Emergency handlers
        self.emergency_handlers = []
        self.critical_failure_count = 0
        self.max_critical_failures = 3
        
        # Start background tasks
        self._start_monitoring_tasks()
    
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # Order status monitoring
        threading.Thread(target=self._monitor_pending_orders, daemon=True).start()
        
        # Failed order retry processing
        threading.Thread(target=self._process_failed_orders, daemon=True).start()
        
        # Health check monitoring
        threading.Thread(target=self._health_check_loop, daemon=True).start()
    
    async def execute_order(self, order_request: OrderRequest) -> ExecutionResponse:
        """Execute order with comprehensive retry and redundancy"""
        start_time = time.time()
        
        # Validate order request
        if not self._validate_order_request(order_request):
            return ExecutionResponse(
                success=False,
                error_message="Invalid order request",
                timestamp=datetime.now()
            )
        
        # Check circuit breakers
        if not self._check_circuit_breakers():
            return ExecutionResponse(
                success=False,
                error_message="All circuit breakers open",
                timestamp=datetime.now()
            )
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Execute with retries
        response = await self._execute_with_retries(order_request)
        
        # Update statistics
        execution_time = (time.time() - start_time) * 1000
        self._update_execution_stats(response, execution_time)
        
        # Handle critical failures
        if not response.success and order_request.priority == "CRITICAL":
            await self._handle_critical_failure(order_request, response)
        
        return response
    
    async def _execute_with_retries(self, order_request: OrderRequest) -> ExecutionResponse:
        """Execute order with retry logic"""
        retry_policy = self.retry_policies.get('order_placement')
        last_response = None
        
        for attempt in range(retry_policy.max_retries + 1):
            order_request.attempt_count = attempt
            order_request.last_attempt = datetime.now()
            
            # Try primary endpoint first
            if self.circuit_breakers['primary'].can_execute():
                response = await self._execute_single_attempt(order_request, 'primary')
                
                if response.success:
                    self.circuit_breakers['primary'].record_success()
                    return response
                else:
                    self.circuit_breakers['primary'].record_failure()
                    last_response = response
            
            # Try backup endpoints
            for i, backup_name in enumerate(['backup_1', 'backup_2']):
                if self.circuit_breakers[backup_name].can_execute():
                    response = await self._execute_single_attempt(
                        order_request, backup_name, use_backup=True
                    )
                    
                    if response.success:
                        self.circuit_breakers[backup_name].record_success()
                        return response
                    else:
                        self.circuit_breakers[backup_name].record_failure()
                        last_response = response
            
            # Check if we should retry
            if attempt < retry_policy.max_retries:
                error_type = self._classify_error(last_response.error_message if last_response else "unknown")
                
                if retry_policy.should_retry(attempt, error_type):
                    delay = retry_policy.get_delay(attempt)
                    logger.warning(f"Order execution failed (attempt {attempt + 1}), retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    break
            
        # All retries exhausted
        if last_response:
            last_response.retry_count = order_request.attempt_count
            return last_response
        
        return ExecutionResponse(
            success=False,
            error_message="All execution attempts failed",
            retry_count=order_request.attempt_count,
            timestamp=datetime.now()
        )
    
    async def _execute_single_attempt(self, order_request: OrderRequest, 
                                    endpoint_name: str, use_backup: bool = False) -> ExecutionResponse:
        """Execute single order attempt"""
        start_time = time.time()
        
        try:
            # Prepare order parameters
            params = self._prepare_order_params(order_request)
            
            # Add authentication
            params = self._add_authentication(params)
            
            # Select endpoint
            base_url = self.backup_base_urls[0] if use_backup else self.primary_base_url
            
            # Make API request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=order_request.timeout_seconds)) as session:
                url = f"{base_url}/api/v3/order"
                
                async with session.post(url, json=params) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Success response
                        return ExecutionResponse(
                            success=True,
                            order_id=data.get('orderId'),
                            status=OrderStatus.SUBMITTED,
                            response_time_ms=int(response_time),
                            execution_path=endpoint_name,
                            timestamp=datetime.now()
                        )
                    
                    elif response.status == 429:  # Rate limit
                        error_msg = "Rate limit exceeded"
                        
                    elif response.status >= 500:  # Server error
                        error_msg = f"Server error: {response.status}"
                        
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get('msg', f'HTTP {response.status}')
                    
                    return ExecutionResponse(
                        success=False,
                        error_message=error_msg,
                        response_time_ms=int(response_time),
                        execution_path=endpoint_name,
                        timestamp=datetime.now()
                    )
                    
        except asyncio.TimeoutError:
            return ExecutionResponse(
                success=False,
                error_message="Request timeout",
                response_time_ms=(time.time() - start_time) * 1000,
                execution_path=endpoint_name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return ExecutionResponse(
                success=False,
                error_message=f"Execution error: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                execution_path=endpoint_name,
                timestamp=datetime.now()
            )
    
    async def emergency_close_position(self, symbol: str, side: str, quantity: float) -> ExecutionResponse:
        """Emergency position closure with maximum redundancy"""
        logger.critical(f"EMERGENCY CLOSE: {symbol} {side} {quantity}")
        
        # Create emergency order request
        emergency_request = OrderRequest(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',  # Opposite side to close
            order_type='MARKET',
            quantity=quantity,
            priority="CRITICAL",
            emergency_order=True,
            max_retries=10,
            timeout_seconds=5
        )
        
        # Try all available methods
        methods = [
            self._emergency_market_order,
            self._emergency_limit_order_aggressive,
            self._emergency_partial_close
        ]
        
        for method in methods:
            try:
                response = await method(emergency_request)
                if response.success:
                    logger.info(f"Emergency close successful via {method.__name__}")
                    return response
            except Exception as e:
                logger.error(f"Emergency method {method.__name__} failed: {e}")
        
        # If all methods fail, add to manual intervention queue
        self._trigger_manual_intervention(emergency_request)
        
        return ExecutionResponse(
            success=False,
            error_message="All emergency close methods failed - manual intervention required",
            timestamp=datetime.now()
        )
    
    async def _emergency_market_order(self, request: OrderRequest) -> ExecutionResponse:
        """Emergency market order execution"""
        # Use highest priority and shortest timeout
        request.timeout_seconds = 3
        request.priority = "CRITICAL"
        
        return await self._execute_with_retries(request)
    
    async def _emergency_limit_order_aggressive(self, request: OrderRequest) -> ExecutionResponse:
        """Emergency limit order with aggressive pricing"""
        try:
            # Get current market price
            current_price = await self._get_current_price(request.symbol)
            if not current_price:
                return ExecutionResponse(success=False, error_message="Cannot get current price")
            
            # Set aggressive limit price (2% worse than market)
            if request.side == 'BUY':
                request.price = current_price * 1.02
            else:
                request.price = current_price * 0.98
            
            request.order_type = 'LIMIT'
            request.time_in_force = 'IOC'  # Immediate or Cancel
            
            return await self._execute_with_retries(request)
            
        except Exception as e:
            return ExecutionResponse(success=False, error_message=f"Aggressive limit order failed: {e}")
    
    async def _emergency_partial_close(self, request: OrderRequest) -> ExecutionResponse:
        """Emergency partial position closure"""
        try:
            # Split into smaller orders
            chunk_size = request.quantity / 3
            successful_closes = 0
            
            for i in range(3):
                chunk_request = OrderRequest(
                    symbol=request.symbol,
                    side=request.side,
                    order_type='MARKET',
                    quantity=chunk_size,
                    priority="CRITICAL",
                    timeout_seconds=3
                )
                
                response = await self._execute_with_retries(chunk_request)
                if response.success:
                    successful_closes += 1
            
            if successful_closes > 0:
                return ExecutionResponse(
                    success=True,
                    filled_quantity=successful_closes * chunk_size,
                    error_message=f"Partial close: {successful_closes}/3 chunks successful"
                )
            
            return ExecutionResponse(success=False, error_message="All partial closes failed")
            
        except Exception as e:
            return ExecutionResponse(success=False, error_message=f"Partial close failed: {e}")
    
    def _monitor_pending_orders(self):
        """Background thread to monitor pending orders"""
        while True:
            try:
                # Check pending orders for timeouts
                current_time = datetime.now()
                timeout_orders = []
                
                for order_id, order_info in self.pending_orders.items():
                    if (current_time - order_info['timestamp']).total_seconds() > order_info['timeout']:
                        timeout_orders.append(order_id)
                
                # Handle timeout orders
                for order_id in timeout_orders:
                    asyncio.create_task(self._handle_order_timeout(order_id))
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
                time.sleep(10)
    
    def _process_failed_orders(self):
        """Background thread to process failed orders"""
        while True:
            try:
                if not self.failed_orders_queue.empty():
                    failed_order = self.failed_orders_queue.get()
                    
                    # Attempt to reprocess failed order
                    asyncio.create_task(self._reprocess_failed_order(failed_order))
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed order processing error: {e}")
                time.sleep(5)
    
    def _health_check_loop(self):
        """Background health check for all systems"""
        while True:
            try:
                # Check API connectivity
                asyncio.create_task(self._health_check_apis())
                
                # Check circuit breaker states
                self._log_circuit_breaker_states()
                
                # Check execution statistics
                self._log_execution_stats()
                
                time.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(60)
    
    async def _health_check_apis(self):
        """Check health of all API endpoints"""
        endpoints = [
            ('primary', self.primary_base_url),
            ('backup_1', self.backup_base_urls[0] if self.backup_base_urls else None)
        ]
        
        for name, url in endpoints:
            if not url:
                continue
                
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{url}/api/v3/ping") as response:
                        if response.status == 200:
                            logger.debug(f"Health check OK: {name}")
                        else:
                            logger.warning(f"Health check failed: {name} - {response.status}")
                            
            except Exception as e:
                logger.warning(f"Health check error: {name} - {e}")
    
    def _trigger_manual_intervention(self, request: OrderRequest):
        """Trigger manual intervention for critical failures"""
        logger.critical(f"MANUAL INTERVENTION REQUIRED: {request.symbol} {request.side} {request.quantity}")
        
        # Add to emergency intervention queue
        intervention_data = {
            'timestamp': datetime.now(),
            'request': asdict(request),
            'failure_reason': 'All automated methods failed'
        }
        
        # This could trigger alerts, save to emergency file, etc.
        self._save_emergency_intervention(intervention_data)
    
    def _save_emergency_intervention(self, data: Dict):
        """Save emergency intervention data"""
        try:
            filename = f"emergency_intervention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.critical(f"Emergency intervention saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save emergency intervention: {e}")
    
    # Utility methods
    def _validate_order_request(self, request: OrderRequest) -> bool:
        """Validate order request"""
        if not request.symbol or not request.side or not request.order_type:
            return False
        if request.quantity <= 0:
            return False
        if request.order_type == 'LIMIT' and not request.price:
            return False
        return True
    
    def _check_circuit_breakers(self) -> bool:
        """Check if any circuit breaker allows execution"""
        return any(cb.can_execute() for cb in self.circuit_breakers.values())
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type for retry logic"""
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'connection_error'
        elif 'rate limit' in error_lower or 'too many requests' in error_lower:
            return 'rate_limit'
        elif 'server error' in error_lower or '5' in error_lower[:3]:
            return 'server_error'
        else:
            return 'unknown'
    
    def _prepare_order_params(self, request: OrderRequest) -> Dict:
        """Prepare order parameters for API"""
        params = {
            'symbol': request.symbol,
            'side': request.side,
            'type': request.order_type,
            'quantity': request.quantity,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        
        if request.price:
            params['price'] = request.price
        if request.time_in_force:
            params['timeInForce'] = request.time_in_force
        if request.client_order_id:
            params['newClientOrderId'] = request.client_order_id
            
        return params
    
    def _add_authentication(self, params: Dict) -> Dict:
        """Add MEXC authentication to parameters"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        return params
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                url = f"{self.primary_base_url}/api/v3/ticker/price?symbol={symbol}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['price'])
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
        return None
    
    def _update_execution_stats(self, response: ExecutionResponse, execution_time: float):
        """Update execution statistics"""
        self.execution_stats['total_orders'] += 1
        
        if response.success:
            self.execution_stats['successful_orders'] += 1
        else:
            self.execution_stats['failed_orders'] += 1
        
        if response.retry_count > 0:
            self.execution_stats['retry_orders'] += 1
        
        # Update average response time
        current_avg = self.execution_stats['avg_response_time']
        total_orders = self.execution_stats['total_orders']
        self.execution_stats['avg_response_time'] = (
            (current_avg * (total_orders - 1) + execution_time) / total_orders
        )
    
    def _log_circuit_breaker_states(self):
        """Log circuit breaker states"""
        for name, cb in self.circuit_breakers.items():
            if cb.state != "CLOSED":
                logger.warning(f"Circuit breaker {name}: {cb.state} (failures: {cb.failure_count})")
    
    def _log_execution_stats(self):
        """Log execution statistics"""
        stats = self.execution_stats
        success_rate = (stats['successful_orders'] / max(stats['total_orders'], 1)) * 100
        
        logger.info(f"Execution Stats - Total: {stats['total_orders']}, "
                   f"Success Rate: {success_rate:.1f}%, "
                   f"Avg Response: {stats['avg_response_time']:.0f}ms, "
                   f"Retries: {stats['retry_orders']}")

class AsyncRateLimiter:
    """Async rate limiter for API calls"""
    
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token for API call"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.requests_per_second, 
                            self.tokens + elapsed * self.requests_per_second)
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
            else:
                sleep_time = (1.0 - self.tokens) / self.requests_per_second
                await asyncio.sleep(sleep_time)
                self.tokens = 0.0

def create_execution_engine(config: Dict) -> RobustExecutionEngine:
    """Factory function to create execution engine"""
    return RobustExecutionEngine(config)

# Example usage
async def main():
    """Test the execution engine"""
    config = {
        'MEXC_API_KEY': 'your_api_key',
        'MEXC_SECRET_KEY': 'your_secret_key',
        'api_limits': {
            'orders_per_second': 5
        }
    }
    
    engine = RobustExecutionEngine(config)
    
    # Test order
    order = OrderRequest(
        symbol='BTCUSDT',
        side='BUY',
        order_type='MARKET',
        quantity=0.001,
        priority='HIGH'
    )
    
    response = await engine.execute_order(order)
    print(f"Order execution: {response.success}, {response.error_message}")

if __name__ == "__main__":
    asyncio.run(main()) 