#!/usr/bin/env python3
"""
Integration Example: Robust Trading System
Shows how to integrate execution engine and failsafe manager
"""

import asyncio
import logging
from datetime import datetime
from hf_config import get_hf_config
from execution_engine import RobustExecutionEngine, OrderRequest
from failsafe_manager import initialize_failsafe, AlertLevel
from high_frequency_sr_bot import HighFrequencyTradingBot

logger = logging.getLogger(__name__)

class IntegratedTradingSystem:
    """Example of fully integrated trading system with all redundancies"""
    
    def __init__(self):
        # Load configuration
        self.config = get_hf_config()
        
        # Initialize robust execution engine
        self.execution_engine = RobustExecutionEngine(self.config)
        
        # Initialize failsafe manager
        self.failsafe = initialize_failsafe(self.config, self.execution_engine)
        
        # Initialize trading bot
        self.trading_bot = HighFrequencyTradingBot(self.config)
        
        # Override bot's execution engine with robust version
        self.trading_bot.execution_engine = self.execution_engine
        
        logger.info("Integrated trading system initialized with full redundancy")
    
    async def start_trading(self):
        """Start trading with full protection"""
        try:
            logger.info("Starting integrated trading system...")
            
            # Register emergency positions with failsafe
            self._setup_position_monitoring()
            
            # Start main trading loop
            await self.trading_bot.start_high_frequency_trading()
            
        except Exception as e:
            logger.critical(f"Critical error in trading system: {e}")
            
            # Trigger emergency shutdown
            await self.failsafe.emergency_shutdown(f"Critical trading error: {e}")
    
    def _setup_position_monitoring(self):
        """Setup position monitoring with failsafe"""
        
        # Override position opening to register with failsafe
        original_open_position = self.trading_bot._place_hf_order
        
        async def monitored_place_order(signal, position_size):
            """Place order with failsafe monitoring"""
            # Send heartbeat
            self.failsafe.heartbeat()
            
            try:
                # Execute order with robust engine
                result = await original_open_position(signal, position_size)
                
                if result:
                    # Register position with failsafe
                    from failsafe_manager import EmergencyPosition
                    
                    emergency_pos = EmergencyPosition(
                        symbol=signal.symbol,
                        side=signal.signal_type,
                        quantity=position_size,
                        entry_price=signal.entry_price,
                        current_price=signal.entry_price,
                        unrealized_pnl=0.0,
                        entry_time=datetime.now(),
                        position_id=f"pos_{signal.symbol}_{int(datetime.now().timestamp())}",
                        priority=1 if signal.urgency == 'IMMEDIATE' else 2
                    )
                    
                    self.failsafe.add_emergency_position(emergency_pos)
                    logger.info(f"Position registered with failsafe: {emergency_pos.position_id}")
                
                return result
                
            except Exception as e:
                # Trigger execution failure event
                self.failsafe.trigger_event(
                    'execution_failure',
                    AlertLevel.ERROR,
                    f"Order execution failed: {e}",
                    {
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type,
                        'error': str(e)
                    }
                )
                raise
        
        # Replace method
        self.trading_bot._place_hf_order = monitored_place_order
    
    async def test_redundancy_systems(self):
        """Test all redundancy and retry systems"""
        logger.info("Testing redundancy systems...")
        
        # Test 1: Order execution with retries
        test_order = OrderRequest(
            symbol='BTCUSDT',
            side='BUY',
            order_type='MARKET',
            quantity=0.001,
            priority='HIGH'
        )
        
        logger.info("Testing order execution with retries...")
        response = await self.execution_engine.execute_order(test_order)
        logger.info(f"Order test result: {response.success}, retries: {response.retry_count}")
        
        # Test 2: Emergency position closure
        if response.success:
            logger.info("Testing emergency position closure...")
            emergency_response = await self.execution_engine.emergency_close_position(
                'BTCUSDT', 'BUY', 0.001
            )
            logger.info(f"Emergency close result: {emergency_response.success}")
        
        # Test 3: Failsafe event handling
        logger.info("Testing failsafe event system...")
        self.failsafe.trigger_event(
            'test_event',
            AlertLevel.WARNING,
            "Testing failsafe system",
            {'test': True}
        )
        
        # Test 4: System health monitoring
        logger.info("Testing system health monitoring...")
        status = self.failsafe.get_system_status()
        logger.info(f"System status: {status['status']}")
        
        logger.info("All redundancy tests completed")

async def main():
    """Main integration example"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create integrated system
        system = IntegratedTradingSystem()
        
        # Test redundancy systems
        await system.test_redundancy_systems()
        
        # Start trading (commented out for safety)
        # await system.start_trading()
        
    except KeyboardInterrupt:
        logger.info("Integration test stopped by user")
    except Exception as e:
        logger.error(f"Integration test error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 