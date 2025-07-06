# Main Trading System Integration and Live Trading
# Part 6 of the complete LLM + RL Options Trading System

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import threading
import queue
import time
import sys
import os
from pathlib import Path
import signal
import schedule

# Import all our custom modules (would be separate files in production)
# from nse_data_infrastructure import NSEDataManager, DataStorageManager
# from llm_signal_generation import FinancialLLMSignalGenerator, TradingSignal
# from rl_environment_training import PPOAgent, OptionsEnvironment
# from risk_portfolio_management import RiskManager, PositionSizer, RiskMetrics
# from backtrader_integration import OptionsBacktester, LLMRLOptionsStrategy

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Main system configuration"""
    # Data sources
    primary_data_provider: str = "nsepy"
    backup_data_provider: str = "nsepy"
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # Trading parameters
    initial_capital: float = 1000000  # 10 Lakhs
    max_daily_trades: int = 50
    trading_frequency: int = 300  # 5 minutes in seconds
    risk_check_frequency: int = 60  # 1 minute in seconds
    
    # Model paths
    rl_model_path: str = "models/ppo_options_agent.pkl"
    llm_model_path: str = "models/financial_llm"
    
    # Risk management
    max_portfolio_risk: float = 0.15  # 15% max portfolio risk
    max_position_size: float = 0.05   # 5% max position size
    stop_loss_threshold: float = 0.20  # 20% stop loss
    
    # System settings
    log_level: str = "INFO"
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    save_state_frequency: int = 3600  # 1 hour in seconds
    
    # API keys and credentials
    truedata_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    broker_api_key: Optional[str] = None

@dataclass
class TradingSystemState:
    """Current state of the trading system"""
    is_running: bool = False
    last_update: datetime = field(default_factory=datetime.now)
    current_portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    active_positions: Dict = field(default_factory=dict)
    pending_orders: List = field(default_factory=list)
    last_signals: Dict = field(default_factory=dict)
    last_rl_action: Optional[int] = None
    system_alerts: List = field(default_factory=list)

class TradingSignalManager:
    """Manages trading signals from various sources"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.signal_history = []
        self.signal_weights = {
            'llm_fundamental': 0.4,
            'llm_technical': 0.35,
            'llm_sentiment': 0.25
        }
        
    async def generate_ensemble_signal(self, symbol: str) -> Dict:
        """Generate ensemble trading signal"""
        
        try:
            # This would integrate with the actual LLM signal generator
            # For now, using simplified logic
            
            signals = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'fundamental': {
                    'direction': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'strength': np.random.uniform(0.3, 0.8),
                    'confidence': np.random.uniform(0.5, 0.9)
                },
                'technical': {
                    'direction': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'strength': np.random.uniform(0.3, 0.8),
                    'confidence': np.random.uniform(0.5, 0.9)
                },
                'sentiment': {
                    'direction': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'strength': np.random.uniform(0.3, 0.8),
                    'confidence': np.random.uniform(0.5, 0.9)
                }
            }
            
            # Calculate ensemble signal
            ensemble_signal = self._calculate_ensemble_signal(signals)
            
            # Store signal history
            self.signal_history.append(ensemble_signal)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            return ensemble_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return self._get_neutral_signal(symbol)
    
    def _calculate_ensemble_signal(self, signals: Dict) -> Dict:
        """Calculate ensemble signal from individual signals"""
        
        # Simple weighted average approach
        total_score = 0.0
        total_confidence = 0.0
        direction_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        
        for signal_type, weight in self.signal_weights.items():
            signal_key = signal_type.replace('llm_', '')
            
            if signal_key in signals:
                signal = signals[signal_key]
                
                # Vote for direction
                direction_votes[signal['direction']] += weight
                
                # Calculate weighted score
                direction_multiplier = 1 if signal['direction'] == 'bullish' else -1 if signal['direction'] == 'bearish' else 0
                score = signal['strength'] * signal['confidence'] * direction_multiplier * weight
                total_score += score
                total_confidence += signal['confidence'] * weight
        
        # Determine final direction
        final_direction = max(direction_votes, key=direction_votes.get)
        
        return {
            'symbol': signals['symbol'],
            'timestamp': signals['timestamp'],
            'direction': final_direction,
            'strength': min(1.0, abs(total_score)),
            'confidence': min(1.0, total_confidence),
            'component_signals': signals,
            'signal_type': 'ensemble'
        }
    
    def _get_neutral_signal(self, symbol: str) -> Dict:
        """Return neutral signal in case of errors"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.0,
            'signal_type': 'neutral_fallback'
        }

class OrderManager:
    """Manages order execution and tracking"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.pending_orders = []
        self.executed_orders = []
        self.order_id_counter = 1
        
    async def submit_order(self, order_details: Dict) -> str:
        """Submit trading order"""
        
        order_id = f"ORDER_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        order = {
            'order_id': order_id,
            'symbol': order_details['symbol'],
            'action': order_details['action'],  # 'BUY' or 'SELL'
            'quantity': order_details['quantity'],
            'order_type': order_details.get('order_type', 'MARKET'),
            'price': order_details.get('price'),
            'timestamp': datetime.now(),
            'status': 'PENDING',
            'broker_order_id': None
        }
        
        try:
            if self.config.enable_live_trading:
                # Submit to actual broker
                broker_order_id = await self._submit_to_broker(order)
                order['broker_order_id'] = broker_order_id
                order['status'] = 'SUBMITTED'
            elif self.config.enable_paper_trading:
                # Paper trading simulation
                await self._simulate_order_execution(order)
            
            self.pending_orders.append(order)
            logger.info(f"Order submitted: {order_id} - {order['action']} {order['quantity']} {order['symbol']}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order['status'] = 'FAILED'
            order['error'] = str(e)
            return order_id
    
    async def _submit_to_broker(self, order: Dict) -> str:
        """Submit order to actual broker (placeholder)"""
        
        # This would integrate with actual broker APIs like:
        # - Zerodha Kite API
        # - Upstox API
        # - Angel Broking API
        # - ICICI Direct API
        
        # For now, simulate submission
        await asyncio.sleep(0.1)
        return f"BROKER_{order['order_id']}"
    
    async def _simulate_order_execution(self, order: Dict):
        """Simulate order execution for paper trading"""
        
        # Simulate realistic execution delay
        await asyncio.sleep(np.random.uniform(0.1, 2.0))
        
        # Simulate execution with slight slippage
        if order['order_type'] == 'MARKET':
            slippage = np.random.uniform(-0.01, 0.01)  # ±1% slippage
            execution_price = order.get('price', 100) * (1 + slippage)
        else:
            execution_price = order['price']
        
        order['execution_price'] = execution_price
        order['execution_time'] = datetime.now()
        order['status'] = 'EXECUTED'
        
        logger.info(f"Paper trade executed: {order['order_id']} at {execution_price:.2f}")
    
    async def update_order_status(self):
        """Update status of pending orders"""
        
        for order in self.pending_orders[:]:
            try:
                if order['status'] == 'PENDING' and self.config.enable_live_trading:
                    # Check with broker for order status
                    status = await self._check_broker_order_status(order['broker_order_id'])
                    order['status'] = status
                
                if order['status'] in ['EXECUTED', 'CANCELLED', 'FAILED']:
                    self.executed_orders.append(order)
                    self.pending_orders.remove(order)
                    
            except Exception as e:
                logger.error(f"Error updating order status: {e}")
    
    async def _check_broker_order_status(self, broker_order_id: str) -> str:
        """Check order status with broker"""
        # Placeholder for actual broker API integration
        return 'EXECUTED'
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of specific order"""
        
        for order in self.pending_orders + self.executed_orders:
            if order['order_id'] == order_id:
                return order
        return None

class PortfolioManager:
    """Manages portfolio positions and P&L tracking"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.positions = {}
        self.cash_balance = config.initial_capital
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.pnl_history = []
        
    def update_position(self, symbol: str, action: str, quantity: int, price: float):
        """Update position after trade execution"""
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0.0,
                'total_cost': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        
        if action == 'BUY':
            # Add to position
            new_quantity = position['quantity'] + quantity
            new_total_cost = position['total_cost'] + (quantity * price * 100)  # 100 shares per option contract
            
            if new_quantity > 0:
                position['avg_price'] = new_total_cost / (new_quantity * 100)
            
            position['quantity'] = new_quantity
            position['total_cost'] = new_total_cost
            
            # Update cash
            self.cash_balance -= (quantity * price * 100 + 20)  # Include commission
            
        elif action == 'SELL':
            # Reduce position
            if position['quantity'] >= quantity:
                # Calculate realized P&L
                realized_pnl = (price - position['avg_price']) * quantity * 100
                position['realized_pnl'] += realized_pnl
                self.total_pnl += realized_pnl
                
                # Update position
                position['quantity'] -= quantity
                if position['quantity'] > 0:
                    position['total_cost'] -= (quantity * position['avg_price'] * 100)
                else:
                    position['total_cost'] = 0.0
                
                # Update cash
                self.cash_balance += (quantity * price * 100 - 20)  # Subtract commission
                
                logger.info(f"Position update: {symbol} {action} {quantity} at {price:.2f}, Realized P&L: {realized_pnl:.2f}")
            else:
                logger.warning(f"Insufficient position to sell: {symbol}, have {position['quantity']}, trying to sell {quantity}")
    
    def update_unrealized_pnl(self, market_prices: Dict[str, float]):
        """Update unrealized P&L based on current market prices"""
        
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if position['quantity'] > 0 and symbol in market_prices:
                current_price = market_prices[symbol]
                unrealized_pnl = (current_price - position['avg_price']) * position['quantity'] * 100
                position['unrealized_pnl'] = unrealized_pnl
                total_unrealized += unrealized_pnl
        
        return total_unrealized
    
    def calculate_portfolio_value(self, market_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        
        unrealized_pnl = self.update_unrealized_pnl(market_prices)
        total_value = self.cash_balance + sum(pos['total_cost'] for pos in self.positions.values()) + unrealized_pnl
        
        return total_value
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        
        return {
            'cash_balance': self.cash_balance,
            'total_positions': len([pos for pos in self.positions.values() if pos['quantity'] > 0]),
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'positions': {k: v for k, v in self.positions.items() if v['quantity'] > 0}
        }

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = TradingSystemState()
        
        # Initialize components
        self.signal_manager = TradingSignalManager(config)
        self.order_manager = OrderManager(config)
        self.portfolio_manager = PortfolioManager(config)
        
        # System control
        self.shutdown_event = asyncio.Event()
        self.tasks = []
        
        # Data and model components (would be initialized from actual classes)
        self.data_manager = None  # NSEDataManager(config.__dict__)
        self.rl_agent = None      # PPOAgent loaded from saved model
        self.risk_manager = None  # RiskManager(config.__dict__)
        
        # State persistence
        self.state_file = "trading_system_state.pkl"
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Trading system initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def start_system(self):
        """Start the trading system"""
        
        logger.info("Starting trading system...")
        
        # Load previous state if available
        self._load_system_state()
        
        # Initialize components
        await self._initialize_components()
        
        # Start main trading loop
        self.state.is_running = True
        
        # Create and start background tasks
        self.tasks = [
            asyncio.create_task(self._main_trading_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._order_monitoring_loop()),
            asyncio.create_task(self._state_persistence_loop()),
            asyncio.create_task(self._performance_monitoring_loop())
        ]
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Trading system started successfully")
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
        
        # Cleanup
        await self._shutdown_system()
    
    async def _initialize_components(self):
        """Initialize all system components"""
        
        try:
            # Initialize data manager
            # self.data_manager = NSEDataManager(self.config.__dict__)
            
            # Load RL agent
            if os.path.exists(self.config.rl_model_path):
                logger.info(f"Loading RL agent from {self.config.rl_model_path}")
                # self.rl_agent = pickle.load(open(self.config.rl_model_path, 'rb'))
            else:
                logger.warning("RL model not found, creating new agent")
                # self.rl_agent = PPOAgent(state_size=36, action_size=122)
            
            # Initialize risk manager
            # self.risk_manager = RiskManager(self.config.__dict__)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def _main_trading_loop(self):
        """Main trading logic loop"""
        
        logger.info("Starting main trading loop")
        
        while self.state.is_running:
            try:
                # Generate trading signals
                signal = await self.signal_manager.generate_ensemble_signal('NIFTY')
                self.state.last_signals['NIFTY'] = signal
                
                # Get current market data
                market_data = await self._get_current_market_data()
                
                if market_data and signal:
                    # Prepare RL state
                    rl_state = self._prepare_rl_state(signal, market_data)
                    
                    # Get RL action
                    if self.rl_agent:
                        rl_action, _ = self.rl_agent.select_action(rl_state, training=False)
                        self.state.last_rl_action = rl_action
                        
                        # Execute trading decision
                        await self._execute_trading_decision(signal, rl_action, market_data)
                
                # Update portfolio metrics
                await self._update_portfolio_metrics(market_data)
                
                # Sleep until next iteration
                await asyncio.sleep(self.config.trading_frequency)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _risk_monitoring_loop(self):
        """Risk monitoring and management loop"""
        
        logger.info("Starting risk monitoring loop")
        
        while self.state.is_running:
            try:
                # Get current portfolio state
                portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                
                # Perform risk checks
                if self.risk_manager and portfolio_summary['total_positions'] > 0:
                    # Convert positions to risk format
                    risk_positions = []
                    for symbol, position in portfolio_summary['positions'].items():
                        risk_position_data = {
                            'symbol': symbol,
                            'market_value': position['total_cost'],
                            'delta': 0.5,  # Would get from actual options data
                            'gamma': 0.01,
                            'theta': -2.0,
                            'vega': 15.0,
                            'portfolio_value': self.state.current_portfolio_value
                        }
                        # risk_positions.append(self.risk_manager.calculate_position_risk(risk_position_data))
                    
                    # Calculate portfolio risk metrics
                    # risk_metrics = self.risk_manager.calculate_portfolio_risk(risk_positions, self.state.current_portfolio_value)
                    
                    # Check risk limits
                    # risk_checks = self.risk_manager.check_risk_limits(risk_positions, self.state.current_portfolio_value, risk_metrics)
                    
                    # Handle risk violations
                    # await self._handle_risk_violations(risk_checks)
                
                await asyncio.sleep(self.config.risk_check_frequency)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _order_monitoring_loop(self):
        """Order status monitoring loop"""
        
        logger.info("Starting order monitoring loop")
        
        while self.state.is_running:
            try:
                # Update order statuses
                await self.order_manager.update_order_status()
                
                # Process executed orders
                for order in self.order_manager.executed_orders[-10:]:  # Last 10 executed orders
                    if order['status'] == 'EXECUTED' and 'processed' not in order:
                        await self._process_executed_order(order)
                        order['processed'] = True
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in order monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _state_persistence_loop(self):
        """System state persistence loop"""
        
        logger.info("Starting state persistence loop")
        
        while self.state.is_running:
            try:
                self._save_system_state()
                await asyncio.sleep(self.config.save_state_frequency)
                
            except Exception as e:
                logger.error(f"Error in state persistence loop: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring and reporting loop"""
        
        logger.info("Starting performance monitoring loop")
        
        while self.state.is_running:
            try:
                # Calculate daily performance
                current_time = datetime.now()
                
                if current_time.hour == 0 and current_time.minute == 0:  # Start of new day
                    await self._generate_daily_report()
                
                # Real-time performance updates
                await self._update_performance_metrics()
                
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def _get_current_market_data(self) -> Optional[Dict]:
        """Get current market data"""
        
        try:
            # Mock market data for demo
            return {
                'NIFTY': {
                    'ltp': 18000 + np.random.normal(0, 100),
                    'change': np.random.normal(0, 50),
                    'volume': np.random.randint(5000000, 15000000),
                    'timestamp': datetime.now()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def _prepare_rl_state(self, signal: Dict, market_data: Dict) -> np.ndarray:
        """Prepare state vector for RL agent"""
        
        state_vector = []
        
        # Market features
        nifty_data = market_data.get('NIFTY', {})
        state_vector.extend([
            nifty_data.get('ltp', 18000) / 20000,  # Normalized price
            nifty_data.get('change', 0) / 1000,    # Normalized change
            nifty_data.get('volume', 10000000) / 20000000  # Normalized volume
        ])
        
        # Portfolio features
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        state_vector.extend([
            portfolio_summary['cash_balance'] / self.config.initial_capital,
            portfolio_summary['total_pnl'] / self.config.initial_capital,
            min(portfolio_summary['total_positions'] / 10, 1.0)  # Normalized position count
        ])
        
        # Signal features
        direction_encoding = {'bullish': 1, 'bearish': -1, 'neutral': 0}
        state_vector.extend([
            signal.get('strength', 0),
            signal.get('confidence', 0),
            direction_encoding.get(signal.get('direction', 'neutral'), 0)
        ])
        
        # Pad to fixed size
        target_size = 36
        while len(state_vector) < target_size:
            state_vector.append(0.0)
        
        return np.array(state_vector[:target_size], dtype=np.float32)
    
    async def _execute_trading_decision(self, signal: Dict, rl_action: int, market_data: Dict):
        """Execute trading decision based on signal and RL action"""
        
        try:
            # Decode RL action (simplified)
            if rl_action == 0:  # Hold
                return
            elif rl_action >= 121:  # Close all positions
                await self._close_all_positions()
                return
            
            # Simple action decoding for demo
            action_types = ['buy_call', 'buy_put', 'sell_call', 'sell_put']
            action_idx = (rl_action - 1) % len(action_types)
            action_type = action_types[action_idx]
            
            quantity = 1 + ((rl_action - 1) // len(action_types)) % 3  # 1, 2, or 3 contracts
            
            # Check signal strength threshold
            if signal['strength'] < 0.5 or signal['confidence'] < 0.6:
                return
            
            # Generate option details
            underlying_price = market_data['NIFTY']['ltp']
            strike_offset = ((rl_action - 1) // (len(action_types) * 3)) % 10 - 5  # -5 to +5
            strike_price = round(underlying_price + strike_offset * 100, -2)
            
            option_symbol = f"NIFTY{int(strike_price)}{'CE' if 'call' in action_type else 'PE'}"
            option_price = self._calculate_option_price(underlying_price, strike_price, 'call' if 'call' in action_type else 'put')
            
            # Submit order
            if 'buy' in action_type:
                order_details = {
                    'symbol': option_symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': option_price,
                    'order_type': 'MARKET'
                }
                
                order_id = await self.order_manager.submit_order(order_details)
                logger.info(f"Submitted buy order: {order_id}")
                
        except Exception as e:
            logger.error(f"Error executing trading decision: {e}")
    
    def _calculate_option_price(self, underlying_price: float, strike_price: float, option_type: str) -> float:
        """Calculate simplified option price"""
        
        if option_type == 'call':
            intrinsic = max(0, underlying_price - strike_price)
            time_value = max(10, 50 - abs(underlying_price - strike_price) * 0.1)
        else:
            intrinsic = max(0, strike_price - underlying_price)
            time_value = max(10, 50 - abs(underlying_price - strike_price) * 0.1)
        
        return intrinsic + time_value
    
    async def _close_all_positions(self):
        """Close all open positions"""
        
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        
        for symbol, position in portfolio_summary['positions'].items():
            if position['quantity'] > 0:
                # Calculate current option price
                underlying_price = 18000  # Would get from market data
                option_price = self._calculate_option_price(underlying_price, 18000, 'call')  # Simplified
                
                order_details = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': position['quantity'],
                    'price': option_price,
                    'order_type': 'MARKET'
                }
                
                order_id = await self.order_manager.submit_order(order_details)
                logger.info(f"Submitted close order: {order_id}")
    
    async def _process_executed_order(self, order: Dict):
        """Process executed order and update portfolio"""
        
        try:
            self.portfolio_manager.update_position(
                symbol=order['symbol'],
                action=order['action'],
                quantity=order['quantity'],
                price=order.get('execution_price', order.get('price', 0))
            )
            
            logger.info(f"Processed executed order: {order['order_id']}")
            
        except Exception as e:
            logger.error(f"Error processing executed order: {e}")
    
    async def _update_portfolio_metrics(self, market_data: Optional[Dict]):
        """Update portfolio metrics and state"""
        
        try:
            if market_data:
                # Update current prices (simplified)
                current_prices = {}
                for symbol in self.portfolio_manager.positions.keys():
                    current_prices[symbol] = self._calculate_option_price(18000, 18000, 'call')  # Simplified
                
                self.state.current_portfolio_value = self.portfolio_manager.calculate_portfolio_value(current_prices)
            
            self.state.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            # Calculate daily P&L
            daily_return = (self.state.current_portfolio_value - self.config.initial_capital) / self.config.initial_capital
            
            logger.info(f"Portfolio Value: ${self.state.current_portfolio_value:,.2f}, Daily Return: {daily_return:.2%}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _generate_daily_report(self):
        """Generate daily performance report"""
        
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            report = {
                'date': datetime.now().date(),
                'portfolio_value': self.state.current_portfolio_value,
                'cash_balance': portfolio_summary['cash_balance'],
                'total_pnl': portfolio_summary['total_pnl'],
                'daily_pnl': portfolio_summary['daily_pnl'],
                'active_positions': portfolio_summary['total_positions'],
                'total_trades': len(self.order_manager.executed_orders)
            }
            
            # Save report
            report_file = f"daily_reports/report_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs('daily_reports', exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Daily report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    def _save_system_state(self):
        """Save current system state to file"""
        
        try:
            state_data = {
                'system_state': self.state.__dict__,
                'portfolio_state': self.portfolio_manager.__dict__,
                'order_history': self.order_manager.executed_orders[-100:],  # Last 100 orders
                'signal_history': self.signal_manager.signal_history[-100:],  # Last 100 signals
                'timestamp': datetime.now()
            }
            
            with open(self.state_file, 'wb') as f:
                pickle.dump(state_data, f)
            
            logger.debug("System state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    def _load_system_state(self):
        """Load previous system state from file"""
        
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'rb') as f:
                    state_data = pickle.load(f)
                
                # Restore state (simplified)
                logger.info("Previous system state loaded successfully")
            else:
                logger.info("No previous state file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def _shutdown_system(self):
        """Gracefully shutdown the trading system"""
        
        logger.info("Shutting down trading system...")
        
        self.state.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close all positions if in live trading mode
        if self.config.enable_live_trading:
            await self._close_all_positions()
        
        # Save final state
        self._save_system_state()
        
        logger.info("Trading system shutdown complete")

# Main execution
async def main():
    """Main function to run the trading system"""
    
    # Configuration
    config = SystemConfig(
        initial_capital=1000000,
        enable_paper_trading=True,
        enable_live_trading=False,
        log_level="INFO",
        trading_frequency=300,  # 5 minutes
        risk_check_frequency=60  # 1 minute
    )
    
    # Create and start trading system
    trading_system = TradingSystem(config)
    
    try:
        await trading_system.start_system()
    except KeyboardInterrupt:
        logger.info("Trading system interrupted by user")
    except Exception as e:
        logger.error(f"Trading system error: {e}")
    finally:
        logger.info("Trading system terminated")

if __name__ == "__main__":
    # Run the trading system
    asyncio.run(main())