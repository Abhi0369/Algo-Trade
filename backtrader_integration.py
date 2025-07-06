# Backtrader Integration and Backtesting Framework
# Part 5 of the complete LLM + RL Options Trading System

import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import pickle
import json
import os
from collections import defaultdict

# Import our custom modules
from dataclasses import dataclass
import gym
from gym import spaces

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_cash: float = 1000000
    commission: float = 0.001
    slippage: float = 0.001
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    data_timeframe: str = "1D"
    options_expiry_days: int = 30
    max_positions: int = 10
    risk_free_rate: float = 0.05

class OptionsData(bt.feeds.PandasData):
    """Custom options data feed for backtrader"""
    
    lines = ('strike', 'option_type', 'iv', 'delta', 'gamma', 'theta', 'vega', 'open_interest')
    
    params = (
        ('strike', -1),
        ('option_type', -1),
        ('iv', -1),
        ('delta', -1),
        ('gamma', -1),
        ('theta', -1),
        ('vega', -1),
        ('open_interest', -1),
    )

class LLMRLOptionsStrategy(bt.Strategy):
    """Main trading strategy integrating LLM signals and RL decisions"""
    
    params = (
        ('llm_signal_generator', None),
        ('rl_agent', None),
        ('risk_manager', None),
        ('position_sizer', None),
        ('rebalance_frequency', 5),  # Rebalance every 5 days
        ('lookback_period', 252),    # 1 year lookback for risk calculations
        ('min_signal_strength', 0.3),
        ('min_signal_confidence', 0.5),
    )
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.data_volume = self.datas[0].volume
        
        # Strategy state
        self.rebalance_counter = 0
        self.portfolio_history = []
        self.trade_history = []
        self.risk_metrics_history = []
        self.signal_history = []
        
        # Performance tracking
        self.initial_portfolio_value = self.broker.getvalue()
        self.max_portfolio_value = self.initial_portfolio_value
        self.max_drawdown = 0.0
        
        # Options tracking
        self.options_positions = {}
        self.underlying_price_history = []
        
        # RL environment state
        self.rl_state_history = []
        self.rl_action_history = []
        self.rl_reward_history = []
        
        logger.info(f"LLMRLOptionsStrategy initialized with {len(self.datas)} data feeds")
    
    def start(self):
        """Called at strategy start"""
        logger.info("Strategy started")
        print(f"Starting portfolio value: ${self.broker.getvalue():,.2f}")
    
    def next(self):
        """Main strategy logic called on each bar"""
        
        current_date = self.datas[0].datetime.date(0)
        current_price = self.data_close[0]
        current_volume = self.data_volume[0]
        
        # Update price history
        self.underlying_price_history.append({
            'date': current_date,
            'price': current_price,
            'volume': current_volume
        })
        
        # Keep only recent history
        if len(self.underlying_price_history) > self.p.lookback_period:
            self.underlying_price_history = self.underlying_price_history[-self.p.lookback_period:]
        
        # Rebalance logic
        self.rebalance_counter += 1
        
        if self.rebalance_counter >= self.p.rebalance_frequency:
            self._execute_rebalance()
            self.rebalance_counter = 0
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
        
        # Close expiring options
        self._close_expiring_options()
    
    def _execute_rebalance(self):
        """Execute portfolio rebalancing with LLM signals and RL decisions"""
        
        try:
            # Generate LLM signals
            current_market_data = self._prepare_market_data()
            llm_signal = None
            
            if self.p.llm_signal_generator:
                # In real implementation, this would be async
                llm_signal = self._get_llm_signal_sync(current_market_data)
                self.signal_history.append(llm_signal)
            
            # Prepare RL state
            rl_state = self._prepare_rl_state(llm_signal)
            
            # Get RL action
            rl_action = None
            if self.p.rl_agent and len(rl_state) > 0:
                rl_action, _ = self.p.rl_agent.select_action(rl_state, training=False)
                self.rl_action_history.append(rl_action)
            
            # Execute trading decisions
            if llm_signal and rl_action is not None:
                self._execute_trading_decision(llm_signal, rl_action)
            
            # Risk management checks
            if self.p.risk_manager:
                self._perform_risk_checks()
        
        except Exception as e:
            logger.error(f"Error in rebalancing: {e}")
    
    def _get_llm_signal_sync(self, market_data: Dict) -> Dict:
        """Get LLM signal synchronously (mock implementation)"""
        
        # Mock LLM signal generation
        # In real implementation, this would call the actual LLM signal generator
        
        current_price = self.data_close[0]
        price_change = (current_price - self.data_close[-5]) / self.data_close[-5] if len(self.data_close) > 5 else 0
        
        # Simple signal based on price momentum and volume
        volume_ratio = self.data_volume[0] / np.mean([self.data_volume[-i] for i in range(1, 6)]) if len(self.data_volume) > 5 else 1.0
        
        if price_change > 0.02 and volume_ratio > 1.2:
            direction = 'bullish'
            strength = min(0.8, abs(price_change) * 10)
        elif price_change < -0.02 and volume_ratio > 1.2:
            direction = 'bearish'
            strength = min(0.8, abs(price_change) * 10)
        else:
            direction = 'neutral'
            strength = 0.3
        
        return {
            'symbol': 'NIFTY',
            'direction': direction,
            'strength': strength,
            'confidence': 0.6 + np.random.random() * 0.3,  # Random confidence between 0.6-0.9
            'timestamp': datetime.now(),
            'signal_type': 'ensemble',
            'metadata': {
                'price_change': price_change,
                'volume_ratio': volume_ratio
            }
        }
    
    def _prepare_market_data(self) -> Dict:
        """Prepare market data for LLM signal generation"""
        
        lookback = min(20, len(self.data_close))
        
        return {
            'symbol': 'NIFTY',
            'current_price': self.data_close[0],
            'price_history': [self.data_close[-i] for i in range(lookback)],
            'volume_history': [self.data_volume[-i] for i in range(lookback)],
            'date': self.datas[0].datetime.date(0)
        }
    
    def _prepare_rl_state(self, llm_signal: Optional[Dict]) -> np.ndarray:
        """Prepare state vector for RL agent"""
        
        state_vector = []
        
        # Market features
        current_price = self.data_close[0]
        state_vector.extend([
            current_price / 20000,  # Normalized price
            (current_price - self.data_close[-1]) / self.data_close[-1] if len(self.data_close) > 1 else 0,  # Price change
            self.data_volume[0] / 10000000,  # Normalized volume
        ])
        
        # Portfolio features
        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        positions_count = len([order for order in self.broker.get_orders_open()])
        
        state_vector.extend([
            cash / self.initial_portfolio_value,  # Normalized cash
            portfolio_value / self.initial_portfolio_value,  # Normalized portfolio value
            positions_count / 10,  # Normalized position count
        ])
        
        # Technical indicators (simplified)
        lookback = min(20, len(self.data_close))
        if lookback > 5:
            sma_5 = np.mean([self.data_close[-i] for i in range(1, 6)])
            sma_20 = np.mean([self.data_close[-i] for i in range(1, min(21, lookback+1))])
            
            state_vector.extend([
                current_price / sma_5 - 1,  # Price vs SMA5
                current_price / sma_20 - 1,  # Price vs SMA20
                (sma_5 / sma_20 - 1) if sma_20 > 0 else 0,  # SMA5 vs SMA20
            ])
        else:
            state_vector.extend([0, 0, 0])
        
        # LLM signal features
        if llm_signal:
            direction_encoding = {'bullish': 1, 'bearish': -1, 'neutral': 0}
            state_vector.extend([
                llm_signal['strength'],
                llm_signal['confidence'],
                direction_encoding.get(llm_signal['direction'], 0)
            ])
        else:
            state_vector.extend([0, 0, 0])
        
        # Risk features
        if len(self.portfolio_history) > 1:
            recent_returns = [h['return'] for h in self.portfolio_history[-10:]]
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0
            state_vector.append(volatility)
        else:
            state_vector.append(0)
        
        # Pad to fixed size if needed
        target_size = 16
        while len(state_vector) < target_size:
            state_vector.append(0.0)
        
        return np.array(state_vector[:target_size], dtype=np.float32)
    
    def _execute_trading_decision(self, llm_signal: Dict, rl_action: int):
        """Execute trading decision based on LLM signal and RL action"""
        
        # Check signal thresholds
        if (llm_signal['strength'] < self.p.min_signal_strength or 
            llm_signal['confidence'] < self.p.min_signal_confidence):
            return
        
        # Decode RL action
        action_type, strike_offset, quantity = self._decode_rl_action(rl_action)
        
        if action_type == 'hold':
            return
        elif action_type == 'close_all':
            self._close_all_positions()
            return
        
        # Generate options to trade
        current_price = self.data_close[0]
        available_options = self._generate_available_options(current_price)
        
        if not available_options:
            return
        
        # Select option based on strike offset
        strike_idx = min(strike_offset, len(available_options) - 1)
        selected_option = available_options[strike_idx]
        
        # Execute trade
        if 'buy' in action_type:
            self._execute_options_buy(selected_option, action_type, quantity, llm_signal)
        elif 'sell' in action_type:
            self._execute_options_sell(selected_option, action_type, quantity)
    
    def _decode_rl_action(self, action: int) -> Tuple[str, int, int]:
        """Decode RL action into trading components"""
        
        # Action encoding: similar to the RL environment
        action_types = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 'close_all']
        
        if action == 0:
            return 'hold', 0, 0
        elif action >= len(action_types) * 30 - 1:  # Last action
            return 'close_all', 0, 0
        
        # Decode structured action
        action_idx = action - 1
        num_strikes = 10
        num_quantities = 3
        
        action_type_idx = action_idx // (num_strikes * num_quantities)
        remaining = action_idx % (num_strikes * num_quantities)
        
        strike_offset = remaining // num_quantities
        quantity_idx = remaining % num_quantities
        
        action_type = action_types[min(action_type_idx + 1, len(action_types) - 2)]
        quantities = [1, 3, 5]
        quantity = quantities[quantity_idx]
        
        return action_type, strike_offset, quantity
    
    def _generate_available_options(self, underlying_price: float) -> List[Dict]:
        """Generate available options for trading"""
        
        options = []
        
        # Generate strikes around current price
        for i in range(-5, 6):  # 10 strikes
            strike = round(underlying_price + i * 100, -2)
            
            for option_type in ['CE', 'PE']:
                # Simplified option pricing
                if option_type == 'CE':
                    intrinsic = max(0, underlying_price - strike)
                    time_value = max(10, 50 - abs(underlying_price - strike) * 0.1)
                else:
                    intrinsic = max(0, strike - underlying_price)
                    time_value = max(10, 50 - abs(underlying_price - strike) * 0.1)
                
                option_price = intrinsic + time_value
                
                options.append({
                    'symbol': f"NIFTY{int(strike)}{option_type}",
                    'strike': strike,
                    'option_type': option_type,
                    'price': option_price,
                    'delta': 0.5 if option_type == 'CE' else -0.5,
                    'gamma': 0.01,
                    'theta': -2.0,
                    'vega': 15.0,
                    'expiry': self.datas[0].datetime.date(0) + timedelta(days=self.p.options_expiry_days)
                })
        
        return options
    
    def _execute_options_buy(self, option: Dict, action_type: str, quantity: int, llm_signal: Dict):
        """Execute options buy order"""
        
        # Position sizing
        if self.p.position_sizer:
            portfolio_value = self.broker.getvalue()
            current_vol = self._calculate_current_volatility()
            
            optimal_quantity = self.p.position_sizer.calculate_position_size(
                signal_strength=llm_signal['strength'],
                signal_confidence=llm_signal['confidence'],
                current_volatility=current_vol,
                portfolio_value=portfolio_value,
                option_price=option['price'],
                option_delta=option['delta']
            )
            
            quantity = min(quantity, optimal_quantity)
        
        if quantity <= 0:
            return
        
        # Check available cash
        order_value = option['price'] * quantity * 100 + 20  # 100 shares per contract + commission
        
        if order_value > self.broker.getcash():
            logger.warning("Insufficient cash for options purchase")
            return
        
        # Create synthetic order (backtrader doesn't natively support options)
        self._create_synthetic_options_position(option, quantity, 'buy', llm_signal)
        
        logger.info(f"Bought {quantity} {option['symbol']} at {option['price']:.2f}")
    
    def _execute_options_sell(self, option: Dict, action_type: str, quantity: int):
        """Execute options sell order (close existing positions)"""
        
        # Find matching positions to close
        matching_positions = [
            pos_id for pos_id, pos in self.options_positions.items()
            if (pos['strike'] == option['strike'] and 
                pos['option_type'] == option['option_type'] and
                pos['quantity'] > 0)
        ]
        
        if not matching_positions:
            logger.warning(f"No matching positions to sell for {option['symbol']}")
            return
        
        # Close positions (FIFO)
        total_closed = 0
        for pos_id in matching_positions:
            if total_closed >= quantity:
                break
            
            position = self.options_positions[pos_id]
            close_quantity = min(quantity - total_closed, position['quantity'])
            
            # Calculate P&L
            entry_price = position['entry_price']
            current_price = option['price']
            pnl = (current_price - entry_price) * close_quantity * 100
            
            # Update position
            position['quantity'] -= close_quantity
            if position['quantity'] <= 0:
                del self.options_positions[pos_id]
            
            # Record trade
            self.trade_history.append({
                'action': 'sell',
                'symbol': option['symbol'],
                'quantity': close_quantity,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'date': self.datas[0].datetime.date(0)
            })
            
            total_closed += close_quantity
            
            logger.info(f"Sold {close_quantity} {option['symbol']} at {current_price:.2f}, P&L: {pnl:.2f}")
    
    def _create_synthetic_options_position(self, option: Dict, quantity: int, action: str, llm_signal: Dict):
        """Create synthetic options position for tracking"""
        
        position_id = f"{option['symbol']}_{datetime.now().timestamp()}"
        
        position = {
            'symbol': option['symbol'],
            'strike': option['strike'],
            'option_type': option['option_type'],
            'quantity': quantity,
            'entry_price': option['price'],
            'entry_date': self.datas[0].datetime.date(0),
            'expiry_date': option['expiry'],
            'delta': option['delta'],
            'gamma': option['gamma'],
            'theta': option['theta'],
            'vega': option['vega'],
            'signal_data': llm_signal.copy()
        }
        
        self.options_positions[position_id] = position
        
        # Update cash (synthetic)
        order_value = option['price'] * quantity * 100 + 20
        # Note: In real backtrader, this would be handled by the broker
        
        # Record trade
        self.trade_history.append({
            'action': action,
            'symbol': option['symbol'],
            'quantity': quantity,
            'price': option['price'],
            'date': self.datas[0].datetime.date(0),
            'position_id': position_id
        })
    
    def _close_all_positions(self):
        """Close all open options positions"""
        
        current_price = self.data_close[0]
        
        for pos_id in list(self.options_positions.keys()):
            position = self.options_positions[pos_id]
            
            # Generate current option price
            current_option_price = self._calculate_current_option_price(position, current_price)
            
            # Calculate P&L
            pnl = (current_option_price - position['entry_price']) * position['quantity'] * 100
            
            # Record trade
            self.trade_history.append({
                'action': 'close',
                'symbol': position['symbol'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': current_option_price,
                'pnl': pnl,
                'date': self.datas[0].datetime.date(0)
            })
            
            del self.options_positions[pos_id]
            
            logger.info(f"Closed {position['quantity']} {position['symbol']}, P&L: {pnl:.2f}")
    
    def _close_expiring_options(self):
        """Close options positions nearing expiry"""
        
        current_date = self.datas[0].datetime.date(0)
        current_price = self.data_close[0]
        
        for pos_id in list(self.options_positions.keys()):
            position = self.options_positions[pos_id]
            
            days_to_expiry = (position['expiry_date'] - current_date).days
            
            if days_to_expiry <= 1:  # Close positions 1 day before expiry
                current_option_price = self._calculate_current_option_price(position, current_price)
                
                # Calculate P&L
                pnl = (current_option_price - position['entry_price']) * position['quantity'] * 100
                
                # Record trade
                self.trade_history.append({
                    'action': 'expire_close',
                    'symbol': position['symbol'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_option_price,
                    'pnl': pnl,
                    'date': current_date
                })
                
                del self.options_positions[pos_id]
                
                logger.info(f"Closed expiring {position['quantity']} {position['symbol']}, P&L: {pnl:.2f}")
    
    def _calculate_current_option_price(self, position: Dict, underlying_price: float) -> float:
        """Calculate current option price based on position data"""
        
        strike = position['strike']
        option_type = position['option_type']
        
        # Simplified option pricing
        if option_type == 'CE':
            intrinsic = max(0, underlying_price - strike)
            time_value = max(5, 30 - abs(underlying_price - strike) * 0.1)  # Reduced time value
        else:
            intrinsic = max(0, strike - underlying_price)
            time_value = max(5, 30 - abs(underlying_price - strike) * 0.1)
        
        return intrinsic + time_value
    
    def _calculate_current_volatility(self) -> float:
        """Calculate current market volatility"""
        
        if len(self.underlying_price_history) < 10:
            return 0.20  # Default volatility
        
        prices = [h['price'] for h in self.underlying_price_history[-20:]]
        returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        
        current_value = self.broker.getvalue()
        cash = self.broker.getcash()
        
        # Calculate returns
        portfolio_return = (current_value - self.initial_portfolio_value) / self.initial_portfolio_value
        
        # Update max portfolio value and drawdown
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        
        current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Calculate options portfolio value
        options_value = 0
        current_price = self.data_close[0]
        
        for position in self.options_positions.values():
            current_option_price = self._calculate_current_option_price(position, current_price)
            position_value = current_option_price * position['quantity'] * 100
            options_value += position_value
        
        # Record portfolio state
        portfolio_state = {
            'date': self.datas[0].datetime.date(0),
            'total_value': current_value,
            'cash': cash,
            'options_value': options_value,
            'underlying_price': current_price,
            'return': portfolio_return,
            'max_drawdown': self.max_drawdown,
            'num_positions': len(self.options_positions)
        }
        
        self.portfolio_history.append(portfolio_state)
    
    def _perform_risk_checks(self):
        """Perform risk management checks"""
        
        if not self.p.risk_manager:
            return
        
        # Convert positions to risk format
        current_price = self.data_close[0]
        risk_positions = []
        
        for pos_id, position in self.options_positions.items():
            current_option_price = self._calculate_current_option_price(position, current_price)
            market_value = current_option_price * position['quantity'] * 100
            
            risk_position = {
                'position_id': pos_id,
                'symbol': position['symbol'],
                'market_value': market_value,
                'delta': position['delta'],
                'gamma': position['gamma'],
                'theta': position['theta'],
                'vega': position['vega'],
                'portfolio_value': self.broker.getvalue()
            }
            
            risk_positions.append(self.p.risk_manager.calculate_position_risk(risk_position))
        
        # Calculate portfolio risk
        portfolio_value = self.broker.getvalue()
        historical_returns = None
        
        if len(self.portfolio_history) > 10:
            returns = [h['return'] for h in self.portfolio_history[-252:]]  # Last year
            historical_returns = np.array(returns)
        
        risk_metrics = self.p.risk_manager.calculate_portfolio_risk(
            risk_positions, portfolio_value, historical_returns
        )
        
        # Check risk limits
        risk_checks = self.p.risk_manager.check_risk_limits(
            risk_positions, portfolio_value, risk_metrics
        )
        
        # Handle violations
        if risk_checks['violations']:
            logger.warning(f"Risk violations detected: {list(risk_checks['violations'].keys())}")
            
            # Take risk mitigation actions
            if 'concentration_single_position' in risk_checks['violations']:
                self._reduce_position_concentrations()
            
            if 'delta_exposure' in risk_checks['violations']:
                # Could implement delta hedging here
                pass
        
        # Store risk metrics
        self.risk_metrics_history.append({
            'date': self.datas[0].datetime.date(0),
            'risk_metrics': risk_metrics,
            'risk_checks': risk_checks
        })
    
    def _reduce_position_concentrations(self):
        """Reduce position concentrations if limits are breached"""
        
        if not self.options_positions:
            return
        
        # Find largest position
        largest_position = None
        largest_value = 0
        current_price = self.data_close[0]
        
        for pos_id, position in self.options_positions.items():
            current_option_price = self._calculate_current_option_price(position, current_price)
            position_value = current_option_price * position['quantity'] * 100
            
            if position_value > largest_value:
                largest_value = position_value
                largest_position = (pos_id, position)
        
        if largest_position:
            pos_id, position = largest_position
            
            # Reduce position by 50%
            reduce_quantity = position['quantity'] // 2
            
            if reduce_quantity > 0:
                current_option_price = self._calculate_current_option_price(position, current_price)
                
                # Calculate P&L for reduced portion
                pnl = (current_option_price - position['entry_price']) * reduce_quantity * 100
                
                # Update position
                position['quantity'] -= reduce_quantity
                
                # Record trade
                self.trade_history.append({
                    'action': 'risk_reduce',
                    'symbol': position['symbol'],
                    'quantity': reduce_quantity,
                    'entry_price': position['entry_price'],
                    'exit_price': current_option_price,
                    'pnl': pnl,
                    'date': self.datas[0].datetime.date(0),
                    'reason': 'concentration_risk'
                })
                
                logger.info(f"Reduced {reduce_quantity} {position['symbol']} due to concentration risk")
    
    def stop(self):
        """Called when strategy stops"""
        
        final_value = self.broker.getvalue()
        total_return = (final_value - self.initial_portfolio_value) / self.initial_portfolio_value
        
        print(f"\nStrategy Results:")
        print(f"Initial Portfolio Value: ${self.initial_portfolio_value:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")
        print(f"Total Trades: {len(self.trade_history)}")
        print(f"Final Positions: {len(self.options_positions)}")
        
        # Save results
        self._save_results()
        
        logger.info("Strategy completed")
    
    def _save_results(self):
        """Save strategy results to files"""
        
        results = {
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history,
            'risk_metrics_history': self.risk_metrics_history,
            'signal_history': self.signal_history,
            'final_positions': self.options_positions,
            'performance_summary': {
                'initial_value': self.initial_portfolio_value,
                'final_value': self.broker.getvalue(),
                'total_return': (self.broker.getvalue() - self.initial_portfolio_value) / self.initial_portfolio_value,
                'max_drawdown': self.max_drawdown,
                'total_trades': len(self.trade_history)
            }
        }
        
        # Create results directory
        os.makedirs('backtest_results', exist_ok=True)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'backtest_results/llm_rl_options_strategy_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")

class OptionsBacktester:
    """Main backtesting orchestrator"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cerebro = bt.Cerebro()
        
        # Setup cerebro
        self.cerebro.broker.setcash(config.initial_cash)
        self.cerebro.broker.setcommission(commission=config.commission)
        
        # Add analyzers
        self.cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        
        logger.info("OptionsBacktester initialized")
    
    def add_data(self, data_df: pd.DataFrame, name: str = 'NIFTY'):
        """Add underlying data to backtest"""
        
        # Ensure proper datetime index
        if 'Date' in data_df.columns:
            data_df.set_index('Date', inplace=True)
        
        data_df.index = pd.to_datetime(data_df.index)
        
        # Create backtrader data feed
        data_feed = bt.feeds.PandasData(
            dataname=data_df,
            name=name,
            fromdate=pd.to_datetime(self.config.start_date),
            todate=pd.to_datetime(self.config.end_date)
        )
        
        self.cerebro.adddata(data_feed)
        logger.info(f"Added data feed: {name}")
    
    def add_strategy(self, strategy_class, **strategy_params):
        """Add trading strategy to backtest"""
        
        self.cerebro.addstrategy(strategy_class, **strategy_params)
        logger.info(f"Added strategy: {strategy_class.__name__}")
    
    def run_backtest(self) -> Dict:
        """Run the backtest and return results"""
        
        logger.info("Starting backtest...")
        
        initial_value = self.cerebro.broker.getvalue()
        
        # Run backtest
        results = self.cerebro.run()
        
        final_value = self.cerebro.broker.getvalue()
        
        # Extract analyzer results
        strategy_results = results[0]
        
        analyzer_results = {}
        for analyzer_name in ['sharpe', 'drawdown', 'returns', 'trades']:
            if hasattr(strategy_results.analyzers, analyzer_name):
                analyzer = getattr(strategy_results.analyzers, analyzer_name)
                analyzer_results[analyzer_name] = analyzer.get_analysis()
        
        # Compile results
        backtest_results = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': (final_value - initial_value) / initial_value,
            'analyzer_results': analyzer_results,
            'config': self.config.__dict__
        }
        
        logger.info(f"Backtest completed. Return: {backtest_results['total_return']:.2%}")
        
        return backtest_results
    
    def plot_results(self):
        """Plot backtest results"""
        try:
            self.cerebro.plot(style='candlestick', volume=False)
        except Exception as e:
            logger.warning(f"Could not plot results: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    def generate_sample_data():
        """Generate sample NIFTY data for testing"""
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        price_data = []
        price = 18000
        
        for date in dates:
            # Random walk with trend
            daily_return = np.random.normal(0.001, 0.02)  # Slight positive trend with 2% daily vol
            price *= (1 + daily_return)
            
            # Ensure realistic ranges
            price = max(15000, min(price, 22000))
            
            price_data.append({
                'Date': date,
                'Open': price * (1 + np.random.normal(0, 0.001)),
                'High': price * (1 + abs(np.random.normal(0, 0.01))),
                'Low': price * (1 - abs(np.random.normal(0, 0.01))),
                'Close': price,
                'Volume': np.random.randint(5000000, 15000000)
            })
        
        return pd.DataFrame(price_data)
    
    # Mock components for testing
    class MockLLMSignalGenerator:
        def generate_signals(self, symbol):
            return {
                'direction': np.random.choice(['bullish', 'bearish', 'neutral']),
                'strength': np.random.uniform(0.3, 0.8),
                'confidence': np.random.uniform(0.5, 0.9)
            }
    
    class MockRLAgent:
        def __init__(self):
            self.action_size = 122  # Same as RL environment
        
        def select_action(self, state, training=False):
            return np.random.randint(0, self.action_size), 0.0
    
    # Run backtest
    def run_example_backtest():
        """Run example backtest"""
        
        # Generate data
        data_df = generate_sample_data()
        
        # Configuration
        config = BacktestConfig(
            initial_cash=1000000,
            commission=0.001,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Create backtester
        backtester = OptionsBacktester(config)
        
        # Add data
        backtester.add_data(data_df)
        
        # Create mock components
        llm_generator = MockLLMSignalGenerator()
        rl_agent = MockRLAgent()
        
        # Add strategy
        backtester.add_strategy(
            LLMRLOptionsStrategy,
            llm_signal_generator=llm_generator,
            rl_agent=rl_agent,
            risk_manager=None,  # Would use real risk manager
            position_sizer=None  # Would use real position sizer
        )
        
        # Run backtest
        results = backtester.run_backtest()
        
        print("\nBacktest Results:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Initial Value: ${results['initial_value']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        
        # Print analyzer results
        if 'sharpe' in results['analyzer_results']:
            sharpe = results['analyzer_results']['sharpe'].get('sharperatio', 'N/A')
            print(f"Sharpe Ratio: {sharpe}")
        
        if 'drawdown' in results['analyzer_results']:
            max_dd = results['analyzer_results']['drawdown'].get('max', {}).get('drawdown', 'N/A')
            print(f"Max Drawdown: {max_dd}")
    
    # Run the example
    run_example_backtest()