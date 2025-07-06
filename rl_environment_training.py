# RL Environment and Training System
# Part 3 of the complete LLM + RL Options Trading System

import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class OptionsPosition:
    """Represents an options position"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'CE' or 'PE'
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    positions: List[OptionsPosition]
    underlying_price: float
    total_value: float
    pnl: float
    delta_exposure: float
    gamma_exposure: float
    theta_exposure: float
    vega_exposure: float
    max_drawdown: float
    sharpe_ratio: float

class OptionsEnvironment(gym.Env):
    """Custom options trading environment for RL"""
    
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 options_data: pd.DataFrame,
                 initial_cash: float = 100000,
                 max_positions: int = 10,
                 commission: float = 20.0,
                 slippage: float = 0.01):
        
        super(OptionsEnvironment, self).__init__()
        
        # Environment parameters
        self.historical_data = historical_data
        self.options_data = options_data
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.commission = commission
        self.slippage = slippage
        
        # Current state
        self.current_step = 0
        self.cash = initial_cash
        self.positions = []
        self.portfolio_history = []
        self.trade_history = []
        
        # RL specific
        self.state_size = self._calculate_state_size()
        self.action_size = self._calculate_action_size()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_size,), 
            dtype=np.float32
        )
        
        # Risk management parameters
        self.max_position_value = initial_cash * 0.1  # 10% max per position
        self.max_delta_exposure = 0.5  # 50% delta neutral threshold
        self.max_total_risk = initial_cash * 0.3  # 30% max total risk
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_returns = []
        
        logger.info(f"Initialized OptionsEnvironment with {len(historical_data)} historical records")
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = []
        self.portfolio_history = []
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one trading step"""
        
        # Execute action
        reward, info = self._execute_action(action)
        
        # Update positions and portfolio state
        self._update_positions()
        
        # Calculate portfolio metrics
        portfolio_state = self._calculate_portfolio_state()
        
        # Record state
        self.portfolio_history.append(portfolio_state)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        next_obs = self._get_observation()
        
        return next_obs, reward, done, info
    
    def _execute_action(self, action) -> Tuple[float, Dict]:
        """Execute trading action and return reward"""
        
        info = {'action_type': 'hold', 'trade_executed': False}
        
        # Decode action
        action_type, strike_idx, quantity = self._decode_action(action)
        
        if action_type == 'hold':
            reward = self._calculate_hold_reward()
            return reward, info
        
        # Get current market data
        current_data = self._get_current_market_data()
        if current_data is None:
            return -0.1, info  # Penalty for invalid state
        
        # Get available options
        available_options = self._get_available_options(current_data)
        
        if not available_options or strike_idx >= len(available_options):
            return -0.1, info  # Penalty for invalid action
        
        selected_option = available_options[strike_idx]
        
        # Execute trade based on action type
        if action_type in ['buy_call', 'buy_put']:
            reward = self._execute_buy_order(selected_option, action_type, quantity)
            info['action_type'] = action_type
            info['trade_executed'] = True
            
        elif action_type in ['sell_call', 'sell_put']:
            reward = self._execute_sell_order(selected_option, action_type, quantity)
            info['action_type'] = action_type
            info['trade_executed'] = True
            
        elif action_type == 'close_all':
            reward = self._close_all_positions()
            info['action_type'] = 'close_all'
            info['trade_executed'] = True
        
        else:
            reward = -0.1  # Penalty for invalid action
        
        return reward, info
    
    def _execute_buy_order(self, option_data: Dict, action_type: str, quantity: int) -> float:
        """Execute buy order for options"""
        
        option_type = 'CE' if 'call' in action_type else 'PE'
        
        # Calculate order value
        option_price = option_data['ltp'] * (1 + self.slippage)  # Add slippage
        order_value = option_price * quantity * 100 + self.commission  # 100 shares per contract
        
        # Check if we have enough cash
        if order_value > self.cash:
            return -0.2  # Penalty for insufficient funds
        
        # Check position limits
        if len(self.positions) >= self.max_positions:
            return -0.1  # Penalty for too many positions
        
        # Check position size limit
        if order_value > self.max_position_value:
            return -0.1  # Penalty for position too large
        
        # Execute trade
        position = OptionsPosition(
            symbol=option_data['symbol'],
            strike=option_data['strike'],
            expiry=option_data['expiry'],
            option_type=option_type,
            quantity=quantity,
            entry_price=option_price,
            current_price=option_price,
            entry_time=datetime.now(),
            delta=option_data.get('delta', 0),
            gamma=option_data.get('gamma', 0),
            theta=option_data.get('theta', 0),
            vega=option_data.get('vega', 0)
        )
        
        self.positions.append(position)
        self.cash -= order_value
        
        # Record trade
        self.trade_history.append({
            'action': 'buy',
            'symbol': position.symbol,
            'quantity': quantity,
            'price': option_price,
            'timestamp': datetime.now(),
            'value': order_value
        })
        
        # Calculate reward (immediate cost + potential reward for good positioning)
        portfolio_delta = self._calculate_portfolio_delta()
        delta_reward = -abs(portfolio_delta) * 0.1  # Reward for delta neutral
        
        return -0.1 + delta_reward  # Small cost for trading + delta reward
    
    def _execute_sell_order(self, option_data: Dict, action_type: str, quantity: int) -> float:
        """Execute sell order (close existing positions)"""
        
        option_type = 'CE' if 'call' in action_type else 'PE'
        strike = option_data['strike']
        
        # Find matching position to close
        matching_positions = [
            p for p in self.positions 
            if p.option_type == option_type and p.strike == strike
        ]
        
        if not matching_positions:
            return -0.1  # Penalty for trying to sell non-existent position
        
        # Close position (FIFO)
        position = matching_positions[0]
        close_quantity = min(quantity, position.quantity)
        
        # Calculate close value
        close_price = option_data['ltp'] * (1 - self.slippage)  # Subtract slippage
        close_value = close_price * close_quantity * 100 - self.commission
        
        # Calculate P&L
        entry_value = position.entry_price * close_quantity * 100
        pnl = close_value - entry_value
        
        # Update position or remove if fully closed
        if close_quantity == position.quantity:
            self.positions.remove(position)
        else:
            position.quantity -= close_quantity
        
        self.cash += close_value
        
        # Record trade
        self.trade_history.append({
            'action': 'sell',
            'symbol': position.symbol,
            'quantity': close_quantity,
            'price': close_price,
            'timestamp': datetime.now(),
            'pnl': pnl
        })
        
        # Reward based on P&L
        return pnl / 1000.0  # Scale P&L to reasonable reward range
    
    def _close_all_positions(self) -> float:
        """Close all open positions"""
        total_pnl = 0.0
        
        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            current_data = self._get_current_market_data()
            if current_data is None:
                continue
            
            # Find current option price
            matching_options = [
                opt for opt in self._get_available_options(current_data)
                if opt['strike'] == position.strike and 
                   ('CE' in opt['symbol'] if position.option_type == 'CE' else 'PE' in opt['symbol'])
            ]
            
            if matching_options:
                option_data = matching_options[0]
                close_price = option_data['ltp'] * (1 - self.slippage)
                close_value = close_price * position.quantity * 100 - self.commission
                entry_value = position.entry_price * position.quantity * 100
                pnl = close_value - entry_value
                
                total_pnl += pnl
                self.cash += close_value
                
                # Record trade
                self.trade_history.append({
                    'action': 'close',
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'price': close_price,
                    'timestamp': datetime.now(),
                    'pnl': pnl
                })
        
        # Clear all positions
        self.positions = []
        
        return total_pnl / 1000.0  # Scale P&L
    
    def _calculate_hold_reward(self) -> float:
        """Calculate reward for holding current positions"""
        
        if not self.positions:
            return 0.0  # No reward/penalty for holding cash
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        theta_decay = 0.0
        
        current_data = self._get_current_market_data()
        if current_data is None:
            return -0.1
        
        available_options = self._get_available_options(current_data)
        
        for position in self.positions:
            # Find current option price
            matching_options = [
                opt for opt in available_options
                if opt['strike'] == position.strike and 
                   ('CE' in opt['symbol'] if position.option_type == 'CE' else 'PE' in opt['symbol'])
            ]
            
            if matching_options:
                current_price = matching_options[0]['ltp']
                position.current_price = current_price
                
                # Calculate unrealized P&L
                position_pnl = (current_price - position.entry_price) * position.quantity * 100
                unrealized_pnl += position_pnl
                
                # Calculate theta decay
                theta_decay += position.theta * position.quantity * 100
        
        # Reward/penalty based on P&L and time decay
        pnl_reward = unrealized_pnl / 1000.0
        theta_penalty = abs(theta_decay) / 1000.0
        
        return pnl_reward - theta_penalty
    
    def _decode_action(self, action: int) -> Tuple[str, int, int]:
        """Decode action integer into trading action"""
        
        # Action space: [hold, buy_call, buy_put, sell_call, sell_put, close_all]
        # For each buy/sell action, we have multiple strikes and quantities
        
        action_types = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 'close_all']
        
        if action == 0:
            return 'hold', 0, 0
        elif action == len(action_types) - 1:
            return 'close_all', 0, 0
        
        # Decode buy/sell actions
        action_idx = action - 1  # Remove hold action
        num_strikes = 10  # Number of strike choices
        num_quantities = 3  # Number of quantity choices [1, 3, 5]
        
        action_type_idx = action_idx // (num_strikes * num_quantities)
        remaining = action_idx % (num_strikes * num_quantities)
        
        strike_idx = remaining // num_quantities
        quantity_idx = remaining % num_quantities
        
        action_type = action_types[action_type_idx + 1]  # Skip 'hold'
        quantities = [1, 3, 5]
        quantity = quantities[quantity_idx]
        
        return action_type, strike_idx, quantity
    
    def _calculate_action_size(self) -> int:
        """Calculate total action space size"""
        # hold + close_all + (4 action types * 10 strikes * 3 quantities)
        return 1 + 1 + (4 * 10 * 3)
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        
        # Market data features
        market_features = self._get_market_features()
        
        # Portfolio features
        portfolio_features = self._get_portfolio_features()
        
        # Technical indicators
        technical_features = self._get_technical_features()
        
        # LLM signal features (will be integrated separately)
        signal_features = self._get_signal_features()
        
        # Combine all features
        observation = np.concatenate([
            market_features,
            portfolio_features,
            technical_features,
            signal_features
        ])
        
        return observation.astype(np.float32)
    
    def _get_market_features(self) -> np.ndarray:
        """Get market-related features"""
        current_data = self._get_current_market_data()
        
        if current_data is None:
            return np.zeros(10)
        
        # Underlying price features
        features = [
            current_data.get('ltp', 0),
            current_data.get('change', 0),
            current_data.get('change_percent', 0),
            current_data.get('volume', 0) / 1000000,  # Scale volume
            current_data.get('high', 0),
            current_data.get('low', 0),
            current_data.get('open', 0),
            current_data.get('vix', 20),  # Default VIX
            current_data.get('pcr', 1),   # Put-Call Ratio
            self.current_step / len(self.historical_data)  # Time progress
        ]
        
        return np.array(features)
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Get portfolio-related features"""
        
        portfolio_state = self._calculate_portfolio_state()
        
        features = [
            portfolio_state.cash / self.initial_cash,  # Normalized cash
            portfolio_state.total_value / self.initial_cash,  # Normalized portfolio value
            portfolio_state.pnl / self.initial_cash,  # Normalized P&L
            len(self.positions) / self.max_positions,  # Position count ratio
            portfolio_state.delta_exposure,
            portfolio_state.gamma_exposure,
            portfolio_state.theta_exposure,
            portfolio_state.vega_exposure,
            portfolio_state.max_drawdown,
            portfolio_state.sharpe_ratio if portfolio_state.sharpe_ratio is not None else 0
        ]
        
        return np.array(features)
    
    def _get_technical_features(self) -> np.ndarray:
        """Get technical analysis features"""
        
        if self.current_step < 20:
            return np.zeros(10)
        
        # Get recent price data
        recent_data = self.historical_data.iloc[max(0, self.current_step-20):self.current_step+1]
        
        if len(recent_data) < 10:
            return np.zeros(10)
        
        close_prices = recent_data['Close'].values
        
        # Calculate technical indicators
        try:
            import talib
            
            # Trend indicators
            sma_10 = talib.SMA(close_prices, timeperiod=10)[-1] if len(close_prices) >= 10 else close_prices[-1]
            sma_20 = talib.SMA(close_prices, timeperiod=min(20, len(close_prices)))[-1] if len(close_prices) >= 5 else close_prices[-1]
            
            # Momentum indicators
            rsi = talib.RSI(close_prices, timeperiod=min(14, len(close_prices)))[-1] if len(close_prices) >= 5 else 50
            
            # Volatility
            atr = talib.ATR(recent_data['High'].values, recent_data['Low'].values, close_prices, timeperiod=min(14, len(close_prices)))[-1] if len(close_prices) >= 5 else 0
            
            features = [
                close_prices[-1] / sma_10 - 1,  # Price vs SMA10
                close_prices[-1] / sma_20 - 1,  # Price vs SMA20
                rsi / 100 - 0.5,  # Normalized RSI
                atr / close_prices[-1],  # Normalized ATR
                (close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) >= 5 else 0,  # 5-day return
                (close_prices[-1] - close_prices[-10]) / close_prices[-10] if len(close_prices) >= 10 else 0,  # 10-day return
                np.std(close_prices[-10:]) / np.mean(close_prices[-10:]) if len(close_prices) >= 10 else 0,  # Volatility
                recent_data['Volume'].iloc[-1] / recent_data['Volume'].mean() - 1,  # Volume ratio
                (recent_data['High'].iloc[-1] - recent_data['Low'].iloc[-1]) / recent_data['Close'].iloc[-1],  # Daily range
                0  # Placeholder for additional indicator
            ]
            
        except ImportError:
            # Fallback if talib not available
            features = [
                (close_prices[-1] - np.mean(close_prices[-10:])) / np.mean(close_prices[-10:]) if len(close_prices) >= 10 else 0,
                (close_prices[-1] - close_prices[-1]) / close_prices[-1] if len(close_prices) >= 2 else 0,
                0, 0, 0, 0, 0, 0, 0, 0
            ]
        
        return np.array(features)
    
    def _get_signal_features(self) -> np.ndarray:
        """Get LLM signal features (placeholder for integration)"""
        # This will be integrated with the LLM signal generator
        # For now, return dummy features
        return np.zeros(6)  # [fundamental_strength, fundamental_direction, technical_strength, technical_direction, sentiment_strength, sentiment_direction]
    
    def _calculate_state_size(self) -> int:
        """Calculate total state vector size"""
        return 10 + 10 + 10 + 6  # market + portfolio + technical + signals
    
    def _get_current_market_data(self) -> Optional[Dict]:
        """Get current market data for the step"""
        if self.current_step >= len(self.historical_data):
            return None
        
        current_row = self.historical_data.iloc[self.current_step]
        return current_row.to_dict()
    
    def _get_available_options(self, current_data: Dict) -> List[Dict]:
        """Get available options for current step"""
        
        # Filter options data for current timestamp
        current_timestamp = current_data.get('timestamp', datetime.now())
        underlying_price = current_data.get('ltp', 0)
        
        # Generate option strikes around current price
        strikes = []
        for i in range(-5, 6):  # 10 strikes total
            strike = round(underlying_price + i * 100, -2)  # Round to nearest 100
            strikes.append(strike)
        
        # Create options data (simplified)
        options = []
        for strike in strikes:
            for option_type in ['CE', 'PE']:
                # Simplified option pricing
                if option_type == 'CE':
                    intrinsic = max(0, underlying_price - strike)
                    time_value = max(10, 50 - abs(underlying_price - strike) * 0.1)
                else:
                    intrinsic = max(0, strike - underlying_price)
                    time_value = max(10, 50 - abs(underlying_price - strike) * 0.1)
                
                ltp = intrinsic + time_value
                
                options.append({
                    'symbol': f"NIFTY{strike}{option_type}",
                    'strike': strike,
                    'option_type': option_type,
                    'ltp': ltp,
                    'expiry': '2024-12-26',
                    'delta': 0.5 if option_type == 'CE' else -0.5,
                    'gamma': 0.01,
                    'theta': -2.0,
                    'vega': 15.0
                })
        
        return options
    
    def _update_positions(self):
        """Update all position values and Greeks"""
        current_data = self._get_current_market_data()
        if current_data is None:
            return
        
        available_options = self._get_available_options(current_data)
        
        for position in self.positions:
            # Find matching option
            matching_options = [
                opt for opt in available_options
                if opt['strike'] == position.strike and opt['option_type'] == position.option_type
            ]
            
            if matching_options:
                option_data = matching_options[0]
                position.current_price = option_data['ltp']
                position.delta = option_data['delta']
                position.gamma = option_data['gamma']
                position.theta = option_data['theta']
                position.vega = option_data['vega']
    
    def _calculate_portfolio_state(self) -> PortfolioState:
        """Calculate current portfolio state"""
        
        # Calculate position values
        total_position_value = 0.0
        total_pnl = 0.0
        delta_exposure = 0.0
        gamma_exposure = 0.0
        theta_exposure = 0.0
        vega_exposure = 0.0
        
        for position in self.positions:
            position_value = position.current_price * position.quantity * 100
            position_pnl = (position.current_price - position.entry_price) * position.quantity * 100
            
            total_position_value += position_value
            total_pnl += position_pnl
            
            delta_exposure += position.delta * position.quantity * 100
            gamma_exposure += position.gamma * position.quantity * 100
            theta_exposure += position.theta * position.quantity * 100
            vega_exposure += position.vega * position.quantity * 100
        
        total_value = self.cash + total_position_value
        
        # Calculate performance metrics
        returns = [h.pnl / self.initial_cash for h in self.portfolio_history] if self.portfolio_history else [0]
        max_drawdown = self._calculate_max_drawdown(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        return PortfolioState(
            cash=self.cash,
            positions=self.positions.copy(),
            underlying_price=self._get_current_market_data().get('ltp', 0) if self._get_current_market_data() else 0,
            total_value=total_value,
            pnl=total_pnl,
            delta_exposure=delta_exposure / self.initial_cash,  # Normalized
            gamma_exposure=gamma_exposure / self.initial_cash,
            theta_exposure=theta_exposure / self.initial_cash,
            vega_exposure=vega_exposure / self.initial_cash,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )
    
    def _calculate_portfolio_delta(self) -> float:
        """Calculate total portfolio delta"""
        total_delta = sum(pos.delta * pos.quantity * 100 for pos in self.positions)
        return total_delta / self.initial_cash  # Normalized
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> Optional[float]:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return None
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return None
        
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def _is_episode_done(self) -> bool:
        """Check if episode is finished"""
        
        # Episode ends if:
        # 1. We've reached the end of data
        # 2. Portfolio value drops too much
        # 3. Maximum steps reached
        
        if self.current_step >= len(self.historical_data) - 1:
            return True
        
        portfolio_state = self._calculate_portfolio_state()
        if portfolio_state.total_value < self.initial_cash * 0.5:  # 50% loss limit
            return True
        
        if self.current_step > 1000:  # Maximum episode length
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            portfolio_state = self._calculate_portfolio_state()
            print(f"Step: {self.current_step}")
            print(f"Cash: ${portfolio_state.cash:,.2f}")
            print(f"Total Value: ${portfolio_state.total_value:,.2f}")
            print(f"P&L: ${portfolio_state.pnl:,.2f}")
            print(f"Positions: {len(self.positions)}")
            print(f"Delta Exposure: {portfolio_state.delta_exposure:.3f}")
            print("-" * 40)

class PPOAgent:
    """PPO Agent for options trading"""
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 device: str = 'cpu'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        # Networks
        self.policy = ActorCriticNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory()
        
        # Training metrics
        self.training_history = []
        
    def select_action(self, state, training=True):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
            dist = Categorical(action_probs)
            
            if training:
                action = dist.sample()
            else:
                action = torch.argmax(action_probs, dim=-1)
            
            action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.item()
    
    def store_transition(self, state, action, action_logprob, reward, done):
        """Store transition in memory"""
        self.memory.store(state, action, action_logprob, reward, done)
    
    def update(self):
        """Update policy using PPO"""
        
        # Calculate discounted rewards
        rewards = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Convert to tensors
        old_states = torch.stack(self.memory.states).to(self.device)
        old_actions = torch.tensor(self.memory.actions).to(self.device)
        old_logprobs = torch.tensor(self.memory.logprobs).to(self.device)
        
        # PPO update
        total_loss = 0
        for _ in range(self.k_epochs):
            # Evaluate actions
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Calculate advantage
            advantages = rewards - state_values.squeeze()
            
            # Ratio for surrogate loss
            ratios = torch.exp(action_logprobs - old_logprobs)
            
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Combined loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), rewards)
            entropy_loss = -dist_entropy.mean()
            
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Record training metrics
        self.training_history.append({
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        })
        
        # Clear memory
        self.memory.clear()
        
        return total_loss.item()

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        shared_output = self.shared(x)
        action_probs = self.actor(shared_output)
        state_value = self.critic(shared_output)
        return action_probs, state_value

class PPOMemory:
    """Memory buffer for PPO"""
    
    def __init__(self):
        self.clear()
    
    def store(self, state, action, action_logprob, reward, done):
        self.states.append(torch.FloatTensor(state))
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

# Training function
def train_ppo_agent(env, agent, episodes=1000, max_steps=1000):
    """Train PPO agent on options trading environment"""
    
    episode_rewards = []
    episode_returns = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action, action_logprob = agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, action_logprob, reward, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update policy
        if len(agent.memory.states) > 0:
            loss = agent.update()
        
        episode_rewards.append(episode_reward)
        
        # Calculate episode return
        portfolio_state = env._calculate_portfolio_state()
        episode_return = (portfolio_state.total_value - env.initial_cash) / env.initial_cash
        episode_returns.append(episode_return)
        
        # Logging
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_return = np.mean(episode_returns[-50:])
            logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, Avg Return = {avg_return:.3f}")
    
    return episode_rewards, episode_returns

# Example usage
if __name__ == "__main__":
    # Generate dummy historical data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    historical_data = pd.DataFrame({
        'Date': dates,
        'Open': 18000 + np.cumsum(np.random.randn(len(dates)) * 50),
        'High': 18000 + np.cumsum(np.random.randn(len(dates)) * 50) + 100,
        'Low': 18000 + np.cumsum(np.random.randn(len(dates)) * 50) - 100,
        'Close': 18000 + np.cumsum(np.random.randn(len(dates)) * 50),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Create environment
    env = OptionsEnvironment(historical_data, pd.DataFrame(), initial_cash=100000)
    
    # Create agent
    agent = PPOAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        lr=3e-4,
        device='cpu'
    )
    
    # Train agent
    logger.info("Starting PPO training...")
    episode_rewards, episode_returns = train_ppo_agent(env, agent, episodes=100, max_steps=100)
    
    logger.info(f"Training completed. Final average return: {np.mean(episode_returns[-10:]):.3f}")
