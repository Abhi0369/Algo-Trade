# Risk Management and Portfolio Management System
# Part 4 of the complete LLM + RL Options Trading System

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import json
import warnings
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio analysis"""
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    volatility: float = 0.0

@dataclass
class PositionRisk:
    """Risk metrics for individual positions"""
    position_id: str
    symbol: str
    market_value: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    var_contribution: float
    stress_test_pnl: Dict[str, float] = field(default_factory=dict)
    concentration_risk: float = 0.0

@dataclass
class RiskLimit:
    """Risk limit definition"""
    name: str
    limit_type: str  # 'absolute', 'percentage', 'greek'
    limit_value: float
    warning_threshold: float
    current_value: float = 0.0
    breach_count: int = 0
    last_breach: Optional[datetime] = None

class RiskEngine(ABC):
    """Abstract base class for risk engines"""
    
    @abstractmethod
    def calculate_var(self, portfolio_returns: np.ndarray, confidence_level: float = 0.95) -> float:
        pass
    
    @abstractmethod
    def calculate_expected_shortfall(self, portfolio_returns: np.ndarray, confidence_level: float = 0.95) -> float:
        pass
    
    @abstractmethod
    def stress_test(self, positions: List[PositionRisk], scenarios: Dict[str, float]) -> Dict[str, float]:
        pass

class MonteCarloRiskEngine(RiskEngine):
    """Monte Carlo simulation based risk engine"""
    
    def __init__(self, num_simulations: int = 10000, time_horizon: int = 1):
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
    
    def calculate_var(self, portfolio_returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(portfolio_returns)
        
        # Calculate VaR at confidence level
        index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[index] if index < len(sorted_returns) else 0.0
        
        return var
    
    def calculate_expected_shortfall(self, portfolio_returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        var = self.calculate_var(portfolio_returns, confidence_level)
        
        # Calculate average of returns worse than VaR
        tail_returns = portfolio_returns[portfolio_returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        expected_shortfall = -np.mean(tail_returns)
        return expected_shortfall
    
    def stress_test(self, positions: List[PositionRisk], scenarios: Dict[str, float]) -> Dict[str, float]:
        """Perform stress testing on portfolio"""
        stress_results = {}
        
        for scenario_name, market_shock in scenarios.items():
            total_pnl = 0.0
            
            for position in positions:
                # Calculate P&L impact based on Greeks
                underlying_move = market_shock
                vol_move = market_shock * 0.3  # Assume vol moves 30% of underlying move
                time_decay = -abs(position.theta)  # Daily theta decay
                
                # Delta P&L
                delta_pnl = position.delta * underlying_move * position.market_value / 100
                
                # Gamma P&L (second order)
                gamma_pnl = 0.5 * position.gamma * (underlying_move ** 2) * position.market_value / 100
                
                # Vega P&L
                vega_pnl = position.vega * vol_move * position.market_value / 100
                
                # Theta P&L
                theta_pnl = time_decay * position.market_value / 100
                
                position_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
                total_pnl += position_pnl
            
            stress_results[scenario_name] = total_pnl
        
        return stress_results
    
    def monte_carlo_simulation(self, 
                              positions: List[PositionRisk], 
                              underlying_vol: float = 0.2,
                              correlation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Run Monte Carlo simulation for portfolio P&L"""
        
        if not positions:
            return np.array([0.0])
        
        # Generate random price movements
        dt = self.time_horizon / 252  # Convert to fraction of year
        price_moves = np.random.normal(0, underlying_vol * np.sqrt(dt), self.num_simulations)
        
        portfolio_pnl = np.zeros(self.num_simulations)
        
        for i, price_move in enumerate(price_moves):
            sim_pnl = 0.0
            
            for position in positions:
                # Calculate position P&L for this simulation
                delta_pnl = position.delta * price_move * position.market_value / 100
                gamma_pnl = 0.5 * position.gamma * (price_move ** 2) * position.market_value / 100
                theta_pnl = position.theta * dt * position.market_value / 100
                
                position_pnl = delta_pnl + gamma_pnl + theta_pnl
                sim_pnl += position_pnl
            
            portfolio_pnl[i] = sim_pnl
        
        return portfolio_pnl

class ParametricRiskEngine(RiskEngine):
    """Parametric risk engine using normal distribution assumptions"""
    
    def __init__(self):
        pass
    
    def calculate_var(self, portfolio_returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate parametric VaR"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Parametric VaR
        var = -(mean_return + z_score * std_return)
        
        return var
    
    def calculate_expected_shortfall(self, portfolio_returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate parametric Expected Shortfall"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Expected shortfall for normal distribution
        lambda_val = stats.norm.pdf(z_score) / (1 - confidence_level)
        expected_shortfall = -(mean_return - std_return * lambda_val)
        
        return expected_shortfall
    
    def stress_test(self, positions: List[PositionRisk], scenarios: Dict[str, float]) -> Dict[str, float]:
        """Parametric stress testing"""
        # Use same logic as Monte Carlo for simplicity
        monte_carlo_engine = MonteCarloRiskEngine()
        return monte_carlo_engine.stress_test(positions, scenarios)

class RiskManager:
    """Main risk management class"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize risk engines
        self.monte_carlo_engine = MonteCarloRiskEngine(
            num_simulations=config.get('monte_carlo_simulations', 10000)
        )
        self.parametric_engine = ParametricRiskEngine()
        
        # Risk limits
        self.risk_limits = self._initialize_risk_limits(config)
        
        # Historical data for risk calculations
        self.portfolio_returns_history = []
        self.risk_metrics_history = []
        
        # Stress test scenarios
        self.stress_scenarios = {
            'market_crash': -0.20,     # 20% market drop
            'volatility_spike': 0.10,   # 10% underlying rise with vol spike
            'black_swan': -0.30,       # 30% market crash
            'flash_crash': -0.15,      # 15% rapid decline
            'moderate_correction': -0.10  # 10% correction
        }
        
        logger.info("Risk Manager initialized with parametric and Monte Carlo engines")
    
    def _initialize_risk_limits(self, config: Dict) -> List[RiskLimit]:
        """Initialize risk limits from configuration"""
        limits_config = config.get('risk_limits', {})
        
        default_limits = [
            RiskLimit('portfolio_var_95', 'percentage', 0.05, 0.04),  # 5% VaR limit, 4% warning
            RiskLimit('portfolio_var_99', 'percentage', 0.10, 0.08),  # 10% VaR limit
            RiskLimit('max_drawdown', 'percentage', 0.15, 0.12),     # 15% max drawdown
            RiskLimit('delta_exposure', 'percentage', 0.50, 0.40),   # 50% delta exposure
            RiskLimit('gamma_exposure', 'percentage', 0.30, 0.25),   # 30% gamma exposure
            RiskLimit('theta_exposure', 'absolute', 500, 400),       # Daily theta limit
            RiskLimit('vega_exposure', 'percentage', 0.20, 0.15),    # 20% vega exposure
            RiskLimit('concentration_single_position', 'percentage', 0.10, 0.08),  # 10% single position
            RiskLimit('concentration_single_strike', 'percentage', 0.15, 0.12),    # 15% single strike
            RiskLimit('leverage_ratio', 'absolute', 2.0, 1.8),       # 2x leverage limit
        ]
        
        # Override with config values
        risk_limits = []
        for limit in default_limits:
            config_limit = limits_config.get(limit.name, {})
            if config_limit:
                limit.limit_value = config_limit.get('limit_value', limit.limit_value)
                limit.warning_threshold = config_limit.get('warning_threshold', limit.warning_threshold)
            risk_limits.append(limit)
        
        return risk_limits
    
    def calculate_portfolio_risk(self, 
                                positions: List[PositionRisk], 
                                portfolio_value: float,
                                historical_returns: Optional[np.ndarray] = None) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not positions:
            return RiskMetrics()
        
        # Use historical returns if available, otherwise simulate
        if historical_returns is not None and len(historical_returns) > 0:
            portfolio_returns = historical_returns
        else:
            # Generate simulated returns for risk calculation
            portfolio_returns = self.monte_carlo_engine.monte_carlo_simulation(positions)
            portfolio_returns = portfolio_returns / portfolio_value  # Convert to returns
        
        # Calculate VaR
        var_95 = self.monte_carlo_engine.calculate_var(portfolio_returns, 0.95)
        var_99 = self.monte_carlo_engine.calculate_var(portfolio_returns, 0.99)
        
        # Calculate Expected Shortfall
        expected_shortfall = self.monte_carlo_engine.calculate_expected_shortfall(portfolio_returns, 0.95)
        
        # Calculate other risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming 5% risk-free rate)
        risk_free_rate = 0.05
        mean_return = np.mean(portfolio_returns) * 252  # Annualized
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        risk_metrics = RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            volatility=volatility
        )
        
        # Store for historical tracking
        self.risk_metrics_history.append({
            'timestamp': datetime.now(),
            'risk_metrics': risk_metrics,
            'portfolio_value': portfolio_value
        })
        
        return risk_metrics
    
    def check_risk_limits(self, 
                         positions: List[PositionRisk], 
                         portfolio_value: float,
                         risk_metrics: RiskMetrics) -> Dict[str, Dict]:
        """Check all risk limits and return violations"""
        
        violations = {}
        warnings = {}
        
        # Calculate current exposure values
        total_delta = sum(pos.delta * pos.market_value for pos in positions)
        total_gamma = sum(pos.gamma * pos.market_value for pos in positions)
        total_theta = sum(pos.theta * pos.market_value for pos in positions)
        total_vega = sum(pos.vega * pos.market_value for pos in positions)
        
        # Check each limit
        for limit in self.risk_limits:
            current_value = 0.0
            
            if limit.name == 'portfolio_var_95':
                current_value = risk_metrics.var_95
            elif limit.name == 'portfolio_var_99':
                current_value = risk_metrics.var_99
            elif limit.name == 'max_drawdown':
                current_value = abs(risk_metrics.max_drawdown)
            elif limit.name == 'delta_exposure':
                current_value = abs(total_delta) / portfolio_value
            elif limit.name == 'gamma_exposure':
                current_value = abs(total_gamma) / portfolio_value
            elif limit.name == 'theta_exposure':
                current_value = abs(total_theta)
            elif limit.name == 'vega_exposure':
                current_value = abs(total_vega) / portfolio_value
            elif limit.name == 'concentration_single_position':
                max_position = max((pos.market_value for pos in positions), default=0)
                current_value = max_position / portfolio_value
            elif limit.name == 'leverage_ratio':
                total_exposure = sum(abs(pos.market_value) for pos in positions)
                current_value = total_exposure / portfolio_value
            
            # Update limit current value
            limit.current_value = current_value
            
            # Check for violations
            if current_value > limit.limit_value:
                violations[limit.name] = {
                    'current_value': current_value,
                    'limit_value': limit.limit_value,
                    'breach_ratio': current_value / limit.limit_value,
                    'limit_type': limit.limit_type
                }
                limit.breach_count += 1
                limit.last_breach = datetime.now()
                
            # Check for warnings
            elif current_value > limit.warning_threshold:
                warnings[limit.name] = {
                    'current_value': current_value,
                    'warning_threshold': limit.warning_threshold,
                    'limit_value': limit.limit_value,
                    'warning_ratio': current_value / limit.warning_threshold
                }
        
        if violations:
            logger.warning(f"Risk limit violations detected: {list(violations.keys())}")
        if warnings:
            logger.info(f"Risk limit warnings: {list(warnings.keys())}")
        
        return {'violations': violations, 'warnings': warnings}
    
    def stress_test_portfolio(self, positions: List[PositionRisk]) -> Dict[str, Dict]:
        """Perform comprehensive stress testing"""
        
        # Monte Carlo stress test
        mc_results = self.monte_carlo_engine.stress_test(positions, self.stress_scenarios)
        
        # Parametric stress test
        param_results = self.parametric_engine.stress_test(positions, self.stress_scenarios)
        
        # Additional scenario analysis
        custom_scenarios = {
            'india_specific_crisis': -0.25,    # India-specific market crisis
            'global_recession': -0.18,         # Global recession impact
            'currency_devaluation': -0.12,     # INR devaluation impact
            'interest_rate_shock': 0.08,       # Interest rate spike
            'earnings_disappointment': -0.15   # Broad earnings miss
        }
        
        custom_results = self.monte_carlo_engine.stress_test(positions, custom_scenarios)
        
        return {
            'standard_scenarios': mc_results,
            'parametric_scenarios': param_results,
            'custom_scenarios': custom_results
        }
    
    def calculate_position_risk(self, position_data: Dict) -> PositionRisk:
        """Calculate risk metrics for individual position"""
        
        # Extract position details
        symbol = position_data['symbol']
        market_value = position_data['market_value']
        delta = position_data.get('delta', 0)
        gamma = position_data.get('gamma', 0)
        theta = position_data.get('theta', 0)
        vega = position_data.get('vega', 0)
        rho = position_data.get('rho', 0)
        
        # Calculate VaR contribution (simplified)
        var_contribution = abs(delta * market_value * 0.02)  # 2% market move
        
        # Calculate concentration risk
        portfolio_value = position_data.get('portfolio_value', 1000000)
        concentration_risk = market_value / portfolio_value
        
        # Stress test scenarios for individual position
        stress_scenarios = {
            'underlying_down_10': -0.10,
            'underlying_up_10': 0.10,
            'vol_spike_50': 0.0,  # Pure volatility move
            'time_decay_1day': 0.0  # Pure time decay
        }
        
        stress_test_pnl = {}
        for scenario, move in stress_scenarios.items():
            if 'vol_spike' in scenario:
                pnl = vega * 0.5 * market_value / 100  # 50% vol increase
            elif 'time_decay' in scenario:
                pnl = theta * market_value / 100  # 1 day time decay
            else:
                # Price movement scenarios
                delta_pnl = delta * move * market_value / 100
                gamma_pnl = 0.5 * gamma * (move ** 2) * market_value / 100
                pnl = delta_pnl + gamma_pnl
            
            stress_test_pnl[scenario] = pnl
        
        return PositionRisk(
            position_id=position_data.get('position_id', f"{symbol}_{datetime.now().timestamp()}"),
            symbol=symbol,
            market_value=market_value,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            var_contribution=var_contribution,
            stress_test_pnl=stress_test_pnl,
            concentration_risk=concentration_risk
        )
    
    def calculate_optimal_hedge_ratio(self, positions: List[PositionRisk], hedge_instrument: str = 'NIFTY_FUTURE') -> float:
        """Calculate optimal hedge ratio for delta hedging"""
        
        if not positions:
            return 0.0
        
        # Calculate total portfolio delta
        total_delta = sum(pos.delta * pos.market_value for pos in positions)
        
        # Assume hedge instrument has delta of 1 (futures)
        hedge_delta = 1.0
        
        # Calculate hedge ratio
        hedge_ratio = -total_delta / hedge_delta
        
        return hedge_ratio
    
    def generate_risk_report(self, 
                           positions: List[PositionRisk], 
                           portfolio_value: float,
                           risk_metrics: RiskMetrics) -> Dict:
        """Generate comprehensive risk report"""
        
        # Risk limit checks
        risk_checks = self.check_risk_limits(positions, portfolio_value, risk_metrics)
        
        # Stress test results
        stress_results = self.stress_test_portfolio(positions)
        
        # Portfolio exposures
        total_delta = sum(pos.delta * pos.market_value for pos in positions)
        total_gamma = sum(pos.gamma * pos.market_value for pos in positions)
        total_theta = sum(pos.theta * pos.market_value for pos in positions)
        total_vega = sum(pos.vega * pos.market_value for pos in positions)
        
        # Position concentration analysis
        position_concentrations = [
            {
                'symbol': pos.symbol,
                'market_value': pos.market_value,
                'concentration': pos.market_value / portfolio_value,
                'var_contribution': pos.var_contribution
            }
            for pos in positions
        ]
        
        # Sort by concentration
        position_concentrations.sort(key=lambda x: x['concentration'], reverse=True)
        
        # Recommended hedge ratio
        hedge_ratio = self.calculate_optimal_hedge_ratio(positions)
        
        risk_report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'total_value': portfolio_value,
                'number_of_positions': len(positions),
                'delta_exposure': total_delta,
                'gamma_exposure': total_gamma,
                'theta_exposure': total_theta,
                'vega_exposure': total_vega
            },
            'risk_metrics': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'expected_shortfall': risk_metrics.expected_shortfall,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'volatility': risk_metrics.volatility
            },
            'risk_limit_status': risk_checks,
            'stress_test_results': stress_results,
            'position_concentrations': position_concentrations[:10],  # Top 10
            'hedge_recommendations': {
                'optimal_hedge_ratio': hedge_ratio,
                'hedge_instrument': 'NIFTY_FUTURE'
            },
            'risk_alerts': self._generate_risk_alerts(risk_checks, stress_results)
        }
        
        return risk_report
    
    def _generate_risk_alerts(self, risk_checks: Dict, stress_results: Dict) -> List[Dict]:
        """Generate risk alerts based on current state"""
        alerts = []
        
        # Risk limit violation alerts
        for violation_name, violation_data in risk_checks.get('violations', {}).items():
            alerts.append({
                'type': 'VIOLATION',
                'severity': 'HIGH',
                'message': f"Risk limit violation: {violation_name}",
                'details': violation_data,
                'timestamp': datetime.now().isoformat()
            })
        
        # Risk limit warning alerts
        for warning_name, warning_data in risk_checks.get('warnings', {}).items():
            alerts.append({
                'type': 'WARNING',
                'severity': 'MEDIUM',
                'message': f"Risk limit warning: {warning_name}",
                'details': warning_data,
                'timestamp': datetime.now().isoformat()
            })
        
        # Stress test alerts
        for scenario, pnl in stress_results.get('standard_scenarios', {}).items():
            if pnl < -50000:  # Alert if stress loss > 50k
                alerts.append({
                    'type': 'STRESS_TEST',
                    'severity': 'HIGH' if pnl < -100000 else 'MEDIUM',
                    'message': f"High stress loss in {scenario}",
                    'details': {'scenario': scenario, 'pnl': pnl},
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts

class PositionSizer:
    """Position sizing based on risk management principles"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.10)  # 10% of portfolio
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Kelly criterion adjustment
        self.volatility_target = config.get('volatility_target', 0.15)  # 15% target volatility
    
    def calculate_position_size(self, 
                              signal_strength: float,
                              signal_confidence: float,
                              current_volatility: float,
                              portfolio_value: float,
                              option_price: float,
                              option_delta: float) -> int:
        """Calculate optimal position size based on risk-adjusted criteria"""
        
        # Base position size using Kelly criterion
        win_rate = signal_confidence
        avg_win = signal_strength * 0.1  # Expected return
        avg_loss = (1 - signal_strength) * 0.05  # Expected loss
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, self.kelly_fraction))  # Cap Kelly fraction
        else:
            kelly_fraction = self.kelly_fraction
        
        # Volatility adjustment
        vol_adjustment = self.volatility_target / max(current_volatility, 0.05)
        vol_adjustment = min(vol_adjustment, 2.0)  # Cap at 2x
        
        # Calculate base position value
        base_position_value = portfolio_value * kelly_fraction * vol_adjustment
        
        # Apply maximum position size limit
        max_position_value = portfolio_value * self.max_position_size
        position_value = min(base_position_value, max_position_value)
        
        # Convert to number of contracts
        contract_value = option_price * 100  # 100 shares per contract
        if contract_value > 0:
            num_contracts = int(position_value / contract_value)
        else:
            num_contracts = 0
        
        # Ensure minimum viable position
        if num_contracts > 0 and num_contracts < 1:
            num_contracts = 1
        
        return max(0, num_contracts)
    
    def calculate_stop_loss(self, 
                           entry_price: float,
                           option_type: str,
                           portfolio_value: float,
                           max_loss_percent: float = 0.02) -> float:
        """Calculate stop loss level"""
        
        # Maximum loss as percentage of portfolio
        max_loss_value = portfolio_value * max_loss_percent
        
        # Calculate stop loss price
        if option_type in ['CE', 'call']:
            # For calls, stop loss is below entry price
            stop_loss_price = entry_price * 0.5  # 50% stop loss for options
        else:
            # For puts, stop loss is also below entry price
            stop_loss_price = entry_price * 0.5
        
        return max(stop_loss_price, 0.05)  # Minimum stop loss of 0.05

# Example usage
if __name__ == "__main__":
    async def test_risk_management():
        """Test risk management system"""
        
        # Configuration
        config = {
            'monte_carlo_simulations': 10000,
            'risk_limits': {
                'portfolio_var_95': {'limit_value': 0.05, 'warning_threshold': 0.04},
                'max_drawdown': {'limit_value': 0.15, 'warning_threshold': 0.12}
            },
            'max_position_size': 0.10,
            'kelly_fraction': 0.25,
            'volatility_target': 0.15
        }
        
        # Initialize risk manager
        risk_manager = RiskManager(config)
        position_sizer = PositionSizer(config)
        
        # Create sample positions
        positions = [
            PositionRisk(
                position_id="pos_1",
                symbol="NIFTY18000CE",
                market_value=50000,
                delta=0.6,
                gamma=0.01,
                theta=-15,
                vega=25,
                rho=5
            ),
            PositionRisk(
                position_id="pos_2",
                symbol="NIFTY17500PE",
                market_value=30000,
                delta=-0.4,
                gamma=0.008,
                theta=-12,
                vega=20,
                rho=-3
            )
        ]
        
        portfolio_value = 1000000
        
        # Calculate risk metrics
        risk_metrics = risk_manager.calculate_portfolio_risk(
            positions, 
            portfolio_value,
            np.random.normal(0, 0.02, 252)  # Simulated daily returns
        )
        
        print("Risk Metrics:")
        print(f"  VaR 95%: {risk_metrics.var_95:.4f}")
        print(f"  VaR 99%: {risk_metrics.var_99:.4f}")
        print(f"  Expected Shortfall: {risk_metrics.expected_shortfall:.4f}")
        print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {risk_metrics.max_drawdown:.4f}")
        
        # Generate risk report
        risk_report = risk_manager.generate_risk_report(positions, portfolio_value, risk_metrics)
        
        print(f"\nRisk Report Generated:")
        print(f"  Number of violations: {len(risk_report['risk_limit_status']['violations'])}")
        print(f"  Number of warnings: {len(risk_report['risk_limit_status']['warnings'])}")
        print(f"  Number of alerts: {len(risk_report['risk_alerts'])}")
        
        # Test position sizing
        position_size = position_sizer.calculate_position_size(
            signal_strength=0.7,
            signal_confidence=0.8,
            current_volatility=0.20,
            portfolio_value=portfolio_value,
            option_price=50,
            option_delta=0.5
        )
        
        print(f"\nRecommended position size: {position_size} contracts")
        
        # Test stop loss calculation
        stop_loss = position_sizer.calculate_stop_loss(
            entry_price=50,
            option_type='CE',
            portfolio_value=portfolio_value
        )
        
        print(f"Recommended stop loss: {stop_loss:.2f}")
    
    # Run the test
    asyncio.run(test_risk_management())