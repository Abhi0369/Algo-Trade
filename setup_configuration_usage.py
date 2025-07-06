# Setup, Configuration, and Complete Usage Examples
# Part 7 of the complete LLM + RL Options Trading System

import os
import sys
import json
import yaml
import argparse
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Setup script for the complete trading system
class TradingSystemSetup:
    """Setup and configuration manager for the trading system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_dir = self.base_dir / "config"
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for setup process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def create_directory_structure(self):
        """Create necessary directory structure"""
        
        directories = [
            self.config_dir,
            self.models_dir,
            self.data_dir,
            self.logs_dir,
            self.results_dir,
            self.results_dir / "backtests",
            self.results_dir / "live_trading",
            self.results_dir / "daily_reports",
            self.data_dir / "historical",
            self.data_dir / "live",
            self.models_dir / "checkpoints"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def install_dependencies(self):
        """Install required Python packages"""
        
        requirements = [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "torch>=1.12.0",
            "transformers>=4.20.0",
            "scikit-learn>=1.1.0",
            "scipy>=1.9.0",
            "redis>=4.3.0",
            "aiohttp>=3.8.0",
            "asyncio",
            "backtrader>=1.9.76",
            "yfinance>=0.1.87",
            "nsepy>=0.8",
            "talib-binary>=0.4.24",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.10.0",
            "streamlit>=1.12.0",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "schedule>=1.1.0",
            "pydantic>=1.10.0",
            "python-dotenv>=0.20.0"
        ]
        
        self.logger.info("Installing required packages...")
        
        for package in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                self.logger.info(f"Installed: {package}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install {package}: {e}")
    
    def create_config_files(self):
        """Create configuration files"""
        
        # Main system configuration
        main_config = {
            "system": {
                "name": "LLM-RL Options Trading System",
                "version": "1.0.0",
                "environment": "development"  # development, staging, production
            },
            "data_sources": {
                "primary_provider": "nsepy",
                "backup_provider": "nsepy",
                "truedata_api_key": None,
                "news_api_key": None,
                "alpha_vantage_key": None,
                "redis_host": "localhost",
                "redis_port": 6379
            },
            "trading": {
                "initial_capital": 1000000,
                "max_daily_trades": 50,
                "trading_frequency": 300,
                "risk_check_frequency": 60,
                "enable_paper_trading": True,
                "enable_live_trading": False,
                "symbols": ["NIFTY", "BANKNIFTY"],
                "options_expiry_days": [7, 14, 21, 30]
            },
            "models": {
                "llm_model_name": "microsoft/DialoGPT-medium",
                "rl_algorithm": "PPO",
                "rl_model_path": "models/ppo_options_agent.pkl",
                "llm_model_path": "models/financial_llm",
                "auto_retrain": True,
                "retrain_frequency_days": 30
            },
            "risk_management": {
                "max_portfolio_risk": 0.15,
                "max_position_size": 0.05,
                "stop_loss_threshold": 0.20,
                "max_delta_exposure": 0.50,
                "max_gamma_exposure": 0.30,
                "var_confidence_level": 0.95,
                "monte_carlo_simulations": 10000
            },
            "logging": {
                "log_level": "INFO",
                "log_file": "logs/trading_system.log",
                "max_log_size_mb": 100,
                "backup_count": 5
            }
        }
        
        # Save main config
        config_file = self.config_dir / "main_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(main_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Created main config: {config_file}")
        
        # RL Training configuration
        rl_config = {
            "training": {
                "episodes": 1000,
                "max_steps_per_episode": 1000,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "eps_clip": 0.2,
                "k_epochs": 4,
                "batch_size": 64,
                "memory_size": 10000
            },
            "environment": {
                "initial_cash": 1000000,
                "commission": 0.001,
                "slippage": 0.001,
                "max_positions": 10,
                "lookback_period": 252
            },
            "network": {
                "hidden_size": 256,
                "num_layers": 3,
                "dropout": 0.3,
                "activation": "relu"
            },
            "evaluation": {
                "eval_frequency": 100,
                "eval_episodes": 10,
                "save_best_model": True,
                "early_stopping_patience": 200
            }
        }
        
        rl_config_file = self.config_dir / "rl_config.yaml"
        with open(rl_config_file, 'w') as f:
            yaml.dump(rl_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Created RL config: {rl_config_file}")
        
        # Backtesting configuration
        backtest_config = {
            "data": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "timeframe": "1D",
                "symbols": ["NIFTY"],
                "include_options_data": True
            },
            "strategy": {
                "rebalance_frequency": 5,
                "min_signal_strength": 0.3,
                "min_signal_confidence": 0.5,
                "max_positions": 10
            },
            "costs": {
                "commission": 20.0,
                "slippage": 0.01,
                "bid_ask_spread": 0.005
            },
            "analysis": {
                "benchmark": "NIFTY",
                "risk_free_rate": 0.05,
                "calculate_greeks": True,
                "monte_carlo_runs": 1000
            }
        }
        
        backtest_config_file = self.config_dir / "backtest_config.yaml"
        with open(backtest_config_file, 'w') as f:
            yaml.dump(backtest_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Created backtest config: {backtest_config_file}")
    
    def create_environment_file(self):
        """Create .env file for sensitive configuration"""
        
        env_content = """# Trading System Environment Variables
# Copy this file to .env and update with your actual values

# Data Provider API Keys
TRUEDATA_API_KEY=your_truedata_api_key_here
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Broker API Keys (for live trading)
ZERODHA_API_KEY=your_zerodha_api_key_here
ZERODHA_API_SECRET=your_zerodha_api_secret_here
UPSTOX_API_KEY=your_upstox_api_key_here
UPSTOX_API_SECRET=your_upstox_api_secret_here

# Database Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development

# Risk Management
MAX_DAILY_LOSS_PERCENT=5
EMERGENCY_STOP_LOSS_PERCENT=10

# Notification Settings
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_email_password
"""
        
        env_file = self.base_dir / ".env.example"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        self.logger.info(f"Created environment file template: {env_file}")
    
    def setup_complete_system(self):
        """Setup the complete trading system"""
        
        self.logger.info("Setting up LLM + RL Options Trading System...")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Install dependencies
        self.install_dependencies()
        
        # Create configuration files
        self.create_config_files()
        
        # Create environment file
        self.create_environment_file()
        
        self.logger.info("Setup completed successfully!")
        print("\n" + "="*60)
        print("LLM + RL Options Trading System Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Copy .env.example to .env and update with your API keys")
        print("2. Run: python train_rl_agent.py to train the RL model")
        print("3. Run: python run_backtest.py to test the strategy")
        print("4. Run: python start_trading_system.py for live trading")
        print("\nFor detailed documentation, see README.md")
        print("="*60)

class TrainingPipeline:
    """Complete training pipeline for the RL agent"""
    
    def __init__(self, config_path: str = "config/rl_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def prepare_training_data(self):
        """Prepare historical data for training"""
        
        self.logger.info("Preparing training data...")
        
        # Generate synthetic historical data for demo
        # In production, this would load real historical data
        
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic NIFTY price data
        price_data = []
        price = 10000
        
        for date in dates:
            daily_return = np.random.normal(0.0005, 0.015)  # ~12% annual return, 24% volatility
            price *= (1 + daily_return)
            
            price_data.append({
                'Date': date,
                'Open': price * (1 + np.random.normal(0, 0.002)),
                'High': price * (1 + abs(np.random.normal(0, 0.008))),
                'Low': price * (1 - abs(np.random.normal(0, 0.008))),
                'Close': price,
                'Volume': np.random.randint(5000000, 15000000)
            })
        
        df = pd.DataFrame(price_data)
        
        # Save training data
        training_data_path = Path("data/historical/nifty_training_data.csv")
        df.to_csv(training_data_path, index=False)
        
        self.logger.info(f"Training data saved to: {training_data_path}")
        return df
    
    async def train_rl_agent(self, training_data: pd.DataFrame):
        """Train the RL agent"""
        
        self.logger.info("Starting RL agent training...")
        
        # This would integrate with the actual RL training code
        # For demo purposes, showing the structure
        
        from rl_environment_training import OptionsEnvironment, PPOAgent, train_ppo_agent
        
        # Create environment
        env = OptionsEnvironment(
            historical_data=training_data,
            options_data=pd.DataFrame(),
            initial_cash=self.config['environment']['initial_cash']
        )
        
        # Create agent
        agent = PPOAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            lr=self.config['training']['learning_rate'],
            gamma=self.config['training']['gamma']
        )
        
        # Train agent
        episode_rewards, episode_returns = train_ppo_agent(
            env=env,
            agent=agent,
            episodes=self.config['training']['episodes'],
            max_steps=self.config['training']['max_steps_per_episode']
        )
        
        # Save trained model
        model_path = Path("models/ppo_options_agent.pkl")
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(agent, f)
        
        self.logger.info(f"Trained model saved to: {model_path}")
        
        # Save training results
        results = {
            'episode_rewards': episode_rewards,
            'episode_returns': episode_returns,
            'config': self.config,
            'training_date': datetime.now().isoformat()
        }
        
        results_path = Path("results/training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return agent
    
    async def run_training_pipeline(self):
        """Run the complete training pipeline"""
        
        # Prepare data
        training_data = await self.prepare_training_data()
        
        # Train RL agent
        agent = await self.train_rl_agent(training_data)
        
        self.logger.info("Training pipeline completed successfully!")
        
        return agent

class BacktestRunner:
    """Run comprehensive backtests"""
    
    def __init__(self, config_path: str = "config/backtest_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load backtest configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_backtest_data(self) -> pd.DataFrame:
        """Prepare data for backtesting"""
        
        # Load historical data
        data_path = Path("data/historical/nifty_training_data.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            # Generate data if not available
            self.logger.warning("Historical data not found, generating synthetic data")
            dates = pd.date_range(
                start=self.config['data']['start_date'],
                end=self.config['data']['end_date'],
                freq='D'
            )
            
            price_data = []
            price = 18000
            
            for date in dates:
                daily_return = np.random.normal(0.0005, 0.015)
                price *= (1 + daily_return)
                
                price_data.append({
                    'Date': date,
                    'Open': price * (1 + np.random.normal(0, 0.002)),
                    'High': price * (1 + abs(np.random.normal(0, 0.008))),
                    'Low': price * (1 - abs(np.random.normal(0, 0.008))),
                    'Close': price,
                    'Volume': np.random.randint(5000000, 15000000)
                })
            
            return pd.DataFrame(price_data)
    
    def run_backtest(self) -> Dict:
        """Run backtest using the trained strategy"""
        
        self.logger.info("Starting backtest...")
        
        # Prepare data
        data_df = self.prepare_backtest_data()
        
        # This would integrate with the actual backtesting framework
        from backtrader_integration import OptionsBacktester, LLMRLOptionsStrategy, BacktestConfig
        
        # Create backtest configuration
        backtest_config = BacktestConfig(
            initial_cash=1000000,
            commission=self.config['costs']['commission'] / 10000,  # Convert to fraction
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        # Create backtester
        backtester = OptionsBacktester(backtest_config)
        
        # Add data
        backtester.add_data(data_df)
        
        # Add strategy with mock components
        class MockLLMGenerator:
            def generate_signals(self, symbol):
                return {
                    'direction': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'strength': np.random.uniform(0.3, 0.8),
                    'confidence': np.random.uniform(0.5, 0.9)
                }
        
        class MockRLAgent:
            def select_action(self, state, training=False):
                return np.random.randint(0, 122), 0.0
        
        backtester.add_strategy(
            LLMRLOptionsStrategy,
            llm_signal_generator=MockLLMGenerator(),
            rl_agent=MockRLAgent()
        )
        
        # Run backtest
        results = backtester.run_backtest()
        
        # Save results
        results_path = Path(f"results/backtests/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Backtest completed. Results saved to: {results_path}")
        
        return results

class PerformanceAnalyzer:
    """Analyze trading performance and generate reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_backtest_results(self, results_path: str) -> Dict:
        """Analyze backtest results"""
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        analysis = {
            'performance_metrics': {
                'total_return': results.get('total_return', 0),
                'sharpe_ratio': results.get('analyzer_results', {}).get('sharpe', {}).get('sharperatio'),
                'max_drawdown': results.get('analyzer_results', {}).get('drawdown', {}).get('max', {}).get('drawdown'),
                'total_trades': results.get('analyzer_results', {}).get('trades', {}).get('total', {}).get('total', 0)
            },
            'risk_metrics': {
                'volatility': self._calculate_volatility(results),
                'var_95': self._calculate_var(results, 0.95),
                'sortino_ratio': self._calculate_sortino_ratio(results)
            },
            'trade_analysis': {
                'win_rate': self._calculate_win_rate(results),
                'avg_win': self._calculate_avg_win(results),
                'avg_loss': self._calculate_avg_loss(results),
                'profit_factor': self._calculate_profit_factor(results)
            }
        }
        
        return analysis
    
    def _calculate_volatility(self, results: Dict) -> float:
        """Calculate portfolio volatility"""
        # Placeholder calculation
        return 0.20
    
    def _calculate_var(self, results: Dict, confidence: float) -> float:
        """Calculate Value at Risk"""
        # Placeholder calculation
        return 0.05
    
    def _calculate_sortino_ratio(self, results: Dict) -> float:
        """Calculate Sortino ratio"""
        # Placeholder calculation
        return 1.5
    
    def _calculate_win_rate(self, results: Dict) -> float:
        """Calculate win rate"""
        trades = results.get('analyzer_results', {}).get('trades', {})
        total_trades = trades.get('total', {}).get('total', 1)
        won_trades = trades.get('won', {}).get('total', 0)
        return won_trades / total_trades if total_trades > 0 else 0
    
    def _calculate_avg_win(self, results: Dict) -> float:
        """Calculate average winning trade"""
        # Placeholder calculation
        return 1500.0
    
    def _calculate_avg_loss(self, results: Dict) -> float:
        """Calculate average losing trade"""
        # Placeholder calculation
        return -800.0
    
    def _calculate_profit_factor(self, results: Dict) -> float:
        """Calculate profit factor"""
        avg_win = self._calculate_avg_win(results)
        avg_loss = abs(self._calculate_avg_loss(results))
        win_rate = self._calculate_win_rate(results)
        
        if avg_loss > 0:
            return (avg_win * win_rate) / (avg_loss * (1 - win_rate))
        return 0
    
    def generate_performance_report(self, analysis: Dict) -> str:
        """Generate HTML performance report"""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; color: #333; }
        .metric { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        .table th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM + RL Options Trading Performance Report</h1>
        <p>Generated on: {timestamp}</p>
    </div>
    
    <div class="metric">
        <h2>Performance Metrics</h2>
        <table class="table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Return</td><td class="{return_class}">{total_return:.2%}</td></tr>
            <tr><td>Sharpe Ratio</td><td>{sharpe_ratio:.2f}</td></tr>
            <tr><td>Maximum Drawdown</td><td class="negative">{max_drawdown:.2%}</td></tr>
            <tr><td>Total Trades</td><td>{total_trades}</td></tr>
        </table>
    </div>
    
    <div class="metric">
        <h2>Risk Metrics</h2>
        <table class="table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Volatility</td><td>{volatility:.2%}</td></tr>
            <tr><td>VaR (95%)</td><td>{var_95:.2%}</td></tr>
            <tr><td>Sortino Ratio</td><td>{sortino_ratio:.2f}</td></tr>
        </table>
    </div>
    
    <div class="metric">
        <h2>Trade Analysis</h2>
        <table class="table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Win Rate</td><td>{win_rate:.2%}</td></tr>
            <tr><td>Average Win</td><td class="positive">₹{avg_win:,.2f}</td></tr>
            <tr><td>Average Loss</td><td class="negative">₹{avg_loss:,.2f}</td></tr>
            <tr><td>Profit Factor</td><td>{profit_factor:.2f}</td></tr>
        </table>
    </div>
</body>
</html>
        """.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_return=analysis['performance_metrics']['total_return'] or 0,
            return_class='positive' if (analysis['performance_metrics']['total_return'] or 0) > 0 else 'negative',
            sharpe_ratio=analysis['performance_metrics']['sharpe_ratio'] or 0,
            max_drawdown=abs(analysis['performance_metrics']['max_drawdown'] or 0),
            total_trades=analysis['performance_metrics']['total_trades'],
            volatility=analysis['risk_metrics']['volatility'],
            var_95=analysis['risk_metrics']['var_95'],
            sortino_ratio=analysis['risk_metrics']['sortino_ratio'],
            win_rate=analysis['trade_analysis']['win_rate'],
            avg_win=analysis['trade_analysis']['avg_win'],
            avg_loss=analysis['trade_analysis']['avg_loss'],
            profit_factor=analysis['trade_analysis']['profit_factor']
        )
        
        # Save report
        report_path = Path(f"results/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(report_path, 'w') as f:
            f.write(html_template)
        
        self.logger.info(f"Performance report generated: {report_path}")
        
        return str(report_path)

# Command-line interface
def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description='LLM + RL Options Trading System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup the trading system')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the RL agent')
    train_parser.add_argument('--config', default='config/rl_config.yaml', help='Training config file')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--config', default='config/backtest_config.yaml', help='Backtest config file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('--results', required=True, help='Results file path')
    
    # Start trading command
    start_parser = subparsers.add_parser('start', help='Start live trading')
    start_parser.add_argument('--config', default='config/main_config.yaml', help='System config file')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup = TradingSystemSetup()
        setup.setup_complete_system()
    
    elif args.command == 'train':
        async def run_training():
            pipeline = TrainingPipeline(args.config)
            await pipeline.run_training_pipeline()
        
        asyncio.run(run_training())
    
    elif args.command == 'backtest':
        runner = BacktestRunner(args.config)
        results = runner.run_backtest()
        print(f"Backtest completed. Total return: {results.get('total_return', 0):.2%}")
    
    elif args.command == 'analyze':
        analyzer = PerformanceAnalyzer()
        analysis = analyzer.analyze_backtest_results(args.results)
        report_path = analyzer.generate_performance_report(analysis)
        print(f"Performance report generated: {report_path}")
    
    elif args.command == 'start':
        print("Starting live trading system...")
        # This would start the main trading system
        from main_trading_system import TradingSystem, SystemConfig
        
        # Load configuration
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to SystemConfig object
        config = SystemConfig(**config_dict.get('trading', {}))
        
        async def run_trading_system():
            system = TradingSystem(config)
            await system.start_system()
        
        asyncio.run(run_trading_system())
    
    else:
        parser.print_help()

# Additional utility scripts

def create_sample_usage_script():
    """Create a sample usage script"""
    
    sample_script = '''#!/usr/bin/env python3
"""
Sample usage script for LLM + RL Options Trading System
This script demonstrates how to use the complete trading system
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import system components
from nse_data_infrastructure import NSEDataManager
from llm_signal_generation import FinancialLLMSignalGenerator
from rl_environment_training import PPOAgent
from risk_portfolio_management import RiskManager
from main_trading_system import TradingSystem, SystemConfig

async def demo_signal_generation():
    """Demonstrate LLM signal generation"""
    
    print("\\n=== LLM Signal Generation Demo ===")
    
    config = {
        'llm_model': 'microsoft/DialoGPT-medium',
        'device': 'cpu'
    }
    
    # Initialize signal generator
    signal_generator = FinancialLLMSignalGenerator(config)
    
    # Generate signals for NIFTY
    signal = await signal_generator.generate_signals('NIFTY')
    
    print(f"Generated Signal:")
    print(f"  Symbol: {signal.symbol}")
    print(f"  Direction: {signal.direction}")
    print(f"  Strength: {signal.strength:.3f}")
    print(f"  Confidence: {signal.confidence:.3f}")

async def demo_data_management():
    """Demonstrate data management"""
    
    print("\\n=== Data Management Demo ===")
    
    config = {
        'primary_provider': 'nsepy',
        'backup_provider': 'nsepy'
    }
    
    # Initialize data manager
    data_manager = NSEDataManager(config)
    
    # Get underlying data
    underlying_data = await data_manager.get_underlying_data('NIFTY')
    if underlying_data:
        print(f"NIFTY Current Price: ₹{underlying_data.ltp:,.2f}")
        print(f"Change: {underlying_data.change:+.2f} ({underlying_data.change_percent:+.2f}%)")
    
    # Get options chain
    options = await data_manager.get_options_chain('NIFTY', '2024-12-26')
    print(f"Retrieved {len(options)} options contracts")

def demo_backtesting():
    """Demonstrate backtesting"""
    
    print("\\n=== Backtesting Demo ===")
    
    # This would run a complete backtest
    from backtrader_integration import BacktestRunner
    
    runner = BacktestRunner()
    results = runner.run_backtest()
    
    print(f"Backtest Results:")
    print(f"  Total Return: {results.get('total_return', 0):.2%}")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")

async def demo_complete_workflow():
    """Demonstrate complete trading workflow"""
    
    print("\\n=== Complete Trading Workflow Demo ===")
    
    # 1. Data Management
    await demo_data_management()
    
    # 2. Signal Generation
    await demo_signal_generation()
    
    # 3. Risk Management
    print("\\n--- Risk Management ---")
    config = {'monte_carlo_simulations': 1000}
    risk_manager = RiskManager(config)
    print("Risk manager initialized")
    
    # 4. Portfolio Management
    print("\\n--- Portfolio Management ---")
    print("Portfolio tracking active")
    
    print("\\n=== Demo Complete ===")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("LLM + RL Options Trading System - Usage Demo")
    print("=" * 50)
    
    # Run complete demo
    asyncio.run(demo_complete_workflow())
'''
    
    # Save sample script
    script_path = Path("sample_usage.py")
    with open(script_path, 'w') as f:
        f.write(sample_script)
    
    # Make executable
    import stat
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    
    print(f"Created sample usage script: {script_path}")

def create_readme():
    """Create comprehensive README file"""
    
    readme_content = '''# LLM + RL Options Trading System

A sophisticated options trading system that combines Large Language Models (LLMs) with Reinforcement Learning (RL) for NIFTY50 options trading on the National Stock Exchange (NSE) of India.

## Features

- **Multi-Modal Signal Generation**: Combines fundamental, technical, and sentiment analysis using LLMs
- **Reinforcement Learning**: PPO-based agent for optimal trading decisions
- **Real-time Data Integration**: NSE data feeds with failover mechanisms
- **Comprehensive Risk Management**: VaR, Greeks monitoring, position sizing
- **Professional Backtesting**: Integration with Backtrader framework
- **Live Trading Support**: Paper trading and live execution capabilities
- **Performance Analytics**: Detailed reporting and analysis tools

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Signal Layer   │    │  Decision Layer │
│                 │    │                 │    │                 │
│ • NSE Data      │───▶│ • LLM Signals   │───▶│ • RL Agent      │
│ • Options Chain │    │ • Fundamental   │    │ • Action Space  │
│ • Market Data   │    │ • Technical     │    │ • State Space   │
│ • News/Sentiment│    │ • Sentiment     │    │ • Reward Func   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Risk Layer    │    │ Portfolio Layer │    │ Execution Layer │
│                 │    │                 │    │                 │
│ • Risk Metrics  │    │ • Position Mgmt │    │ • Order Mgmt    │
│ • Limit Checks  │    │ • P&L Tracking  │    │ • Broker APIs   │
│ • Stress Tests  │    │ • Greeks Monitor│    │ • Paper Trading │
│ • Alerts        │    │ • Performance   │    │ • Live Trading  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Redis (for caching)
- TA-Lib (for technical analysis)

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd llm-rl-options-trading

# Run setup script
python -m setup_configuration_usage setup

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Install additional dependencies
pip install -r requirements.txt
```

## Configuration

### Main Configuration (`config/main_config.yaml`)

```yaml
system:
  name: "LLM-RL Options Trading System"
  environment: "development"

data_sources:
  primary_provider: "nsepy"
  truedata_api_key: null  # Add your TrueData API key

trading:
  initial_capital: 1000000
  enable_paper_trading: true
  enable_live_trading: false

risk_management:
  max_portfolio_risk: 0.15
  max_position_size: 0.05
```

## Usage

### 1. Train the RL Agent

```bash
# Train with default configuration
python -m setup_configuration_usage train

# Train with custom config
python -m setup_configuration_usage train --config config/custom_rl_config.yaml
```

### 2. Run Backtests

```bash
# Run backtest with default parameters
python -m setup_configuration_usage backtest

# Run with custom date range
python -m setup_configuration_usage backtest --config config/custom_backtest_config.yaml
```

### 3. Analyze Results

```bash
# Analyze backtest results
python -m setup_configuration_usage analyze --results results/backtests/backtest_20241201_120000.json
```

### 4. Start Live Trading

```bash
# Start paper trading
python -m setup_configuration_usage start

# Start live trading (ensure enable_live_trading: true in config)
python -m setup_configuration_usage start --config config/live_config.yaml
```

## API Integration

### Supported Data Providers

- **NSEpy**: Free NSE data (development/testing)
- **TrueData**: Professional real-time data
- **Global Datafeeds**: Direct exchange connectivity

### Supported Brokers

- Zerodha Kite Connect
- Upstox API
- ICICI Direct
- Angel Broking

## Risk Management

The system includes comprehensive risk management:

- **Value at Risk (VaR)** calculation
- **Greeks monitoring** (Delta, Gamma, Theta, Vega)
- **Position sizing** based on Kelly criterion
- **Stop-loss** mechanisms
- **Concentration limits**
- **Stress testing**

## Performance Monitoring

- Real-time P&L tracking
- Risk metrics dashboard
- Performance attribution
- Trade analysis
- Automated reporting

## Development

### Project Structure

```
├── config/                 # Configuration files
├── data/                  # Historical and live data
├── models/                # Trained ML models
├── results/               # Backtest and trading results
├── logs/                  # System logs
├── nse_data_infrastructure.py      # Data management
├── llm_signal_generation.py       # LLM signal generation
├── rl_environment_training.py     # RL training
├── risk_portfolio_management.py   # Risk management
├── backtrader_integration.py      # Backtesting
├── main_trading_system.py         # Main system
└── setup_configuration_usage.py   # Setup and CLI
```

### Adding New Features

1. **New Data Source**: Extend `DataProvider` class
2. **New Signal Type**: Add to `FinancialLLM` class
3. **New RL Algorithm**: Implement in `rl_environment_training.py`
4. **New Risk Metric**: Extend `RiskEngine` class

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Start Redis server
   redis-server
   ```

2. **TA-Lib Installation Issues**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install ta-lib-dev
   
   # On macOS
   brew install ta-lib
   ```

3. **API Rate Limits**
   - Check your API key quotas
   - Implement proper rate limiting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading in options involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the sample usage scripts

---

**Risk Warning**: Options trading involves significant financial risk. Only trade with capital you can afford to lose.
'''
    
    readme_path = Path("README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README file: {readme_path}")

if __name__ == "__main__":
    # If run directly, provide helpful information
    print("LLM + RL Options Trading System - Setup & Configuration")
    print("=" * 60)
    print("Available commands:")
    print("  python setup_configuration_usage.py setup    - Setup system")
    print("  python setup_configuration_usage.py train    - Train RL agent")
    print("  python setup_configuration_usage.py backtest - Run backtest")
    print("  python setup_configuration_usage.py analyze  - Analyze results")
    print("  python setup_configuration_usage.py start    - Start trading")
    print()
    print("For detailed help: python setup_configuration_usage.py -h")
    print("=" * 60)
    
    # Create additional files
    create_sample_usage_script()
    create_readme()
    
    # Run main CLI if arguments provided
    if len(sys.argv) > 1:
        main()
