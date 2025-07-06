# NSE Data Infrastructure and Management System
# Part 1 of the complete LLM + RL Options Trading System

import pandas as pd
import numpy as np
import redis
import asyncio
import websocket
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import requests
from dataclasses import dataclass
import sqlite3
from abc import ABC, abstractmethod
import yfinance as yf
import nsepy as nse
import concurrent.futures
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptionsData:
    """Data structure for options information"""
    symbol: str
    expiry: str
    strike: float
    option_type: str  # 'CE' or 'PE'
    ltp: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    change: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    timestamp: datetime

@dataclass
class UnderlyingData:
    """Data structure for underlying asset information"""
    symbol: str
    ltp: float
    open: float
    high: float
    low: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsData]:
        pass
    
    @abstractmethod
    async def get_underlying_data(self, symbol: str) -> UnderlyingData:
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

class NSEPyDataProvider(DataProvider):
    """NSEpy-based data provider for development/testing"""
    
    def __init__(self):
        self.name = "NSEpy"
        
    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsData]:
        """Get options chain data using NSEpy"""
        try:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            
            # Get options chain from NSE
            options_data = []
            
            # Get current underlying price for delta calculation
            underlying = await self.get_underlying_data(symbol)
            
            # Simulate options chain data (replace with actual NSE API)
            strikes = np.arange(underlying.ltp * 0.8, underlying.ltp * 1.2, 50)
            
            for strike in strikes:
                for option_type in ['CE', 'PE']:
                    # Calculate Greeks (simplified Black-Scholes)
                    greeks = self._calculate_greeks(underlying.ltp, strike, expiry_date, option_type)
                    
                    option = OptionsData(
                        symbol=f"{symbol}{expiry}{strike}{option_type}",
                        expiry=expiry,
                        strike=strike,
                        option_type=option_type,
                        ltp=greeks['price'],
                        bid=greeks['price'] * 0.98,
                        ask=greeks['price'] * 1.02,
                        volume=np.random.randint(100, 10000),
                        open_interest=np.random.randint(1000, 100000),
                        change=np.random.uniform(-5, 5),
                        implied_volatility=greeks['iv'],
                        delta=greeks['delta'],
                        gamma=greeks['gamma'],
                        theta=greeks['theta'],
                        vega=greeks['vega'],
                        rho=greeks['rho'],
                        timestamp=datetime.now()
                    )
                    options_data.append(option)
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return []
    
    async def get_underlying_data(self, symbol: str) -> UnderlyingData:
        """Get underlying asset data"""
        try:
            # Use yfinance for demo (replace with NSE API)
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            latest = data.iloc[-1]
            
            return UnderlyingData(
                symbol=symbol,
                ltp=latest['Close'],
                open=latest['Open'],
                high=latest['High'],
                low=latest['Low'],
                volume=int(latest['Volume']),
                change=latest['Close'] - data.iloc[-2]['Close'],
                change_percent=((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close']) * 100,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching underlying data: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def _calculate_greeks(self, S: float, K: float, expiry_date, option_type: str) -> Dict:
        """Calculate option Greeks using simplified Black-Scholes"""
        from scipy.stats import norm
        import math
        
        # Time to expiry in years
        T = (expiry_date - datetime.now().date()).days / 365.0
        r = 0.05  # Risk-free rate
        sigma = 0.2  # Implied volatility (simplified)
        
        # Black-Scholes calculations
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        if option_type == 'CE':
            price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        
        gamma = norm.pdf(d1) / (S*sigma*math.sqrt(T))
        theta = -(S*norm.pdf(d1)*sigma)/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2 if option_type=='CE' else -d2)
        vega = S*norm.pdf(d1)*math.sqrt(T)
        rho = K*T*math.exp(-r*T)*norm.cdf(d2 if option_type=='CE' else -d2)
        
        return {
            'price': max(price, 0.05),  # Minimum price
            'delta': delta,
            'gamma': gamma,
            'theta': theta/365,  # Daily theta
            'vega': vega/100,    # Per 1% volatility change
            'rho': rho/100,      # Per 1% interest rate change
            'iv': sigma
        }

class TrueDataProvider(DataProvider):
    """TrueData API provider for production"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.truedata.in"
        self.name = "TrueData"
    
    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsData]:
        """Get options chain from TrueData API"""
        # Implementation would use TrueData's actual API
        # This is a placeholder structure
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/options/{symbol}/{expiry}"
        
        # Replace with actual API call
        logger.info(f"Fetching options chain from TrueData for {symbol} expiry {expiry}")
        return []
    
    async def get_underlying_data(self, symbol: str) -> UnderlyingData:
        """Get underlying data from TrueData"""
        # Placeholder for TrueData API implementation
        logger.info(f"Fetching underlying data from TrueData for {symbol}")
        return None
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data from TrueData"""
        logger.info(f"Fetching historical data from TrueData for {symbol}")
        return pd.DataFrame()

class DataCache:
    """Redis-based data caching system"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.default_ttl = 60  # 1 minute default TTL
    
    def set_options_chain(self, symbol: str, expiry: str, data: List[OptionsData], ttl: int = None):
        """Cache options chain data"""
        key = f"options:{symbol}:{expiry}"
        # Convert to JSON-serializable format
        cache_data = [
            {
                'symbol': opt.symbol,
                'strike': opt.strike,
                'option_type': opt.option_type,
                'ltp': opt.ltp,
                'bid': opt.bid,
                'ask': opt.ask,
                'volume': opt.volume,
                'open_interest': opt.open_interest,
                'iv': opt.implied_volatility,
                'delta': opt.delta,
                'gamma': opt.gamma,
                'theta': opt.theta,
                'vega': opt.vega,
                'timestamp': opt.timestamp.isoformat()
            }
            for opt in data
        ]
        
        self.redis_client.setex(key, ttl or self.default_ttl, json.dumps(cache_data))
    
    def get_options_chain(self, symbol: str, expiry: str) -> Optional[List[Dict]]:
        """Retrieve cached options chain data"""
        key = f"options:{symbol}:{expiry}"
        cached_data = self.redis_client.get(key)
        
        if cached_data:
            return json.loads(cached_data)
        return None
    
    def set_underlying_data(self, symbol: str, data: UnderlyingData, ttl: int = None):
        """Cache underlying data"""
        key = f"underlying:{symbol}"
        cache_data = {
            'symbol': data.symbol,
            'ltp': data.ltp,
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'volume': data.volume,
            'change': data.change,
            'change_percent': data.change_percent,
            'timestamp': data.timestamp.isoformat()
        }
        
        self.redis_client.setex(key, ttl or self.default_ttl, json.dumps(cache_data))
    
    def get_underlying_data(self, symbol: str) -> Optional[Dict]:
        """Retrieve cached underlying data"""
        key = f"underlying:{symbol}"
        cached_data = self.redis_client.get(key)
        
        if cached_data:
            return json.loads(cached_data)
        return None

class NSEDataManager:
    """Main data management class coordinating all data sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache = DataCache()
        
        # Initialize data providers
        self.providers = {}
        
        # NSEpy provider (always available)
        self.providers['nsepy'] = NSEPyDataProvider()
        
        # TrueData provider (if configured)
        if config.get('truedata_api_key'):
            self.providers['truedata'] = TrueDataProvider(config['truedata_api_key'])
        
        # Set primary and backup providers
        self.primary_provider = config.get('primary_provider', 'nsepy')
        self.backup_provider = config.get('backup_provider', 'nsepy')
        
        logger.info(f"Initialized NSEDataManager with primary: {self.primary_provider}")
    
    async def get_options_chain(self, symbol: str, expiry: str, use_cache: bool = True) -> List[OptionsData]:
        """Get options chain with fallback and caching"""
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get_options_chain(symbol, expiry)
            if cached_data:
                logger.info(f"Retrieved options chain from cache for {symbol}")
                return self._convert_cached_options(cached_data)
        
        # Try primary provider
        try:
            data = await self.providers[self.primary_provider].get_options_chain(symbol, expiry)
            if data:
                # Cache the successful result
                self.cache.set_options_chain(symbol, expiry, data)
                logger.info(f"Retrieved options chain from {self.primary_provider}")
                return data
        except Exception as e:
            logger.warning(f"Primary provider {self.primary_provider} failed: {e}")
        
        # Fallback to backup provider
        try:
            data = await self.providers[self.backup_provider].get_options_chain(symbol, expiry)
            if data:
                self.cache.set_options_chain(symbol, expiry, data)
                logger.info(f"Retrieved options chain from backup {self.backup_provider}")
                return data
        except Exception as e:
            logger.error(f"Backup provider {self.backup_provider} also failed: {e}")
        
        return []
    
    async def get_underlying_data(self, symbol: str, use_cache: bool = True) -> Optional[UnderlyingData]:
        """Get underlying data with fallback and caching"""
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get_underlying_data(symbol)
            if cached_data:
                logger.info(f"Retrieved underlying data from cache for {symbol}")
                return self._convert_cached_underlying(cached_data)
        
        # Try primary provider
        try:
            data = await self.providers[self.primary_provider].get_underlying_data(symbol)
            if data:
                self.cache.set_underlying_data(symbol, data)
                logger.info(f"Retrieved underlying data from {self.primary_provider}")
                return data
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")
        
        # Fallback to backup provider
        try:
            data = await self.providers[self.backup_provider].get_underlying_data(symbol)
            if data:
                self.cache.set_underlying_data(symbol, data)
                logger.info(f"Retrieved underlying data from backup {self.backup_provider}")
                return data
        except Exception as e:
            logger.error(f"Backup provider also failed: {e}")
        
        return None
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data"""
        try:
            return await self.providers[self.primary_provider].get_historical_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def _convert_cached_options(self, cached_data: List[Dict]) -> List[OptionsData]:
        """Convert cached data back to OptionsData objects"""
        return [
            OptionsData(
                symbol=opt['symbol'],
                expiry='',  # Will be set from context
                strike=opt['strike'],
                option_type=opt['option_type'],
                ltp=opt['ltp'],
                bid=opt['bid'],
                ask=opt['ask'],
                volume=opt['volume'],
                open_interest=opt['open_interest'],
                change=0,  # Not cached
                implied_volatility=opt['iv'],
                delta=opt['delta'],
                gamma=opt['gamma'],
                theta=opt['theta'],
                vega=opt['vega'],
                rho=0,  # Not cached
                timestamp=datetime.fromisoformat(opt['timestamp'])
            )
            for opt in cached_data
        ]
    
    def _convert_cached_underlying(self, cached_data: Dict) -> UnderlyingData:
        """Convert cached data back to UnderlyingData object"""
        return UnderlyingData(
            symbol=cached_data['symbol'],
            ltp=cached_data['ltp'],
            open=cached_data['open'],
            high=cached_data['high'],
            low=cached_data['low'],
            volume=cached_data['volume'],
            change=cached_data['change'],
            change_percent=cached_data['change_percent'],
            timestamp=datetime.fromisoformat(cached_data['timestamp'])
        )

class DataStorageManager:
    """Manages persistent storage of market data"""
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS underlying_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp DATETIME,
                ltp REAL,
                open REAL,
                high REAL,
                low REAL,
                volume INTEGER,
                change_val REAL,
                change_percent REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                underlying TEXT,
                expiry DATE,
                strike REAL,
                option_type TEXT,
                timestamp DATETIME,
                ltp REAL,
                bid REAL,
                ask REAL,
                volume INTEGER,
                open_interest INTEGER,
                iv REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_underlying_symbol_time ON underlying_data(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_symbol_time ON options_data(underlying, timestamp)')
        
        conn.commit()
        conn.close()
    
    def store_underlying_data(self, data: UnderlyingData):
        """Store underlying data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO underlying_data 
            (symbol, timestamp, ltp, open, high, low, volume, change_val, change_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.symbol, data.timestamp, data.ltp, data.open, data.high,
            data.low, data.volume, data.change, data.change_percent
        ))
        
        conn.commit()
        conn.close()
    
    def store_options_data(self, data: List[OptionsData], underlying_symbol: str):
        """Store options data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for option in data:
            cursor.execute('''
                INSERT INTO options_data 
                (symbol, underlying, expiry, strike, option_type, timestamp, ltp, bid, ask, 
                 volume, open_interest, iv, delta, gamma, theta, vega)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                option.symbol, underlying_symbol, option.expiry, option.strike, 
                option.option_type, option.timestamp, option.ltp, option.bid, option.ask,
                option.volume, option.open_interest, option.implied_volatility,
                option.delta, option.gamma, option.theta, option.vega
            ))
        
        conn.commit()
        conn.close()

# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    config = {
        'primary_provider': 'nsepy',
        'backup_provider': 'nsepy',
        'truedata_api_key': None,  # Set your TrueData API key here
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    async def test_data_manager():
        """Test the data manager functionality"""
        data_manager = NSEDataManager(config)
        storage_manager = DataStorageManager()
        
        # Test underlying data
        underlying = await data_manager.get_underlying_data('NIFTY')
        if underlying:
            print(f"Underlying Data: {underlying}")
            storage_manager.store_underlying_data(underlying)
        
        # Test options chain
        options = await data_manager.get_options_chain('NIFTY', '2024-12-26')
        if options:
            print(f"Retrieved {len(options)} options")
            storage_manager.store_options_data(options, 'NIFTY')
            
            # Print first few options
            for opt in options[:5]:
                print(f"Option: {opt.strike} {opt.option_type} - LTP: {opt.ltp}, Delta: {opt.delta:.3f}")
    
    # Run the test
    asyncio.run(test_data_manager())