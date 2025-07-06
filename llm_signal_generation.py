# LLM Signal Generation System
# Part 2 of the complete LLM + RL Options Trading System

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, AutoConfig
)
from datasets import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from dataclasses import dataclass
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: str  # 'fundamental', 'technical', 'sentiment', 'ensemble'
    direction: str    # 'bullish', 'bearish', 'neutral'
    strength: float   # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict    # Additional signal-specific data

@dataclass
class MarketData:
    """Comprehensive market data for LLM analysis"""
    price_data: pd.DataFrame
    news_data: List[Dict]
    earnings_data: Dict
    technical_indicators: Dict
    economic_indicators: Dict
    sentiment_scores: Dict

class FinancialDataCollector:
    """Collects multi-modal financial data for LLM processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.news_api_key = config.get('news_api_key')
        self.alpha_vantage_key = config.get('alpha_vantage_key')
        
    async def collect_comprehensive_data(self, symbol: str) -> MarketData:
        """Collect all types of financial data for a symbol"""
        
        # Parallel data collection
        tasks = [
            self._get_price_data(symbol),
            self._get_news_data(symbol),
            self._get_earnings_data(symbol),
            self._get_economic_indicators(),
        ]
        
        price_data, news_data, earnings_data, economic_data = await asyncio.gather(*tasks)
        
        # Calculate technical indicators
        technical_indicators = self._calculate_technical_indicators(price_data)
        
        # Initial sentiment analysis
        sentiment_scores = await self._analyze_news_sentiment(news_data)
        
        return MarketData(
            price_data=price_data,
            news_data=news_data,
            earnings_data=earnings_data,
            technical_indicators=technical_indicators,
            economic_indicators=economic_data,
            sentiment_scores=sentiment_scores
        )
    
    async def _get_price_data(self, symbol: str) -> pd.DataFrame:
        """Get historical price data"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1y", interval="1d")
            return data
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return pd.DataFrame()
    
    async def _get_news_data(self, symbol: str) -> List[Dict]:
        """Get recent news data"""
        news_data = []
        
        if not self.news_api_key:
            # Simulate news data for demo
            return [
                {
                    'title': f'{symbol} shows strong quarterly results',
                    'content': f'Company {symbol} reported better than expected earnings...',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'source': 'Demo News',
                    'sentiment': 0.7
                },
                {
                    'title': f'Market volatility affects {symbol} trading',
                    'content': f'Recent market conditions have impacted {symbol}...',
                    'timestamp': datetime.now() - timedelta(hours=8),
                    'source': 'Demo News',
                    'sentiment': -0.3
                }
            ]
        
        # Real implementation would use news APIs
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.news_api_key}"
                async with session.get(url) as response:
                    data = await response.json()
                    
                    for article in data.get('articles', [])[:10]:
                        news_data.append({
                            'title': article['title'],
                            'content': article['description'],
                            'timestamp': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                            'source': article['source']['name'],
                            'sentiment': 0.0  # Will be calculated later
                        })
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
        
        return news_data
    
    async def _get_earnings_data(self, symbol: str) -> Dict:
        """Get earnings and fundamental data"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            
            return {
                'pe_ratio': info.get('trailingPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'market_cap': info.get('marketCap', 0),
                'last_earnings_date': info.get('lastEarningsDate'),
                'next_earnings_date': info.get('nextEarningsDate')
            }
        except Exception as e:
            logger.error(f"Error fetching earnings data: {e}")
            return {}
    
    async def _get_economic_indicators(self) -> Dict:
        """Get macroeconomic indicators"""
        # Simulate economic data for demo
        return {
            'gdp_growth': 6.5,
            'inflation_rate': 4.2,
            'interest_rate': 6.0,
            'unemployment_rate': 7.8,
            'currency_rate': 83.2,  # USD/INR
            'crude_oil_price': 85.5,
            'gold_price': 2050.0,
            'vix': 18.5
        }
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate technical analysis indicators"""
        if price_data.empty:
            return {}
        
        high = price_data['High'].values
        low = price_data['Low'].values
        close = price_data['Close'].values
        volume = price_data['Volume'].values
        
        indicators = {}
        
        try:
            # Trend indicators
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1]
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)[-1]
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)[-1]
            
            # Momentum indicators
            indicators['rsi'] = talib.RSI(close, timeperiod=14)[-1]
            indicators['macd'], indicators['macd_signal'], _ = talib.MACD(close)
            indicators['macd'] = indicators['macd'][-1]
            indicators['macd_signal'] = indicators['macd_signal'][-1]
            
            # Volatility indicators
            indicators['bollinger_upper'], indicators['bollinger_middle'], indicators['bollinger_lower'] = talib.BBANDS(close)
            indicators['bollinger_upper'] = indicators['bollinger_upper'][-1]
            indicators['bollinger_lower'] = indicators['bollinger_lower'][-1]
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            # Volume indicators
            indicators['volume_sma'] = talib.SMA(volume.astype(float), timeperiod=20)[-1]
            indicators['obv'] = talib.OBV(close, volume.astype(float))[-1]
            
            # Pattern recognition
            indicators['doji'] = talib.CDLDOJI(high[-20:], high[-20:], low[-20:], close[-20:])[-1]
            indicators['hammer'] = talib.CDLHAMMER(high[-20:], high[-20:], low[-20:], close[-20:])[-1]
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    async def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment of news data"""
        if not news_data:
            return {'overall_sentiment': 0.0, 'sentiment_scores': []}
        
        # Simple sentiment analysis (replace with FinBERT)
        sentiment_scores = []
        for news in news_data:
            # Simplified sentiment based on keywords
            positive_keywords = ['growth', 'profit', 'strong', 'bullish', 'positive', 'gain']
            negative_keywords = ['loss', 'decline', 'bearish', 'negative', 'fall', 'weak']
            
            content = (news.get('title', '') + ' ' + news.get('content', '')).lower()
            
            positive_count = sum(1 for word in positive_keywords if word in content)
            negative_count = sum(1 for word in negative_keywords if word in content)
            
            if positive_count > negative_count:
                sentiment = 0.5 + (positive_count - negative_count) * 0.1
            elif negative_count > positive_count:
                sentiment = -0.5 - (negative_count - positive_count) * 0.1
            else:
                sentiment = 0.0
            
            sentiment = max(-1.0, min(1.0, sentiment))
            sentiment_scores.append(sentiment)
            news['sentiment'] = sentiment
        
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_scores': sentiment_scores,
            'news_count': len(news_data)
        }

class FinancialLLM:
    """Financial LLM for signal generation"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # For classification tasks
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 3  # bullish, bearish, neutral
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                config=config,
                ignore_mismatched_sizes=True
            )
        except:
            # Fallback to a simpler model
            self.model = self._create_simple_classifier()
        
        self.model.to(device)
        self.model.eval()
        
        # Label mapping
        self.label_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
        
    def _create_simple_classifier(self):
        """Create a simple neural network classifier"""
        class SimpleFinancialClassifier(nn.Module):
            def __init__(self, vocab_size=50257, embed_dim=768, num_classes=3):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.fc1 = nn.Linear(embed_dim, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, num_classes)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids).mean(dim=1)  # Average pooling
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                logits = self.fc3(x)
                return type('obj', (object,), {'logits': logits})()
        
        return SimpleFinancialClassifier()
    
    def analyze_fundamental_data(self, earnings_data: Dict, economic_data: Dict) -> TradingSignal:
        """Analyze fundamental data and generate signal"""
        
        # Create text representation of fundamental data
        text = self._create_fundamental_text(earnings_data, economic_data)
        
        # Get LLM prediction
        prediction = self._predict_from_text(text)
        
        # Calculate strength based on fundamental ratios
        strength = self._calculate_fundamental_strength(earnings_data)
        
        return TradingSignal(
            symbol="",  # Will be set by caller
            signal_type="fundamental",
            direction=prediction['direction'],
            strength=strength,
            confidence=prediction['confidence'],
            timestamp=datetime.now(),
            metadata={'earnings_data': earnings_data, 'text_input': text}
        )
    
    def analyze_technical_data(self, technical_indicators: Dict, price_data: pd.DataFrame) -> TradingSignal:
        """Analyze technical data and generate signal"""
        
        # Create text representation of technical data
        text = self._create_technical_text(technical_indicators, price_data)
        
        # Get LLM prediction
        prediction = self._predict_from_text(text)
        
        # Calculate strength based on technical indicators
        strength = self._calculate_technical_strength(technical_indicators)
        
        return TradingSignal(
            symbol="",
            signal_type="technical",
            direction=prediction['direction'],
            strength=strength,
            confidence=prediction['confidence'],
            timestamp=datetime.now(),
            metadata={'technical_indicators': technical_indicators, 'text_input': text}
        )
    
    def analyze_sentiment_data(self, news_data: List[Dict], sentiment_scores: Dict) -> TradingSignal:
        """Analyze sentiment data and generate signal"""
        
        # Create text representation of sentiment data
        text = self._create_sentiment_text(news_data, sentiment_scores)
        
        # Get LLM prediction
        prediction = self._predict_from_text(text)
        
        # Use sentiment scores as strength
        strength = abs(sentiment_scores.get('overall_sentiment', 0.0))
        
        return TradingSignal(
            symbol="",
            signal_type="sentiment",
            direction=prediction['direction'],
            strength=strength,
            confidence=prediction['confidence'],
            timestamp=datetime.now(),
            metadata={'sentiment_scores': sentiment_scores, 'text_input': text}
        )
    
    def _predict_from_text(self, text: str) -> Dict:
        """Get prediction from text input"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
            
            direction = self.label_map[predicted_class]
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in LLM prediction: {e}")
            return {'direction': 'neutral', 'confidence': 0.0}
    
    def _create_fundamental_text(self, earnings_data: Dict, economic_data: Dict) -> str:
        """Create text representation of fundamental data"""
        text_parts = []
        
        # Earnings data
        pe_ratio = earnings_data.get('pe_ratio', 0)
        if pe_ratio > 0:
            if pe_ratio < 15:
                text_parts.append("The stock appears undervalued with low PE ratio")
            elif pe_ratio > 25:
                text_parts.append("The stock appears overvalued with high PE ratio")
            else:
                text_parts.append("The stock has moderate valuation")
        
        # Revenue growth
        revenue_growth = earnings_data.get('revenue_growth', 0)
        if revenue_growth > 0.1:
            text_parts.append("Strong revenue growth indicates positive business momentum")
        elif revenue_growth < -0.1:
            text_parts.append("Declining revenue raises concerns about business performance")
        
        # Economic indicators
        gdp_growth = economic_data.get('gdp_growth', 0)
        inflation = economic_data.get('inflation_rate', 0)
        
        if gdp_growth > 5:
            text_parts.append("Strong economic growth supports market optimism")
        elif gdp_growth < 2:
            text_parts.append("Weak economic growth creates market uncertainty")
        
        if inflation > 6:
            text_parts.append("High inflation may pressure corporate margins")
        
        return ". ".join(text_parts) if text_parts else "Limited fundamental data available"
    
    def _create_technical_text(self, technical_indicators: Dict, price_data: pd.DataFrame) -> str:
        """Create text representation of technical data"""
        text_parts = []
        
        # RSI analysis
        rsi = technical_indicators.get('rsi', 50)
        if rsi > 70:
            text_parts.append("RSI indicates overbought conditions suggesting potential correction")
        elif rsi < 30:
            text_parts.append("RSI shows oversold conditions indicating potential bounce")
        
        # Moving average analysis
        sma_20 = technical_indicators.get('sma_20', 0)
        sma_50 = technical_indicators.get('sma_50', 0)
        current_price = price_data['Close'].iloc[-1] if not price_data.empty else 0
        
        if current_price > sma_20 > sma_50:
            text_parts.append("Price above moving averages shows bullish trend")
        elif current_price < sma_20 < sma_50:
            text_parts.append("Price below moving averages indicates bearish trend")
        
        # MACD analysis
        macd = technical_indicators.get('macd', 0)
        macd_signal = technical_indicators.get('macd_signal', 0)
        
        if macd > macd_signal:
            text_parts.append("MACD bullish crossover suggests upward momentum")
        elif macd < macd_signal:
            text_parts.append("MACD bearish crossover indicates downward pressure")
        
        return ". ".join(text_parts) if text_parts else "Limited technical data available"
    
    def _create_sentiment_text(self, news_data: List[Dict], sentiment_scores: Dict) -> str:
        """Create text representation of sentiment data"""
        text_parts = []
        
        overall_sentiment = sentiment_scores.get('overall_sentiment', 0)
        
        if overall_sentiment > 0.3:
            text_parts.append("Recent news sentiment is predominantly positive")
        elif overall_sentiment < -0.3:
            text_parts.append("Recent news sentiment is predominantly negative")
        else:
            text_parts.append("Recent news sentiment is neutral")
        
        # Add key news headlines
        for news in news_data[:3]:
            text_parts.append(f"News: {news.get('title', '')}")
        
        return ". ".join(text_parts) if text_parts else "No recent news available"
    
    def _calculate_fundamental_strength(self, earnings_data: Dict) -> float:
        """Calculate signal strength from fundamental data"""
        strength = 0.5  # Base strength
        
        # PE ratio contribution
        pe_ratio = earnings_data.get('pe_ratio', 0)
        if 10 <= pe_ratio <= 20:
            strength += 0.2
        elif pe_ratio > 30:
            strength -= 0.2
        
        # Growth contribution
        revenue_growth = earnings_data.get('revenue_growth', 0)
        earnings_growth = earnings_data.get('earnings_growth', 0)
        
        if revenue_growth > 0.15:
            strength += 0.2
        if earnings_growth > 0.15:
            strength += 0.1
        
        return max(0.0, min(1.0, strength))
    
    def _calculate_technical_strength(self, technical_indicators: Dict) -> float:
        """Calculate signal strength from technical indicators"""
        strength = 0.5
        
        # RSI contribution
        rsi = technical_indicators.get('rsi', 50)
        if rsi > 70 or rsi < 30:
            strength += 0.2
        
        # MACD contribution
        macd = technical_indicators.get('macd', 0)
        macd_signal = technical_indicators.get('macd_signal', 0)
        
        if abs(macd - macd_signal) > 0.1:
            strength += 0.2
        
        return max(0.0, min(1.0, strength))

class SignalEnsemble:
    """Ensemble multiple signals for final trading decision"""
    
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights for different signal types
        self.weights = weights or {
            'fundamental': 0.4,
            'technical': 0.35,
            'sentiment': 0.25
        }
        
        # Ensemble model for meta-learning
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def combine_signals(self, signals: List[TradingSignal]) -> TradingSignal:
        """Combine multiple signals into final ensemble signal"""
        
        if not signals:
            return TradingSignal(
                symbol="",
                signal_type="ensemble",
                direction="neutral",
                strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={}
            )
        
        # Simple weighted average approach
        weighted_score = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        
        direction_scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        
        for signal in signals:
            weight = self.weights.get(signal.signal_type, 0.33)
            
            # Convert direction to score
            if signal.direction == 'bullish':
                score = signal.strength
                direction_scores['bullish'] += weight
            elif signal.direction == 'bearish':
                score = -signal.strength
                direction_scores['bearish'] += weight
            else:
                score = 0
                direction_scores['neutral'] += weight
            
            weighted_score += score * weight * signal.confidence
            total_weight += weight
            confidence_sum += signal.confidence
        
        # Normalize
        if total_weight > 0:
            final_score = weighted_score / total_weight
            avg_confidence = confidence_sum / len(signals)
        else:
            final_score = 0.0
            avg_confidence = 0.0
        
        # Determine final direction
        if final_score > 0.1:
            final_direction = 'bullish'
        elif final_score < -0.1:
            final_direction = 'bearish'
        else:
            final_direction = 'neutral'
        
        return TradingSignal(
            symbol=signals[0].symbol,
            signal_type="ensemble",
            direction=final_direction,
            strength=abs(final_score),
            confidence=avg_confidence,
            timestamp=datetime.now(),
            metadata={
                'component_signals': [s.__dict__ for s in signals],
                'direction_scores': direction_scores,
                'final_score': final_score
            }
        )
    
    def train_meta_model(self, historical_signals: List[Dict], returns: List[float]):
        """Train meta-model on historical signal performance"""
        
        if len(historical_signals) < 50:
            logger.warning("Insufficient data for meta-model training")
            return
        
        # Prepare features
        features = []
        for signal_set in historical_signals:
            feature_vector = self._extract_features(signal_set)
            features.append(feature_vector)
        
        X = np.array(features)
        y = np.array([1 if r > 0 else 0 for r in returns])  # Binary classification
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.meta_model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Meta-model trained with accuracy: {self.meta_model.score(X_scaled, y):.3f}")
    
    def _extract_features(self, signal_set: Dict) -> List[float]:
        """Extract numerical features from signal set"""
        features = []
        
        for signal_type in ['fundamental', 'technical', 'sentiment']:
            signal = signal_set.get(signal_type)
            if signal:
                features.extend([
                    signal.strength,
                    signal.confidence,
                    1.0 if signal.direction == 'bullish' else -1.0 if signal.direction == 'bearish' else 0.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return features

class FinancialLLMSignalGenerator:
    """Main class orchestrating LLM-based signal generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_collector = FinancialDataCollector(config)
        self.llm = FinancialLLM(
            model_name=config.get('llm_model', 'microsoft/DialoGPT-medium'),
            device=config.get('device', 'cpu')
        )
        self.ensemble = SignalEnsemble(config.get('signal_weights'))
        
        # Performance tracking
        self.signal_history = []
        self.performance_metrics = {}
    
    async def generate_signals(self, symbol: str) -> TradingSignal:
        """Generate comprehensive trading signals for a symbol"""
        
        logger.info(f"Generating signals for {symbol}")
        
        # Collect comprehensive market data
        market_data = await self.data_collector.collect_comprehensive_data(symbol)
        
        # Generate individual signals
        signals = []
        
        # Fundamental signal
        if market_data.earnings_data:
            fundamental_signal = self.llm.analyze_fundamental_data(
                market_data.earnings_data,
                market_data.economic_indicators
            )
            fundamental_signal.symbol = symbol
            signals.append(fundamental_signal)
        
        # Technical signal
        if market_data.technical_indicators:
            technical_signal = self.llm.analyze_technical_data(
                market_data.technical_indicators,
                market_data.price_data
            )
            technical_signal.symbol = symbol
            signals.append(technical_signal)
        
        # Sentiment signal
        if market_data.news_data:
            sentiment_signal = self.llm.analyze_sentiment_data(
                market_data.news_data,
                market_data.sentiment_scores
            )
            sentiment_signal.symbol = symbol
            signals.append(sentiment_signal)
        
        # Combine signals
        ensemble_signal = self.ensemble.combine_signals(signals)
        ensemble_signal.symbol = symbol
        
        # Store for performance tracking
        self._track_signal_performance(ensemble_signal, signals)
        
        logger.info(f"Generated {ensemble_signal.direction} signal with strength {ensemble_signal.strength:.3f}")
        
        return ensemble_signal
    
    def _track_signal_performance(self, ensemble_signal: TradingSignal, component_signals: List[TradingSignal]):
        """Track signal performance for continuous improvement"""
        
        signal_record = {
            'timestamp': ensemble_signal.timestamp,
            'symbol': ensemble_signal.symbol,
            'ensemble_signal': ensemble_signal.__dict__,
            'component_signals': [s.__dict__ for s in component_signals]
        }
        
        self.signal_history.append(signal_record)
        
        # Keep only recent history (last 1000 signals)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def update_performance_metrics(self, symbol: str, actual_return: float):
        """Update performance metrics with actual trading results"""
        
        # Find recent signals for this symbol
        recent_signals = [
            s for s in self.signal_history[-50:]  # Last 50 signals
            if s['symbol'] == symbol
        ]
        
        if recent_signals:
            latest_signal = recent_signals[-1]
            
            # Calculate signal accuracy
            predicted_direction = latest_signal['ensemble_signal']['direction']
            actual_direction = 'bullish' if actual_return > 0 else 'bearish' if actual_return < 0 else 'neutral'
            
            is_correct = predicted_direction == actual_direction
            
            # Update metrics
            if symbol not in self.performance_metrics:
                self.performance_metrics[symbol] = {
                    'total_signals': 0,
                    'correct_signals': 0,
                    'accuracy': 0.0,
                    'avg_return': 0.0,
                    'returns': []
                }
            
            metrics = self.performance_metrics[symbol]
            metrics['total_signals'] += 1
            if is_correct:
                metrics['correct_signals'] += 1
            
            metrics['accuracy'] = metrics['correct_signals'] / metrics['total_signals']
            metrics['returns'].append(actual_return)
            metrics['avg_return'] = np.mean(metrics['returns'])
            
            logger.info(f"Signal accuracy for {symbol}: {metrics['accuracy']:.3f}")

# Example usage
if __name__ == "__main__":
    async def test_signal_generator():
        """Test the LLM signal generation system"""
        
        config = {
            'news_api_key': None,  # Set your news API key
            'alpha_vantage_key': None,  # Set your Alpha Vantage key
            'llm_model': 'microsoft/DialoGPT-medium',
            'device': 'cpu',
            'signal_weights': {
                'fundamental': 0.4,
                'technical': 0.4,
                'sentiment': 0.2
            }
        }
        
        generator = FinancialLLMSignalGenerator(config)
        
        # Generate signals for NIFTY
        signal = await generator.generate_signals('NIFTY')
        
        print(f"Generated Signal:")
        print(f"  Direction: {signal.direction}")
        print(f"  Strength: {signal.strength:.3f}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Type: {signal.signal_type}")
        
        # Simulate performance tracking
        generator.update_performance_metrics('NIFTY', 0.02)  # 2% return
    
    # Run the test
    asyncio.run(test_signal_generator())