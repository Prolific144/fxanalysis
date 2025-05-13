import logging
import pandas as pd
import numpy as np
import talib
from datetime import datetime, time, timedelta
import pytz
import MetaTrader5 as mt5
from typing import Dict, List, Any
from reports import PDFReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "data": {
        "source": "mt5",
        "symbols": ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "AUDUSD", "NZDUSD"],
        "timeframe_mapping": {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
    },
    "sessions": { #Kenya sessions
        "Sydney":    {"start": "00:00", "end": "09:00"},
        "Tokyo":     {"start": "03:00", "end": "12:00"},
        "Asian":     {"start": "02:00", "end": "11:00"},
        "London":    {"start": "11:00", "end": "20:00"},
        "New York":  {"start": "16:00", "end": "01:00"}  # next day
    },
    "market": {
        "timezone": "Africa/Nairobi",
        "open_days": [0, 1, 2, 3, 4]  # Monday to Friday
    }
}

class MT5DataProvider:
    """Handles connection and data retrieval from MetaTrader 5."""
    def __init__(self):
        if not mt5.initialize():
            logger.error("Failed to initialize MT5 connection")
            raise ConnectionError("MT5 initialization failed")
        logger.info("MT5 connection established")

    def __del__(self):
        mt5.shutdown()
        logger.info("MT5 connection closed")

    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Fetches market data from MT5 for the specified symbol and timeframe."""
        try:
            if timeframe not in CONFIG["data"]["timeframe_mapping"]:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            mt5_timeframe = CONFIG["data"]["timeframe_mapping"][timeframe]
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                raise ValueError(f"No data returned for {symbol}")
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            })
            
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
                
            return df[['time'] + required_columns]
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise

class TechnicalIndicators:
    """Calculates technical indicators for market analysis."""
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.indicators = {}

    def calculate_all(self) -> Dict[str, Any]:
        """Calculates all technical indicators."""
        self._calculate_trend_indicators()
        self._calculate_momentum_indicators()
        self._calculate_volatility_indicators()
        self._determine_market_regime()
        return self.indicators

    def _calculate_trend_indicators(self):
        close = self.data['Close']
        
        ma_50 = talib.SMA(close, timeperiod=50)
        ma_200 = talib.SMA(close, timeperiod=200)
        ma_50_cross = "Bullish" if ma_50.iloc[-1] > ma_200.iloc[-1] else "Bearish"
        
        adx = talib.ADX(self.data['High'], self.data['Low'], close, timeperiod=14)
        
        self.indicators.update({
            "Trend": {
                "MA_50": ma_50.iloc[-1],
                "MA_200": ma_200.iloc[-1],
                "MA_Cross": ma_50_cross,
                "ADX": adx.iloc[-1],
                "ADX_Strength": "Strong" if adx.iloc[-1] > 25 else "Weak" if adx.iloc[-1] < 20 else "Moderate"
            }
        })

    def _calculate_momentum_indicators(self):
        close = self.data['Close']
        
        rsi = talib.RSI(close, timeperiod=14)
        rsi_value = rsi.iloc[-1]
        rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        
        macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_signal = "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"
        
        slowk, slowd = talib.STOCH(self.data['High'], self.data['Low'], close, 
                                  fastk_period=14, slowk_period=3, slowd_period=3)
        stoch_signal = "Overbought" if slowk.iloc[-1] > 80 else "Oversold" if slowk.iloc[-1] < 20 else "Neutral"
        
        self.indicators.update({
            "Momentum": {
                "RSI": rsi_value,
                "RSI_Signal": rsi_signal,
                "MACD": macd.iloc[-1],
                "MACD_Signal": signal.iloc[-1],
                "MACD_Direction": macd_signal,
                "Stochastic_K": slowk.iloc[-1],
                "Stochastic_D": slowd.iloc[-1],
                "Stochastic_Signal": stoch_signal
            }
        })

    def _calculate_volatility_indicators(self):
        close = self.data['Close']
        
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        bb_width = (upper - lower) / middle * 100
        
        atr = talib.ATR(self.data['High'], self.data['Low'], close, timeperiod=14)
        
        self.indicators.update({
            "Volatility": {
                "BB_Upper": upper.iloc[-1],
                "BB_Middle": middle.iloc[-1],
                "BB_Lower": lower.iloc[-1],
                "BB_Width": bb_width.iloc[-1],
                "ATR": atr.iloc[-1]
            }
        })

    def _determine_market_regime(self):
        adx = self.indicators["Trend"]["ADX"]
        bb_width = self.indicators["Volatility"]["BB_Width"]
        
        if adx > 25 and bb_width > 5:
            regime = "Trending"
        elif adx < 20 and bb_width < 5:
            regime = "Range-bound"
        else:
            regime = "Transitional"
            
        self.indicators["Market_Regime"] = regime

class SessionAnalyzer:
    """Analyzes trading sessions for performance metrics."""
    def __init__(self, symbol: str, data_provider: MT5DataProvider):
        self.symbol = symbol
        self.data_provider = data_provider
        self.sessions = CONFIG["sessions"]
        self.timezone = pytz.timezone(CONFIG["market"]["timezone"])
        self.data = None

    def fetch_data(self, start_date: datetime, end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Fetches and prepares session data."""
        self.data = self.data_provider.fetch_data(self.symbol, start_date, end_date, timeframe)
        self.data["Hour"] = self.data["time"].dt.hour
        self.data["Date"] = self.data["time"].dt.date
        return self.data

    def _assign_session(self, hour: int) -> str:
        """Assigns a trading session based on the hour."""
        for session, times in self.sessions.items():
            start_hour = int(times["start"].split(":")[0])
            end_hour = int(times["end"].split(":")[0])
            
            if start_hour <= hour < end_hour or (start_hour > end_hour and (hour >= start_hour or hour < end_hour)):
                return session
        return "No Session"

    def analyze_sessions(self) -> pd.DataFrame:
        """Analyzes session performance metrics."""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
            
        self.data["Session"] = self.data["Hour"].apply(self._assign_session)
        valid_sessions = self.data[self.data["Session"] != "No Session"]
        
        if valid_sessions.empty:
            return pd.DataFrame()
            
        grouped = valid_sessions.groupby(["Date", "Session"])
        
        metrics = grouped.agg({
            "Open": "first",
            "Close": "last",
            "High": "max",
            "Low": "min",
            "Volume": "sum"
        }).reset_index()
        
        metrics["Price_Change"] = metrics["Close"] - metrics["Open"]
        metrics["Pct_Change"] = (metrics["Price_Change"] / metrics["Open"]) * 100
        metrics["Range"] = metrics["High"] - metrics["Low"]
        
        atr_values = []
        for _, group in valid_sessions.groupby(["Date", "Session"]):
            atr = talib.ATR(group["High"], group["Low"], group["Close"], timeperiod=14)
            atr_values.append(atr.iloc[-1] if not atr.empty else np.nan)
        metrics["ATR"] = atr_values
        
        return metrics

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Returns performance metrics for all sessions."""
        metrics = self.analyze_sessions()
        if metrics.empty:
            return []
            
        session_stats = metrics.groupby("Session").agg({
            "Pct_Change": ["mean", "std"],
            "ATR": "mean",
            "Volume": "mean"
        }).reset_index()
        
        session_stats.columns = ["Session", "Avg_Return", "Return_Std", "Avg_ATR", "Avg_Volume"]
        session_stats["Risk_Adj_Return"] = session_stats["Avg_Return"] / session_stats["Return_Std"]
        
        return session_stats.to_dict(orient="records")

    def get_top_sessions(self, n: int = 2) -> List[Dict[str, Any]]:
        """Returns the top performing sessions based on risk-adjusted return."""
        all_sessions = self.get_all_sessions()
        if not all_sessions:
            return []
            
        top_sessions = sorted(all_sessions, key=lambda x: x["Risk_Adj_Return"], reverse=True)[:n]
        return top_sessions

    def get_current_market_status(self) -> Dict[str, Any]:
        """Returns the current market status based on time and sessions."""
        now = datetime.now(self.timezone)
        current_time = now.time()
        weekday = now.weekday()
        
        status = {
            "Time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Day": "Weekday" if weekday in CONFIG["market"]["open_days"] else "Weekend",
            "Active_Sessions": [],
            "Recommended_Action": "No trading"
        }
        
        if weekday not in CONFIG["market"]["open_days"]:
            return status
            
        for session, times in self.sessions.items():
            start_hour = int(times["start"].split(":")[0])
            end_hour = int(times["end"].split(":")[0])
            start_time = time(start_hour, 0)
            end_time = time(end_hour, 0)
            
            if (start_time <= current_time < end_time) or (start_hour > end_hour and (current_time >= start_time or current_time < end_time)):
                status["Active_Sessions"].append(session)
                
        if status["Active_Sessions"]:
            status["Recommended_Action"] = "Consider trading"
            
        return status

class TradingStrategy:
    """Generates trading signals based on technical indicators."""
    def __init__(self):
        self.indicators = None
        
    def generate_signal(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a trading signal based on indicator analysis."""
        try:
            signal = {
                "Direction": "Neutral",
                "Confidence": 50,
                "Factors": [],
                "Entry_Level": None,
                "Stop_Loss": None,
                "Take_Profit": None
            }
            
            trend = indicators["Trend"]
            if trend["MA_Cross"] == "Bullish":
                signal["Factors"].append("MA Bullish")
                signal["Confidence"] += 15
            elif trend["MA_Cross"] == "Bearish":
                signal["Factors"].append("MA Bearish")
                signal["Confidence"] -= 15
                
            if trend["ADX_Strength"] == "Strong":
                signal["Factors"].append("Strong Trend")
                signal["Confidence"] += 10
                
            momentum = indicators["Momentum"]
            if momentum["RSI_Signal"] == "Oversold":
                signal["Factors"].append("RSI Oversold")
                signal["Confidence"] += 10
            elif momentum["RSI_Signal"] == "Overbought":
                signal["Factors"].append("RSI Overbought")
                signal["Confidence"] -= 10
                
            if momentum["MACD_Direction"] == "Bullish":
                signal["Factors"].append("MACD Bullish")
                signal["Confidence"] += 10
            elif momentum["MACD_Direction"] == "Bearish":
                signal["Factors"].append("MACD Bearish")
                signal["Confidence"] -= 10
                
            if momentum["Stochastic_Signal"] == "Oversold":
                signal["Factors"].append("Stoch Oversold")
                signal["Confidence"] += 5
            elif momentum["Stochastic_Signal"] == "Overbought":
                signal["Factors"].append("Stoch Overbought")
                signal["Confidence"] -= 5
                
            volatility = indicators["Volatility"]
            if indicators["Market_Regime"] == "Trending" and volatility["BB_Width"] > 5:
                signal["Factors"].append("High Volatility Trend")
                signal["Confidence"] += 10
                
            if signal["Confidence"] >= 70:
                signal["Direction"] = "Strong Buy"
                last_close = data["Close"].iloc[-1]
                signal["Entry_Level"] = last_close
                signal["Stop_Loss"] = last_close - volatility["ATR"] * 1.5
                signal["Take_Profit"] = last_close + volatility["ATR"] * 2
            elif signal["Confidence"] >= 60:
                signal["Direction"] = "Buy"
                last_close = data["Close"].iloc[-1]
                signal["Entry_Level"] = last_close
                signal["Stop_Loss"] = last_close - volatility["ATR"] * 1.5
                signal["Take_Profit"] = last_close + volatility["ATR"] * 2
            elif signal["Confidence"] <= 30:
                signal["Direction"] = "Strong Sell"
                last_close = data["Close"].iloc[-1]
                signal["Entry_Level"] = last_close
                signal["Stop_Loss"] = last_close + volatility["ATR"] * 1.5
                signal["Take_Profit"] = last_close - volatility["ATR"] * 2
            elif signal["Confidence"] <= 40:
                signal["Direction"] = "Sell"
                last_close = data["Close"].iloc[-1]
                signal["Entry_Level"] = last_close
                signal["Stop_Loss"] = last_close + volatility["ATR"] * 1.5
                signal["Take_Profit"] = last_close - volatility["ATR"] * 2
                
            signal["Confidence"] = max(0, min(100, signal["Confidence"]))
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            raise

class MarketAnalysis:
    """Performs comprehensive market analysis."""
    def __init__(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.data_provider = MT5DataProvider()
        self.strategy = TradingStrategy()
        self.session_analyzer = SessionAnalyzer(symbol, self.data_provider)
        self.data = None
        self.analysis_result = None

    def run_analysis(self) -> Dict[str, Any]:
        """Runs the full market analysis process."""
        try:
            logger.info(f"Starting analysis for {self.symbol} ({self.timeframe}) from {self.start_date} to {self.end_date}")
            
            self.data = self.session_analyzer.fetch_data(self.start_date, self.end_date, self.timeframe)
            if self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")
                
            indicators = TechnicalIndicators(self.data).calculate_all()
            signal = self.strategy.generate_signal(self.data, indicators)
            all_sessions = self.session_analyzer.get_all_sessions()
            market_status = self.session_analyzer.get_current_market_status()
            
            report = {
                "metadata": {
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "period": f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
                    "generated_at": datetime.now(self.session_analyzer.timezone).strftime("%Y-%m-%d %H:%M:%S")
                },
                "market_status": market_status,
                "trading_signal": signal,
                "technical_indicators": indicators,
                "session_analysis": all_sessions,
                "summary": self._generate_summary(indicators, signal, all_sessions, market_status),
                "data": self.data
            }
            
            self.analysis_result = report
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed for {self.symbol}: {e}")
            raise
            
    def generate_pdf_report(self, filename: str) -> str:
        """Generates a PDF report using the analysis results."""
        if self.analysis_result is None:
            raise ValueError("No analysis results available. Run analysis first.")
        report_generator = PDFReportGenerator(self.analysis_result)
        return report_generator.generate_report(filename)
            
    def _generate_summary(self, indicators: Dict[str, Any], signal: Dict[str, Any], 
                         sessions: List[Dict[str, Any]], market_status: Dict[str, Any]) -> str:
        """Generates a textual summary of the analysis."""
        summary = []
        
        summary.append("="*80)
        summary.append(f"MARKET ANALYSIS REPORT - {self.symbol} ({self.timeframe})".center(80))
        summary.append("="*80)
        summary.append(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        summary.append(f"Generated at: {market_status['Time']}")
        summary.append("")
        
        summary.append("-"*80)
        summary.append("MARKET STATUS".center(80))
        summary.append("-"*80)
        summary.append(f"Day: {market_status['Day']}")
        summary.append(f"Active Sessions: {', '.join(market_status['Active_Sessions']) or 'None'}")
        summary.append(f"Recommended Action: {market_status['Recommended_Action']}")
        summary.append("")
        
        summary.append("-"*80)
        summary.append("TRADING SIGNAL".center(80))
        summary.append("-"*80)
        summary.append(f"Direction: {signal['Direction']}")
        summary.append(f"Confidence: {signal['Confidence']}%")
        summary.append(f"Factors: {', '.join(signal['Factors']) or 'None'}")
        if signal["Entry_Level"] is not None:
            summary.append(f"Entry Level: {signal['Entry_Level']:.5f}")
            summary.append(f"Stop Loss: {signal['Stop_Loss']:.5f}")
            summary.append(f"Take Profit: {signal['Take_Profit']:.5f}")
        summary.append("")
        
        summary.append("-"*80)
        summary.append("TECHNICAL INDICATORS".center(80))
        summary.append("-"*80)
        
        trend = indicators["Trend"]
        summary.append(f"Trend: MA50/200 {trend['MA_Cross']} (50: {trend['MA_50']:.5f}, 200: {trend['MA_200']:.5f})")
        summary.append(f"ADX: {trend['ADX']:.2f} ({trend['ADX_Strength']})")
        summary.append(f"Market Regime: {indicators['Market_Regime']}")
        summary.append("")
        
        momentum = indicators["Momentum"]
        summary.append(f"RSI: {momentum['RSI']:.2f} ({momentum['RSI_Signal']})")
        summary.append(f"MACD: {momentum['MACD']:.5f} > Signal: {momentum['MACD_Signal']:.5f} ({momentum['MACD_Direction']})")
        summary.append(f"Stochastic: K={momentum['Stochastic_K']:.2f}, D={momentum['Stochastic_D']:.2f} ({momentum['Stochastic_Signal']})")
        summary.append("")
        
        vol = indicators["Volatility"]
        summary.append(f"ATR: {vol['ATR']:.5f}")
        summary.append(f"Bollinger Bands Width: {vol['BB_Width']:.2f}% (Upper: {vol['BB_Upper']:.5f}, Lower: {vol['BB_Lower']:.5f})")
        summary.append("")
        
        if sessions:
            summary.append("-"*80)
            summary.append("TRADING SESSIONS".center(80))
            summary.append("-"*80)
            for session in sessions:
                summary.append(f"Session: {session['Session']}")
                summary.append(f"  Avg Return: {session['Avg_Return']:.2f}%")
                summary.append(f"  Return Std: {session['Return_Std']:.2f}%")
                summary.append(f"  Risk-Adj Return: {session['Risk_Adj_Return']:.2f}")
                summary.append(f"  Avg ATR: {session['Avg_ATR']:.5f}")
                summary.append(f"  Avg Volume: {session['Avg_Volume']:,.0f}")
                summary.append("")
        
        return "\n".join(summary)

def generate_market_analysis(symbol, timeframe: str, days_back: int = 30) -> Dict[str, Any]:
    """Convenience function to generate analysis for the last N days."""
    end_date = datetime.now(pytz.timezone(CONFIG["market"]["timezone"]))
    start_date = end_date - timedelta(days=days_back)
    
    analyzer = MarketAnalysis(symbol, timeframe, start_date, end_date)
    return analyzer.run_analysis()