import streamlit as st
import logging
import os
import sys
from datetime import datetime, timedelta
from market_analysis import (
    MarketAnalysis, CONFIG,
    YFinanceDataProvider, PDFReportGenerator
)
import pytz
import subprocess
import plotly.graph_objects as go

# Ensure logs and reports directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Configure logging
log_file = f"logs/market_analysis_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom CSS for styling
def local_css():
    css = """
    <style>
        .stMetric {
            background-color: rgb(154, 172, 206);
            padding: 10px;
            border-radius: 5px;
            border: 1px solid rgb(154, 172, 206);
            margin-bottom: 10px;
        }
        .stMetric label {
            font-weight: bold;
            color: #333;
        }
        .stMetric .metric-value {
            font-size: 1.2em;
            color: #1e90ff;
        }
        .stButton>button {
            background-color: #1e90ff;
            color: white;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #104e8b;
        }
        .expander-header {
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def get_analysis_dates(period_text, start_date, end_date):
    """Retrieves the start and end dates for analysis."""
    if period_text == "Last 7 days":
        days_back = 7
    elif period_text == "Last 30 days":
        days_back = 30
    elif period_text == "Last 90 days":
        days_back = 90
    else:  # Custom
        return start_date, end_date
    
    end_date = datetime.now(pytz.timezone(CONFIG["market"]["timezone"]))
    start_date = end_date - timedelta(days=days_back)
    return start_date.date(), end_date.date()

def run_analysis(symbol, timeframe, start_date, end_date, data_provider):
    """Runs the market analysis and returns the results."""
    if start_date >= end_date:
        st.error("Error: Start date must be before end date")
        return None
    
    try:
        with st.spinner('Running analysis... This may take a moment...'):
            tz = pytz.timezone(CONFIG["market"]["timezone"])
            start_dt = tz.localize(datetime.combine(start_date, datetime.min.time()))
            end_dt = tz.localize(datetime.combine(end_date, datetime.max.time()))
            
            analyzer = MarketAnalysis(symbol, timeframe, start_dt, end_dt)
            analysis_results = analyzer.run_analysis()
        return analysis_results
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        st.error(f"Failed to generate analysis: {str(e)}")
        return None

def create_gauge_chart(value, title, min_val=0, max_val=100):
    """Create a gauge chart for confidence metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=10))
    return fig

def display_results(analysis_results):
    """Displays analysis results using Streamlit with enhanced UI."""
    if not analysis_results:
        return

    st.header("📊 Analysis Results")
    st.markdown("---")

    # Overview Section
    with st.expander("📌 Overview", expanded=True):
        st.write(analysis_results.get("summary", "No summary available"))

    st.markdown("---")

    # Trading Signal Section
    with st.expander("🚦 Trading Signal", expanded=True):
        signal = analysis_results.get("trading_signal", {})
        
        # Create columns for signal metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction = signal.get('Direction', 'N/A')
            direction_color = "green" if direction == "Bullish" else "red" if direction == "Bearish" else "gray"
            st.metric(label="**Direction**", value=direction, delta=None, 
                     delta_color="normal", help="Market direction based on analysis")
        
        with col2:
            confidence = signal.get('Confidence', 0)
            st.plotly_chart(create_gauge_chart(confidence, "Confidence"), 
                            use_container_width=True)
        
        with col3:
            entry_level = signal.get("Entry_Level")
            if entry_level is not None:
                st.metric(label="**Entry Level**", value=f"{entry_level:.5f}")
        
        # Key Factors
        st.subheader("Key Factors")
        factors = signal.get('Factors', [])
        if factors:
            factors_cols = st.columns(2)
            for i, factor in enumerate(factors):
                with factors_cols[i % 2]:
                    st.info(f"• {factor}")
        else:
            st.write("No key factors provided.")
        
        # Trade Levels (if available)
        if signal.get("Entry_Level") is not None:
            st.subheader("Trade Levels")
            level_col1, level_col2, level_col3 = st.columns(3)
            with level_col1:
                st.metric(label="Entry", value=f"{signal.get('Entry_Level', 0):.5f}")
            with level_col2:
                st.metric(label="Stop Loss", value=f"{signal.get('Stop_Loss', 0):.5f}")
            with level_col3:
                st.metric(label="Take Profit", value=f"{signal.get('Take_Profit', 0):.5f}")
    
    st.markdown("---")

    # Technical Indicators Section
    with st.expander("📈 Technical Indicators", expanded=False):
        indicators = analysis_results.get("technical_indicators", {})
        
        # Market Regime
        regime_col1, regime_col2 = st.columns([1, 3])
        with regime_col1:
            regime = indicators.get('Market_Regime', 'N/A')
            regime_color = "green" if "Bull" in regime else "red" if "Bear" in regime else "orange"
            st.markdown(f"**Market Regime:** <span style='color:{regime_color};font-weight:bold;'>{regime}</span>", 
                        unsafe_allow_html=True)
        
        # Other indicators in tabs
        tab1, tab2, tab3 = st.tabs(["Trend Indicators", "Momentum Indicators", "Volatility Indicators"])
        
        with tab1:
            trend_indicators = {k: v for k, v in indicators.items() if k in ["MA", "EMA", "ADX"]}
            if trend_indicators:
                for k, v in trend_indicators.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            st.metric(label=sub_k, value=f"{sub_v:.5f}" if isinstance(sub_v, float) else sub_v)
            else:
                st.write("No trend indicators available.")
        
        with tab2:
            momentum_indicators = {k: v for k, v in indicators.items() if k in ["RSI", "MACD", "Stochastic"]}
            if momentum_indicators:
                for k, v in momentum_indicators.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            st.metric(label=sub_k, value=f"{sub_v:.5f}" if isinstance(sub_v, float) else sub_v)
            else:
                st.write("No momentum indicators available.")
        
        with tab3:
            vol_indicators = {k: v for k, v in indicators.items() if k in ["ATR", "Bollinger_Bands"]}
            if vol_indicators:
                for k, v in vol_indicators.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            st.metric(label=sub_k, value=f"{sub_v:.5f}" if isinstance(sub_v, float) else sub_v)
            else:
                st.write("No volatility indicators available.")
    
    st.markdown("---")

    # Session Analysis Section
    with st.expander("⏰ Session Analysis", expanded=False):
        sessions = analysis_results.get("session_analysis", [])
        if not sessions:
            st.warning("No session data available")
        else:
            session_tabs = st.tabs([session.get('Session', f"Session {i+1}") for i, session in enumerate(sessions)])
            
            for i, tab in enumerate(session_tabs):
                with tab:
                    session = sessions[i]
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Average Return", f"{session.get('Avg_Return', 0):.2f}%")
                        st.metric("Return Std Dev", f"{session.get('Return_Std', 0):.2f}%")
                    
                    with col2:
                        st.metric("Risk-Adjusted Return", f"{session.get('Risk_Adj_Return', 0):.2f}")
                        st.metric("Average ATR", f"{session.get('Avg_ATR', 0):.5f}")
                    
                    st.metric("Average Volume", f"{session.get('Avg_Volume', 0):,.0f}")
    
    st.markdown("---")

    # Market Status Section
    with st.expander("🌐 Market Status", expanded=False):
        status = analysis_results.get("market_status", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Time", status.get('Time', 'N/A'))
            st.metric("Day of Week", status.get('Day', 'N/A'))
        
        with col2:
            active_sessions = status.get('Active_Sessions', ['None'])
            st.metric("Active Sessions", ", ".join(active_sessions))
            
            rec_action = status.get('Recommended_Action', 'N/A')
            action_color = "green" if rec_action == "Trade" else "red" if rec_action == "Avoid" else "orange"
            st.markdown(f"**Recommended Action:** <span style='color:{action_color};font-weight:bold;'>{rec_action}</span>", 
                       unsafe_allow_html=True)

def export_pdf_report(analysis_results):
    """Exports the current analysis as a PDF report and offers a download."""
    if not analysis_results:
        st.warning("No analysis to export. Please run analysis first.")
        return

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"MarketAnalysis_{analysis_results.get('metadata', {}).get('symbol', 'Unknown')}_{timestamp}.pdf"
        file_path = os.path.join("reports", default_name)

        pdf_generator = PDFReportGenerator(analysis_results)
        pdf_generator.generate_report(file_path)

        with open(file_path, "rb") as file:
            st.download_button(
                label="📥 Download PDF Report",
                data=file,
                file_name=default_name,
                mime="application/pdf",
                use_container_width=True
            )

        st.success(f"PDF report successfully generated: {default_name}")
        
        # Optionally try to open the report
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.call(["open", file_path])
            else:
                subprocess.call(["xdg-open", file_path])
        except Exception as e:
            st.warning(f"Could not automatically open the report: {e}")

    except Exception as e:
        logger.error(f"Failed to generate PDF: {str(e)}")
        st.error(f"Failed to generate PDF report: {str(e)}")


def main():
    # Set page config
    st.set_page_config(
        page_title="Advanced Market Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    local_css()

    with st.sidebar:
        st.title("⚙️ Analysis Parameters")
        st.markdown("---")
        
        try:
            data_provider = YFinanceDataProvider()
        except Exception as e:
            logger.error(f"Failed to initialize data provider: {e}")
            st.error("Failed to initialize data provider.")
            return

        symbol = st.selectbox("Symbol:", CONFIG.get("data", {}).get("symbols", []))
        timeframe = st.selectbox("Timeframe:", 
                                list(CONFIG.get("data", {}).get("timeframe_mapping", {}).keys()), 
                               index=2 if len(CONFIG.get("data", {}).get("timeframe_mapping", {})) > 2 else 0)
        
        period = st.selectbox("Period:", 
                            ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"])
        
        start_date = None
        end_date = None
        if period == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                         datetime.now().date() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End Date", datetime.now().date())
        else:
            end_date = datetime.now().date()
            if period == "Last 7 days":
                start_date = end_date - timedelta(days=7)
            elif period == "Last 30 days":
                start_date = end_date - timedelta(days=30)
            elif period == "Last 90 days":
                start_date = end_date - timedelta(days=90)
        
        st.markdown("---")
        if st.button("🚀 Run Analysis", use_container_width=True):
            st.session_state['analysis_results'] = run_analysis(
                symbol, timeframe, start_date, end_date, data_provider
            )

    # Main content area
    st.title("📈 Advanced Market Analysis Tool")
    st.markdown("Analyze market conditions and generate trading signals with comprehensive technical analysis.")
    st.markdown("---")

    # Display results if they exist in session state
    if 'analysis_results' in st.session_state:
        display_results(st.session_state['analysis_results'])
        
        # Export section
        st.markdown("---")
        st.subheader("📤 Export Results")
        export_pdf_report(st.session_state['analysis_results'])
        
    # Add some info in the sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Developed by SnooG Analytics**")
        st.markdown("---")
        st.markdown("**About This Tool**")
        st.markdown("""
        This advanced market analysis tool provides:
        - Comprehensive technical analysis
        - Trading signals with confidence levels
        - Session-based performance metrics
        - Market status monitoring
        """)

if __name__ == "__main__":
    main()