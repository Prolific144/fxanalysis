import logging
from typing import List, Any, Dict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepTogether, LongTable, FrameBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.rl_config import defaultPageSize
from reportlab.platypus import PageTemplate, BaseDocTemplate, Frame
from reportlab.lib.utils import ImageReader
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import talib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration for PDF settings
PDF_CONFIG = {
    "logo_path": "assets/logo.png",
    "chart_dpi": 150,
    "chart_size": (6.5, 3.5)
}

class CustomDocTemplate(BaseDocTemplate):
    """Custom document template with header and footer."""
    def __init__(self, filename, symbol, generated_at, **kwargs):
        self.generated_at = generated_at
        super().__init__(filename, **kwargs)
        self.symbol = symbol
        self.addPageTemplates([
            PageTemplate(
                id='AllPages',
                frames=[Frame(
                    self.leftMargin, self.bottomMargin,
                    self.width, self.height - 0.75*inch,  # Space for header
                    id='normal'
                )],
                onPage=self.add_header_footer
            )
        ])

    def add_header_footer(self, canvas, doc):
        """Adds header and footer to each page."""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica-Bold', 10)
        header_text = f"Market Analysis Report - {self.symbol}"
        canvas.drawCentredString(doc.width/2 + doc.leftMargin, doc.height + doc.topMargin - 0.5*inch, header_text)
        canvas.line(doc.leftMargin, doc.height + doc.topMargin - 0.6*inch, 
                   doc.width + doc.leftMargin, doc.height + doc.topMargin - 0.6*inch)
        
        # Footer
        canvas.setFont('Helvetica', 8)
        page_number = f"Page {doc.page}"
        timestamp = f"Generated: {self.generated_at}"
        canvas.drawString(inch, 0.75*inch, page_number)
        canvas.drawRightString(doc.width + inch, 0.75*inch, timestamp)
        
        canvas.restoreState()

class PDFReportGenerator:
    """Generates advanced PDF reports from market analysis data."""
    def __init__(self, analysis_data: Dict[str, Any]):
        self.analysis = analysis_data
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Creates custom styles for the PDF report, avoiding redefinition."""
        if 'Header1' not in self.styles.byName:
            self.styles.add(ParagraphStyle(
                name='Header1',
                parent=self.styles['Heading1'],
                fontSize=16,
                leading=20,
                alignment=TA_CENTER,
                spaceAfter=12,
                textColor=colors.HexColor('#2c3e50')
            ))
        
        if 'Header2' not in self.styles.byName:
            self.styles.add(ParagraphStyle(
                name='Header2',
                parent=self.styles['Heading2'],
                fontSize=12,
                leading=16,
                spaceAfter=8,
                textColor=colors.HexColor('#3498db')
            ))
        
        if 'BodyText' not in self.styles.byName:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['BodyText'],
                fontSize=10,
                leading=14,
                spaceAfter=6,
                fontName='Helvetica'
            ))
        
        if 'SignalText' not in self.styles.byName:
            self.styles.add(ParagraphStyle(
                name='SignalText',
                parent=self.styles['BodyText'],
                fontSize=11,
                leading=15,
                backColor=colors.HexColor('#f8f9fa'),
                borderPadding=(6, 4, 6, 4),
                borderColor=colors.HexColor('#dee2e6'),
                borderWidth=1
            ))
        
        if 'TOCEntry' not in self.styles.byName:
            self.styles.add(ParagraphStyle(
                name='TOCEntry',
                parent=self.styles['BodyText'],
                fontSize=10,
                leading=14,
                leftIndent=10,
                spaceAfter=4
            ))

    def generate_report(self, filename: str) -> str:
        """Generates a PDF report and saves it to the specified file."""
        doc = CustomDocTemplate(
            filename,
            symbol=self.analysis['metadata']['symbol'],
            generated_at=self.analysis['metadata']['generated_at'],
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        elements = []
        
        # Manual Contents Page
        elements.extend(self._create_contents_page())
        
        # Report sections
        elements.extend(self._create_cover_page())
        elements.extend(self._create_summary_section())
        elements.extend(self._create_signal_section())
        elements.extend(self._create_indicators_section())
        elements.extend(self._create_session_analysis())
        elements.extend(self._create_charts_section())
        
        doc.build(elements)
        return filename

    def _create_contents_page(self) -> List[Any]:
        """Creates a manual contents page with section titles and placeholder page numbers."""
        elements = []
        
        elements.append(Paragraph("Contents", self.styles['Header1']))
        elements.append(Spacer(1, 0.25*inch))
        
        contents = [
            ["1", "Executive Summary", "2"],
            ["2", "Detailed Trading Signal", "3"],
            ["3", "Technical Indicators", "4"],
            ["3.1", "Trend Indicators", "4"],
            ["3.2", "Momentum Indicators", "4"],
            ["3.3", "Volatility Indicators", "5"],
            ["4", "Session Analysis", "6"],
            ["4.1", "Session Performance Metrics", "6"],
            ["4.2", "Session Average Returns", "7"],
            ["5", "Price and Indicator Charts", "8"],
            ["5.1", "Price and Volume Analysis", "8"],
            ["5.2", "Relative Strength Index (RSI)", "9"]
        ]
        
        contents_table = Table(contents, colWidths=[0.5*inch, 4*inch, 1*inch])
        contents_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('LEFTPADDING', (1, 0), (1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0, colors.transparent),
        ]))
        
        elements.append(contents_table)
        elements.append(Spacer(1, 0.5*inch))
        elements.append(PageBreak())
        
        return elements

    def _create_cover_page(self) -> List[Any]:
        elements = []
        
        try:
            logo = Image(PDF_CONFIG["logo_path"], width=2*inch, height=1*inch)
            elements.append(Spacer(1, 0.5*inch))
            elements.append(KeepTogether(logo))
            elements.append(Spacer(1, 0.5*inch))
        except:
            logger.warning("Logo file not found, skipping logo in PDF")
        
        title = Paragraph(
            f"MARKET ANALYSIS REPORT<br/>{self.analysis['metadata']['symbol']}",
            self.styles['Header1']
        )
        elements.append(KeepTogether(title))
        elements.append(Spacer(1, 0.5*inch))
        
        intro_text = """
        This report provides a comprehensive analysis of market trends, trading signals, 
        technical indicators, and session performance. Generated using advanced analytical 
        tools, it offers actionable insights for informed trading decisions.
        """
        elements.append(Paragraph(intro_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.25*inch))
        
        meta_data = [
            ["Timeframe:", self.analysis['metadata']['timeframe']],
            ["Period:", self.analysis['metadata']['period']],
            ["Generated:", self.analysis['metadata']['generated_at']]
        ]
        
        meta_table = Table(meta_data, colWidths=[1.5*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        
        elements.append(KeepTogether(meta_table))
        elements.append(Spacer(1, 0.5*inch))
        
        status = self.analysis['market_status']
        status_text = f"""
        <b>Current Market Status:</b><br/>
        Time: {status['Time']}<br/>
        Day: {status['Day']}<br/>
        Active Sessions: {', '.join(status['Active_Sessions']) or 'None'}<br/>
        Recommended Action: {status['Recommended_Action']}
        """
        
        status_elements = [
            Paragraph("Market Status", self.styles['Header2']),
            Paragraph(status_text, self.styles['BodyText'])
        ]
        elements.append(KeepTogether(status_elements))
        elements.append(Spacer(1, 0.8*inch))
        
        elements.append(PageBreak())
        
        return elements

    def _create_summary_section(self) -> List[Any]:
        elements = []
        elements.append(Paragraph("Executive Summary", self.styles['Header1']))
        
        signal = self.analysis['trading_signal']
        summary_text = f"""
        The analysis of {self.analysis['metadata']['symbol']} suggests a <b>{signal['Direction']}</b> signal with 
        {signal['Confidence']}% confidence. This recommendation is based on the following factors: 
        {', '.join(signal['Factors'])}.
        """
        
        summary_elements = [
            Spacer(1, 0.25*inch),
            Paragraph(summary_text, self.styles['BodyText'])
        ]
        
        if signal["Entry_Level"] is not None:
            trade_text = f"""
            <b>Trade Recommendation:</b><br/>
            Entry Level: {signal['Entry_Level']:.5f}<br/>
            Stop Loss: {signal['Stop_Loss']:.5f}<br/>
            Take Profit: {signal['Take_Profit']:.5f}
            """
            summary_elements.append(Paragraph(trade_text, self.styles['SignalText']))
        
        summary_elements.extend([
            Spacer(1, 0.5*inch),
            Paragraph("<hr>", self.styles['BodyText'])
        ])
        
        elements.append(KeepTogether(summary_elements))
        return elements

    def _create_signal_section(self) -> List[Any]:
        elements = []
        elements.append(Paragraph("Detailed Trading Signal", self.styles['Header1']))
        
        signal = self.analysis['trading_signal']
        signal_data = [
            ["Signal Direction:", signal['Direction']],
            ["Confidence Level:", f"{signal['Confidence']}%"],
            ["Key Factors:", ", ".join(signal['Factors'])],
        ]
        
        if signal["Entry_Level"] is not None:
            signal_data.extend([
                ["Entry Level:", f"{signal['Entry_Level']:.5f}"],
                ["Stop Loss:", f"{signal['Stop_Loss']:.5f}"],
                ["Take Profit:", f"{signal['Take_Profit']:.5f}"],
            ])
        
        signal_table = Table(signal_data, colWidths=[2*inch, 4*inch])
        signal_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#ffffff')),
        ]))
        
        signal_elements = [
            Spacer(1, 0.25*inch),
            signal_table,
            Spacer(1, 0.5*inch),
            Paragraph("<hr>", self.styles['BodyText'])
        ]
        
        elements.append(KeepTogether(signal_elements))
        return elements

    def _create_indicators_section(self) -> List[Any]:
        elements = []
        elements.append(Paragraph("Technical Indicators", self.styles['Header1']))
        
        indicators = self.analysis['technical_indicators']
        
        # Trend Indicators
        trend_elements = [
            Paragraph("Trend Indicators", self.styles['Header2'])
        ]
        trend = indicators['Trend']
        trend_data = [
            ["Moving Averages (50/200):", f"{trend['MA_Cross']}"],
            ["MA 50:", f"{trend['MA_50']:.5f}"],
            ["MA 200:", f"{trend['MA_200']:.5f}"],
            ["ADX:", f"{trend['ADX']:.2f} ({trend['ADX_Strength']})"],
            ["Market Regime:", indicators['Market_Regime']]
        ]
        
        trend_table = Table(trend_data, colWidths=[2.5*inch, 3.5*inch])
        trend_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ]))
        
        trend_elements.extend([
            Spacer(1, 0.25*inch),
            trend_table,
            Spacer(1, 0.25*inch)
        ])
        
        # Momentum Indicators
        momentum_elements = [
            Paragraph("Momentum Indicators", self.styles['Header2'])
        ]
        momentum = indicators['Momentum']
        momentum_data = [
            ["RSI:", f"{momentum['RSI']:.2f} ({momentum['RSI_Signal']})"],
            ["MACD Line:", f"{momentum['MACD']:.5f}"],
            ["MACD Signal:", f"{momentum['MACD_Signal']:.5f}"],
            ["MACD Direction:", momentum['MACD_Direction']],
            ["Stochastic %K:", f"{momentum['Stochastic_K']:.2f}"],
            ["Stochastic %D:", f"{momentum['Stochastic_D']:.2f}"],
            ["Stochastic Signal:", momentum['Stochastic_Signal']]
        ]
        
        momentum_table = Table(momentum_data, colWidths=[2.5*inch, 3.5*inch])
        momentum_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ]))
        
        momentum_elements.extend([
            Spacer(1, 0.25*inch),
            momentum_table,
            Spacer(1, 0.25*inch)
        ])
        
        # Volatility Indicators
        volatility_elements = [
            Paragraph("Volatility Indicators", self.styles['Header2'])
        ]
        vol = indicators['Volatility']
        vol_data = [
            ["ATR (14):", f"{vol['ATR']:.5f}"],
            ["Bollinger Bands Width:", f"{vol['BB_Width']:.2f}%"],
            ["BB Upper:", f"{vol['BB_Upper']:.5f}"],
            ["BB Middle:", f"{vol['BB_Middle']:.5f}"],
            ["BB Lower:", f"{vol['BB_Lower']:.5f}"]
        ]
        
        vol_table = Table(vol_data, colWidths=[2.5*inch, 3.5*inch])
        vol_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ]))
        
        volatility_elements.extend([
            Spacer(1, 0.25*inch),
            vol_table,
            Spacer(1, 0.5*inch),
            Paragraph("<hr>", self.styles['BodyText'])
        ])
        
        elements.extend([
            KeepTogether(trend_elements),
            KeepTogether(momentum_elements),
            KeepTogether(volatility_elements)
        ])
        return elements

    def _create_session_analysis(self) -> List[Any]:
        elements = []
        elements.append(Paragraph("Session Analysis", self.styles['Header1']))
        
        sessions = self.analysis['session_analysis']
        if not sessions:
            elements.append(Paragraph("No session data available", self.styles['BodyText']))
            return elements
        
        # Session performance table
        session_elements = [
            Paragraph("Session Performance Metrics", self.styles['Header2']),
            Spacer(1, 0.25*inch)
        ]
        session_data = [["Session", "Avg Return", "Return Std", "Risk-Adj Return", "Avg ATR", "Avg Volume"]]
        for session in sorted(sessions, key=lambda x: x['Session']):
            session_data.append([
                session['Session'],
                f"{session['Avg_Return']:.2f}%",
                f"{session['Return_Std']:.2f}%",
                f"{session['Risk_Adj_Return']:.2f}",
                f"{session['Avg_ATR']:.5f}",
                f"{session['Avg_Volume']:,.0f}"
            ])
        
        session_table = LongTable(session_data, colWidths=[1.2*inch]*6, splitByRow=True)
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8f9fa')),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f8f9fa')),
        ]))
        
        session_elements.append(session_table)
        session_elements.append(Spacer(1, 0.25*inch))
        
        # Session returns bar chart
        try:
            fig, ax = plt.subplots(figsize=PDF_CONFIG["chart_size"])
            session_df = pd.DataFrame(sessions).sort_values('Session')
            sns.barplot(x='Avg_Return', y='Session', hue='Session', data=session_df, palette='Blues_d', ax=ax, legend=False)
            ax.set_title(f"{self.analysis['metadata']['symbol']} Session Returns")
            ax.set_xlabel("Average Return (%)")
            ax.set_ylabel("Session")
            ax.grid(True, axis='x')
            
            for i, v in enumerate(session_df['Avg_Return']):
                ax.text(v, i, f"{v:.2f}%", va='center', ha='left' if v < 0 else 'right', color='black')
            
            fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
            img_data = BytesIO()
            fig.savefig(img_data, format='png', dpi=PDF_CONFIG["chart_dpi"], bbox_inches='tight')
            img_data.seek(0)
            plt.close()
            
            session_chart = Image(img_data, width=PDF_CONFIG["chart_size"][0]*inch, height=PDF_CONFIG["chart_size"][1]*inch)
            chart_elements = [
                Paragraph("Session Average Returns", self.styles['Header2']),
                Spacer(1, 0.25*inch),
                session_chart
            ]
            session_elements.extend([KeepTogether(chart_elements)])
            
        except Exception as e:
            logger.error(f"Error generating session chart: {e}")
            session_elements.append(Paragraph("Could not generate session chart", self.styles['BodyText']))
        
        session_elements.extend([
            Spacer(1, 0.5*inch),
            Paragraph("<hr>", self.styles['BodyText'])
        ])
        
        elements.append(KeepTogether(session_elements))
        return elements

    def _create_charts_section(self) -> List[Any]:
        elements = []
        elements.append(Paragraph("Price and Indicator Charts", self.styles['Header1']))
        
        try:
            data = self.analysis['data']
            if data is None or data.empty:
                raise ValueError("No data available for chart generation")
                
            # Price chart with MA and Bollinger Bands
            price_elements = [
                Paragraph("Price and Volume Analysis", self.styles['Header2']),
                Spacer(1, 0.25*inch)
            ]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PDF_CONFIG["chart_size"], sharex=True, 
                                         gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
            
            ax1.plot(data['time'], data['Close'], label='Close Price', color='#2c3e50', linewidth=1.5)
            ma_50 = talib.SMA(data['Close'], timeperiod=50)
            ma_200 = talib.SMA(data['Close'], timeperiod=200)
            ax1.plot(data['time'], ma_50, label='MA50', color='#e67e22', linestyle='--')
            ax1.plot(data['time'], ma_200, label='MA200', color='#3498db', linestyle='--')
            
            upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            ax1.fill_between(data['time'], upper, lower, color='gray', alpha=0.1, label='Bollinger Bands')
            
            ax1.set_title(f"{self.analysis['metadata']['symbol']} Price Analysis")
            ax1.set_ylabel("Price")
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            ax2.bar(data['time'], data['Volume'], color='#6c757d', alpha=0.6)
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Time")
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15, hspace=0.1)
            img_data = BytesIO()
            fig.savefig(img_data, format='png', dpi=PDF_CONFIG["chart_dpi"], bbox_inches='tight')
            img_data.seek(0)
            plt.close()
            
            price_chart = Image(img_data, width=PDF_CONFIG["chart_size"][0]*inch, height=PDF_CONFIG["chart_size"][1]*inch)
            price_elements.append(price_chart)
            price_elements.append(Spacer(1, 0.25*inch))
            elements.append(KeepTogether(price_elements))
            
            # RSI chart
            rsi_elements = [
                Paragraph("Relative Strength Index (RSI)", self.styles['Header2']),
                Spacer(1, 0.25*inch)
            ]
            fig, ax = plt.subplots(figsize=PDF_CONFIG["chart_size"])
            rsi = talib.RSI(data['Close'], timeperiod=14)
            ax.plot(data['time'], rsi, label='RSI', color='#2c3e50')
            ax.axhline(y=70, color='red', linestyle='--', label='Overbought', alpha=0.7)
            ax.axhline(y=30, color='green', linestyle='--', label='Oversold', alpha=0.7)
            ax.fill_between(data['time'], 70, 100, color='red', alpha=0.1)
            ax.fill_between(data['time'], 0, 30, color='green', alpha=0.1)
            
            ax.set_title(f"{self.analysis['metadata']['symbol']} RSI")
            ax.set_xlabel("Time")
            ax.set_ylabel("RSI")
            ax.set_ylim(0, 100)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
            img_data = BytesIO()
            fig.savefig(img_data, format='png', dpi=PDF_CONFIG["chart_dpi"], bbox_inches='tight')
            img_data.seek(0)
            plt.close()
            
            rsi_chart = Image(img_data, width=PDF_CONFIG["chart_size"][0]*inch, height=PDF_CONFIG["chart_size"][1]*inch)
            rsi_elements.append(rsi_chart)
            elements.append(KeepTogether(rsi_elements))
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            elements.append(Paragraph("Could not generate charts", self.styles['BodyText']))
        
        return elements