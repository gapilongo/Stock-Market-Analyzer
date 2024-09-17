# File: app/main.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker symbol and date range.")
            return pd.DataFrame()
        return data
    except Exception as e:
        logging.error(f"Error loading stock data: {e}")
        st.error(f"An error occurred while loading data for {ticker}. Please try again.")
        return pd.DataFrame()

def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving averages for the stock data."""
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    return data

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def create_comparison_chart(ticker_data: dict) -> go.Figure:
    
    fig = go.Figure()
    
    for ticker, data in ticker_data.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name=f"{ticker} Close Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name=f"{ticker} 20-day SMA"))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name=f"{ticker} 50-day SMA"))

    fig.update_layout(
        title="Stock Price and Moving Averages Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Indicators",
        hovermode="x unified"
    )
    return fig


def create_comparison_rsi_chart(ticker_data: dict) -> go.Figure:
    fig = go.Figure()

    # Plot RSI for each ticker
    for ticker, data in ticker_data.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name=f"{ticker} RSI"))

    # Add horizontal lines for overbought (70) and oversold (30) levels
    fig.add_hline(y=70, line=dict(color="red", width=2, dash="dash"), annotation_text="Overbought", annotation_position="top right")
    fig.add_hline(y=30, line=dict(color="green", width=2, dash="dash"), annotation_text="Oversold", annotation_position="bottom right")

    # Update layout
    fig.update_layout(
        title="RSI Comparison",
        xaxis_title="Date",
        yaxis_title="RSI",
        legend_title="Ticker",
        height=500
    )
    return fig

def main():
    st.set_page_config(page_title="Advanced Stock Market Analyzer", layout="wide")
    st.title("Advanced Stock Market Analyzer")

    # Sidebar for user input
    st.sidebar.header("Input Parameters")
    tickers = st.sidebar.text_area("Enter stock tickers (separated by commas):", value="AAPL, MSFT").upper().split(',')
    start_date = st.sidebar.date_input("Start date", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End date", value=datetime.now())

    if st.sidebar.button("Analyze"):
        ticker_data = {}  # To store stock data for each ticker
        for ticker in tickers:
            ticker = ticker.strip()  # Clean the ticker input
            if not ticker:
                continue  # Skip empty ticker symbols

            logging.info(f"Analyzing stock: {ticker} from {start_date} to {end_date}")

            # Load and process data
            data = load_stock_data(ticker, start_date, end_date)
            if not data.empty:
                data = calculate_moving_averages(data)
                data = calculate_rsi(data)
                ticker_data[ticker] = data  # Store the processed data

        # If we have data for at least one stock, proceed to generate the comparison charts
        if ticker_data:
            # Create and display comparison price chart
            price_chart = create_comparison_chart(ticker_data)
            st.plotly_chart(price_chart, use_container_width=True)

            # Create and display RSI comparison chart
            rsi_chart = create_comparison_rsi_chart(ticker_data)  # You'll need to define this similarly
            st.plotly_chart(rsi_chart, use_container_width=True)

            # Display recent data for each ticker
            st.subheader("Recent Data")
            for ticker, data in ticker_data.items():
                st.write(f"**{ticker}** Stock Analysis")
                st.dataframe(data.tail().style.format({'Close': '${:.2f}', 'SMA_20': '${:.2f}', 'SMA_50': '${:.2f}', 'RSI': '{:.2f}'}))

                # Technical Analysis for each stock
                last_close = data['Close'].iloc[-1]
                sma_20 = data['SMA_20'].iloc[-1]
                sma_50 = data['SMA_50'].iloc[-1]
                rsi = data['RSI'].iloc[-1]

                if last_close > sma_20 > sma_50:
                    st.write(f"{ticker}: The stock is in an **uptrend**. The current price is above both the 20-day and 50-day SMAs.")
                elif last_close < sma_20 < sma_50:
                    st.write(f"{ticker}: The stock is in a **downtrend**. The current price is below both the 20-day and 50-day SMAs.")
                else:
                    st.write(f"{ticker}: The stock is showing **mixed signals**. Consider additional indicators for a clearer picture.")

                if rsi > 70:
                    st.write(f"{ticker}: The RSI indicates that the stock may be **overbought**.")
                elif rsi < 30:
                    st.write(f"{ticker}: The RSI indicates that the stock may be **oversold**.")
                else:
                    st.write(f"{ticker}: The RSI is **neutral**, indicating neither overbought nor oversold conditions.")


if __name__ == "__main__":
    main() 
