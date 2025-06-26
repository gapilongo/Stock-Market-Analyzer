# File: app/main.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

MAX_RSI_VALUE = 100
BASE_RS = 1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        data = get_stock_history(ticker, start_date, end_date)
        if not validate_stock_data(data, ticker):
            return pd.DataFrame()
        return data
    except Exception as e:
        logging.error(f"Error loading stock data: {e}")
        st.error(f"An error occurred while loading data for {ticker}. Please try again.")
        return pd.DataFrame()


def get_stock_history(ticker, start_date, end_date):
    return yf.Ticker(ticker).history(start=start_date, end=end_date)


def validate_stock_data(data: pd.DataFrame, ticker: str) -> bool:
    if data.empty:
        st.error(f"No data found for {ticker}. Please check the ticker symbol and date range.")
        return False
    return True


def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    data = add_moving_average(data, 20)
    data = add_moving_average(data, 50)
    return data


def add_moving_average(data: pd.DataFrame, period: int):
    data[f"SMA_{period}"] = data['Close'].rolling(window=period).mean()
    return data


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = MAX_RSI_VALUE - (MAX_RSI_VALUE / (BASE_RS + rs))
    return data


def sma_crossover_strategy(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['Signal'] = 0
    data['Signal'][20:] = (data['SMA_20'][20:] > data['SMA_50'][20:]).astype(int)
    data['Position'] = data['Signal'].diff()
    data['Market Return'] = data['Close'].pct_change()
    data['Strategy Return'] = data['Market Return'] * data['Signal'].shift(1)
    data['Cumulative Market Return'] = (1 + data['Market Return']).cumprod() - 1
    data['Cumulative Strategy Return'] = (1 + data['Strategy Return'].fillna(0)).cumprod() - 1
    return data


def create_candlestick_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=f"{ticker} Candlesticks",
                increasing_line_color='#2ECC71',  # Green for bullish
                decreasing_line_color='#E74C3C'   # Red for bearish
            ),
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                line=dict(color='blue', width=1),
                name="20-day SMA"
            ),
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                line=dict(color='orange', width=1),
                name="50-day SMA"
            )
        ]
    )
    
    # Add volume as subplot
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color='rgba(100, 100, 100, 0.5)',
            yaxis="y2"
        )
    )
    
    fig.update_layout(
        title=f"{ticker} Candlestick Chart with Volume",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend_title="Indicators",
        hovermode="x unified",
        height=600
    )
    return fig


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
    for ticker, data in ticker_data.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name=f"{ticker} RSI"))
    fig.add_hline(y=70, line=dict(color="red", width=2, dash="dash"), annotation_text="Overbought", annotation_position="top right")
    fig.add_hline(y=30, line=dict(color="green", width=2, dash="dash"), annotation_text="Oversold", annotation_position="bottom right")
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

    st.sidebar.header("Input Parameters")
    tickers = st.sidebar.text_area("Enter stock tickers (separated by commas):", value="AAPL, MSFT").upper().split(',')
    start_date = st.sidebar.date_input("Start date", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End date", value=datetime.now())

    if st.sidebar.button("Analyze"):
        ticker_data = {}
        for ticker in tickers:
            ticker = ticker.strip()
            if not ticker:
                continue

            logging.info(f"Analyzing stock: {ticker} from {start_date} to {end_date}")
            data = load_stock_data(ticker, start_date, end_date)
            if not data.empty:
                data = calculate_moving_averages(data)
                data = calculate_rsi(data)
                data = sma_crossover_strategy(data)
                ticker_data[ticker] = data

        if ticker_data:
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Candlestick Charts", "Multi-Stock Comparison", "Recent Data & Analysis"])
            
            with tab1:
                # Display candlestick charts with volume for each stock
                for ticker, data in ticker_data.items():
                    st.subheader(f"{ticker} Candlestick Chart")
                    candlestick_chart = create_candlestick_chart(data, ticker)
                    st.plotly_chart(candlestick_chart, use_container_width=True)

            with tab2:
                # Display multi-stock comparison line chart
                st.subheader("Price Comparison")
                price_chart = create_comparison_chart(ticker_data)
                st.plotly_chart(price_chart, use_container_width=True)

                # RSI comparison
                st.subheader("RSI Comparison")
                rsi_chart = create_comparison_rsi_chart(ticker_data)
                st.plotly_chart(rsi_chart, use_container_width=True)

            with tab3:
                # Display performance metrics and recent data
                st.subheader("Recent Data & Technical Analysis")
                for ticker, data in ticker_data.items():
                    with st.expander(f"{ticker} Analysis"):
                        st.write(f"**Recent Data**")
                        st.dataframe(data.tail().style.format({'Close': '${:.2f}', 'SMA_20': '${:.2f}', 'SMA_50': '${:.2f}', 'RSI': '{:.2f}'}))

                        last_close = data['Close'].iloc[-1]
                        sma_20 = data['SMA_20'].iloc[-1]
                        sma_50 = data['SMA_50'].iloc[-1]
                        rsi = data['RSI'].iloc[-1]

                        st.write(f"**Trend Analysis**")
                        if last_close > sma_20 > sma_50:
                            st.write(f"The stock is in an **uptrend**. The current price is above both the 20-day and 50-day SMAs.")
                        elif last_close < sma_20 < sma_50:
                            st.write(f"The stock is in a **downtrend**. The current price is below both the 20-day and 50-day SMAs.")
                        else:
                            st.write(f"The stock is showing **mixed signals**. Consider additional indicators for a clearer picture.")

                        st.write(f"**RSI Analysis**")
                        if rsi > 70:
                            st.write(f"The RSI indicates that the stock may be **overbought**.")
                        elif rsi < 30:
                            st.write(f"The RSI indicates that the stock may be **oversold**.")
                        else:
                            st.write(f"The RSI is **neutral**, indicating neither overbought nor oversold conditions.")

                        final_strategy_return = data['Cumulative Strategy Return'].iloc[-1] * 100
                        final_market_return = data['Cumulative Market Return'].iloc[-1] * 100
                        num_trades = data['Position'].abs().sum()

                        st.write(f"**SMA Crossover Strategy Performance**")
                        st.write(f"- Total Strategy Return: {final_strategy_return:.2f}%")
                        st.write(f"- Total Market Return: {final_market_return:.2f}%")
                        st.write(f"- Number of Trades Executed: {int(num_trades)}")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative Market Return'], name="Market Return"))
                        fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative Strategy Return'], name="Strategy Return"))
                        fig.update_layout(
                            title=f"Cumulative Returns: Market vs SMA Crossover Strategy",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()