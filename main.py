# File: app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Import make_subplots
from datetime import datetime, timedelta
import logging
import requests as rq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- CONSTANTS ---
MAX_RSI_VALUE = 100
BASE_RS = 1

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DATA FETCHING & VALIDATION ---
def load_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No data found for ticker: {ticker}. It might be delisted or the symbol is incorrect.")
            return pd.DataFrame()
        return data
    except Exception as e:
        logging.error(f"Error loading stock data for {ticker}: {e}")
        st.error(f"An error occurred while loading data for {ticker}. Please try again.")
        return pd.DataFrame()

def fetch_crypto_data(crypto_id, days=90) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    try:
        response = rq.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        prices = data["prices"]
        # Note: CoinGecko free API only provides 'Close' price, not OHLC for daily data.
        df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("Date", inplace=True)
        df.drop("Timestamp", axis=1, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching crypto data for {crypto_id}: {e}")
        return pd.DataFrame()

# --- TECHNICAL ANALYSIS CALCULATIONS ---
def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    data["SMA_20"] = data['Close'].rolling(window=20).mean()
    data["SMA_50"] = data['Close'].rolling(window=50).mean()
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
    data.loc[data['SMA_20'] > data['SMA_50'], 'Signal'] = 1
    data['Position'] = data['Signal'].diff()
    data['Market Return'] = data['Close'].pct_change()
    data['Strategy Return'] = data['Market Return'] * data['Signal'].shift(1)
    data['Cumulative Market Return'] = (1 + data['Market Return']).cumprod()
    data['Cumulative Strategy Return'] = (1 + data['Strategy Return'].fillna(0)).cumprod()
    return data

# --- SENTIMENT ANALYSIS ---
def get_sentiment_analysis(query: str, api_key: str) -> pd.DataFrame:
    if not api_key:
        st.warning("NewsAPI key is not provided. Skipping sentiment analysis.")
        return pd.DataFrame()
    url = f'https://newsapi.org/v2/everything'
    try:
        response = rq.get(url, params={'q': query, 'apiKey': api_key, 'language': 'en'})
        response.raise_for_status()
        data = response.json()
        analyzer = SentimentIntensityAnalyzer()
        results = []
        for article in data.get('articles', []):
            title = article.get('title', '')
            if title:
                sentiment = analyzer.polarity_scores(title)
                results.append({'Title': title, 'Sentiment Score': sentiment['compound'], 'Source': article.get('source', {}).get('name', 'N/A')})
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Failed to fetch news for '{query}': {e}")
        return pd.DataFrame()

# --- PLOTTING FUNCTIONS ---
def create_line_comparison_chart(ticker_data: dict, data_type: str) -> go.Figure:
    fig = go.Figure()
    for ticker, data in ticker_data.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name=f"{ticker} Close", mode='lines'))
    fig.update_layout(
        title=f"{data_type} Close Price Comparison",
        xaxis_title="Date", yaxis_title="Price (USD)", legend_title="Ticker", hovermode="x unified"
    )
    return fig

# NEW: Function to create a detailed Candlestick chart with a Volume Subplot
def create_candlestick_chart(ticker_name: str, data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05, subplot_titles=(f'{ticker_name} Price Action', 'Trading Volume'),
                       row_heights=[0.7, 0.3])

    # --- Candlestick and SMAs ---
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'],
                                 name='OHLC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='20-SMA', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='50-SMA', line=dict(color='purple', width=1)), row=1, col=1)

    # --- Buy/Sell Signals ---
    buy_signals = data[data['Position'] == 1]
    sell_signals = data[data['Position'] == -1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], name='Buy Signal', mode='markers',
                             marker=dict(symbol='triangle-up', color='green', size=12)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], name='Sell Signal', mode='markers',
                             marker=dict(symbol='triangle-down', color='red', size=12)), row=1, col=1)

    # --- Volume Subplot ---
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='grey'), row=2, col=1)

    fig.update_layout(
        title_text=f"{ticker_name} Detailed Analysis",
        legend_title="Indicators",
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

def create_comparison_rsi_chart(ticker_data: dict) -> go.Figure:
    fig = go.Figure()
    for ticker, data in ticker_data.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name=f"{ticker} RSI"))
    fig.add_hline(y=70, line=dict(color="red", width=1, dash="dash"), annotation_text="Overbought")
    fig.add_hline(y=30, line=dict(color="green", width=1, dash="dash"), annotation_text="Oversold")
    fig.update_layout(
        title="Relative Strength Index (RSI) Comparison",
        xaxis_title="Date", yaxis_title="RSI", legend_title="Ticker", height=400
    )
    return fig

# --- MAIN APPLICATION ---
def main():
    st.set_page_config(page_title="Advanced Market Analyzer", layout="wide")
    st.title("📈 Advanced Stock & Crypto Analyzer")

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Input Parameters")
    data_type = st.sidebar.selectbox("Select Asset Type:", ["Stocks", "Cryptocurrency"])
    news_api_key = st.sidebar.text_input("Enter NewsAPI Key (Optional):", type="password", help="Get a free key from newsapi.org")

    if data_type == "Stocks":
        tickers_input = st.sidebar.text_area("Enter stock tickers (comma-separated):", value="AAPL, MSFT, GOOG").upper()
        tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
        start_date = st.sidebar.date_input("Start date", value=datetime.now() - timedelta(days=365))
        end_date = st.sidebar.date_input("End date", value=datetime.now())
    else: # Cryptocurrency
        crypto_options = {"Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum", "Dogecoin (DOGE)": "dogecoin", "Solana (SOL)": "solana", "Ripple (XRP)": "ripple", "Cardano (ADA)": "cardano"}
        crypto_names = list(crypto_options.keys())
        selected_cryptos = st.sidebar.multiselect("Choose Cryptocurrencies:", crypto_names, default=["Bitcoin (BTC)", "Ethereum (ETH)"])
        crypto_days = st.sidebar.slider("Days of historical data", min_value=30, max_value=365, value=180)

    if st.sidebar.button("Analyze", type="primary"):
        ticker_data = {}

        # Data Loading Block...
        if data_type == "Stocks":
            # ... (omitted for brevity, no changes)
            if not tickers:
                st.warning("Please enter at least one stock ticker.")
                return
            for ticker in tickers:
                data = load_stock_data(ticker, start_date, end_date)
                if not data.empty:
                    ticker_data[ticker] = data
        else:
            # ... (omitted for brevity, no changes)
            if not selected_cryptos:
                st.warning("Please select at least one cryptocurrency.")
                return
            for crypto_name in selected_cryptos:
                crypto_id = crypto_options[crypto_name]
                data = fetch_crypto_data(crypto_id, days=crypto_days)
                if not data.empty:
                    ticker_data[crypto_name] = data
        
        if not ticker_data:
            st.error("No valid data could be loaded for the selected assets. Please check your inputs.")
            return
            
        # Process all loaded dataframes
        for name, data in ticker_data.items():
            data = calculate_moving_averages(data)
            data = calculate_rsi(data)
            data = sma_crossover_strategy(data)
            ticker_data[name] = data

        st.header("Comparative Analysis")
        # RENAMED: Using the simple line chart for multi-asset comparison
        st.plotly_chart(create_line_comparison_chart(ticker_data, data_type), use_container_width=True)
        st.plotly_chart(create_comparison_rsi_chart(ticker_data), use_container_width=True)

        st.header("Individual Asset Breakdown")
        for name, data in ticker_data.items():
            with st.expander(f"Analysis for {name}", expanded=True):
                
                # NEW: Display the detailed candlestick chart for stocks
                if data_type == "Stocks":
                    st.plotly_chart(create_candlestick_chart(name, data), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Performance Snapshot")
                    final_strategy_return = (data['Cumulative Strategy Return'].iloc[-1] - 1) * 100
                    final_market_return = (data['Cumulative Market Return'].iloc[-1] - 1) * 100
                    st.metric("Total Strategy Return", f"{final_strategy_return:.2f}%")
                    st.metric("Total Market Return (Buy & Hold)", f"{final_market_return:.2f}%")
                    st.metric("Number of Trades", f"{int(data['Position'].abs().sum() / 2)}")

                    st.subheader("Current Signals")
                    last_close = data['Close'].iloc[-1]
                    rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) else 'N/A'
                    # ... (omitted for brevity, no changes)
                    if last_close > data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
                        st.success(f"**Uptrend**: Price is above both short and long-term moving averages.")
                    elif last_close < data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1]:
                        st.error(f"**Downtrend**: Price is below both moving averages.")
                    else:
                        st.warning(f"**Mixed Signals**: Price is between the moving averages.")
                    if isinstance(rsi, float):
                        if rsi > 70:
                            st.error(f"**Overbought**: RSI is {rsi:.2f}, suggesting a potential pullback.")
                        elif rsi < 30:
                            st.success(f"**Oversold**: RSI is {rsi:.2f}, suggesting a potential bounce.")
                        else:
                            st.info(f"**Neutral**: RSI is {rsi:.2f}, indicating balanced momentum.")
                
                with col2:
                    st.subheader("Recent Data Points")
                    st.dataframe(data.tail().style.format(precision=2))

                st.subheader("Strategy Performance vs. Market")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative Market Return'], name="Market (Buy & Hold)"))
                fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative Strategy Return'], name="SMA Crossover Strategy"))
                fig.update_layout(title=f"{name}: Cumulative Returns Growth", xaxis_title="Date", yaxis_title="Cumulative Growth (1 = 100%)", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader(f"News Sentiment for '{name}'")
                sentiment_df = get_sentiment_analysis(name, news_api_key)
                if not sentiment_df.empty:
                    st.dataframe(sentiment_df, use_container_width=True)
                else:
                    st.info("Could not retrieve sentiment data.")

if __name__ == "__main__":
    main()