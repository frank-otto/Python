import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import requests

# Download NLTK resources
nltk.download('vader_lexicon')

# Initialize the Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Set Streamlit app theme
st.set_page_config(page_title="Innovative Stock Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #0e1117;
    }
    .sidebar .sidebar-content .block-container {
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Innovative Stock Analysis App")

# Dow Jones significant stocks
dow_jones_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ', 'JPM', 'PG', 'DIS', 'NVDA', 'V', 'HD', 
    'UNH', 'MA', 'PFE', 'MRK', 'KO', 'PEP', 'BAC', 'CSCO', 'XOM', 'GS', 'IBM', 
    'INTC', 'MCD', 'MMM', 'NKE', 'TRV', 'WBA', 'WMT'
]

# Select stock ticker and time period
ticker = st.selectbox("Select a stock ticker:", dow_jones_tickers)
period = st.selectbox("Select the time period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)  # Default to 1 year

# Ensure stock is a yfinance Ticker object
stock = yf.Ticker(ticker)

# Fetch historical data
data = stock.history(period=period)
data.reset_index(inplace=True)

# Stock price chart
st.header("Stock Price Trend")
fig_price = px.line(data, x='Date', y='Close', title=f'{ticker} Stock Price Trend', labels={'Close': 'Close Price', 'Date': 'Date'})
fig_price.update_layout(xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark', title_font=dict(size=20))
st.plotly_chart(fig_price)

# 3D Sector Performance Comparison
st.header("3D Sector Performance Comparison")
st.write("""
    The 3D bubble chart below represents sector performance with additional dimensions:
    - **Performance**: The primary metric representing the performance of the sector.
    - **Average P/E Ratio**: The average price-to-earnings ratio of companies in the sector.
    - **Market Cap**: The total market capitalization of the sector.
    - **Bubble Size**: Represents the market capitalization.
""")

# Example data for sector performance (this should be replaced with actual data)
sector_performance_data = {
    "Sector": ["Technology", "Health Care", "Financials", "Consumer Discretionary"],
    "Performance": [12.3, 8.5, 7.2, 9.1],
    "Average PE Ratio": [25.3, 18.5, 15.2, 20.1],
    "Market Cap": [8e12, 4e12, 3e12, 2.5e12]
}
sector_df = pd.DataFrame(sector_performance_data)

fig_sector_3d = px.scatter_3d(sector_df, x='Performance', y='Average PE Ratio', z='Market Cap',
                              size='Market Cap', color='Sector', hover_name='Sector',
                              title='3D Sector Performance Comparison',
                              labels={'Performance': 'Performance (%)', 'Average PE Ratio': 'Average P/E Ratio', 'Market Cap': 'Market Cap'},
                              template='plotly_dark')
fig_sector_3d.update_layout(title_font=dict(size=20))
st.plotly_chart(fig_sector_3d)

# Forecasting and Predictions
st.header("Forecasting and Predictions")

# Model selection
model_choice = st.selectbox("Select a forecasting model:", ["Gradient Boosting Regressor (GBR)", "Linear Regression (LR)", "Random Forest Regressor (RFR)", "ARIMA"])
model_explanations = {
    "Gradient Boosting Regressor (GBR)": "An ensemble learning method that builds multiple decision trees in a sequential manner, where each tree corrects the errors of its predecessor. It is effective for capturing complex patterns in the data.",
    "Linear Regression (LR)": "A simple yet powerful statistical method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.",
    "Random Forest Regressor (RFR)": "An ensemble learning method that constructs multiple decision trees and merges them to get a more accurate and stable prediction. It is less likely to overfit compared to individual decision trees.",
    "ARIMA": "A statistical model used for time series forecasting that captures standard temporal structures in the data."
}
st.write(model_explanations[model_choice])

# Prepare data for forecasting
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature engineering
data['Day'] = data.index.day
data['Month'] = data.index.month
data['Year'] = data.index.year
data['DayOfWeek'] = data.index.dayofweek

# Select features and target
X = data[['Day', 'Month', 'Year', 'DayOfWeek']]
y = data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and forecast based on the selected model
if model_choice == "Gradient Boosting Regressor (GBR)":
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    future_dates = pd.date_range(start=data.index[-1], periods=30)
    future_dates_df = pd.DataFrame({'Date': future_dates})
    future_dates_df['Date'] = pd.to_datetime(future_dates_df['Date'])  # Ensure DateTime format
    future_dates_df['Day'] = future_dates_df['Date'].dt.day
    future_dates_df['Month'] = future_dates_df['Date'].dt.month
    future_dates_df['Year'] = future_dates_df['Date'].dt.year
    future_dates_df['DayOfWeek'] = future_dates_df['Date'].dt.dayofweek
    forecast = model.predict(future_dates_df[['Day', 'Month', 'Year', 'DayOfWeek']])
elif model_choice == "Linear Regression (LR)":
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_dates = pd.date_range(start=data.index[-1], periods=30)
    future_dates_df = pd.DataFrame({'Date': future_dates})
    future_dates_df['Date'] = pd.to_datetime(future_dates_df['Date'])  # Ensure DateTime format
    future_dates_df['Day'] = future_dates_df['Date'].dt.day
    future_dates_df['Month'] = future_dates_df['Date'].dt.month
    future_dates_df['Year'] = future_dates_df['Date'].dt.year
    future_dates_df['DayOfWeek'] = future_dates_df['Date'].dt.dayofweek
    forecast = model.predict(future_dates_df[['Day', 'Month', 'Year', 'DayOfWeek']])
elif model_choice == "Random Forest Regressor (RFR)":
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    future_dates = pd.date_range(start=data.index[-1], periods=30)
    future_dates_df = pd.DataFrame({'Date': future_dates})
    future_dates_df['Date'] = pd.to_datetime(future_dates_df['Date'])  # Ensure DateTime format
    future_dates_df['Day'] = future_dates_df['Date'].dt.day
    future_dates_df['Month'] = future_dates_df['Date'].dt.month
    future_dates_df['Year'] = future_dates_df['Date'].dt.year
    future_dates_df['DayOfWeek'] = future_dates_df['Date'].dt.dayofweek
    forecast = model.predict(future_dates_df[['Day', 'Month', 'Year', 'DayOfWeek']])
elif model_choice == "ARIMA":
    model = ARIMA(y, order=(5, 1, 0))
    model_fit = model.fit()
    future_dates = pd.date_range(start=data.index[-1], periods=30)
    forecast = model_fit.forecast(steps=30)
    future_dates_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    future_dates_df.set_index('Date', inplace=True)
    forecast = future_dates_df['Forecast']

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
fig_forecast.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Predicted Prices'))
fig_forecast.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark', title_font=dict(size=20))
st.plotly_chart(fig_forecast)

# Correlation Analysis
st.header("Correlation Analysis")
st.write("""
    The correlation matrix shows the correlation coefficients between different stocks.
    - **Correlation Coefficient**: A value between -1 and 1 that measures the linear relationship between two variables.
    - **Positive Correlation**: Indicates that the stocks tend to move in the same direction.
    - **Negative Correlation**: Indicates that the stocks tend to move in opposite directions.
    - **Zero Correlation**: Indicates no linear relationship between the stocks.
""")
correlation_data = pd.DataFrame()

for stock_ticker in dow_jones_tickers:
    ticker_data = yf.Ticker(stock_ticker).history(period="1y")['Close']
    correlation_data[stock_ticker] = ticker_data

corr_matrix = correlation_data.corr()

# 3D Correlation Matrix
fig_corr_3d = go.Figure(data=[go.Surface(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index)])
fig_corr_3d.update_layout(title='3D Correlation Matrix', autosize=True,
                          scene=dict(xaxis_title='Stocks', yaxis_title='Stocks', zaxis_title='Correlation'),
                          template='plotly_dark', title_font=dict(size=20))
st.plotly_chart(fig_corr_3d)

# 3D Stock Analysis: Price, Volume, and P/E Ratio
st.header("3D Stock Analysis: Price, Volume, and P/E Ratio")
st.write("""
    The 3D plot below shows the relationship between the stock price, trading volume, and the price-to-earnings (P/E) ratio.
    - **Stock Price**: The closing price of the stock.
    - **Trading Volume**: The number of shares traded during the period.
    - **P/E Ratio**: The price-to-earnings ratio, which is a valuation metric.
""")

# Fetch additional data for P/E ratio
info = stock.info
data['Volume'] = data['Volume']
data['Close'] = data['Close']
data['PE Ratio'] = info['trailingPE'] if 'trailingPE' in info else np.nan  # Handle cases where P/E ratio is not available

# Drop rows with NaN values
data.dropna(subset=['Volume', 'Close', 'PE Ratio'], inplace=True)

fig_3d_pe = px.scatter_3d(data, x='Close', y='Volume', z='PE Ratio', color='Close', title='3D Stock Analysis: Price, Volume, and P/E Ratio', labels={'Close': 'Stock Price', 'Volume': 'Volume', 'PE Ratio': 'P/E Ratio'})
fig_3d_pe.update_layout(template='plotly_dark', title_font=dict(size=20))
st.plotly_chart(fig_3d_pe)

# 3D Stock Analysis: SMAs and Volume
st.header("3D Stock Analysis: SMAs and Volume")
st.write("""
    The 3D plot below shows the relationship between the 30-day and 60-day Simple Moving Averages (SMA) and trading volume.
    - **Short-Term vs Long-Term Trends**: The 30-day SMA represents short-term trends, while the 60-day SMA represents longer-term trends.
    - **Crossovers**: Bullish signals may occur when the 30-day SMA crosses above the 60-day SMA, and bearish signals when it crosses below.
    - **Volume Peaks and Troughs**: High trading volumes often accompany significant price moves and can indicate strong interest in the stock.
""")

data['SMA_30'] = data['Close'].rolling(window=30).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()

# Drop rows with NaN values
data.dropna(subset=['SMA_30', 'SMA_60', 'Volume'], inplace=True)

fig_3d_sma = px.scatter_3d(data, x='SMA_30', y='SMA_60', z='Volume', color='Close', title='3D Stock Analysis: Moving Averages vs Volume', labels={'SMA_30': '30-Day SMA', 'SMA_60': '60-Day SMA', 'Volume': 'Volume', 'Close': 'Close Price'})
fig_3d_sma.update_layout(template='plotly_dark', title_font=dict(size=20))
st.plotly_chart(fig_3d_sma)

st.write("This app provides comprehensive analysis and visualization tools for modern stock analysis.")
