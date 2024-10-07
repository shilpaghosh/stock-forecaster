import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# Function to prepare data for LSTM prediction
def prepare_data_for_prediction(data, look_back=60):
    x = []
    x.append(data[-look_back:])
    return np.array(x)

# Function to forecast the stock price for the next 3 months
def forecast_stock(stock_ticker, model_file, scaler):
    df = yf.download(stock_ticker, period='1y')
    data = df['Adj Close'].values.reshape(-1, 1)

    # Normalize the data
    scaled_data = scaler.fit_transform(data)

    # Prepare for prediction
    look_back = 180  # Use the last 180 days to predict the next day
    last_180_days_scaled = prepare_data_for_prediction(scaled_data, look_back)
    last_180_days_scaled = np.reshape(last_180_days_scaled, (1, last_180_days_scaled.shape[1], 1))

    # Load the pre-trained model
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        st.error(f"Model file {model_file} not found. Please train the model first.")
        return None, None

    # Predict the next 6 months
    predictions = []
    for _ in range(180):  # 90 days (3 months)
        pred_price = model.predict(last_180_days_scaled)
        predictions.append(pred_price[0][0])
        last_180_days_scaled = np.roll(last_180_days_scaled, -1, axis=1)
        last_180_days_scaled[0, -1, 0] = pred_price[0][0]

    # Inverse transform the predicted prices
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Return actual and forecasted prices for comparison
    return data.flatten(), predictions.flatten()

# Streamlit app
st.title("Best Performing Stocks - 2024 (Last 6 Months) with Forecast")

# Dropdown for stock selection
stocks = ['NVDA', 'TSLA', 'META', 'AAPL', 'MSFT']  # Example of top stocks
selected_stock = st.selectbox("Select a stock to forecast:", stocks)

model_file = f'models/{selected_stock}_model.h5'



# Fetch stock data
data = yf.download(selected_stock, period='6mo')


st.write(f"## {selected_stock} Stock Performance Data")
st.dataframe(data.tail())

# Forecast for next 3 months
if os.path.exists(model_file):
    # Load scaler and predict
    scaler = MinMaxScaler(feature_range=(0, 1))  # Placeholder, adjust as needed
    actual_prices, forecasted_prices = forecast_stock(selected_stock, model_file, scaler)

    if forecasted_prices is not None:
        # Create a DataFrame for actual and forecasted prices
        forecast_dates = pd.date_range(start=pd.Timestamp.today(), periods=180) # six months
        actual_dates = pd.date_range(end=pd.Timestamp.today(), periods=len(actual_prices))

        forecast_df = pd.DataFrame({
            'Date': np.concatenate([actual_dates, forecast_dates]),
            'Price': np.concatenate([actual_prices, np.full(len(forecast_dates), np.nan)]),
            'Type': ['Actual'] * len(actual_prices) + ['Forecasted'] * len(forecast_dates)
        })

        # Create a Plotly figure with two traces (one for actual, one for forecast)
        fig = go.Figure()

        # Add actual prices trace (red line)
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_prices,
            mode='lines',
            name='Actual Prices',
            line=dict(color='red', width=2)
        ))

        # Add forecasted prices trace (green line)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecasted_prices,
            mode='lines',
            name='Forecasted Prices',
            line=dict(color='green', width=2)
        ))

        # Update layout
        fig.update_layout(
            title=f"Actual vs Forecasted Stock Prices for {selected_stock}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            legend_title="Legend",
        )

        # Display the plot
        st.plotly_chart(fig)

else:
    st.error(f"No pre-trained model found for {selected_stock}. Please train the model using the training script.")
