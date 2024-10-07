import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import os

# Function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Output layer for stock price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to prepare data for LSTM
def prepare_data(data, look_back=60):
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def train_and_save_model(stock_ticker, model_file='stock_model.h5'):
    # Download stock data
    df = yf.download(stock_ticker, period='1y')
    data = df['Adj Close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare the data for LSTM
    look_back = 180  # Use the last 180 days to predict the next day
    x_train, y_train = prepare_data(scaled_data, look_back)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for LSTM

    # Create and train the model
    model = create_lstm_model(input_shape=x_train.shape)
    model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=True)

    # Save the trained model
    model.save(model_file)
    print(f"Model trained and saved to {model_file}")

    return scaler

# Train and save the model for NVIDIA (example)
if __name__ == "__main__":
    stocks = ['NVDA', 'TSLA', 'META', 'AAPL', 'MSFT']
    for stock in stocks:
        print("Training {}".format(stock))
        train_and_save_model(stock, model_file="models/{}_model.h5".format(stock))

