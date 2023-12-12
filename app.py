from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import cryptocompare
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import threading
from flask import g

app = Flask(__name__)

# Global variable for sequence length
sequence_length = 10

# Function to create a plot and store it in the Flask context
def create_plot_threaded(closing_prices, predictions, split, crypto_symbol):
    # Ensure that Matplotlib operations are performed in the main thread
    with app.app_context():
        base64_plot = create_plot(closing_prices, predictions, split, crypto_symbol)
        return base64_plot

# Function to get historical price data for a given cryptocurrency
def get_crypto_data(crypto_symbol, limit=2000):
    historical_data = cryptocompare.get_historical_price_day(crypto_symbol, currency='USD', limit=limit,
                                                             toTs=pd.Timestamp.now())
    df = pd.DataFrame(historical_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')
    df = df.sort_index()
    return df

# Function to preprocess data and create sequences for the LSTM model
def preprocess_data(data, sequence_length):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    sequences = []
    targets = []
    for i in range(len(data_scaled) - sequence_length):
        seq = data_scaled[i:i + sequence_length]
        target = data_scaled[i + sequence_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets), scaler

# Function to build and train the LSTM model
def build_lstm_model(sequence_length):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to make predictions
def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions_original_scale = scaler.inverse_transform(predictions)
    return predictions_original_scale

# Function to create a plot and return it as a base64-encoded image
def create_plot(closing_prices, predictions, split, crypto_symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(closing_prices, label='Actual Prices', color='black')
    plt.plot(np.arange(split + sequence_length, len(closing_prices)), predictions, label='Predictions', color='blue')
    plt.xlabel('Days')
    plt.ylabel(f'{crypto_symbol} Price (USD)')
    plt.title(f'{crypto_symbol} Price Prediction using LSTM')
    plt.legend()

    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    # Encode the BytesIO object as base64
    base64_img = base64.b64encode(img_data.read()).decode('utf-8')

    plt.close()  # Close the plot to avoid memory leaks
    return base64_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<crypto_symbol>')
def predict(crypto_symbol):
    crypto_data = get_crypto_data(crypto_symbol)
    closing_prices = crypto_data['close'].values

    sequences, targets, scaler = preprocess_data(closing_prices, sequence_length)
    split = int(0.8 * len(sequences))
    X_train, y_train = sequences[:split], targets[:split]
    X_test, y_test = sequences[split:], targets[split:]

    model = build_lstm_model(sequence_length)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

    predictions = make_predictions(model, X_test, scaler)
    base64_plot = create_plot_threaded(closing_prices, predictions, split, crypto_symbol)

    return jsonify(predictions=predictions.tolist(), actual=closing_prices.tolist(), split=split, plot=base64_plot)

@app.before_request
def before_request():
    import matplotlib
    matplotlib.use('Agg')

if __name__ == "__main__":
    # Use the 'Agg' backend to avoid GUI-related issues
    app.run(debug=True)
