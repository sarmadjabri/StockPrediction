import pandas as pd   #Testing how to use commits
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Flatten, Dot
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from scipy.stats import norm

# request the use of the math formula
use_black_scholes = input("Do you want to use the Black-Scholes equation as a feature? (yes/no): ")

# Define the stock ticker and time period
stock_ticker = 'TSLA'
start_date = '2021-01-01'
end_date = '2022-02-26'

# Download the historical data from Yahoo Finance
df = yf.download(stock_ticker, start=start_date, end=end_date)

# Calculate technical indicators
df = df[['Close']].copy()
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()
df['RSI'] = df['Close'].ewm(span=14, adjust=False).mean() / df['Close'].ewm(span=14, adjust=False).std()
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Bollinger_High'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
df['Bollinger_Low'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()

# Define correlated stocks and download za data
correlated_stocks = ['F', 'GM', 'NIO']
correlated_data = []
for stock in correlated_stocks:
    stock_data = yf.download(stock, start=start_date, end=end_date)
    stock_data = stock_data[['Close']].rename(columns={'Close': f'{stock}_Close'})
    stock_data[f'{stock}_MA_50'] = stock_data[f'{stock}_Close'].rolling(window=50).mean()
    stock_data[f'{stock}_MA_200'] = stock_data[f'{stock}_Close'].rolling(window=200).mean()
    stock_data[f'{stock}_RSI'] = stock_data[f'{stock}_Close'].ewm(span=14, adjust=False).mean() / stock_data[f'{stock}_Close'].ewm(span=14, adjust=False).std()
    stock_data[f'{stock}_MACD'] = stock_data[f'{stock}_Close'].ewm(span=12, adjust=False).mean() - stock_data[f'{stock}_Close'].ewm(span=26, adjust=False).mean()
    stock_data[f'{stock}_MACD_Signal'] = stock_data[f'{stock}_MACD'].ewm(span=9, adjust=False).mean()
    stock_data[f'{stock}_Bollinger_High'] = stock_data[f'{stock}_Close'].rolling(window=20).mean() + 2 * stock_data[f'{stock}_Close'].rolling(window=20).std()
    stock_data[f'{stock}_Bollinger_Low'] = stock_data[f'{stock}_Close'].rolling(window=20).mean() - 2 * stock_data[f'{stock}_Close'].rolling(window=20).std()
    correlated_data.append(stock_data)

# Concatenate all correlated stock data
df = df.join(pd.concat(correlated_data, axis=1), how='left')

# Drop rows with NaN values
df = df.dropna()

# Define the Black-Scholes/Merton equation function
def black_scholes(S, K, t, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)

    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    elif option_type == 'put':
        return K*np.exp(-r*t)*norm.cdf(-d2) - S*norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

if use_black_scholes.lower() == 'yes':
    # Calculate the Black-Scholes equation kjlds
        S = df['Close'].values
        K = 100  # PRICE strike
        t = 1  # time to expiration (years)
        r = 0.05  # risk-free interest rate
        sigma = 0.2  # vol
        option_type = 'call'

        bsm_values = [black_scholes(S[i], K, t, r, sigma, option_type) for i in range(len(S))]

        # Add the Black-Scholes/Merton equation values
        df['BSM_Call'] = bsm_values

        # Create a new feature that combines the Black-Scholes equation values with the other features
        df['Combined_Feature'] = df['BSM_Call'] * df['Close']

# Create lagged features
def create_lagged_features(df, n_lags):
    lagged_features = [df.shift(lag) for lag in range(1, n_lags + 1)]
    lagged_features.append(df)
    df_lagged = pd.concat(lagged_features, axis=1)
    df_lagged.columns = [f'{col}_lag_{lag}' for lag in range(1, n_lags + 1) for col in df.columns] + list(df.columns)
    return df_lagged.dropna()

n_lags = 10
df_lagged = create_lagged_features(df, n_lags)

# target matrices
n_timesteps = 60
n_features = len(df_lagged.columns)
feature_matrix = []
target = []

for i in range(n_timesteps, len(df_lagged)):
    features = df_lagged.iloc[i-n_timesteps:i].values
    target_value = df_lagged.iloc[i]['Close']

    feature_matrix.append(features)
    target.append(target_value)

feature_matrix = np.array(feature_matrix)
target = np.array(target)

# Scale the data
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Flatten feature_matrix
n_feature_columns = feature_matrix.shape[2]  # Number of features per time step
feature_matrix_reshaped = feature_matrix.reshape(-1, n_feature_columns)
feature_matrix_scaled = scaler_features.fit_transform(feature_matrix_reshaped)

# Reshape to og
feature_matrix = feature_matrix_scaled.reshape(feature_matrix.shape)

# Scale the target variable
target = target.reshape(-1, 1)
target = scaler_target.fit_transform(target).flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.2, random_state=42)

# Define the model with api
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = LSTM(units=200, return_sequences=True)(inputs)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = LSTM(units=100, return_sequences=True)(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = LSTM(units=50, return_sequences=True)(x)  # Ensure return_sequences=True for attention mechanism
x = Dropout(0.3)(x)
x = BatchNormalization()(x)

# Attention Layer
attention = Dense(50, activation='softmax')(x)  # Adjusted to match LSTM output dimensions
context = Dot(axes=[1, 1])([x, attention])  # Ensure correct axes for dot product
x = Flatten()(context)

outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# stop overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=2)

# Prepare for predictions
future_data = df_lagged.iloc[-n_timesteps:].copy()

# scale future data
future_data = future_data.values

# scale data
future_data_reshaped = future_data.reshape(-1, n_feature_columns)
future_data_scaled = scaler_features.transform(future_data_reshaped)

# Change shape for future_data_scaled
future_data_final = future_data_scaled.reshape(1, n_timesteps, n_feature_columns)

# Predict the actual pric
future_prediction_scaled = model.predict(future_data_final)
future_prediction = scaler_target.inverse_transform(future_prediction_scaled)

print(f'Prediction for the next time step: ${future_prediction[0][0]:.2f}')