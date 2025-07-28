import os
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense
import keras.saving
import yfinance as yf
import talib
from fredapi import Fred
from datetime import datetime, timedelta
import pandas_ta as ta
import requests
import logging
from functools import lru_cache
import time

load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable()
class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        """Initialize layer weights based on input shape."""
        super(AdditiveAttention, self).build(input_shape)

    def call(self, query, values):
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super(AdditiveAttention, self).get_config()
        config.update({
            'units': self.units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def remove_outliers(df, columns, window=21):
    df = df.copy()
    for col in columns:
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, min_periods=1).std()
        z_scores = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        df.loc[abs(z_scores) > 3, col] = rolling_mean
    return df

def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    logger.info(f"Fetched {symbol} data: {df.shape}, Close range: {df['Close'].min():.2f}-{df['Close'].max():.2f}")
    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

def get_stock_data(ticker, start_date, end_date, interval='1d'):
    time.sleep(5)
    api_key = "UlJNQTlPbk52NEJfQWVUOXp2TTZGUHNVOG9ZY3pDNmRITHhaSkpZbWZPaz0"
    '''
    days = {'1y': 365, '6mo': 180, '1mo': 30}.get(period, 365)
    start_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    '''
    url = f"https://api.marketdata.app/v1/stocks/candles/{interval}/{ticker}"
    params = {
        'from': start_date,
        'to': end_date
    }
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            'Date': pd.to_datetime(data['t'], unit='s'),
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v'],
            'Dividends': 0,
            'Stock Splits': 0
        })
        df.set_index('Date', inplace=True)
        df['Adj Close'] = df['Close']
       
        logger.info(f"Fetched {ticker} data: {df.shape}, Close range: {df['Close'].min():.2f}-{df['Close'].max():.2f}")
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def fetch_macro_indicators(start_date, end_date):
    for attempt in range(3):
        try:
            fred = Fred(api_key=os.getenv('FRED_API_KEY', 'your_api_key'))
            gdp = fred.get_series('GDP', start_date, end_date)
            fed_rate = fred.get_series('FEDFUNDS', start_date, end_date)
            cpi = fred.get_series('CPIAUCNS', start_date, end_date)
            vix = yf.Ticker('^VIX').history(start=start_date, end=end_date)['Close']

            if isinstance(vix.index, pd.DatetimeIndex):
                vix.index = vix.index.tz_localize(None)
            for series in [gdp, fed_rate, cpi]:
                if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
                    series.index = series.index.tz_localize(None)

            macro = pd.DataFrame({
                'GDP': gdp, 'Fed_Rate': fed_rate, 'CPI': cpi, 'VIX': vix
            }).interpolate(method='linear').ffill().bfill()
            macro.index = pd.to_datetime(macro.index).tz_localize(None)
            logger.info(f"Macro data fetched: {macro.shape}, index tz: {macro.index.tz}")
            return macro
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                logger.error("Failed to fetch macro data. Using defaults.")
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                macro = pd.DataFrame({
                    'GDP': np.zeros(len(dates)),
                    'Fed_Rate': np.zeros(len(dates)),
                    'CPI': np.zeros(len(dates)),
                    'VIX': np.zeros(len(dates))
                }, index=dates)
                return macro

def fetch_sentiment(symbol, start_date, end_date):
    try:
        api_key = os.getenv('ALPHA_VANTAGE_KEY', 'your_api_key')
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        response = requests.get(url).json()
        sentiment_data = {}
        for item in response.get('feed', []):
            date = pd.to_datetime(item['time_published']).tz_localize(None).date()
            score = float(item.get('ticker_sentiment_score', 0))
            if start_date.date() <= date <= end_date.date():
                if date in sentiment_data:
                    sentiment_data[date].append(score)
                else:
                    sentiment_data[date] = [score]
        dates = pd.date_range(start_date, end_date)
        sentiment = pd.Series([
            np.mean(sentiment_data.get(date.date(), [0])) for date in dates
        ], index=dates).fillna(0)
        x_sentiment = sentiment.rolling(5).mean().fillna(0)
        sentiment_diff = sentiment.diff().fillna(0)
        sentiment_vol = sentiment.rolling(5).std().fillna(0)
        logger.info(f"Fetched real sentiment for {symbol}: {len(sentiment)} entries")
        return sentiment, x_sentiment, sentiment_diff, sentiment_vol
    except Exception as e:
        logger.warning(f"Failed to fetch sentiment: {e}. Using zeros.")
        dates = pd.date_range(start=start_date, end=end_date)
        zero_series = pd.Series(np.zeros(len(dates)), index=dates)
        return zero_series, zero_series, zero_series, zero_series

def compute_technical_indicators(df):
    df = df.copy()
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
    df['Bollinger_Mid'] = talib.SMA(df['Close'], timeperiod=20)
    df['EMA_9'] = talib.EMA(df['Close'], timeperiod=9)
    df['EMA_21'] = talib.EMA(df['Close'], timeperiod=21)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50) #Add SMA_50 to be used in predict script

    log_close = np.log1p(df['Close'])
    df['Returns'] = log_close.pct_change().fillna(0)
    df['Volatility'] = df['Returns'].rolling(21).std().fillna(0)

    df['Momentum_3'] = log_close.pct_change(3).fillna(0)
    df['Momentum_5'] = log_close.pct_change(5).fillna(0)
    df['Lag_Return_1'] = df['Returns'].shift(1).fillna(0)
    df['Lag_Return_3'] = df['Returns'].shift(3).fillna(0)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

   # df['XLK_Corr'] = df['Close'].rolling(21).corr(df['XLK_Close']).fillna(0)
    df['Volume_Return'] = df['Returns'] * df['Volume'].rolling(5).mean().fillna(0)

    # Add rolling statistics
    windows = [3, 7, 21]
    for w in windows:
        df[f'Rolling_Mean_{w}'] = df['Close'].rolling(w).mean()
        df[f'Rolling_Std_{w}'] = df['Close'].rolling(w).std()
        df[f'Rolling_Min_{w}'] = df['Close'].rolling(w).min()
        df[f'Rolling_Max_{w}'] = df['Close'].rolling(w).max()

    return df

def preprocess_features(df, lookback, symbol, interval='1d'):
    df = df.copy()

    # Fetch XLK data early
   # xl = yf.Ticker('XLK').history(start=df.index.min(), end=df.index.max(), auto_adjust=True)
   # xl.index = pd.to_datetime(xl.index).tz_localize(None)
   # df['XLK_Close'] = xl['Close'].reindex(df.index, method='ffill')
    start_date = df.index.min()
    end_date = df.index.max()
    df_spy = get_stock_data('SPY', start_date, end_date, interval=interval)
    df['SPY_Close'] = df_spy['Close'].reindex(df.index, method='ffill')
    df_gld = get_stock_data('GLD', start_date, end_date, interval = interval) # Gold price
    df['GLD_Close'] = df_gld['Close'].reindex(df.index, method='ffill')
    df_vix = get_stock_data('VIXY', start_date, end_date, interval = interval) # VIx Short-Term
    df['VIX_Close'] = df_vix['Close'].reindex(df.index, method='ffill')

    # Apply outlier removal to price columns
    price_cols = ['Open', 'High', 'Low', 'Close']
    df = remove_outliers(df, price_cols, window=21)

    # Compute technical indicators
    df = compute_technical_indicators(df)

    # Apply log transformation to price columns
    price_cols = ['Open', 'High', 'Low', 'Close', 'VWAP']
    df[price_cols] = np.log1p(df[price_cols])

    # Fetch macro indicators
    #macro = fetch_macro_indicators(df.index.min(), df.index.max())
    #df = df.join(macro, how='left')

    # Fetch sentiment
#    sentiment, x_sentiment, sentiment_diff, sentiment_vol = fetch_sentiment(symbol, df.index.min(), df.index.max())
#    df['Sentiment'] = sentiment.reindex(df.index, method='ffill')
#    df['X_Sentiment'] = x_sentiment.reindex(df.index, method='ffill')
#    df['Sentiment_Diff'] = sentiment_diff.reindex(df.index, method='ffill')
#    df['Sentiment_Vol'] = sentiment_vol.reindex(df.index, method='ffill')

    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

    # Validate for NaNs/infinites
    if df.isna().any().any() or np.any(np.isinf(df)):
        logger.warning("Data contains NaNs or infinites after imputation")
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

    # Scale features
    scaler = create_scaler()
    features_to_scale = df.drop(columns=['Close', 'Dividends', 'Stock Splits']).columns
    scaled_features = scaler.fit_transform(df[features_to_scale])
    df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale, index=df.index)
    df_scaled['Close'] = df['Close']

    # Define feature names
    keep_cols = ['Volume', 'Bollinger_Mid', 'EMA_9', 'EMA_21', 
                 'Returns', 'Lag_Return_3', 'VWAP', 'Volume_Return',
                 'Rolling_Std_3', 'Rolling_Mean_3', 
                 'Rolling_Std_21', 'Rolling_Mean_21',
                 'Rolling_Std_7', 'Rolling_Mean_7',
                 'GLD_Close', 'VIX_Close', 'SPY_Close',
                 'High', 'RSI', 'MACD',
                 

            #'Open',  'Low',   'MACD_Signal','Volatility',
            #'OBV', 'Momentum_3', 'Momentum_5', 'Lag_Return_1',
            #'XLK_Corr', 'XLK_Close',
            # 'Sentiment', 'X_Sentiment','Sentiment_Diff', 'Sentiment_Vol',
            # 'Rolling_Max_3',
            # 'Rolling_Max_7',
            # 'Rolling_Min_21', 'Rolling_Max_21','VIX',
            
            #Minimal impact features
            # 
            #'Rolling_Min_3', 'Rolling_Min_7',  'Fed_Rate',
        
    ]
      
    df_scaled = df_scaled[[col for col in keep_cols if col in df_scaled.columns] + ['Close']]
    feature_names = [col for col in df_scaled.columns if col != 'Close']

    #logger.info(f"Feature names: {feature_names}")
    return df_scaled, scaler, feature_names

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    if np.any(np.isnan(Xs)) or np.any(np.isinf(Xs)):
        logger.warning("Sequences contain NaNs or infinites")
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"Sequence shapes: X={Xs.shape}, y={ys.shape}")
    return Xs, ys

def create_predict_fn(model, lookback, num_features):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, lookback, num_features], dtype=tf.float32)])
    def predict_fn(inputs):
        return model(inputs, training=False)
    return predict_fn

def load_artifacts(symbol, horizon):
    artifacts_path = os.path.join("saved_models", f"{symbol}_artifacts_horizon{horizon}.pkl")
    if os.path.exists(artifacts_path):
        artifacts = joblib.load(artifacts_path)
        logger.info(f"Loaded artifacts from {artifacts_path}")
        return artifacts
    else:
        raise FileNotFoundError(f"Artifacts not found at {artifacts_path}")
    
def create_scaler():
    return RobustScaler()