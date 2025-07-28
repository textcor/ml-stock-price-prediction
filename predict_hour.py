import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import os
from utils import fetch_stock_data, preprocess_features, create_sequences, get_stock_data ,load_artifacts, AdditiveAttention
from datetime import datetime, timedelta

from technical import calculate_indicators, detect_pattern, get_indicators_values, print_pivot_points,calculate_pivot_points, print_recommendations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def predict_price(symbol, horizon=1):
    """
    Predict the stock price direction based on a trained GRU model with improved stability.
    """
    try:
        # Load model and artifacts
        model_path = os.path.join("saved_models", f"{symbol}_horizon{horizon}.keras")
        artifacts = load_artifacts(symbol, horizon)
        scaler = artifacts['scaler']
        feature_names = artifacts['feature_names']
        lookback = artifacts['lookback']

        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
        logger.info(f"Loaded model for horizon={horizon}, lookback={lookback}, expected features: {len(feature_names)}")

    except Exception as e:
        logger.error(f"❌ Error loading model/artifacts for horizon {horizon}: {e}")
        return None

    try:
        # Fetch and preprocess data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=360)
        logger.info(f"Getting data from {start_date} to {end_date}")
        df = get_stock_data(symbol, start_date, end_date, interval='1h')
        df_scaled, _, _ = preprocess_features(df, lookback, symbol)

        # Align features with training
        X = df_scaled[feature_names].values
        X_scaled = scaler.transform(X)
        sequence = X_scaled[-lookback:]
        sequence = sequence.reshape((1, lookback, len(feature_names)))
        logger.info(f"Prepared sequence with shape: {sequence.shape}")

        # Make prediction
        prediction = model.predict(sequence, verbose=0)[0][0]
        logger.info(f"Raw prediction: {prediction}")

        if not np.isfinite(prediction):
            logger.error("❌ Model prediction is NaN or infinite")
            return None

        probability = np.clip(prediction, 0.1, 0.9)  # Broader clipping to avoid extreme values
        direction = "UP" if probability > 0.5 else "DOWN"
        confidence = min(1.0, max(0.0, 2 * abs(probability - 0.5)))  # Cap confidence at 100%

        # Reverse log transformation for last_close
        last_close_log = df_scaled['Close'].iloc[-1]
        last_close = np.expm1(last_close_log)

        # Smooth daily return with EMA (24-hour window)
        price_returns_series = df['Close'].pct_change().rolling(window=24, min_periods=1).mean()
        price_returns = price_returns_series.iloc[-1] if len(price_returns_series) > 0 else 0.0
        if np.isnan(price_returns):  # Explicitly handle NaN
            price_returns = 0.0
        price_returns = np.clip(price_returns, -0.05, 0.05)  # Cap at ±5%

        # Volatility with longer lookback (720 hours = 30 days)
        recent_returns = df['Close'].pct_change().rolling(window=720, min_periods=1).std()
        volatility = recent_returns.iloc[-1] if len(recent_returns) > 0 else 0.0
        if np.isnan(volatility):  # Explicitly handle NaN
            volatility = 0.0
        volatility = np.clip(volatility, 0, 0.1)  # Cap volatility at 10%

        # Calculate technical indicators
        df = calculate_indicators(df)
        recommendation = get_indicators_values(df)
        analysis = {'pivot_points': calculate_pivot_points(df)}

        # Adjust trend factor based on technical signals
        trend_factor = 1.0
        if 'SMA_50' in df.columns:
            trend_factor = 1.5 if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1] else 0.5
        
        # Incorporate technical recommendations (e.g., MACD, VWAP)

        # Count buy/sell signals, handling non-string values
        buy_signals = sum(1 for r in recommendation.values() if isinstance(r, str) and r.lower() == 'buy')
        sell_signals = sum(1 for r in recommendation.values() if isinstance(r, str) and r.lower() == 'sell')
        if buy_signals > sell_signals:
            trend_factor *= 1.2  # Boost for bullish signals
        elif sell_signals > buy_signals:
            trend_factor *= 0.8  # Reduce for bearish signals

        logger.info(f"Buy signals: {buy_signals}, Sell signals: {sell_signals}")
        # Adjust expected return based on direction
        base_return = abs(price_returns) * confidence * trend_factor
        expected_return = base_return if direction == "UP" else -base_return
        logger.info(f"Smooth Daily Return: {price_returns:.6f}")
        logger.info(f"Volatility (720-hours): {volatility:.6f}")
        logger.info(f"Trend Factor: {trend_factor:.2f}")
        logger.info(f"Expected Return (adjusted): {expected_return:.6f}")

        # Simulate future prices with controlled randomness
        predicted_changes = last_close * (1 + expected_return * horizon + volatility * np.random.normal(0, 0.5, 1000) * np.sqrt(horizon))
        future_price = np.mean(predicted_changes)
        confidence_interval = 1.96 * np.std(predicted_changes)

        # Sanity check: Ensure projected price aligns with direction
        if direction == "DOWN" and future_price > last_close:
            logger.warning(f"Projected price (${future_price:.2f}) contradicts DOWN direction. Adjusting...")
            future_price = last_close * (1 - abs(expected_return) * horizon)  # Force downward adjustment
        elif direction == "UP" and future_price < last_close:
            logger.warning(f"Projected price (${future_price:.2f}) contradicts UP direction. Adjusting...")
            future_price = last_close * (1 + abs(expected_return) * horizon)  # Force upward adjustment

        # Percent changes
        total_percent_change = ((future_price - last_close) / last_close) * 100

        # Log technical analysis
        print_recommendations(recommendation)
        print_pivot_points(analysis)
        detect_pattern(df)

        logger.info("\n📈 Prediction Results:")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Horizon: {horizon} hour(s)")
        logger.info(f"Direction: {direction}")
        logger.info(f"Confidence: {confidence:.2%}")
        logger.info(f"Probability (GRU): {probability:.4f}")
        logger.info(f"Last Close: ${last_close:.2f}")
        logger.info(f"Projected Price: ${future_price:.2f}")
        logger.info(f"95% Confidence Interval: ±${confidence_interval:.2f}")
        logger.info(f"Total Percent Change: {total_percent_change:.2f}%")

        return {
            'symbol': symbol,
            'horizon': horizon,
            'direction': direction,
            'confidence': confidence,
            'probability': probability,
            'last_close': last_close,
            'projected_price': future_price,
            'confidence_interval': confidence_interval,
            'total_percent_change': total_percent_change
        }

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--symbol', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon in hours')
    args = parser.parse_args()

    result = predict_price(args.symbol, args.horizon)