ML Stock Price Prediction

This repository contains code for predicting stock price direction using advanced machine learning techniques, specifically focusing on hourly predictions. The project aims to demonstrate how historical stock data, combined with deep learning models and technical analysis, can be leveraged to build robust predictive models.
Table of Contents

    Project Overview

    Features

    Technologies Used

    Installation

    Usage

    Data

    Model Architecture & Training

    Prediction Logic

    Results

    Contributing

    License

Project Overview

The "ML Stock Price Prediction" project focuses on applying a sophisticated GRU (Gated Recurrent Unit) deep learning model with attention mechanisms to forecast hourly stock price movements (UP/DOWN). It integrates hyperparameter optimization, k-fold cross-validation, and technical analysis to enhance prediction accuracy and provide comprehensive insights. The repository serves as a practical example of data-driven, short-term stock market analysis and model interpretability.
Features

    Hourly Price Direction Prediction: Specialized for predicting stock price direction (UP/DOWN) on an hourly basis.

    Advanced GRU Model: Utilizes a Bidirectional GRU network with Conv1D layers, Layer Normalization, and Additive Attention for robust feature extraction and sequence processing.

    Hyperparameter Optimization (HPO): Employs Optuna for automated and efficient tuning of model hyperparameters to find the optimal configuration.

    K-Fold Cross-Validation: Implements k-fold cross-validation during final model training for more reliable performance evaluation and to reduce overfitting.

    Data Preprocessing & Augmentation: Includes scripts for cleaning, normalizing, and preparing historical stock data, along with data augmentation techniques to improve model generalization.

    SHAP for Model Interpretability: Integrates SHAP (SHapley Additive exPlanations) to explain individual predictions and understand feature importance, enhancing model transparency.

    Technical Analysis Integration: Incorporates various technical indicators (e.g., SMA, EMA, MACD, RSI, Bollinger Bands, VWAP) and pivot points to enrich the input features and provide additional context for predictions.

    Confidence & Direction Output: Provides not only the predicted direction (UP/DOWN) but also a confidence score and projected price with a confidence interval.

    GPU/CPU Support: Configurable to train models on either GPU (if available) or CPU.

Technologies Used

    Python: The primary programming language.

    Pandas: For data manipulation and analysis.

    NumPy: For numerical operations.

    TensorFlow/Keras: For building and training deep learning models (GRU, Attention).

    Optuna: For hyperparameter optimization.

    yfinance: For fetching historical stock data.

    Scikit-learn: For data preprocessing utilities (e.g., StandardScaler, compute_class_weight).

    SHAP: For model interpretability.

    Matplotlib: For generating SHAP summary plots.

    Joblib: For saving and loading model artifacts.

Installation

To get a copy of the project up and running on your local machine, follow these steps:

    Clone the repository:

    git clone https://github.com/textcor/ml-stock-price-prediction.git
    cd ml-stock-price-prediction

    Create a virtual environment (recommended):

    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`

    Install the required dependencies:

    pip install -r requirements.txt

    (Note: A requirements.txt file should be created based on the train_hour.py and predict_hour.py dependencies. If not present, install manually: pip install pandas numpy tensorflow optuna yfinance scikit-learn shap matplotlib joblib)

Usage

The repository contains two main scripts: train_hour.py for model training and predict_hour.py for making predictions.
Training the Model (train_hour.py)

This script trains a GRU model for a specified stock symbol and prediction horizon. It includes hyperparameter optimization and k-fold cross-validation.

python train_hour.py --symbols <STOCK_SYMBOL(S)> --horizon <PREDICTION_HORIZON_HOURS> --lookback <LOOKBACK_HOURS> [--finetune] [--cpu] [--clean_db] [--shap]

Arguments:

    --symbols: Comma-separated stock ticker symbols (e.g., AAPL,MSFT). Required.

    --horizon: The prediction horizon in hours (e.g., 1 for 1 hour ahead). Default is 5.

    --lookback: The number of past hours to consider for sequence creation (e.g., 60). Default is 60.

    --finetune: (Optional) If set, the script attempts to load an existing model and continue training/fine-tuning.

    --cpu: (Optional) If set, forces training on CPU even if a GPU is available.

    --clean_db: (Optional) If set, clears the Optuna study database before starting optimization.

    --shap: (Optional) If set, generates SHAP explanation plots after training.

Example:

python train_hour.py --symbols AAPL --horizon 1 --lookback 90 --shap

Making Predictions (predict_hour.py)

This script uses a trained model to predict the price direction for a given stock symbol and horizon.

python predict_hour.py --symbol <STOCK_SYMBOL> --horizon <PREDICTION_HORIZON_HOURS>

Arguments:

    --symbol: The stock ticker symbol (e.g., AAPL). Required.

    --horizon: The prediction horizon in hours (must match the horizon used during training). Default is 1.

Example:

python predict_hour.py --symbol MSFT --horizon 1

The script will output the predicted direction (UP/DOWN), confidence, last close price, projected price, confidence interval, and total percentage change, along with technical analysis recommendations.
Data

The project fetches historical hourly stock data using yfinance. The data includes Open, High, Low, Close, and Volume.
The preprocess_features function handles scaling and feature engineering, which includes various technical indicators.
Model Architecture & Training

The core of the prediction system is a deep learning model built with TensorFlow/Keras:

    Input Layer: Takes sequences of historical hourly data.

    Convolutional Layer (Conv1D): Extracts local features from the time series, with a residual connection for stable learning.

    Bidirectional GRU Layers: Processes sequential data in both forward and backward directions to capture dependencies.

    Layer Normalization: Applied after convolutional and GRU layers to stabilize training.

    Additive Attention Mechanism: Focuses the model on the most relevant parts of the input sequence for prediction.

    Dense Layers: Final classification layers with a sigmoid activation for binary output (UP/DOWN).

Training Process:

    Data Acquisition: Fetches 1 year of hourly data using yfinance.

    Feature Engineering: Calculates various technical indicators (e.g., moving averages, RSI, MACD, Bollinger Bands, VWAP) and adds them as features.

    Hyperparameter Optimization: Optuna is used to find the best combination of gru_units, dropout, learning_rate, batch_size, lookback, attention_units, and conv_filters.

    K-Fold Cross-Validation: The final model is trained using 5-fold cross-validation, improving robustness and providing a more reliable estimate of model performance (AUC score).

    Early Stopping & Learning Rate Reduction: Callbacks are used to prevent overfitting and adjust the learning rate during training.

    Model Saving: The best performing model and its associated artifacts (scaler, feature names, best parameters) are saved for future predictions.

    SHAP Explanations: Optionally generates SHAP summary plots to visualize feature importance and understand model decisions.

Prediction Logic

The predict_hour.py script performs the following steps:

    Load Artifacts: Loads the trained GRU model, data scaler, feature names, and lookback period from saved files.

    Fetch & Preprocess Latest Data: Retrieves the most recent hourly data for the specified symbol and preprocesses it using the loaded scaler and feature names.

    Sequence Preparation: Creates the input sequence for the model based on the lookback period.

    Model Prediction: The GRU model outputs a probability score for the price moving UP.

    Direction & Confidence: Determines the direction (UP/DOWN) and calculates a confidence score based on the probability.

    Technical Analysis: Calculates various technical indicators, pivot points, and generates buy/sell recommendations for the latest data.

    Projected Price & Confidence Interval: Simulates future price movements by incorporating expected return (adjusted by confidence and technical trend factors) and volatility to derive a projected price and a 95% confidence interval.

    Sanity Checks: Adjusts the projected price to ensure it aligns with the predicted direction.

    Output: Prints detailed prediction results, including direction, confidence, last close, projected price, confidence interval, and total percentage change, along with technical analysis insights.

Results

The training process aims to maximize the AUC (Area Under the Receiver Operating Characteristic Curve) metric, which indicates the model's ability to distinguish between UP and DOWN movements. The final cross-validation performance (mean AUC and standard deviation) is logged.

The prediction script provides a comprehensive output for each prediction, including:

    Symbol & Horizon

    Predicted Direction (UP/DOWN)

    Confidence Score

    GRU Probability

    Last Close Price

    Projected Price

    95% Confidence Interval

    Total Percent Change

    Technical Recommendations (Buy/Sell)

    Pivot Points & Pattern Detection

Examples of these results are printed to the console when running predict_hour.py.
Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

    Fork the repository.

    Create a new branch (git checkout -b feature/YourFeature).

    Make your changes.

    Commit your changes (git commit -m 'Add some feature').

    Push to the branch (git push origin feature/YourFeature).

    Open a Pull Request.

License

This project is licensed under the MIT License - see the LICENSE file for details.
