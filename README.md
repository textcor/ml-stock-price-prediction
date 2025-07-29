
# ML Stock Price Prediction (Hourly)

## Overview

This repository contains a sophisticated machine learning system for predicting hourly stock price movements. The project implements a hybrid deep learning architecture combining GRU networks with attention mechanisms and convolutional layers for time-series forecasting. The system provides:

- Hourly price direction predictions (UP/DOWN)
- Confidence scores and probability estimates
- Technical analysis integration (SMA, MACD, VWAP, etc.)
- Volatility-adjusted projections
- SHAP value explanations for model interpretability

## Key Features

### Advanced Model Architecture
- **Bidirectional GRU** with residual connections
- **Conv1D layers** for feature extraction
- **Attention mechanism** to focus on important time steps
- **Layer normalization** for stable training
- **Regularization** with dropout and L2 penalties

### Robust Training Pipeline
- **Optuna hyperparameter optimization**
- **K-fold cross-validation**
- **Class weighting** for imbalanced data
- **Data augmentation** with noise injection
- **Early stopping** and learning rate reduction
- **GPU/CPU compatibility** with memory management

### Comprehensive Prediction System
- **Technical indicator integration** (50+ features)
- **Volatility-adjusted projections**
- **Confidence intervals** for predictions
- **Pivot point analysis**
- **Pattern detection** (Head & Shoulders, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/textcor/ml-stock-price-prediction.git
cd ml-stock-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python train_hour.py --symbols AAPL,MSFT --horizon 5 --lookback 60
```

Options:
- `--symbols`: Comma-separated list of stock symbols
- `--horizon`: Prediction horizon in hours (default: 5)
- `--lookback`: Initial lookback window size (default: 60)
- `--finetune`: Fine-tune existing model
- `--cpu`: Force CPU training
- `--clean_db`: Clear previous optimization studies
- `--shap`: Generate SHAP explanations

### Making Predictions
```bash
python predict_hour.py --symbol AAPL --horizon 5
```

Output includes:
- Price direction (UP/DOWN) with confidence
- Projected price with confidence interval
- Technical analysis recommendations
- Pivot points
- Chart pattern detection

## File Structure

```
├── data/                   # Raw and processed data storage
├── saved_models/           # Trained models and artifacts
├── src/
│   ├── train_hour.py       # Model training pipeline
│   ├── predict_hour.py     # Prediction system
│   ├── technical.py        # Technical analysis functions
│   └── utils.py           # Shared utilities
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── README.md              # This document
```

## Technical Details

### Data Processing
- Hourly price data from Yahoo Finance
- Log-returns transformation
- Robust scaling with outlier handling
- 50+ technical indicators
- Sequential windowing for time-series

### Model Architecture
```python
Input → Conv1D → GRU → Attention → Dense → Output
            ↑_________ Residual _________↑
```

### Hyperparameter Optimization
- GRU units: [64, 128, 256, 512]
- Dropout: [0.2, 0.6]
- Learning rate: [1e-5, 5e-5, 1e-4, 5e-4]
- Lookback window: [30, 60, 90]
- Attention units: [32, 64, 128]

## Performance Metrics

- Evaluated using AUC (Area Under Curve)
- K-fold cross-validation (N=5)
- Validation on 20% holdout set
- Typical performance: 0.65-0.75 AUC

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational purposes only. Predictions should not be considered as financial advice. Always conduct your own research before making investment decisions.