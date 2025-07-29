import argparse
import gc
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Conv1D, LayerNormalization, Bidirectional, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import optuna
import joblib
from optuna.pruners import MedianPruner
from utils import preprocess_features, create_sequences, create_predict_fn, fetch_stock_data,get_stock_data ,create_scaler, AdditiveAttention
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold  # Add this import for k-fold cross-validation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def set_device(use_cpu = False):
    """Set the device to CPU or GPU based on the parameter"""
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("Training on CPU")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                logger.info(f"✅ Enabled memory growth for {gpu}")
                logger.info(f"Training on GPU, {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            except RuntimeError as e:
                logger.error(e)
        else:
            logger.info("No GPU available, training on CPU")


HPO_TRIALS = 100
TRIALS_EPOCHS = 15
FINAL_EPOCHS = 200
MODEL_DIR = "saved_models"
TRIALS_PATIENCE = 8
FINAL_PATIENCE = 30
DATA_DIR = "saved_data"
N_FOLDS = 5  # Number of folds for k-fold cross-validation

tf.keras.backend.clear_session()
gc.collect()

def build_model(gru_units, dropout, learning_rate, lookback, num_features, attention_units, conv_filters):
    inputs = Input(shape=(lookback, num_features), dtype=tf.float32)

    # Convolutional feature extraction with residual connection
    conv = Conv1D(filters=conv_filters, kernel_size=3, activation='relu', padding='same')(inputs)
    conv_norm = LayerNormalization()(conv)
    conv_residual = Add()([inputs, conv_norm]) if num_features == conv_filters else conv_norm

    # Bidirectional GRU layers
    gru1 = Bidirectional(GRU(gru_units, return_sequences=True, kernel_regularizer=l2(0.005)))(conv_residual)
    gru_norm1 = LayerNormalization()(gru1)
    gru2 = GRU(gru_units, return_sequences=True, kernel_regularizer=l2(0.005))(gru_norm1)
    gru_norm2 = LayerNormalization()(gru2)

    # Attention mechanism
    query = Dense(gru_units)(gru_norm2[:, -1, :])[:, tf.newaxis, :]  # Last time step as query
    context_vector, _ = AdditiveAttention(attention_units)(query, gru_norm2)

    # Dense layers
    norm = LayerNormalization()(context_vector)
    dropout_layer = Dropout(dropout)(norm)
    dense = Dense(gru_units // 2, activation='relu')(dropout_layer)
    outputs = Dense(1, activation='sigmoid', dtype=tf.float32)(dense)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    return model


def augment_data(X):
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise

class freeMemory(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    tf.keras.backend.clear_session()
    gc.collect()

def objective(trial, X, y, class_weights, num_features):
    gru_units = trial.suggest_categorical('gru_units', [64, 128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.2, 0.6, step=0.1)
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 5e-5, 1e-4, 5e-4])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lookback = trial.suggest_categorical('lookback', [30, 60, 90])
    attention_units = trial.suggest_categorical('attention_units', [32, 64, 128])
    conv_filters = trial.suggest_categorical('conv_filters', [32, 64, 128])


    logger.info(f"Trial {trial.number} params: gru_units={gru_units}, dropout={dropout}, "
                f"lr={learning_rate}, batch_size={batch_size}, lookback={lookback}, "
                f"attention_units={attention_units}")

    try:
        X_seq, y_seq = create_sequences(X, y, lookback)
        train_size = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_seq[:train_size], y_seq[train_size:]

        logger.info(f"Trial {trial.number}: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

        X_train = augment_data(X_train)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

        model = build_model(gru_units, dropout, learning_rate, lookback, num_features, attention_units, conv_filters)

        tf.keras.backend.clear_session()
        gc.collect()

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=TRIALS_EPOCHS,
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(monitor='val_AUC', mode='max', patience=TRIALS_PATIENCE, restore_best_weights=True),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: trial.report(logs['val_AUC'], epoch)
                ),
                freeMemory(),
            ],
            verbose=0
        )

        tf.keras.backend.clear_session()
        gc.collect()
        val_metrics = model.evaluate(val_dataset, return_dict=True)
        return val_metrics['AUC']
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise

def train_stock_model(symbol, horizon, lookback, fine_tune=False, clean_db=False):
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError(f"lookback must be a positive integer, got {lookback}")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")

    logger.info(f"🚀 Starting training for {symbol} - Lookback: {lookback}, Horizon: {horizon}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    study_db_path = f"saved_models/{symbol}_study_horizon{horizon}.db"
    artifacts_path = os.path.join(MODEL_DIR, f"{symbol}_artifacts_horizon{horizon}.pkl")
    data_path = os.path.join(DATA_DIR, f"{symbol}_data_horizon{horizon}.pkl")
    model_path = os.path.join(MODEL_DIR, f"{symbol}_horizon{horizon}.keras")
    

# Load existing data and model if fine-tuning
    existing_df_scaled = None
    existing_scaler = None
    existing_feature_names = None
    best_params = None
    initial_epoch = 0

    if fine_tune and os.path.exists(artifacts_path):
        artifacts = joblib.load(artifacts_path)
        existing_scaler = artifacts['scaler']
        existing_feature_names = artifacts['feature_names']
        best_params = artifacts.get('best_params', None)
        if os.path.exists(data_path):
            existing_df_scaled = joblib.load(data_path)
            logger.info(f"Loaded existing preprocessed data from {data_path}")
        if os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path} for fine-tuning")
            model = load_model(model_path)
            initial_epoch = artifacts.get('initial_epoch', 0)  # Load initial_epoch from artifacts
            logger.info(f"Resuming from epoch {initial_epoch}")
        else:
            logger.warning(f"Model not found at {model_path}, starting fresh despite fine-tune flag")
            model = None
    else:
        model = None


    if clean_db and os.path.exists(study_db_path):
        os.remove(study_db_path)
        logger.info(f"🧹 Cleared existing study database at {study_db_path}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=364) # I'm dowloading only 1 year till get pay Marketdate plan
    df = get_stock_data(symbol, start_date, end_date, interval = '1h')  
    df_scaled, scaler, feature_names = preprocess_features(df, lookback, symbol)

        # Validate scaler is fitted
    if not hasattr(scaler, 'center_') or not hasattr(scaler, 'scale_'):
        logger.error("Scaler is not fitted before use")
        raise ValueError("Scaler is not fitted")
    
    # Use preprocess_features to get scaled data and scaler
   # df_scaled, scaler, feature_names = preprocess_features(df, lookback, symbol)
   # logger.info(f"NaNs per feature: \n{df_scaled.isna().sum()}")
   # logger.info(f"Feature stats: \n{df_scaled.describe()}")

    log_returns = np.log1p(df_scaled['Close'].pct_change().fillna(0))
   # logger.info(f"Feature correlations with log_returns:\n{df_scaled[feature_names].corrwith(log_returns).abs().sort_values(ascending=False)}")
    y = (log_returns.shift(-horizon) > 0).astype(int).values
    X = df_scaled[feature_names].values

    y = (df_scaled['Close'].shift(-horizon) > df_scaled['Close']).astype(int).values
    X = df_scaled[feature_names].values

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = {0: class_weights[0] * 1.1, 1: class_weights[1] * 0.9}
    logger.info(f"⚖️ Class weights adjusted: {class_weights}")

    logger.info(f"STUDY: {study_db_path}")
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_db_path}")
    study_name = f"{symbol}_{horizon}"

    if clean_db:
        storage._db.execute("DELETE FROM studies WHERE study_name = ?", (study_name,))
        storage._db.commit()

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            load_if_exists=True,
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
    except optuna.exceptions.StorageInternalError:
        logger.warning("Incompatible study detected. Creating new study.")
        storage._db.execute("DELETE FROM studies WHERE study_name = ?", (study_name,))
        storage._db.commit()
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

    # Skip HPO if fine-tuning and best_params exist
    if fine_tune and best_params is not None:
        logger.info(f"Using existing best params for fine-tuning: {best_params}")
    else:
        study.optimize(
            lambda trial: objective(trial, X, y, class_weights, X.shape[-1]),
            n_trials=HPO_TRIALS,
            n_jobs=1
        )
        best_params = study.best_params
        logger.info(f"🌟 Best params: {best_params}")

    X_seq, y_seq = create_sequences(X, y, best_params['lookback'])
    
    # Perform k-fold cross-validation for final model training
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    auc_scores = []
    best_model = None
    best_val_auc = -float('inf')

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_seq)):
        logger.info(f"Fold {fold+1}/{N_FOLDS}")
        X_train, X_val = X_seq[train_idx], X_seq[val_idx]
        y_train, y_val = y_seq[train_idx], y_seq[val_idx]

        X_train = augment_data(X_train)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(best_params['batch_size']).cache().prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(best_params['batch_size']).cache().prefetch(tf.data.AUTOTUNE)

        final_gru_model = build_model(
            best_params['gru_units'],
            best_params['dropout'],
            best_params['learning_rate'],
            best_params['lookback'],
            X.shape[-1],
            best_params['attention_units'],
            best_params['conv_filters']
        )

        tf.keras.backend.clear_session()
        gc.collect()
        logger.info(f"🔨 Training final model on Fold {fold+1}")
        history = final_gru_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=FINAL_EPOCHS,
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(monitor='val_AUC', mode='max', patience=FINAL_PATIENCE, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_AUC', mode='max', factor=0.5, patience=FINAL_PATIENCE//2),
                freeMemory(),
            ],
            verbose=1
        )

        val_metrics = final_gru_model.evaluate(val_dataset, return_dict=True)
        fold_auc = val_metrics['AUC']
        auc_scores.append(fold_auc)
        logger.info(f"Fold {fold+1} AUC: {fold_auc:.3f}")

        # Keep track of the best model based on validation AUC
        if fold_auc > best_val_auc:
            best_val_auc = fold_auc
            best_model = final_gru_model

        tf.keras.backend.clear_session()
        gc.collect()

    # Compute mean and standard deviation of AUC across folds
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    logger.info(f"📊 Final CV Performance across {N_FOLDS} folds: {mean_auc:.3f} ± {std_auc:.3f}")
    logger.info(f"Best model AUC: {best_val_auc}")

    # Use the best model for final predictions
    final_gru_model = best_model
    gru_predict_fn = create_predict_fn(final_gru_model, best_params['lookback'], X.shape[-1])

    
    
    final_gru_model.save(model_path)
    logger.info(f"💾 Model saved to {model_path}")

    # Save scalers and feature names
    scaler = create_scaler()
    X_scaled = scaler.fit_transform(X)
    artifacts = {
        'scaler': scaler,
        'feature_names': feature_names,
        'lookback': best_params['lookback'],
        'best_params' : best_params,
        'best_val_auc': best_val_auc,
        'initial_epoch': initial_epoch + len(history.history['loss'])  # Save updated epoch count
    }
    
    # SAve Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifacts, artifacts_path)
    logger.info(f"💾 Artifacts saved to {artifacts_path}")

    # Save preprocessed data
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(df_scaled, data_path)
    logger.info(f"💾 Preprocessed data saved to {data_path}")

    return final_gru_model, X_val, feature_names, best_params['lookback'], gru_predict_fn

def explain_with_shap(model, X, feature_names,symbol, lookback, horizon ,model_type='gru'):
    if model_type == 'gru':
        predict_fn = create_predict_fn(model, lookback, X.shape[-1])
        n_samples = min(100, X.shape[0])
        X_aggregated = np.mean(X, axis=1)  # Shape: (n_samples, n_features)
        background = X_aggregated[np.random.choice(X_aggregated.shape[0], n_samples, replace=False)]

        def model_predict_2d(X_2d):
            X_3d = np.repeat(X_2d[:, np.newaxis, :], X.shape[1], axis=1)
            return predict_fn(X_3d).numpy()

        explainer = shap.KernelExplainer(model_predict_2d, background)
        shap_values = explainer.shap_values(X_aggregated[:5], nsamples=100)
        shap_values = shap_values.squeeze()  # (5, 46, 1) → (5, 46)
    else:  # Random Forest
        explainer = shap.TreeExplainer(model)
        shap_values_full = explainer.shap_values(X)  # Should be list or (800, 46, 2)
        logging.info(f"shap_values_full type: {type(shap_values_full)}")
        logging.info(f"shap_values_full shape/info: {np.shape(shap_values_full) if isinstance(shap_values_full, np.ndarray) else [v.shape for v in shap_values_full]}")

        # Handle both possible output formats
        if isinstance(shap_values_full, list):
            shap_values = shap_values_full[1][:5]  # Class 1, first 5: (5, 46)
        else:
            shap_values = shap_values_full[:, :, 1][:5]  # (800, 46, 2) → (5, 46)
        logging.info(f"shap_values after indexing shape: {np.shape(shap_values)}")

    # Log shapes for debugging
    logging.info(f"X shape for SHAP: {X.shape}")
    logging.info(f"shap_values final shape: {np.array(shap_values).shape}")

    # Use aggregated data for GRU, raw X for RF
    features_to_plot = X_aggregated[:5] if model_type == 'gru' else X[:5]
    shap.summary_plot(shap_values, features=features_to_plot, feature_names=feature_names)
    plt.savefig(f"{symbol}_{horizon}_shap_summary_{model_type}.png")
    plt.close()
    logging.info(f"SHAP explanation saved as {symbol}_{horizon}_shap_summary_{model_type}.png")

def main():
    parser = argparse.ArgumentParser(description="Train stock price prediction model")
    parser.add_argument('--symbols', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true',default=False, help='Train on CPU instead of GPU')
    parser.add_argument('--clean_db', action='store_true')
    parser.add_argument('--shap', action='store_true', default=False)

    args = parser.parse_args()

    # Set device configuration
   
    set_device(args.cpu)
    symbols = args.symbols.split(',')
    for symbol in symbols:
        final_gru_model, X_val, feature_names, lookback ,_ = train_stock_model(symbol.strip().upper(), 
                          args.horizon, 
                          args.lookback, 
                          args.finetune, 
                          args.clean_db)
        if(args.shap):
            explain_with_shap(final_gru_model, X_val, feature_names, symbol, lookback, horizon = args.horizon ,model_type='gru')

if __name__ == "__main__":
    main()