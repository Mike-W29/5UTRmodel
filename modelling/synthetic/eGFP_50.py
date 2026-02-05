#!/usr/bin/env python3
"""
eGFP50_model.py

Train a deep learning model (CNN + BiLSTM + Attention) to predict mean ribosome load (rl) using egfp_50 data
from 5' UTR sequence and structure joint encodings.

This script contains:
 - robust one-hot encoding for sequence and structure
 - a custom Attention Keras layer that returns both context vector and attention weights
 - model building function
 - preprocessing that fits scalers on training data and reuses them during inference
 - training loop with EarlyStopping and best-weight restore
 - evaluation on a test set with R^2, Spearman and MSE metrics

How to use:
 - Edit DATA_PATH and MODEL_OUT_PATH constants below (or wrap this script with argparse).
 - Run: python train_5utr_model.py

Author: Mike Wang 
Date: 2025-10-31
"""

from typing import Tuple, Dict
import logging
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import spearmanr, linregress

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Configuration / constants
# ---------------------------
DATA_PATH = "~/data/GSM3130435_egfp_unmod_1_structure_feature_table_maxBPspan_30.txt"
MODEL_OUT_PATH = "~models/synthetic/model_eGFP_50.h5"
# Limit used in your original script; keep same slicing for reproducibility
MAX_ROWS = 280_000
SEQ_LEN = 80  # sequence length used in encoding and model input
RANDOM_STATE = 2021
BATCH_SIZE = 128
EPOCHS = 999

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Reproducibility
# ---------------------------
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------
# Utilities: One-hot encoding
# ---------------------------
def one_hot_encode(
    df: pd.DataFrame,
    seqcol: str = "sequence",
    strucol: str = "structure_full",
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create one-hot encodings for sequences and sequence+structure pairs.

    - Sequence one-hot: shape (N, seq_len, 4) with order [A, C, G, T].
    - Structure+sequence joint one-hot: shape (N, seq_len, 12) mapping combinations
      like 'A(' , 'A)', 'A.' etc. to 12 channels.

    The function is robust to sequences shorter than seq_len — shorter sequences are
    padded with zeros. Unknown characters produce zero vectors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sequence and structure columns.
    seqcol : str
        Column name containing nucleotide sequence (string).
    strucol : str
        Column name containing structure characters (string) with characters like '(', ')', '.'.
    seq_len : int
        Fixed length to truncate/pad sequences to.

    Returns
    -------
    seq_vectors : np.ndarray
        One-hot encoded nucleotide array, dtype=float32.
    stru_vectors : np.ndarray
        One-hot encoded joint sequence-structure array, dtype=float32.
    """
    nuc_d: Dict[str, list] = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "U": [0, 0, 0, 1],  # treat U as T if present
        "N": [0, 0, 0, 0],
    }

    # 12-channel mapping for base + structure char
    bases = ["A", "C", "G", "T"]
    struct_symbols = ["(", ")", "."]
    stru_d = {}
    idx = 0
    for b in bases:
        for s in struct_symbols:
            stru_d[f"{b}{s}"] = [1 if i == idx else 0 for i in range(12)]
            idx += 1
    # fallback = zeros
    fallback_12 = [0] * 12

    N = len(df)
    seq_vectors = np.zeros((N, seq_len, 4), dtype=np.float32)
    stru_vectors = np.zeros((N, seq_len, 12), dtype=np.float32)

    seq_series = df[seqcol].fillna("").astype(str).str.upper()
    stru_series = df[strucol].fillna("").astype(str).str.upper()

    for i, (seq, stru) in enumerate(zip(seq_series, stru_series)):
        # truncate or pad
        seq = seq[:seq_len]
        stru = stru[:seq_len]

        L_seq = len(seq)
        # Sequence one-hot
        for j, ch in enumerate(seq):
            seq_vectors[i, j, :] = nuc_d.get(ch, [0, 0, 0, 0])

        # structure+sequence one-hot
        # zip short-circuits on shorter; if structure missing, map to fallback
        for j in range(seq_len):
            if j < len(seq) and j < len(stru):
                key = seq[j] + stru[j]
                stru_vectors[i, j, :] = stru_d.get(key, fallback_12)
            else:
                # padding (zeros) already present
                pass

    return seq_vectors, stru_vectors


# ---------------------------
# Attention layer
# ---------------------------
class Attention(Layer):
    """
    Attention layer for sequence features.

    Returns:
      - context vector c (shape = (batch, features))
      - attention weights a (shape = (batch, steps, 1))
      
    Ref in #https://zhuanlan.zhihu.com/p/97525394
    """
    

    def __init__(self, bias: bool = True, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        self.bias = bias
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.features_dim = 0

    def build(self, input_shape):
        assert len(input_shape) == 3, "Attention expects input shape (batch, steps, features)"
        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
            name=f"{self.name}_W",
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(1,), initializer="zeros", name=f"{self.name}_b", regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None

        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # We return None because attention returns fixed-size context, not a sequence
        return None

    def call(self, x, mask=None):
        """
        x: (batch, steps, features)
        returns: (context_vector, attention_weights)
        """
        features_dim = self.features_dim
        # step_dim is dynamic
        step_dim = tf.shape(x)[1]

        # compute e = tanh(x . W + b)
        flat_x = tf.reshape(x, (-1, features_dim))  # (batch*steps, features)
        e = tf.reshape(tf.matmul(flat_x, tf.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # (batch, steps)
        if self.bias:
            e = e + self.b
        e = tf.tanh(e)

        # attention weights
        a = tf.exp(e)
        if mask is not None:
            a *= tf.cast(mask, tf.float32)
        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)
        a_expanded = tf.expand_dims(a, axis=-1)  # (batch, steps, 1)

        # context vector: weighted sum over steps
        c = tf.reduce_sum(a_expanded * x, axis=1)  # (batch, features)
        return c, a_expanded

    def compute_output_shape(self, input_shape):
        # returns shapes for (context_vector, attention_weights) but Keras expects single shape
        return (input_shape[0], self.features_dim)


# ---------------------------
# Model builder
# ---------------------------
def build_model(
    maxlen: int = SEQ_LEN,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """
    Build the model architecture:
      - Inputs: structure_onehot (maxlen x 12), seq_onehot (maxlen x 4), mfe scalar
      - For structure: small CNN -> two CNN branches -> stacked BiLSTM
      - For sequence: small CNN -> two CNN branches -> stacked BiLSTM
      - Apply Attention to both streams, concatenate with mfe, then dense layers.
    """
    # Inputs
    input_stru = L.Input(shape=(maxlen, 12), name="Input_stru")
    input_seq = L.Input(shape=(maxlen, 4), name="Input_seq")
    input_mfe = L.Input(shape=(1,), name="Input_mfe")

    # ---- structure branch ----
    s = L.Conv1D(32, kernel_size=1, padding="same", name="Conv1D_stru")(input_stru)
    s = L.BatchNormalization()(s)
    s = L.ReLU()(s)

    s_branch1 = L.Conv1D(64, kernel_size=7, padding="same", name="Conv1D_stru_feature2")(s)
    s_branch1 = L.BatchNormalization()(s_branch1)
    s_branch1 = L.ReLU()(s_branch1)

    s_branch2 = L.Conv1D(64, kernel_size=9, padding="same", name="Conv1D_stru_feature3")(s)
    s_branch2 = L.BatchNormalization()(s_branch2)
    s_branch2 = L.ReLU()(s_branch2)

    # stacked BiLSTM for each branch (return_sequences=True to allow Attention)
    lstm_s1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_stru2")(s_branch1)
    lstm_s1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_stru2_1")(lstm_s1)

    lstm_s2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_stru3")(s_branch2)
    lstm_s2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_stru3_1")(lstm_s2)

    merged_s = L.concatenate([lstm_s1, lstm_s2], axis=-1)
    merged_s = L.Conv1D(256, kernel_size=1, padding="same", name="Conv1D_stru_merged")(merged_s)
    merged_s = L.BatchNormalization()(merged_s)
    merged_s = L.ReLU()(merged_s)

    stru_out, attention_weight_stru = Attention(name="Attention_stru")(merged_s)
    stru_out = L.Dropout(0.3)(stru_out)

    # ---- sequence branch ----
    q = L.Conv1D(32, kernel_size=1, padding="same", name="Conv1D_seq")(input_seq)
    q = L.BatchNormalization()(q)
    q = L.ReLU()(q)

    q1 = L.Conv1D(64, kernel_size=3, padding="same", name="Conv1D_seq_feature1")(q)
    q1 = L.BatchNormalization()(q1)
    q1 = L.ReLU()(q1)

    q2 = L.Conv1D(64, kernel_size=5, padding="same", name="Conv1D_seq_feature2")(q)
    q2 = L.BatchNormalization()(q2)
    q2 = L.ReLU()(q2)

    lstm_q1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_seq")(q1)
    lstm_q1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_seq1")(lstm_q1)

    lstm_q2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_seq2")(q2)
    lstm_q2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_seq2_1")(lstm_q2)

    merged_q = L.concatenate([lstm_q1, lstm_q2], axis=-1)
    merged_q = L.Conv1D(256, kernel_size=1, padding="same", name="Conv1D_seq_merged")(merged_q)
    merged_q = L.BatchNormalization()(merged_q)
    merged_q = L.ReLU()(merged_q)

    seq_out, attention_weight_seq = Attention(name="Attention_seq")(merged_q)
    seq_out = L.Dropout(0.3)(seq_out)

    # ---- final layers ----
    x_flat = L.Concatenate()([stru_out, seq_out, input_mfe])
    x = L.Dense(256, activation="relu")(x_flat)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.2)(x)
    out = L.Dense(1, activation="linear", name="out")(x)

    model = tf.keras.Model(inputs=[input_stru, input_seq, input_mfe], outputs=out)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ---------------------------
# Metrics / Evaluation helpers
# ---------------------------
def r2_score_from_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute r-squared using scipy.linregress (same as original script)."""
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    return r_value ** 2


# ---------------------------
# Main training routine
# ---------------------------
def main():
    # ---------------------------
    # Load data
    # ---------------------------
    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep="\t")
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[:MAX_ROWS].copy()
    logger.info("Data shape after slicing: %s", df.shape)

    # split train/test consistent with original script:
    e_test = df.iloc[:20_000].copy()
    e_train = df.iloc[20_000:].copy()
    logger.info("Train / test split sizes: %d / %d", len(e_train), len(e_test))

    # ---------------------------
    # Preprocess inputs
    # ---------------------------
    logger.info("One-hot encoding train sequences (seq_len=%d)", SEQ_LEN)
    seq_e_train, stru_e_train = one_hot_encode(e_train, seqcol="sequence", strucol="structure_full", seq_len=SEQ_LEN)

    # The original script computed 'Norm_mfe' as MFE_full / 80, then MinMax scaled.
    # We'll compute Norm_mfe and then fit a MinMax scaler on training set only and transform both sets.
    e_train["Norm_mfe"] = e_train["MFE_full"] / 80.0
    e_test["Norm_mfe"] = e_test["MFE_full"] / 80.0

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    train_meta = e_train[["Norm_mfe"]].values
    test_meta = e_test[["Norm_mfe"]].values
    scaled_train_meta = mm_scaler.fit_transform(train_meta)
    scaled_test_meta = mm_scaler.transform(test_meta)

    # Scale target rl with StandardScaler fitted on training targets (important!)
    target_scaler = StandardScaler()
    e_train["scaled_rl"] = target_scaler.fit_transform(e_train["rl"].values.reshape(-1, 1)).flatten()
    # Note: we do NOT transform test `rl` here; we will inverse_transform predictions with target_scaler

    # ---------------------------
    # Model build
    # ---------------------------
    logger.info("Building model")
    model = build_model(maxlen=SEQ_LEN)
    model.summary(print_fn=lambda s: logger.info(s))

    # ---------------------------
    # Train/validation split and fit
    # ---------------------------
    # Use the precomputed one-hot encodings and scaled target
    X_train_seq, X_val_seq, X_train_stru, X_val_stru, X_train_meta, X_val_meta, y_train, y_val = train_test_split(
        seq_e_train, stru_e_train, scaled_train_meta, e_train["scaled_rl"].values, test_size=0.2, random_state=42
    )

    # Early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    logger.info("Starting training. Batch size=%d, epochs=%d", BATCH_SIZE, EPOCHS)
    history = model.fit(
        [X_train_stru, X_train_seq, X_train_meta],
        y_train,
        validation_data=([X_val_stru, X_val_seq, X_val_meta], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1,
    )

    # ---------------------------
    # Save model
    # ---------------------------
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    model.save(MODEL_OUT_PATH)
    logger.info("Saved trained model to %s", MODEL_OUT_PATH)

    # ---------------------------
    # Prepare test encodings (from earlier e_test / scaled_test_meta)
    # Note: one-hot for test must be done similarly — if you need the test seq/stru arrays,
    # create them here. In the original snippet they were not shown; we build them now.
    # ---------------------------
    logger.info("One-hot encoding test sequences (seq_len=%d)", SEQ_LEN)
    seq_e_test, stru_e_test = one_hot_encode(e_test, seqcol="sequence", strucol="structure_full", seq_len=SEQ_LEN)

    # ---------------------------
    # Predict on test and inverse-transform predictions
    # ---------------------------
    logger.info("Predicting on test set")
    preds_scaled = model.predict([stru_e_test, seq_e_test, scaled_test_meta]).reshape(-1)
    preds_orig = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

    # Attach predictions
    e_test = e_test.copy()
    e_test["pred"] = preds_orig

    # Evaluate
    r2 = r2_score_from_slope(e_test["rl"].values, e_test["pred"].values)
    spearman_corr, spearman_p = spearmanr(e_test["rl"].values, e_test["pred"].values)
    mse = np.mean((e_test["rl"].values - e_test["pred"].values) ** 2)
    rmse = np.sqrt(mse)

    logger.info("Evaluation results on test set:")
    logger.info("  r-squared (linear regression) = %.4f", r2)
    logger.info("  Spearman rho = %.4f (p=%.3e)", spearman_corr, spearman_p)
    logger.info("  MSE = %.6f, RMSE = %.6f", mse, rmse)

    # Print small table head for quick sanity check
    logger.info("Sample predictions (first 5 rows):")
    logger.info("\n%s", e_test[["rl", "pred"]].head())

    # Optionally return objects for further analysis if imported as a module
    return {
        "model": model,
        "history": history,
        "test_df": e_test,
        "target_scaler": target_scaler,
        "meta_scaler": mm_scaler,
    }


if __name__ == "__main__":
    main()
