#!/usr/bin/env python3
"""
train_egfp_varying_length.py

Train a deep model (CNN + BiLSTM + Attention) on eGFP sequences of varying lengths (25-100 nt),
padded/truncated to fixed length (130) for modeling.

Key features:
 - Robust one-hot encoding for sequence and structure
 - Custom Attention layer returning context vector and attention weights
 - Mask creation inside model (Lambda) and passed to Attention to ignore padded positions
 - Train/test split sampled per-length for balanced evaluation set
 - Target scaling fitted on training set only (no leakage)
 - Logging and modular structure for easy GitHub publishing

Author: Mike Wang 
Date: 2025-10-31
"""
from typing import Tuple, Dict, Any
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Configuration
# ---------------------------
DATA_PATH = "~/data/GSM4084997_varying_length_25to100_structure_feature_table_maxBPspan_30.txt"
MODEL_OUT_PATH = "~/models/synthetic/model_eGFP_25_100.h5"
RANDOM_STATE = 3407
MAX_PAD_LEN = 130        # final fixed input length (as in your original pipeline)
SAMPLE_PER_LEN = 200     # number of examples per length to place into test set
BATCH_SIZE = 128
EPOCHS = 999

# ---------------------------
# Logging & reproducibility
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------
# Utilities
# ---------------------------
def build_test_set_by_length(df: pd.DataFrame, len_col: str = "len", sample_per_len: int = SAMPLE_PER_LEN) -> pd.DataFrame:
    """
    Build a balanced test set by sampling up to `sample_per_len` rows for each length.
    Assumes `df` contains column `len_col` (integer sequence length).
    """
    parts = []
    for length in sorted(df[len_col].unique()):
        subset = df[df[len_col] == length]
        if len(subset) == 0:
            continue
        parts.append(subset.iloc[:sample_per_len].copy())
    if parts:
        test_df = pd.concat(parts, ignore_index=True)
    else:
        test_df = pd.DataFrame(columns=df.columns)
    return test_df


def pad_right_or_left_to_fixed(seq: str, total_len: int = MAX_PAD_LEN, pad_char: str = "N", from_right: bool = True) -> str:
    """
    Pad a sequence with pad_char to `total_len`. If `from_right` True, pad on the left (i.e. keep rightmost bases)
    This mimics the original behaviour: new_seq = 130*'N' + seq; then take last 130 chars.
    """
    if seq is None:
        seq = ""
    seq = str(seq)
    if len(seq) >= total_len:
        # keep rightmost total_len characters
        return seq[-total_len:]
    pad_needed = total_len - len(seq)
    if from_right:
        return pad_char * pad_needed + seq
    else:
        return seq + pad_char * pad_needed


def one_hot_encode(df: pd.DataFrame, seqcol: str, strucol: str, seq_len: int = MAX_PAD_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-hot encode nucleotide sequences and joint (base+structure) categories.

    Returns:
      seq_onehot: shape (N, seq_len, 4), order [A, C, G, T]
      stru_onehot: shape (N, seq_len, 12), mapping combinations (A(, A), A., C(, ...)
    Unknowns/padding -> zero vectors.

    Notes:
      - This function is robust to short sequences (we assume df already contains padded seq strings).
    """
    nuc_map: Dict[str, tuple] = {
        "A": (1, 0, 0, 0),
        "C": (0, 1, 0, 0),
        "G": (0, 0, 1, 0),
        "T": (0, 0, 0, 1),
        "U": (0, 0, 0, 1),  # treat U as T
        "N": (0, 0, 0, 0),
    }

    bases = ["A", "C", "G", "T"]
    struct_symbols = ["(", ")", "."]
    stru_map: Dict[str, list] = {}
    idx = 0
    for b in bases:
        for s in struct_symbols:
            vec = [0] * 12
            vec[idx] = 1
            stru_map[f"{b}{s}"] = vec
            idx += 1
    fallback_12 = [0] * 12

    N = len(df)
    seq_out = np.zeros((N, seq_len, 4), dtype=np.float32)
    stru_out = np.zeros((N, seq_len, 12), dtype=np.float32)

    seq_series = df[seqcol].fillna("").astype(str).str.upper()
    stru_series = df[strucol].fillna("").astype(str).str.upper()

    for i, (sseq, sstru) in enumerate(zip(seq_series, stru_series)):
        # ensure both are exactly seq_len (assume padding already applied in DataFrame)
        if len(sseq) < seq_len:
            sseq = pad_right_or_left_to_fixed(sseq, seq_len, pad_char="N", from_right=True)
        if len(sstru) < seq_len:
            sstru = pad_right_or_left_to_fixed(sstru, seq_len, pad_char="N", from_right=True)

        # Fill sequence one-hot
        for j, ch in enumerate(sseq[:seq_len]):
            seq_out[i, j, :] = nuc_map.get(ch, (0, 0, 0, 0))

        # Fill joint structure one-hot
        for j, (bch, sch) in enumerate(zip(sseq[:seq_len], sstru[:seq_len])):
            key = f"{bch}{sch}"
            stru_out[i, j, :] = stru_map.get(key, fallback_12)

    return seq_out, stru_out


# ---------------------------
# Custom Attention Layer
# ---------------------------
class Attention(Layer):
    """
    Simple attention layer that returns (context_vector, attention_weights).
    Masking support: a boolean mask (shape [batch, steps]) can be passed; masked positions get zero attention.
    Ref in #https://zhuanlan.zhihu.com/p/97525394

    """
    def __init__(self, bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        self.bias = bias

    def build(self, input_shape):
        assert len(input_shape) == 3, "Attention expects a 3D tensor (batch, steps, features)"
        self.W = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name=f"{self.name}_W")
        if self.bias:
            self.b = self.add_weight(shape=(1,), initializer="zeros", name=f"{self.name}_b")
        else:
            self.b = None
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (batch, steps, features)
        features = tf.shape(x)[-1]
        # Compute e = tanh(x dot W + b)  -> (batch, steps)
        flat_x = tf.reshape(x, (-1, features))  # (batch*steps, features)
        e = tf.reshape(tf.matmul(flat_x, tf.reshape(self.W, (features, 1))), (-1, tf.shape(x)[1]))  # (batch, steps)
        if self.bias:
            e = e + self.b
        e = tf.tanh(e)

        a = tf.exp(e)
        if mask is not None:
            # mask is boolean; cast to float
            a = a * tf.cast(mask, dtype=tf.float32)
        denom = tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon()
        a = a / denom
        a_exp = tf.expand_dims(a, axis=-1)  # (batch, steps, 1)

        context = tf.reduce_sum(a_exp * x, axis=1)  # (batch, features)
        return context, a_exp

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])  # context vector shape


# ---------------------------
# Model builder
# ---------------------------
def build_model_with_attention(maxlen: int = MAX_PAD_LEN, hidden_dim: int = 64, dropout: float = 0.3) -> tf.keras.Model:
    """
    Build functional Keras model using:
      - inputs: structure_onehot (maxlen,12), seq_onehot (maxlen,4), scalar mfe
      - CNN -> BiLSTM -> Attention for both sequence and structure streams
      - concatenate context vectors + mfe -> dense head -> linear output
    """
    input_stru = L.Input(shape=(maxlen, 12), name="Input_stru")
    input_seq = L.Input(shape=(maxlen, 4), name="Input_seq")
    input_mfe = L.Input(shape=(1,), name="Input_mfe")

    # Build a boolean mask from sequence (True where any channel non-zero)
    # We use a Lambda layer so mask is part of the graph and usable downstream
    make_mask = L.Lambda(lambda x: tf.reduce_any(tf.not_equal(x, 0), axis=-1), name="make_mask")
    mask_bool = make_mask(input_seq)  # shape: (batch, maxlen)

    # Expand mask to (batch, maxlen, 1) when we want to multiply with conv outputs
    mask_expanded = L.Lambda(lambda m: tf.expand_dims(tf.cast(m, dtype=tf.float32), axis=-1), name="mask_expand")(mask_bool)

    # ---- Structure branch ----
    s = L.Conv1D(32, 1, padding="same", name="Conv1D_stru")(input_stru)
    s = s * mask_expanded
    s = L.BatchNormalization()(s)
    s = L.ReLU()(s)

    s1 = L.Conv1D(64, 7, padding="same", name="Conv1D_stru_f1")(s)
    s1 = s1 * mask_expanded
    s1 = L.BatchNormalization()(s1)
    s1 = L.ReLU()(s1)

    s2 = L.Conv1D(64, 9, padding="same", name="Conv1D_stru_f2")(s)
    s2 = s2 * mask_expanded
    s2 = L.BatchNormalization()(s2)
    s2 = L.ReLU()(s2)

    # BiLSTM stacks (Keras will use mask if layers receive it; we pass mask explicitly to Attention)
    lstm_s1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_s1")(s1)
    lstm_s1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_s1_2")(lstm_s1)

    lstm_s2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_s2")(s2)
    lstm_s2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_s2_2")(lstm_s2)

    merged_s = L.concatenate([lstm_s1, lstm_s2], axis=-1)
    merged_s = L.Conv1D(256, 1, padding="same", name="Conv1D_stru_merged")(merged_s)
    merged_s = merged_s * mask_expanded
    merged_s = L.BatchNormalization()(merged_s)
    merged_s = L.ReLU()(merged_s)

    stru_out, stru_att = Attention(name="Attention_stru")(merged_s, mask=mask_bool)
    stru_out = L.Dropout(0.3)(stru_out)

    # ---- Sequence branch ----
    q = L.Conv1D(32, 1, padding="same", name="Conv1D_seq")(input_seq)
    q = q * mask_expanded
    q = L.BatchNormalization()(q)
    q = L.ReLU()(q)

    q1 = L.Conv1D(64, 3, padding="same", name="Conv1D_seq_f1")(q)
    q1 = q1 * mask_expanded
    q1 = L.BatchNormalization()(q1)
    q1 = L.ReLU()(q1)

    q2 = L.Conv1D(64, 5, padding="same", name="Conv1D_seq_f2")(q)
    q2 = q2 * mask_expanded
    q2 = L.BatchNormalization()(q2)
    q2 = L.ReLU()(q2)

    lstm_q1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_q1")(q1)
    lstm_q1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_q1_2")(lstm_q1)

    lstm_q2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_q2")(q2)
    lstm_q2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_q2_2")(lstm_q2)

    merged_q = L.concatenate([lstm_q1, lstm_q2], axis=-1)
    merged_q = L.Conv1D(256, 1, padding="same", name="Conv1D_seq_merged")(merged_q)
    merged_q = merged_q * mask_expanded
    merged_q = L.BatchNormalization()(merged_q)
    merged_q = L.ReLU()(merged_q)

    seq_out, seq_att = Attention(name="Attention_seq")(merged_q, mask=mask_bool)
    seq_out = L.Dropout(0.3)(seq_out)

    # ---- Head ----
    x = L.Concatenate()([stru_out, seq_out, input_mfe])
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.2)(x)
    out = L.Dense(1, activation="linear", name="out")(x)

    model = tf.keras.Model(inputs=[input_stru, input_seq, input_mfe], outputs=out)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ---------------------------
# Evaluation helpers
# ---------------------------
def r2_from_linreg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    return r_value ** 2


# ---------------------------
# Main training routine
# ---------------------------
def main() -> Dict[str, Any]:
    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep="\t")
    # ensure sequence length col
    df["len"] = df["sequence"].astype(str).apply(len)

    # Build a balanced test set by sequence length (keep first SAMPLE_PER_LEN examples per length)
    e_test = build_test_set_by_length(df, len_col="len", sample_per_len=SAMPLE_PER_LEN)
    logger.info("Constructed test set with %d examples (sample_per_len=%d)", len(e_test), SAMPLE_PER_LEN)

    # Training set = all rows that are not in e_test (avoid overlap)
    # Use index-based difference for robustness
    if len(e_test):
        test_idx = set(e_test.index)
        e_train = df[~df.index.isin(test_idx)].copy()
    else:
        e_train = df.copy()

    # Create padded columns seq130 / stru130 (pad on left to keep rightmost bases)
    e_train["seq130"] = e_train["sequence"].astype(str).apply(lambda s: pad_right_or_left_to_fixed(s, total_len=MAX_PAD_LEN, pad_char="N", from_right=True))
    e_train["stru130"] = e_train["structure_full"].astype(str).apply(lambda s: pad_right_or_left_to_fixed(s, total_len=MAX_PAD_LEN, pad_char="N", from_right=True))
    e_test["seq130"] = e_test["sequence"].astype(str).apply(lambda s: pad_right_or_left_to_fixed(s, total_len=MAX_PAD_LEN, pad_char="N", from_right=True))
    e_test["stru130"] = e_test["structure_full"].astype(str).apply(lambda s: pad_right_or_left_to_fixed(s, total_len=MAX_PAD_LEN, pad_char="N", from_right=True))

    # One-hot encode training data
    logger.info("One-hot encoding training sequences (N=%d, len=%d)", len(e_train), MAX_PAD_LEN)
    train_seq_onehot, train_stru_onehot = one_hot_encode(e_train, seqcol="seq130", strucol="stru130", seq_len=MAX_PAD_LEN)

    # Compute normalized MFE per-sequence length (as in original code)
    e_train["Norm_mfe"] = e_train["MFE_full"] / e_train["len"]
    e_test["Norm_mfe"] = e_test["MFE_full"] / e_test["len"]

    # Fit MinMax scaler on training meta feature(s)
    meta_cols = ["Norm_mfe"]
    mm = MinMaxScaler(feature_range=(0, 1))
    train_meta = mm.fit_transform(e_train[meta_cols].values)
    test_meta = mm.transform(e_test[meta_cols].values) if len(e_test) > 0 else np.zeros((0, len(meta_cols)))

    # Scale target (rl) with StandardScaler fitted on training set only
    target_scaler = StandardScaler()
    e_train["scaled_rl"] = target_scaler.fit_transform(e_train["rl"].values.reshape(-1, 1)).flatten()

    # Build model
    model = build_model_with_attention(maxlen=MAX_PAD_LEN)
    model.summary(print_fn=lambda s: logger.info(s))

    # Train / val split (on training set)
    X_seq_train, X_seq_val, X_stru_train, X_stru_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
        train_seq_onehot, train_stru_onehot, train_meta, e_train["scaled_rl"].values, test_size=0.2, random_state=42
    )

    # Fit
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    logger.info("Start training: batch_size=%d, epochs=%d", BATCH_SIZE, EPOCHS)
    history = model.fit(
        [X_stru_train, X_seq_train, X_meta_train],
        y_train,
        validation_data=([X_stru_val, X_seq_val, X_meta_val], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Save model
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    model.save(MODEL_OUT_PATH)
    logger.info("Saved model to %s", MODEL_OUT_PATH)

    # Prepare test one-hot and predict
    if len(e_test) > 0:
        logger.info("Encoding and predicting on test set (N=%d)", len(e_test))
        test_seq_onehot, test_stru_onehot = one_hot_encode(e_test, seqcol="seq130", strucol="stru130", seq_len=MAX_PAD_LEN)
        preds_scaled = model.predict([test_stru_onehot, test_seq_onehot, test_meta]).reshape(-1)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        e_test = e_test.copy()
        e_test["pred"] = preds

        # Eval metrics
        r2 = r2_from_linreg(e_test["rl"].values, e_test["pred"].values)
        from scipy.stats import spearmanr
        rho, pval = spearmanr(e_test["rl"].values, e_test["pred"].values)
        mse = ((e_test["rl"].values - e_test["pred"].values) ** 2).mean()
        rmse = np.sqrt(mse)

        logger.info("Test evaluation: r2=%.4f, spearman rho=%.4f (p=%.3e), MSE=%.6f, RMSE=%.6f", r2, rho, pval, mse, rmse)
    else:
        logger.warning("Test set empty, skipped evaluation")
        e_test = e_test.copy()

    return {
        "model": model,
        "history": history,
        "train_df": e_train,
        "test_df": e_test,
        "meta_scaler": mm,
        "target_scaler": target_scaler,
    }


if __name__ == "__main__":
    main()
