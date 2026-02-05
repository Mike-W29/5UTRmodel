#!/usr/bin/env python3
"""
HEK293T.py

Train a sequence+structure model (CNN + BiLSTM + Attention) on endogenous HEK293T data.
This script:
 - loads expression metadata and UTR sequence/structure annotations,
 - filters by expression thresholds,
 - merges sequence features with expression table,
 - constructs train/test splits,
 - pads/truncates UTR sequences to fixed length (330) and one-hot encodes them,
 - builds a two-stream model (structure stream + sequence stream) with attention,
 - fits scalers only on the training set and reuses them for test predictions,
 - logs training progress and evaluation metrics.

Save as: HEK293T.py
Author: Mike Wang 
Date: 2025-10-31
"""
from typing import Tuple, Dict, Any
import logging
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Configuration / constants
# ---------------------------
DATA_COUNTS_PATH = "~/data/endogenous/df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt"
TRAIN_SEQ_PATH = "~/data/endogenous/Homo_trian_seq_feature_table_maxBPspan30.txt"
MODEL_OUT_PATH = "~/models/endogenous/endogenesis_model_HKE293T.h5"

RANDOM_STATE = 3407
TF_SEED = RANDOM_STATE
NP_SEED = RANDOM_STATE
PY_RANDOM_SEED = RANDOM_STATE

MIN_RNA_RPKM = 1.0
MIN_RIBO_RPKM = 1.0

MAX_LEN = 330  # fixed input length (pad/truncate to keep rightmost bases)
BATCH_SIZE = 32
EPOCHS = 999
TEST_SPLIT_RATIO = 0.10  # keep 10% as test (you originally used random split 0.9 train)

# ---------------------------
# Logging & reproducibility
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

tf.random.set_seed(TF_SEED)
np.random.seed(NP_SEED)
random.seed(PY_RANDOM_SEED)


# ---------------------------
# Utilities
# ---------------------------
def pad_keep_right(seq: str, total_len: int = MAX_LEN, pad_char: str = "N") -> str:
    """Pad on the left with pad_char so that the rightmost `total_len` bases are kept."""
    if seq is None:
        seq = ""
    s = str(seq)
    if len(s) >= total_len:
        return s[-total_len:]
    return pad_char * (total_len - len(s)) + s


def one_hot_encode(df: pd.DataFrame, seqcol: str, strucol: str, seq_len: int = MAX_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust one-hot encoding:
      - seq_onehot: (N, seq_len, 4) with order [A, C, G, T]
      - stru_onehot: (N, seq_len, 12) mapping base+structure combinations (A(, A), A., ...)

    Unknown/padding -> zero vector.
    Assumes df[seqcol] and df[strucol] contain padded strings of length >= seq_len.
    """
    nuc_map = {"A": (1, 0, 0, 0), "C": (0, 1, 0, 0), "G": (0, 0, 1, 0), "T": (0, 0, 0, 1), "U": (0, 0, 0, 1), "N": (0, 0, 0, 0)}
    bases = ["A", "C", "G", "T"]
    struct_symbols = ["(", ")", "."]
    stru_map = {}
    idx = 0
    for b in bases:
        for s in struct_symbols:
            v = [0] * 12
            v[idx] = 1
            stru_map[f"{b}{s}"] = v
            idx += 1
    fallback_12 = [0] * 12

    N = len(df)
    seq_feat = np.zeros((N, seq_len, 4), dtype=np.float32)
    stru_feat = np.zeros((N, seq_len, 12), dtype=np.float32)

    seq_series = df[seqcol].fillna("").astype(str).str.upper()
    stru_series = df[strucol].fillna("").astype(str).str.upper()

    for i, (sseq, sstru) in enumerate(zip(seq_series, stru_series)):
        if len(sseq) < seq_len:
            sseq = pad_keep_right(sseq, seq_len)
        if len(sstru) < seq_len:
            sstru = pad_keep_right(sstru, seq_len)
        sseq = sseq[:seq_len]
        sstru = sstru[:seq_len]
        for j, ch in enumerate(sseq):
            seq_feat[i, j, :] = nuc_map.get(ch, (0, 0, 0, 0))
        for j, (bch, sch) in enumerate(zip(sseq, sstru)):
            stru_feat[i, j, :] = stru_map.get(f"{bch}{sch}", fallback_12)

    return seq_feat, stru_feat


# ---------------------------
# Custom Attention Layer
# ---------------------------
class Attention(Layer):
    """Attention layer returning (context_vector, attention_weights). Mask supported.
       Ref in #https://zhuanlan.zhihu.com/p/97525394
    """
    def __init__(self, bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        self.bias = bias

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name=f"{self.name}_W")
        if self.bias:
            self.b = self.add_weight(shape=(1,), initializer="zeros", name=f"{self.name}_b")
        else:
            self.b = None
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (batch, steps, features)
        features = tf.shape(x)[-1]
        flat_x = tf.reshape(x, (-1, features))
        e = tf.reshape(tf.matmul(flat_x, tf.reshape(self.W, (features, 1))), (-1, tf.shape(x)[1]))
        if self.bias:
            e = e + self.b
        e = tf.tanh(e)
        a = tf.exp(e)
        if mask is not None:
            a = a * tf.cast(mask, tf.float32)
        denom = tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon()
        a = a / denom
        a_exp = tf.expand_dims(a, axis=-1)
        context = tf.reduce_sum(a_exp * x, axis=1)
        return context, a_exp


# ---------------------------
# Model builder
# ---------------------------
def build_model_with_attention(maxlen: int = MAX_LEN, hidden_dim: int = 64, dropout: float = 0.3) -> tf.keras.Model:
    """
    Build the two-stream model:
      - input_stru: (maxlen, 12)
      - input_seq: (maxlen, 4)
      - input_mfe: (1,)
      - returns scalar continuous output (linear)
    """
    input_stru = L.Input(shape=(maxlen, 12), name="Input_stru")
    input_seq = L.Input(shape=(maxlen, 4), name="Input_seq")
    input_mfe = L.Input(shape=(1,), name="Input_mfe")

    # boolean mask from sequence channels (True where any base exists)
    mask_bool = L.Lambda(lambda x: tf.reduce_any(tf.not_equal(x, 0), axis=-1), name="make_mask")(input_seq)
    mask_exp = L.Lambda(lambda m: tf.expand_dims(tf.cast(m, tf.float32), axis=-1), name="mask_expand")(mask_bool)

    # Structure branch
    s = L.Conv1D(32, 1, padding="same", name="Conv1D_stru")(input_stru)
    s = s * mask_exp
    s = L.BatchNormalization()(s)
    s = L.ReLU()(s)

    s1 = L.Conv1D(64, 7, padding="same", name="Conv1D_stru_f1")(s)
    s1 = s1 * mask_exp
    s1 = L.BatchNormalization()(s1)
    s1 = L.ReLU()(s1)

    s2 = L.Conv1D(64, 9, padding="same", name="Conv1D_stru_f2")(s)
    s2 = s2 * mask_exp
    s2 = L.BatchNormalization()(s2)
    s2 = L.ReLU()(s2)

    lstm_s1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_s1")(s1)
    lstm_s1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_s1_2")(lstm_s1)

    lstm_s2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_s2")(s2)
    lstm_s2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_s2_2")(lstm_s2)

    merged_s = L.concatenate([lstm_s1, lstm_s2], axis=-1)
    merged_s = L.Conv1D(256, 1, padding="same", name="Conv1D_stru_merged")(merged_s)
    merged_s = merged_s * mask_exp
    merged_s = L.BatchNormalization()(merged_s)
    merged_s = L.ReLU()(merged_s)

    stru_out, stru_att = Attention(name="Attention_stru")(merged_s, mask=mask_bool)
    stru_out = L.Dropout(0.3)(stru_out)

    # Sequence branch
    q = L.Conv1D(32, 1, padding="same", name="Conv1D_seq")(input_seq)
    q = q * mask_exp
    q = L.BatchNormalization()(q)
    q = L.ReLU()(q)

    q1 = L.Conv1D(64, 3, padding="same", name="Conv1D_seq_f1")(q)
    q1 = q1 * mask_exp
    q1 = L.BatchNormalization()(q1)
    q1 = L.ReLU()(q1)

    q2 = L.Conv1D(64, 5, padding="same", name="Conv1D_seq_f2")(q)
    q2 = q2 * mask_exp
    q2 = L.BatchNormalization()(q2)
    q2 = L.ReLU()(q2)

    lstm_q1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_q1")(q1)
    lstm_q1 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_q1_2")(lstm_q1)

    lstm_q2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_q2")(q2)
    lstm_q2 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"), name="LSTM_q2_2")(lstm_q2)

    merged_q = L.concatenate([lstm_q1, lstm_q2], axis=-1)
    merged_q = L.Conv1D(256, 1, padding="same", name="Conv1D_seq_merged")(merged_q)
    merged_q = merged_q * mask_exp
    merged_q = L.BatchNormalization()(merged_q)
    merged_q = L.ReLU()(merged_q)

    seq_out, seq_att = Attention(name="Attention_seq")(merged_q, mask=mask_bool)
    seq_out = L.Dropout(0.3)(seq_out)

    # Final head
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
# Main pipeline
# ---------------------------
def main() -> Dict[str, Any]:
    logger.info("Loading expression counts & annotations from %s", DATA_COUNTS_PATH)
    df_counts = pd.read_csv(DATA_COUNTS_PATH, sep=" ", index_col=0)
    # Filter low expressed transcripts
    df_counts = df_counts[(df_counts["rpkm_rnaseq"] > MIN_RNA_RPKM) & (df_counts["rpkm_riboseq"] > MIN_RIBO_RPKM)]
    logger.info("After expression filter: %d rows remain", len(df_counts))

    logger.info("Loading sequence feature table from %s", TRAIN_SEQ_PATH)
    seq_table = pd.read_csv(TRAIN_SEQ_PATH, sep="\t")
    seq_table["len"] = seq_table["utr"].astype(str).apply(len)

    # Keep only transcripts present in both
    seq_filtered = seq_table[seq_table["Transcript_ID"].isin(df_counts["ensembl_tx_id"])]
    df_filtered = df_counts[df_counts["ensembl_tx_id"].isin(seq_filtered["Transcript_ID"])]

    # Merge on transcript id
    df_merged = pd.merge(
        df_filtered,
        seq_filtered[["Transcript_ID", "utr", "Sequence", "MFE_UTR", "structure_UTR", "MFE_full", "structure_full"]],
        left_on="ensembl_tx_id",
        right_on="Transcript_ID",
        how="left",
    )

    # Compute log TE (target) and filter by UTR length range
    df_merged["log_te"] = np.log(df_merged["te"])
    df_merged["len"] = df_merged["utr"].astype(str).apply(len)
    df_merged = df_merged[(df_merged["len"] >= 25) & (df_merged["len"] <= MAX_LEN)]
    logger.info("After merging & length filter: %d rows", len(df_merged))

    # For genes with multiple isoforms, keep one representative (max TE as in original script)
    idx = df_merged.groupby("Sequence")["te"].idxmax()
    df_merged = df_merged.loc[idx].reset_index(drop=True)
    logger.info("After isoform deduplication: %d rows", len(df_merged))

    # Train/test split (randomized)
    indices = df_merged.index.to_list()
    random.Random(RANDOM_STATE).shuffle(indices)
    split_at = int(len(indices) * (1.0 - TEST_SPLIT_RATIO))
    train_idx, test_idx = indices[:split_at], indices[split_at:]
    e_train = df_merged.loc[train_idx].reset_index(drop=True)
    e_test = df_merged.loc[test_idx].reset_index(drop=True)
    logger.info("Train / Test sizes: %d / %d", len(e_train), len(e_test))

    # Create padded sequence/structure strings (keep rightmost MAX_LEN chars)
    e_train["seq330"] = e_train["Sequence"].astype(str).apply(lambda s: pad_keep_right(s, MAX_LEN))
    e_train["stru330"] = e_train["structure_full"].astype(str).apply(lambda s: pad_keep_right(s, MAX_LEN))
    e_test["seq330"] = e_test["Sequence"].astype(str).apply(lambda s: pad_keep_right(s, MAX_LEN))
    e_test["stru330"] = e_test["structure_full"].astype(str).apply(lambda s: pad_keep_right(s, MAX_LEN))

    # Encode
    logger.info("One-hot encoding train set (N=%d)", len(e_train))
    train_seq_onehot, train_stru_onehot = one_hot_encode(e_train, seqcol="seq330", strucol="stru330", seq_len=MAX_LEN)

    # Metadata: normalized MFE per length
    e_train["Norm_mfe"] = e_train["MFE_full"] / e_train["len"]
    e_test["Norm_mfe"] = e_test["MFE_full"] / e_test["len"]

    meta_cols = ["Norm_mfe"]
    mm = MinMaxScaler(feature_range=(0, 1))
    train_meta = mm.fit_transform(e_train[meta_cols].values)
    test_meta = mm.transform(e_test[meta_cols].values) if len(e_test) > 0 else np.zeros((0, len(meta_cols)))

    # Scale target on training set
    target_scaler = StandardScaler()
    e_train["scaled_log_te"] = target_scaler.fit_transform(e_train["log_te"].values.reshape(-1, 1)).flatten()

    # Build model
    model = build_model_with_attention(maxlen=MAX_LEN)
    model.summary(print_fn=lambda s: logger.info(s))

    # Train/validation split on training set
    X_seq_train, X_seq_val, X_stru_train, X_stru_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
        train_seq_onehot, train_stru_onehot, train_meta, e_train["scaled_log_te"].values, test_size=0.2, random_state=42
    )

    # Fit model
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
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
    logger.info("Saved trained model to %s", MODEL_OUT_PATH)

    # Evaluate on test set (if non-empty)
    if len(e_test) > 0:
        logger.info("Encoding and predicting on test set (N=%d)", len(e_test))
        test_seq_onehot, test_stru_onehot = one_hot_encode(e_test, seqcol="seq330", strucol="stru330", seq_len=MAX_LEN)
        preds_scaled = model.predict([test_stru_onehot, test_seq_onehot, test_meta]).reshape(-1)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        e_test = e_test.copy()
        e_test["pred_log_te"] = preds

        # Metrics
        from scipy.stats import linregress, spearmanr
        slope, intercept, r_value, p_value, std_err = linregress(e_test["log_te"].values, e_test["pred_log_te"].values)
        r2 = r_value ** 2
        rho, pval = spearmanr(e_test["log_te"].values, e_test["pred_log_te"].values)
        mse = ((e_test["log_te"].values - e_test["pred_log_te"].values) ** 2).mean()
        rmse = np.sqrt(mse)

        logger.info("Test metrics: r2=%.4f, spearman rho=%.4f (p=%.3e), MSE=%.6f, RMSE=%.6f", r2, rho, pval, mse, rmse)
    else:
        logger.warning("Test set empty: skipping evaluation")

    # Return useful objects for downstream use
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
