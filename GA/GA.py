#!/usr/bin/env python3
"""
ga_optimize_5utr.py

Genetic Algorithm for optimizing 5'UTR sequences using two pretrained models:
 - model_TE: an endogenous TE predictor
 - model_MRL: an MRL (mean ribosome load) predictor

Workflow:
 1. Generate or load an initial population of UTR sequences.
 2. For each candidate, append CDS or EGFP tail, fold with RNAfold (parallel),
    extract structure and MFE, build padded inputs and one-hot encodings.
 3. Predict scores from both models, combine into a fitness score.
 4. Run a GA loop (selection, crossover, mutation, optionally inject user-provided FASTA seqs).
 5. Return ranked final population and a log DataFrame.

Notes:
 - Requires RNAfold in PATH (ViennaRNA package).
 - Requires joblib, Biopython (only if loading FASTA), tensorflow, sklearn, tqdm.
 - Adjust paths to models and scalers below or pass via CLI args.

Author: refactor for user
Date: 2025-10-31
"""
from __future__ import annotations
import argparse
import logging
import os
import random
import subprocess
import time
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio import SeqIO  # only needed if fasta input is used
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tqdm import trange

# ---------------------------
# Configuration (defaults)
# ---------------------------
DEFAULT_MODEL_TE = "~/models/endogenous/endogenesis_model_HKE293T.h5"
DEFAULT_MODEL_MRL = "~/synthetic/model_eGFP_25_100.h5"
DEFAULT_MFE_SCALER_TE = "~/norm_mfe_scaler.pkl"
DEFAULT_MFE_SCALER_MRL = "~/varylength_norm_mfe_scaler.pkl"

SEQ_LEN = 50  # length variable used elsewhere (not the padded length for models)
NUM_THREADS = 16
CDS30 = "ATGGAAGACGCCAAAAACATAAAGAAAGGC"  # tail used for TE model
EGFP = "ATGGGCGAATTAAGTAAGGGCGAGGAGCTG"  # tail used for MRL model
NUCLEOTIDES = ["A", "C", "G", "T"]

MIN_LEN = 25
MAX_LEN = 100

PAD_330 = 330
PAD_130 = 130

# ---------------------------
# Logging & seeds
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RNG_SEED = 3407
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------------------
# Attention class (kept for model loading)
# ---------------------------
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer


class Attention(Layer):
    """
    Small Attention layer used by the pretrained models.
    We keep this here so load_model(..., custom_objects={'Attention': Attention}) works.
    """
    def __init__(self, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.features_dim = 0
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name=f"{self.name}_W",
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight(shape=(1,), initializer="zeros", name=f"{self.name}_b", regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = K.shape(x)[1]
        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            e += self.b
        e = K.tanh(e)
        a = K.exp(e)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        c = K.sum(a * x, axis=1)
        return c, a

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


# ---------------------------
# Helper functions
# ---------------------------
def run_rnafold(sequence: str, timeout: float = 10.0) -> Tuple[Optional[str], Optional[float]]:
    """
    Call RNAfold --noPS --maxBPspan=30 on a single sequence.
    Return (structure_string, mfe) or (None, None) on failure.
    Uses subprocess.run with a timeout for safety.
    """
    try:
        proc = subprocess.run(
            ["RNAfold", "--noPS", "--maxBPspan=30"],
            input=sequence.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        out = proc.stdout.decode()
        # RNAfold prints two lines per sequence:
        # seq
        # structure ( ( ) ) ( -3.10 )
        lines = [ln for ln in out.splitlines() if ln.strip() != ""]
        if len(lines) >= 2 and "(" in lines[1]:
            # structure part could be: ".((((...)))) (-3.21)"
            parts = lines[1].strip().split()
            structure = parts[0]
            # extract last parentheses content as mfe if exists
            if "(" in lines[1] and ")" in lines[1]:
                try:
                    mfe_str = lines[1].split(" (")[-1].split(")")[0]
                    mfe = float(mfe_str)
                except Exception:
                    mfe = None
            else:
                mfe = None
            return structure, mfe
        else:
            return None, None
    except subprocess.TimeoutExpired:
        logger.warning("RNAfold timeout for sequence (len=%d)", len(sequence))
        return None, None
    except FileNotFoundError:
        logger.error("RNAfold not found in PATH. Install ViennaRNA or add to PATH.")
        raise
    except Exception as e:
        logger.exception("RNAfold failed: %s", e)
        return None, None


def pad_right_keep(seq: str, total_len: int, pad_char: str = "N") -> str:
    """Pad left with pad_char so that rightmost `total_len` bases are kept."""
    s = "" if seq is None else str(seq)
    if len(s) >= total_len:
        return s[-total_len:]
    return pad_char * (total_len - len(s)) + s


# one-hot dictionaries (kept as simple lists)
NUC_MAP = {"A": (1, 0, 0, 0), "C": (0, 1, 0, 0), "G": (0, 0, 1, 0), "T": (0, 0, 0, 1), "N": (0, 0, 0, 0)}
# build structure map (12 channels)
STRU_MAP = {}
idx = 0
for b in "ACGT":
    for s in "().":
        v = [0] * 12
        v[idx] = 1
        STRU_MAP[f"{b}{s}"] = v
        idx += 1
STRU_MAP["NN"] = [0] * 12


def one_hot_encode_batch(seqs: List[str], structs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple batch one-hot encoding:
      seqs -> (N, L, 4)
      structs -> (N, L, 12) using base+struct mapping
    Assumes seqs and structs are same length strings (already padded/truncated).
    """
    n = len(seqs)
    if n == 0:
        return np.zeros((0, 0, 4), dtype=np.float32), np.zeros((0, 0, 12), dtype=np.float32)
    L_seq = len(seqs[0])
    seq_out = np.zeros((n, L_seq, 4), dtype=np.float32)
    stru_out = np.zeros((n, L_seq, 12), dtype=np.float32)
    for i, (s, t) in enumerate(zip(seqs, structs)):
        for j, ch in enumerate(s):
            seq_out[i, j, :] = NUC_MAP.get(ch, (0, 0, 0, 0))
        for j, (bch, sch) in enumerate(zip(s, t)):
            stru_out[i, j, :] = STRU_MAP.get(f"{bch}{sch}", [0] * 12)
    return seq_out, stru_out


# ---------------------------
# Core scoring pipeline
# ---------------------------
class Predictor:
    """
    Holds loaded models and mfe scalers and exposes batch scoring method.
    """
    def __init__(self, model_te_path: str, model_mrl_path: str,
                 mfe_scaler_te_path: str, mfe_scaler_mrl_path: str,
                 n_threads: int = NUM_THREADS):
        self.custom_objects = {"Attention": Attention}
        logger.info("Loading TE model from %s", model_te_path)
        self.model_TE = load_model(model_te_path, custom_objects=self.custom_objects)
        logger.info("Loading MRL model from %s", model_mrl_path)
        self.model_MRL = load_model(model_mrl_path, custom_objects=self.custom_objects)

        logger.info("Loading MFE scalers")
        self.mfe_scaler_TE = joblib.load(mfe_scaler_te_path)
        self.mfe_scaler_MRL = joblib.load(mfe_scaler_mrl_path)

        self.n_threads = n_threads

    def batch_fold(self, sequences: List[str], timeout: float = 10.0) -> List[Tuple[Optional[str], Optional[float]]]:
        """
        Run RNAfold in parallel for a list of sequences. Returns list of (structure, mfe).
        We use joblib.Parallel to preserve original style.
        """
        results = Parallel(n_jobs=self.n_threads)(
            delayed(run_rnafold)(seq, timeout) for seq in sequences
        )
        return results

    def score_batch(self, candidate_utrs: List[str], alpha: float = 0.8, beta: float = 0.2) -> List[float]:
        """
        Given a list of UTR strings (not padded), compute combined fitness score:
          score = alpha * TE_pred + beta * MRL_pred

        Steps:
          - append CDS30 / EGFP tails
          - fold with RNAfold (parallel)
          - prepare padded sequences (330 / 130) and one-hot encode
          - scale MFE features and predict with the two models
        Returns a list of floats (same order as input); invalid entries get -1.0.
        """
        if len(candidate_utrs) == 0:
            return []

        # ensure trimmed to MAX_LEN at right (keep rightmost)
        trimmed = [utr[-MAX_LEN:] if len(utr) > MAX_LEN else utr for utr in candidate_utrs]

        # Build full sequences for TE and MRL models
        full_seqs_TE = [s + CDS30 for s in trimmed]
        full_seqs_MRL = [s + EGFP for s in trimmed]

        # Run RNAfold in parallel
        folds_TE = self.batch_fold(full_seqs_TE)
        folds_MRL = self.batch_fold(full_seqs_MRL)

        # validate and collect indices
        valid_idx = [i for i, (a, b) in enumerate(zip(folds_TE, folds_MRL)) if a[0] and b[0] and a[1] is not None and b[1] is not None]
        if len(valid_idx) == 0:
            logger.warning("No valid folds produced for this batch; returning -1 for all.")
            return [-1.0] * len(candidate_utrs)

        # prepare padded inputs according to original script behavior
        seqs_330 = [pad_right_keep(full_seqs_TE[i], PAD_330) for i in valid_idx]
        strus_330 = [pad_right_keep(folds_TE[i][0], PAD_330) for i in valid_idx]
        mfe_te = [folds_TE[i][1] / max(len(full_seqs_TE[i]), 1) for i in valid_idx]

        seqs_130 = [pad_right_keep(full_seqs_MRL[i], PAD_130) for i in valid_idx]
        strus_130 = [pad_right_keep(folds_MRL[i][0], PAD_130) for i in valid_idx]
        mfe_mrl = [folds_MRL[i][1] / max(len(full_seqs_MRL[i]), 1) for i in valid_idx]

        # one-hot encode
        seq_te_enc, stru_te_enc = one_hot_encode_batch(seqs_330, strus_330)
        seq_mrl_enc, stru_mrl_enc = one_hot_encode_batch(seqs_130, strus_130)

        # scale mfe
        mfe_scaled_te = self.mfe_scaler_TE.transform(np.array(mfe_te).reshape(-1, 1))
        mfe_scaled_mrl = self.mfe_scaler_MRL.transform(np.array(mfe_mrl).reshape(-1, 1))

        # predictions
        TE_preds = self.model_TE.predict([stru_te_enc, seq_te_enc, mfe_scaled_te], verbose=0).reshape(-1)
        MRL_preds = self.model_MRL.predict([stru_mrl_enc, seq_mrl_enc, mfe_scaled_mrl], verbose=0).reshape(-1)

        # populate scores list
        scores = [-1.0] * len(candidate_utrs)
        for out_idx, i in enumerate(valid_idx):
            # combine (allow TE_preds and MRL_preds to be numpy scalars)
            score_val = float(alpha * TE_preds[out_idx] + beta * MRL_preds[out_idx])
            scores[i] = score_val
        return scores


# ---------------------------
# GA helpers
# ---------------------------
def generate_random_sequence(length: Optional[int] = None) -> str:
    if length is None:
        length = random.randint(MIN_LEN, MAX_LEN)
    return "".join(random.choices(NUCLEOTIDES, k=length))


def remove_uAUG(seq: str) -> str:
    """
    Randomly mutate the 'ATG' start codons in the UTR region (except preserve Kozak region near the CDS).
    This function follows your original logic: keep last 6 positions (GCCACC region preserved),
    for other ATGs, mutate the 3rd position of 'ATG' to A/C/G randomly.
    """
    s = list(seq)
    for i in range(max(0, len(s) - 6 - 2)):
        if s[i:i + 3] == list("ATG"):
            # change the third base (position i+2) to non-G (choose A/C/G)
            s[i + 2] = random.choice(["A", "C", "G"])
    return "".join(s)


def enforce_kozak(seq: str) -> str:
    """
    Ensure the tail contains 'GCCACC' (Kozak) directly before CDS region analog.
    We preserve the last 6 chars accordingly; this matches original script behaviour.
    """
    if len(seq) < 6:
        return seq + "GCCACC"  # if too short, append
    return seq[:-6] + "GCCACC"


def crossover(parents: List[str], rate: float = 0.5) -> List[str]:
    """
    Single-point crossover that preserves Kozak enforcement on children.
    If population odd, last parent is carried over.
    """
    children: List[str] = []
    parents_copy = parents.copy()
    random.shuffle(parents_copy)
    if len(parents_copy) % 2 == 1:
        children.append(parents_copy[-1])
        parents_copy = parents_copy[:-1]
    for i in range(0, len(parents_copy), 2):
        s1, s2 = parents_copy[i], parents_copy[i + 1]
        if random.random() > rate:
            children += [enforce_kozak(s1), enforce_kozak(s2)]
        else:
            # choose crossover point - avoid very near ends (reserve 6 for Kozak)
            p = random.randint(1, max(1, min(len(s1), len(s2)) - 6))
            c1 = enforce_kozak(s1[:p] + s2[p:])
            c2 = enforce_kozak(s2[:p] + s1[p:])
            children += [c1, c2]
    return children


def mutate(seq: str, rate: float = 0.01) -> str:
    seq_list = list(seq)
    for i in range(len(seq_list) - 6):  # leave last 6 positions (Kozak) intact
        if random.random() < rate:
            seq_list[i] = random.choice(NUCLEOTIDES)
    return "".join(seq_list)


# ---------------------------
# Genetic algorithm main
# ---------------------------
def genetic_algorithm(predictor: Predictor,
                      pop_size: int = 100,
                      generations: int = 50,
                      num_parents: int = 20,
                      mutation_rate: float = 0.01,
                      crossover_rate: float = 0.5,
                      fasta_path: Optional[str] = None,
                      n_threads: int = NUM_THREADS) -> Tuple[List[Tuple[str, float]], pd.DataFrame]:
    """
    Run the GA: returns final ranked list (sequence, score) and a pandas log DataFrame.
    If fasta_path is provided, sequences from the FASTA are included in the initial population
    (and appended if needed to reach pop_size).
    """
    population: List[str] = []
    user_sequences: List[str] = []

    # load user FASTA if provided
    if fasta_path and os.path.exists(fasta_path):
        for rec in SeqIO.parse(fasta_path, "fasta"):
            seq = str(rec.seq).upper()
            population.append(seq)
            user_sequences.append(seq)
        logger.info("Loaded %d sequences from FASTA %s", len(user_sequences), fasta_path)

    # fill remaining population with random sequences
    while len(population) < pop_size:
        seq = enforce_kozak(remove_uAUG(generate_random_sequence()))
        population.append(seq)

    log_records = []
    # GA loop with tqdm/trange progress bar
    for gen in (pbar := trange(generations, desc="GA", ncols=100)):
        t0 = time.time()
        scores = predictor.score_batch(population)
        ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)

        # bookkeeping for progress bar
        best_seq, best_score = ranked[0]
        avg_score = sum(scores) / len(scores)
        pbar.set_postfix({"Gen": gen, "Best": f"{best_score:.4f}", "Avg": f"{avg_score:.4f}", "Time(s)": f"{time.time() - t0:.2f}"})

        # Selection
        parents = [s for s, _ in ranked[:num_parents]]

        # maintain user-provided sequences in generation 0 by replacing from end
        if gen == 0 and user_sequences:
            replace_idx = -1
            for useq in user_sequences:
                if useq not in parents:
                    parents[replace_idx] = useq
                    replace_idx -= 1

        # Produce offspring
        offspring = crossover(parents, rate=crossover_rate)
        offspring = [mutate(s, rate=mutation_rate) for s in offspring]
        offspring = [remove_uAUG(s) for s in offspring]

        # New population: parents + offspring (may exceed pop_size if parents+offspring > pop_size)
        population = parents + offspring
        # trim/populate to exactly pop_size
        if len(population) > pop_size:
            population = population[:pop_size]
        else:
            while len(population) < pop_size:
                population.append(enforce_kozak(remove_uAUG(generate_random_sequence())))

        # compute metrics for log
        log_records.append({
            "Generation": gen,
            "Best Score": float(best_score),
            "Average Score": float(avg_score),
            "Best Sequence": best_seq,
            "Time (s)": float(time.time() - t0),
        })

    # final evaluation and ranking
    final_scores = predictor.score_batch(population)
    final_ranked = sorted(list(zip(population, final_scores)), key=lambda x: x[1], reverse=True)
    log_df = pd.DataFrame(log_records)
    return final_ranked, log_df


# ---------------------------
# CLI entrypoint
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GA optimize 5'UTRs using pretrained TE and MRL models.")
    p.add_argument("--model_te", default=DEFAULT_MODEL_TE, help="Path to TE model (.h5)")
    p.add_argument("--model_mrl", default=DEFAULT_MODEL_MRL, help="Path to MRL model (.h5)")
    p.add_argument("--mfe_scaler_te", default=DEFAULT_MFE_SCALER_TE, help="Path to mfe scaler for TE (joblib .pkl)")
    p.add_argument("--mfe_scaler_mrl", default=DEFAULT_MFE_SCALER_MRL, help="Path to mfe scaler for MRL (joblib .pkl)")
    p.add_argument("--pop_size", type=int, default=100)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--num_parents", type=int, default=20)
    p.add_argument("--mutation_rate", type=float, default=0.01)
    p.add_argument("--crossover_rate", type=float, default=0.5)
    p.add_argument("--fasta", default=None, help="Optional user FASTA to seed initial population")
    p.add_argument("--threads", type=int, default=NUM_THREADS)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("Starting GA optimization with args: %s", args)

    predictor = Predictor(
        model_te_path=args.model_te,
        model_mrl_path=args.model_mrl,
        mfe_scaler_te_path=args.mfe_scaler_te,
        mfe_scaler_mrl_path=args.mfe_scaler_mrl,
        n_threads=args.threads,
    )

    final_ranked, log_df = genetic_algorithm(
        predictor=predictor,
        pop_size=args.pop_size,
        generations=args.generations,
        num_parents=args.num_parents,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        fasta_path=args.fasta,
        n_threads=args.threads,
    )

    # Save top results and log
    out_prefix = "ga_results"
    pd.DataFrame(final_ranked, columns=["sequence", "score"]).to_csv(f"{out_prefix}_final_ranked.csv", index=False)
    log_df.to_csv(f"{out_prefix}_log.csv", index=False)
    logger.info("Saved results to %s_final_ranked.csv and %s_log.csv", out_prefix, out_prefix)
    logger.info("Top 5 sequences:\n%s", pd.DataFrame(final_ranked[:5], columns=["sequence", "score"]).to_string(index=False))


if __name__ == "__main__":
    main()
