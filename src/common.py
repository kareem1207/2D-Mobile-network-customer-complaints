"""Shared utilities: data loading, tokenisation, models, metrics, timing."""
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "code_mixed_complaints.csv"
LABELS = ["negative", "positive", "mixed"]
L2I = {l: i for i, l in enumerate(LABELS)}
EMB_DIM, UNITS = 64, 64
LAYERS = {
    "SimpleRNN": tf.keras.layers.SimpleRNN,
    "LSTM": tf.keras.layers.LSTM,
    "GRU": tf.keras.layers.GRU,
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def load_data():
    df = pd.read_csv(DATA)
    df["tokens"] = df.text.map(tokenize)
    df["y"] = df.label.map(L2I)
    return df


def build_vocab(token_lists, max_size=8000, min_freq=1):
    """Vocabulary from TRAIN only. 0 = padding, 1 = OOV."""
    cnt = Counter(t for toks in token_lists for t in toks)
    words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(token_lists, vocab, maxlen, truncate="pre"):
    """truncate='pre' keeps the LAST maxlen tokens (the real complaint is at the end);
    'post' keeps the first maxlen tokens. Post-padding + masking."""
    X = np.zeros((len(token_lists), maxlen), dtype="int32")
    for i, toks in enumerate(token_lists):
        ids = [vocab.get(t, 1) for t in toks]
        ids = ids[-maxlen:] if truncate == "pre" else ids[:maxlen]
        X[i, :len(ids)] = ids
    return X


def build_model(layer, vocab_size, n_classes=3, bidirectional=False, dropout=0.3, lr=2e-3):
    rnn = LAYERS[layer](UNITS)
    if bidirectional:
        rnn = tf.keras.layers.Bidirectional(rnn)
    m = tf.keras.Sequential([
        tf.keras.Input((None,), dtype="int32"),
        tf.keras.layers.Embedding(vocab_size, EMB_DIM, mask_zero=True, name="emb"),
        rnn,
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_classes, activation="softmax", name="head"),
    ])
    m.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def fit(model, Xtr, ytr, Xva, yva, epochs=40, patience=5, batch=32, verbose=0):
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch, callbacks=cb, verbose=verbose)
    return h


def predict(model, X, batch=256):
    return np.argmax(model.predict(X, batch_size=batch, verbose=0), axis=1)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)


def per_class_f1(y, p):
    return f1_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)


def time_10k(model, X, repeat_to=10_000, batch=256):
    """Seconds to predict 10,000 reviews on CPU (test set tiled up to 10k). One warm-up call first."""
    reps = int(np.ceil(repeat_to / len(X)))
    big = np.tile(X, (reps, 1))[:repeat_to]
    model.predict(big[:batch], batch_size=batch, verbose=0)
    t0 = time.perf_counter()
    model.predict(big, batch_size=batch, verbose=0)
    return time.perf_counter() - t0


def report(y, p):
    return classification_report(y, p, target_names=LABELS, digits=3, zero_division=0)


