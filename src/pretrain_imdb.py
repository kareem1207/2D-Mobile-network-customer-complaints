"""Pre-train SimpleRNN / LSTM / GRU sentiment models on the English IMDB dataset (Keras).

Used for the 'start from the English IMDB model' training strategy. Weights are saved to
artifacts/ (git-ignored; regenerate with this script, ~10 min on 4 CPU cores).

Run:  python src/pretrain_imdb.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

NUM_WORDS, MAXLEN, EPOCHS = 10_000, 200, 4
ART = ROOT / "artifacts"


def load_imdb():
    (xtr, ytr), (xte, yte) = tf.keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    pad = lambda xs: tf.keras.preprocessing.sequence.pad_sequences(xs, maxlen=MAXLEN, padding="post", truncating="pre")
    return pad(xtr), ytr, pad(xte), yte


def imdb_word_index():
    wi = tf.keras.datasets.imdb.get_word_index()
    # keras offsets real words by 3 (0 pad, 1 start, 2 OOV, 3 unused)
    return {w: i + 3 for w, i in wi.items() if i + 3 < NUM_WORDS}


def main():
    ART.mkdir(exist_ok=True)
    xtr, ytr, xte, yte = load_imdb()
    log = {}
    for name, cls in LAYERS.items():
        set_seed(42)
        m = tf.keras.Sequential([
            tf.keras.layers.Embedding(NUM_WORDS, EMB_DIM, mask_zero=True, name="emb"),
            cls(UNITS),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        m.compile(tf.keras.optimizers.Adam(2e-3), "binary_crossentropy", metrics=["accuracy"])
        m.fit(xtr, ytr, validation_split=0.1, epochs=EPOCHS, batch_size=64, verbose=2)
        loss, acc = m.evaluate(xte, yte, verbose=0, batch_size=256)
        print(f"{name}: IMDB test acc {acc:.4f}")
        log[name] = {"imdb_test_acc": float(acc)}
        np.savez(ART / f"imdb_{name}.npz", emb=m.layers[0].get_weights()[0],
                 **{f"rnn{i}": w for i, w in enumerate(m.layers[1].get_weights())})
    json.dump(log, open(ROOT / "results" / "imdb_pretrain.json", "w"), indent=2)


if __name__ == "__main__":
    main()
