"""All experiments for Task 2 (team 14, variant 2D). Run after pretrain_imdb.py:

    python src/experiments.py

Stages (each writes to results/ and figures/):
 1. sequence-length / truncation sweep   -> results/seqlen_sweep.csv, figures/seqlen_sweep.png
 2. 3 recurrent layers x 2 strategies    -> results/comparison_runs.csv, results/comparison_table.csv
 3. gradient-flow profile (vanishing)    -> results/gradient_profile.csv, figures/gradient_profile.png
 4. best model on the TEST split         -> results/best_*.{txt,csv,json}, figures/confusion_matrix.png

Model selection uses the validation split only; the test split is touched in stage 2/4 for reporting.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from pretrain_imdb import imdb_word_index, ART, NUM_WORDS  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, FIG = ROOT / "results", ROOT / "figures"
SEEDS = [0, 1, 2]


def splits(df, maxlen, truncate, vocab):
    out = {}
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        out[s] = (encode(d.tokens.tolist(), vocab, maxlen, truncate), d.y.values, d)
    return out


def transfer_init(model, layer, vocab):
    """Initialise from the English IMDB model: copy recurrent weights, and embedding rows for every
    word that also exists in the IMDB vocabulary. The classification head stays fresh (3 classes)."""
    w = np.load(ART / f"imdb_{layer}.npz")
    wi = imdb_word_index()
    emb = model.get_layer("emb").get_weights()[0]
    hit = 0
    for word, idx in vocab.items():
        j = wi.get(word)
        if j is not None:
            emb[idx] = w["emb"][j]
            hit += 1
    model.get_layer("emb").set_weights([emb])
    rnn = [l for l in model.layers if l.name.startswith(("simple_rnn", "lstm", "gru"))][0]
    rnn.set_weights([w[f"rnn{i}"] for i in range(len(rnn.get_weights()))])
    return hit / len(vocab)


def run_one(layer, strategy, seed, df, vocab, maxlen, truncate, bidirectional=False):
    set_seed(seed)
    S = splits(df, maxlen, truncate, vocab)
    m = build_model(layer, len(vocab) + 2, bidirectional=bidirectional, lr=2e-3 if strategy == "scratch" else 1e-3)
    coverage = float("nan")
    if strategy == "imdb":
        coverage = transfer_init(m, layer, vocab)
    h = fit(m, S["train"][0], S["train"][1], S["val"][0], S["val"][1])
    pv, pt = predict(m, S["val"][0]), predict(m, S["test"][0])
    yt = S["test"][1]
    r = {"layer": ("Bi" if bidirectional else "") + layer, "strategy": strategy, "seed": seed, "maxlen": maxlen,
         "truncate": truncate, "epochs": len(h.history["loss"]), "params": m.count_params(),
         "val_macro_f1": macro_f1(S["val"][1], pv), "test_macro_f1": macro_f1(yt, pt),
         "test_f1_mixed": per_class_f1(yt, pt)[2], "imdb_vocab_coverage": coverage,
         "sec_per_10k": time_10k(m, S["test"][0])}
    return r, m, pt, S


def stage1(df, vocab):
    rows = []
    cache = RES / "seqlen_sweep.csv"
    if cache.exists() and os.environ.get("REUSE_SWEEP"):  # skip the ~10 min sweep when re-running later stages
        d = pd.read_csv(cache)
    else:
        for maxlen in [16, 32, 64, 128]:
            for trunc in ["pre", "post"]:
                for seed in SEEDS:
                    r, *_ = run_one("LSTM", "scratch", seed, df, vocab, maxlen, trunc)
                    rows.append(r)
                    print("seqlen", maxlen, trunc, seed, round(r["val_macro_f1"], 3), flush=True)
        d = pd.DataFrame(rows)
        d.to_csv(cache, index=False)
    g = d.groupby(["truncate", "maxlen"]).agg(val_f1=("val_macro_f1", "mean"), val_sd=("val_macro_f1", "std"),
                                              sec_10k=("sec_per_10k", "mean")).reset_index()
    g.to_csv(RES / "seqlen_sweep_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for t, c in g.groupby("truncate"):
        ax.errorbar(c.maxlen, c.val_f1, yerr=c.val_sd, marker="o", capsize=3,
                    label="keep last tokens" if t == "pre" else "keep first tokens")
    ax.set_xscale("log", base=2); ax.set_xlabel("max sequence length (tokens)"); ax.set_ylabel("validation macro-F1")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "seqlen_sweep.png", dpi=160); plt.close(fig)
    pre = g[g["truncate"] == "pre"]
    maxlen = int(pre.loc[pre.val_f1.idxmax(), "maxlen"])  # best mean validation macro-F1
    print("chosen maxlen", maxlen)
    return maxlen


def bucket(n):
    return "short(<=15)" if n <= 15 else ("medium(16-40)" if n <= 40 else "long(>40)")


def stage2(df, vocab, maxlen):
    rows, preds = [], {}
    configs = [(l, s, False) for l in LAYERS for s in ["scratch", "imdb"]] + \
              [("LSTM", "scratch", True), ("GRU", "scratch", True)]
    for layer, strategy, bi in configs:
        for seed in SEEDS:
            r, m, pt, S = run_one(layer, strategy, seed, df, vocab, maxlen, "pre", bidirectional=bi)
            d = S["test"][2].copy()
            d["pred"] = pt
            for b, g in d.assign(b=d.n_tokens.map(bucket)).groupby("b"):
                r[f"f1_{b}"] = macro_f1(g.y.values, g.pred.values)
            rows.append(r)
            preds[(r["layer"], strategy, seed)] = (pt, m)
            print(r["layer"], strategy, seed, "val", round(r["val_macro_f1"], 3), "test", round(r["test_macro_f1"], 3), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RES / "comparison_runs.csv", index=False)
    agg = {c: ["mean", "std"] for c in ["val_macro_f1", "test_macro_f1", "test_f1_mixed", "sec_per_10k"] +
           [c for c in d.columns if c.startswith("f1_")]}
    agg.update(params=("first"), epochs="mean", imdb_vocab_coverage="mean")
    t = d.groupby(["layer", "strategy"]).agg(agg)
    t.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in t.columns]
    t = t.reset_index()
    t.to_csv(RES / "comparison_table.csv", index=False)
    return d, preds


def stage3(df, vocab):
    """Mean gradient norm of the loss w.r.t. the embedding of the token d steps before the end,
    on long reviews. Quantifies how much a token far from the end can influence the output."""
    maxlen = 128
    S = splits(df, maxlen, "pre", vocab)
    d = S["train"][2]
    sel = d.n_tokens.values >= 60
    X, y = S["train"][0][sel], S["train"][1][sel]
    lens = d.n_tokens.values[sel].clip(max=maxlen)
    prof = {}
    for layer in LAYERS:
        set_seed(0)
        m = build_model(layer, len(vocab) + 2)
        fit(m, S["train"][0], S["train"][1], S["val"][0], S["val"][1])
        emb, rnn, head = m.layers[0], m.layers[1], m.layers[3]
        Xt = tf.constant(X)
        mask = emb.compute_mask(Xt)
        with tf.GradientTape() as tape:
            e = emb(Xt)
            tape.watch(e)
            out = head(rnn(e, mask=mask, training=False))
            loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y, out))
        g = tf.norm(tape.gradient(loss, e), axis=-1).numpy()  # (N, maxlen)
        acc = np.zeros(60)
        for i in range(len(X)):
            acc += g[i, lens[i] - 1 - np.arange(60)]
        prof[layer] = acc / len(X)
    out = pd.DataFrame(prof); out.index.name = "tokens_before_end"
    out.to_csv(RES / "gradient_profile.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for l in LAYERS:
        ax.semilogy(out.index, out[l] / (out[l].iloc[0] + 1e-12), label=l)
    ax.set_xlabel("distance of token from end of review"); ax.set_ylabel("relative gradient norm (log)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "gradient_profile.png", dpi=160); plt.close(fig)


def stage4(df, runs, preds, vocab, maxlen):
    ok = runs[~runs.layer.str.startswith("Bi")]
    best = ok.groupby(["layer", "strategy"]).val_macro_f1.mean().idxmax()
    # best seed of the best config, chosen on validation
    sub = runs[(runs.layer == best[0]) & (runs.strategy == best[1])]
    seed = int(sub.loc[sub.val_macro_f1.idxmax(), "seed"])
    pt, model = preds[(best[0], best[1], seed)]
    test = df[df.split == "test"].copy()
    test["pred"] = pt
    y = test.y.values
    (RES / "best_report.txt").write_text(f"best config (by mean validation macro-F1): {best}, seed {seed}\n\n" + report(y, pt))
    cm = confusion_matrix(y, pt)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RES / "best_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=30); ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=160); plt.close(fig)
    test["correct"] = (test.pred == test.y).astype(int)
    for col in ["category", "phrase_pool", "spelling_noise", "label"]:
        test.groupby(col).agg(n=("correct", "size"), acc=("correct", "mean")).to_csv(RES / f"best_acc_by_{col}.csv")
    test["bucket"] = test.n_tokens.map(bucket)
    test.groupby("bucket").agg(n=("correct", "size"), acc=("correct", "mean")).to_csv(RES / "best_acc_by_bucket.csv")
    test["pred_label"] = test.pred.map(dict(enumerate(LABELS)))
    test[test.correct == 0][["id", "text", "label", "pred_label", "category", "spelling_noise", "phrase_pool"]].to_csv(RES / "best_errors.csv", index=False)
    summary = {"best": list(best), "seed": seed, "maxlen": maxlen,
               "test_macro_f1": macro_f1(y, pt), "f1_per_class": dict(zip(LABELS, per_class_f1(y, pt).tolist())),
               "cohen_kappa_model_vs_gold": float(cohen_kappa_score(y, pt)),
               "sec_per_10k_reviews": time_10k(model, encode(test.tokens.tolist(), vocab, maxlen, "pre")),
               "params": model.count_params()}
    json.dump(summary, open(RES / "best_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


def main():
    df = load_data()
    vocab = build_vocab(df[df.split == "train"].tokens)
    print("vocab size", len(vocab))
    maxlen = stage1(df, vocab)
    runs, preds = stage2(df, vocab, maxlen)
    stage3(df, vocab)
    stage4(df, runs, preds, vocab, maxlen)


if __name__ == "__main__":
    main()
