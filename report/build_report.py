"""Builds report/DL_T14_2D.pdf from results/*.csv (python report/build_report.py)."""
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

R = Path(__file__).resolve().parents[1]
res = lambda f: pd.read_csv(R / "results" / f)
B = ParagraphStyle("b", fontName="Times-Roman", fontSize=12, leading=13.6, spaceAfter=3, alignment=4)
H = ParagraphStyle("h", parent=B, fontName="Times-Bold", fontSize=12, spaceBefore=5, spaceAfter=2, alignment=0)
T = ParagraphStyle("t", parent=B, fontSize=12, leading=14, alignment=1, fontName="Times-Bold", spaceAfter=2)
C = ParagraphStyle("c", parent=B, fontSize=9.5, leading=11, alignment=0, spaceAfter=0)
CB = ParagraphStyle("cb", parent=C, fontName="Times-Bold")
p = lambda t, s=B: Paragraph(t, s)


def table(rows, widths, head=True):
    rows = [[Paragraph(str(c), CB if (head and i == 0) else C) for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6f2")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t


ct = res("comparison_table.csv")
order = ["SimpleRNN", "LSTM", "GRU", "BiLSTM", "BiGRU"]
rows = [["Layer", "Start", "Val F1", "Test macro-F1 (mean ± sd)", "Mixed F1", "s / 10k", "Params"]]
for l in order:
    for s in ["scratch", "imdb"]:
        r = ct[(ct.layer == l) & (ct.strategy == s)]
        if len(r):
            r = r.iloc[0]
            rows.append([l, "IMDB init" if s == "imdb" else "scratch", f"{r.val_macro_f1_mean:.3f}",
                         f"{r.test_macro_f1_mean:.3f} ± {r.test_macro_f1_std:.3f}", f"{r.test_f1_mixed_mean:.3f}",
                         f"{r.sec_per_10k_mean:.2f}", f"{int(r.params_first):,}"])
sw = res("seqlen_sweep_summary.csv")
sw_rows = [["Max length"] + [str(m) for m in [16, 32, 64, 128]]]
for t, name in [("pre", "keep last tokens"), ("post", "keep first tokens")]:
    d = sw[sw["truncate"] == t].sort_values("maxlen")
    sw_rows.append([f"Val F1, {name}"] + [f"{v:.3f}" for v in d.val_f1])
sw_rows.append(["Seconds / 10k reviews"] + [f"{v:.2f}" for v in sw[sw["truncate"] == "pre"].sort_values("maxlen").sec_10k])

img = lambda f, w: Image(str(R / "figures" / f), width=w * mm, height=w * mm * (3.2 / 5 if f != "confusion_matrix.png" else 3.2 / 3.6))
figs = Table([[img("seqlen_sweep.png", 58), img("gradient_profile.png", 58), img("confusion_matrix.png", 52)]], colWidths=[60 * mm, 60 * mm, 54 * mm])
figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))

S = []
S += [p("DL_T14_2D — Sentiment Analysis of Code-Mixed Telugu–English Mobile-Network Complaints", T),
      p("<b>Task 2, Team 14, variant 2D</b> · CM701PC / CM751PC Deep Learning, MGIT (Autonomous) · Code: github.com/kareem1207/2D-Mobile-network-customer-complaints", B)]
S.append(table([["Roll no.", "Role (department allotment)", "Contribution (to be confirmed by the team before submission)"],
                ["23261A6646", "Data lead", "Dataset generator, annotation guide, data card, error analysis"],
                ["23261A6655", "Model lead", "Models, experiments, README / reproducible code"],
                ["23261A6657", "Evaluation &amp; report lead", "Metrics, plots, target checks, report"]], [24, 44, 106]))
S += [p("1. Problem understanding", H),
      p("Customers write Telugu in English letters mixed with English words (“network chala bagundi but bill lo extra charges vesaru”), often with polite "
        "openings and the real complaint last. Three classes are needed: negative, positive and mixed. Targets: macro-F1 ≥ 0.65, F1 on “mixed” ≥ 0.45, "
        "predicting 10,000 reviews in &lt; 60 s on CPU, and a defended maximum sequence length. The trade-off is context against speed: longer sequences and "
        "bidirectional layers read more but cost time (prediction time grows about linearly with length, Table 1). With very little labelled data, a small "
        "model (64-unit layer, 64-d embeddings, 56k–114k parameters) is also the safer choice against overfitting."),
      p("<b>Important caveat.</b> No public labelled Telugu–English mobile-network corpus exists, so the dataset is <b>synthetic</b>, generated from hand-written phrase "
        "banks. All numbers below measure behaviour on this generator, not real customers; they are likely optimistic.", B)]
S += [p("2. Data preparation and exploratory analysis", H),
      p("3,000 reviews (1,000 per class), split 2,100/450/450 train/validation/test with seed 42, no text shared between splits, vocabulary (746 types) built on "
        "train only. Length is 2–83 tokens (median 20); mixed reviews are longer (mean 38 vs 21 tokens), a shortcut we flag in §6. Reviews fall in six structures "
        "(plain, polite opening, long with buried complaint, sarcasm, negation such as “problem ledu”, and distant mixed) and half carry injected spelling "
        "variation. To test generalisation, the last 3 of 10 phrases per aspect and polarity are <i>held out</i> of training, and half of the validation and "
        "test reviews use them. The annotation guide and data card are in the repository; labels are assigned by construction, not by humans."),
      p("3. Models and experiments", H),
      p("Embedding(64, masked) → recurrent layer(64) → Dropout(0.3) → Softmax(3); Adam, early stopping on validation loss, 3 seeds per configuration. "
        "<b>Sequence length (LSTM, validation F1, Table 1):</b>"),
      table(sw_rows, [48, 30, 30, 30, 30]), Spacer(1, 3),
      p("Keeping the <i>last</i> tokens is nearly flat from 16 to 128 tokens, so we chose <b>64</b> (best mean, 0.854; 128 costs 1.8× the time for no gain). "
        "Keeping the <i>first</i> tokens at 16 loses 0.21 F1 because the complaint is at the end. "
        "<b>Layers and training strategies (test split, mean over 3 seeds, length 64):</b>"),
      table(rows, [24, 22, 17, 42, 20, 19, 20]), Spacer(1, 3),
      p("“IMDB init” starts from an English IMDB model of the same layer (IMDB test accuracy: SimpleRNN 0.615, LSTM 0.840, GRU 0.859): recurrent weights are copied, "
        "embedding rows are copied for the 21% of our vocabulary that also occurs in the IMDB vocabulary, and a new 3-class head is trained. SimpleRNN, LSTM and BiGRU (0.814–0.834) "
        "are within seed noise of each other; GRU from scratch is lower (0.775).")]
S += [p("4. Results against every target", H),
      table([["Target", "Required", "Best model (SimpleRNN, scratch, seed 1)", "Met?"],
             ["Macro-F1, test", "≥ 0.65", "0.842 (all configurations 0.775–0.834 mean)", "Yes"],
             ["F1 of “mixed” class", "≥ 0.45", "0.993 (suspiciously easy, see §6)", "Yes"],
             ["Predict 10,000 reviews (4-core CPU)", "&lt; 60 s", "0.55 s (slowest model 1.14 s)", "Yes"],
             ["Maximum sequence length", "defend", "64 tokens, from the sweep above", "Done"]], [52, 22, 84, 16]),
      Spacer(1, 3),
      p("The best configuration was picked by mean validation F1 (SimpleRNN scratch 0.866; BiGRU 0.866 and SimpleRNN with IMDB init 0.863 are statistically tied), and its "
        "best-validation seed is reported. Per class F1: negative 0.760, positive 0.773, mixed 0.993; Cohen’s κ of model vs. gold = 0.763."),
      figs, p("<i>Figure: left, length sweep; centre, gradient reaching earlier tokens; right, test confusion matrix of the best model.</i>", C)]
S += [p("5. Answers to the report questions", H),
      p("<b>Q1 (unfolded graph, vanishing gradients).</b> Unfolding a recurrent net over T tokens multiplies T Jacobians, so a token d steps from the end influences the "
        "output through roughly d factors; LSTM/GRU gates add an (almost) additive path that should keep gradients alive. <b>Our data did not show the textbook "
        "result:</b> SimpleRNN was not worse on long reviews (F1 on reviews &gt; 40 tokens: SimpleRNN 0.864, LSTM 0.853, GRU 0.770). We measured the gradient norm "
        "of the loss w.r.t. each token embedding on long reviews; relative to the last token it fell to 0.058 (SimpleRNN), 0.047 (LSTM) and 0.002 (GRU) at 40 tokens "
        "from the end. All three decay, and the gated GRU decays most. A likely reason is that in this synthetic data the decisive words sit in the last ~20 tokens and "
        "even “distant mixed” reviews are caught by connectors such as <i>kani</i> or <i>but</i>, so no layer has to remember far back. The accuracy figures therefore do "
        "<b>not</b> support the claim that gates help here."),
      p("<b>Q2 (English IMDB start).</b> Mixed evidence. Change in test macro-F1 vs. scratch: SimpleRNN +0.004, LSTM −0.004, GRU +0.038 (sd 0.013–0.027), mean +0.013. "
        "Only the GRU gain is large (about 3 sd of its scratch runs), but with 3 seeds and no similar gain for the other layers we treat it as unproven. The transfer is weak because only 21% of the vocabulary overlaps and the "
        "Romanised-Telugu words, which carry most sentiment here, have no English pre-trained meaning; the IMDB-trained SimpleRNN itself reached only 0.615 on IMDB. "
        "We conclude: no reliable benefit, small possible help for GRU."),
      p("<b>Q3 (error analysis).</b> The best model makes 71 errors on 450 reviews: 39 negative→positive, 30 positive→negative, and only 2 involve “mixed”. "
        "<b>65 of 71 errors (92%) are on held-out phrases</b>: accuracy 0.974 on seen vs 0.700 on unseen wording, because new sentiment words (<i>impressive, okay, "
        "unstable</i>) are out-of-vocabulary. By structure: negation 0.66 (17 of 50 wrong), plain 0.83, polite opening 0.84, sarcasm 0.84 (3 of 19 wrong), long 0.87, "
        "distant mixed 1.00; clean and noisy spelling are similar (0.83 vs 0.85). Among sampled errors, items such as “network busy message ee madhya raaledu” (positive by "
        "absence of a complaint) are only weakly positive, so some error is label ambiguity, but we cannot quantify it: inter-annotator κ has <b>not</b> been measured "
        "because labels are programmatic. A blind 210-item sheet and <font face='Courier'>src/kappa.py</font> are provided; until a second human fills it, "
        "the error share caused by unclear labels is unknown. The one design choice with the largest measured effect is keeping the <i>last</i> tokens "
        "(0.847 vs 0.638 validation F1 at length 16, Δ = 0.209; Table 1).")]
S += [p("6. Limitations and future work", H),
      p("(i) Synthetic, template-based data: real reviews have more vocabulary, slang, emojis, Telugu script and spelling noise; accuracy on real data will be lower. "
        "(ii) Shortcuts: longer reviews and contrastive connectors make “mixed” easy (F1 0.99) and probably inflate it. (iii) Word-level vocabulary cannot read "
        "unseen spellings; character n-grams or a multilingual pre-trained encoder would help. (iv) No human second annotator yet. With more time we would collect "
        "real complaints with consent, double-annotate them, and re-run the layer comparison on data where long-range context is genuinely needed.")]
S += [p("7. References", H),
      p("[1] I. Goodfellow, Y. Bengio and A. Courville, <i>Deep Learning</i>, MIT Press, 2016, Ch. 7, 9, 10. "
        "[2] A. Maas et al., “Learning word vectors for sentiment analysis,” ACL, 2011 (IMDB). "
        "[3] Dravidian CodeMix shared task, FIRE workshop. [4] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” <i>Neural Computation</i>, 1997. "
        "[5] K. Cho et al., “Learning phrase representations using RNN encoder-decoder,” EMNLP, 2014.")]
S += [p("Appendix: use of AI tools", H),
      p("The dataset generator, experiment code, notebook and first draft of this report were produced with Claude Code (Anthropic) at the request of the data lead. "
        "Every number in this report is read from files in <font face='Courier'>results/</font> written by the submitted code (seeds fixed). Each member must read and "
        "understand the code before the viva; the contribution table above is a draft that the team must correct to reflect who actually did what.")]

SimpleDocTemplate(str(R / "report" / "DL_T14_2D.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                  title="DL_T14_2D", author="Team 14").build(S)
