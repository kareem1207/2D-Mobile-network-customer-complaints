"""Builds notebooks/DL_T14_2D.ipynb (then executed with nbconvert; see README)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# DL_T14_2D — Sentiment analysis of code-mixed Telugu–English mobile-network complaints\n"
       "Task 2, Team 14 (MGIT, CM701PC/CM751PC). **Data is synthetic** (see `data/DATA_CARD.md`).\n\n"
       "This notebook (1) inspects the data, (2) runs a quick seeded comparison of SimpleRNN / LSTM / GRU, and "
       "(3) displays the full results produced by `python src/experiments.py` (3 seeds × 8 configurations, sequence-length sweep, "
       "gradient profile, error analysis)."),
    code("import sys, os\nif os.path.basename(os.getcwd()) == 'notebooks':\n    os.chdir('..')\nsys.path.insert(0, os.path.abspath('src'))\nfrom common import *\nimport pandas as pd\nfrom IPython.display import Image, display\n"
         "pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 120)"),
    md("## 1. Data"),
    code("df = load_data()\nprint(df.groupby(['split','label']).size().unstack())\nprint(df.groupby('label').n_tokens.describe().round(1))\n"
         "df[['text','label','category']].sample(6, random_state=1)"),
    md("## 2. Quick run (seed 0, max length 64, keep last tokens) — three recurrent layers trained from scratch"),
    code("vocab = build_vocab(df[df.split=='train'].tokens)\nprint('vocab size', len(vocab))\n"
         "S = {s: (encode(df[df.split==s].tokens.tolist(), vocab, 64, 'pre'), df[df.split==s].y.values) for s in ['train','val','test']}\n"
         "rows = []\nfor name in LAYERS:\n    set_seed(0)\n    m = build_model(name, len(vocab)+2)\n    fit(m, *S['train'], *S['val'])\n"
         "    p = predict(m, S['test'][0])\n    rows.append({'layer': name, 'test_macro_f1': macro_f1(S['test'][1], p), 'f1_mixed': per_class_f1(S['test'][1], p)[2],\n"
         "                 'sec_per_10k': time_10k(m, S['test'][0]), 'params': m.count_params()})\npd.DataFrame(rows).round(3)"),
    md("## 3. Full results (3 seeds each; produced by `src/experiments.py`)\n### Sequence length / truncation sweep (LSTM, validation macro-F1)"),
    code("display(pd.read_csv('results/seqlen_sweep_summary.csv').round(3))\ndisplay(Image('figures/seqlen_sweep.png', width=420))"),
    md("### Three recurrent layers × two training strategies"),
    code("t = pd.read_csv('results/comparison_table.csv')\nt[['layer','strategy','val_macro_f1_mean','test_macro_f1_mean','test_macro_f1_std','test_f1_mixed_mean','sec_per_10k_mean','params_first']].round(3)"),
    md("### Gradient flow to early tokens (vanishing-gradient evidence)"),
    code("g = pd.read_csv('results/gradient_profile.csv', index_col=0)\ndisplay(g.iloc[[0,10,20,40,59]])\ndisplay(Image('figures/gradient_profile.png', width=420))"),
    md("### Best model on the test split"),
    code("print(open('results/best_report.txt').read())\nprint(open('results/best_summary.json').read())\ndisplay(Image('figures/confusion_matrix.png', width=300))"),
    md("### Error analysis"),
    code("for c in ['phrase_pool','category','bucket','spelling_noise']:\n    print(c); display(pd.read_csv(f'results/best_acc_by_{c}.csv').round(3))\n"
         "pd.read_csv('results/best_errors.csv').sample(10, random_state=0)"),
    md("To reproduce everything: `python src/generate_dataset.py && python src/pretrain_imdb.py && python src/experiments.py`."),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/DL_T14_2D.ipynb")
