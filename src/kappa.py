"""Cohen's kappa between the dataset labels (annotator A) and an independent second annotator.

Usage:  python src/kappa.py annotation/key_annotator_a.csv annotation/second_annotator_sheet.csv

The second annotator fills `label_annotator_b` (negative / positive / mixed) in the BLIND sheet,
which does not contain annotator A's labels. Prints kappa, raw agreement and the disagreement matrix.
"""
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

LABELS = ["negative", "positive", "mixed"]


def main(key_path, sheet_path):
    d = pd.read_csv(key_path).merge(pd.read_csv(sheet_path), on="id").dropna(subset=["label_annotator_b"])
    d = d[d.label_annotator_b.astype(str).str.strip() != ""]
    if d.empty:
        sys.exit("label_annotator_b is empty - the second annotator has not labelled the sheet yet.")
    a, b = d["label_annotator_a"], d["label_annotator_b"].str.strip().str.lower()
    print(f"n = {len(d)}  raw agreement = {(a == b).mean():.3f}  Cohen's kappa = {cohen_kappa_score(a, b, labels=LABELS):.3f}")
    print(pd.DataFrame(confusion_matrix(a, b, labels=LABELS), index=LABELS, columns=LABELS))


if __name__ == "__main__":
    main(*sys.argv[1:3])
