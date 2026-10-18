
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CATS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def fleiss_kappa(M, n):
    P = (M * (M - 1)).sum(axis=1) / (n * (n - 1))
    pj = M.sum(axis=0) / (len(M) * n)
    return (P.mean() - (pj ** 2).sum()) / (1 - (pj ** 2).sum())


def main():
    d = pd.read_csv(ROOT / "data" / "kinyarwanda_individual_labels.csv")
    L = d[["label_1", "label_2", "label_3"]].astype(str).apply(lambda c: c.str.upper())
    raw = pd.Series(L.values.ravel()).value_counts().to_dict()
    full = L[L.isin(CATS).all(axis=1)]
    M = np.stack([(full == c).sum(axis=1).values for c in CATS], axis=1)
    out = {
        "tweets_annotated": int(len(d)),
        "raw_label_counts": {k: int(v) for k, v in raw.items()},
        "tweets_with_3_pos_neu_neg_labels": int(len(full)),
        "unanimous_fraction": round(float((full.nunique(axis=1) == 1).mean()), 3),
        "majority_exists_fraction": round(float((full.nunique(axis=1) <= 2).mean()), 3),
        "fleiss_kappa": round(float(fleiss_kappa(M, 3)), 3),
    }
    json.dump(out, open(ROOT / "results" / "annotator_agreement.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
