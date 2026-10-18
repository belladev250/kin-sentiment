
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
rng = np.random.default_rng(0)


def boot_ci(gold, pred, B=2000):
    n = len(gold); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(f1_score(gold[i], pred[i], average="macro"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(gold, a, b):
    from math import comb
    ca, cb = a == gold, b == gold
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = n01 + n10; k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)  # exact binomial
    return n01, n10, p


def main():
    c = pd.read_csv(RES / "classical_test_predictions.csv")
    gold = c.gold.values
    preds = {"E1 (baseline)": c.pred_baseline.values}
    best_c = [col for col in c.columns if col.startswith("pred_E")][0]
    preds[best_c.replace("pred_", "") + " (best classical)"] = c[best_c].values
    n = RES / "neural_test_predictions.csv"
    if n.exists():
        preds["N2 (BiLSTM+attn, best seed)"] = pd.read_csv(n).pred_N2.values
    t = RES / "transformer_test_predictions.csv"
    if t.exists():
        td = pd.read_csv(t); col = [x for x in td.columns if x.startswith("pred_")][0]
        preds[col.replace("pred_", "") + " (best Transformer)"] = td[col].values

    out = {}
    for name, p in preds.items():
        lo, hi = boot_ci(gold, p)
        out[name] = {"macro_f1": round(f1_score(gold, p, average="macro"), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    base = preds["E1 (baseline)"]
    for name, p in preds.items():
        if name.startswith("E1"):
            continue
        n01, n10, pv = mcnemar(gold, base, p)
        out[name]["mcnemar_vs_baseline"] = {"baseline_right_model_wrong": n01, "baseline_wrong_model_right": n10, "p_value": round(pv, 6)}
    json.dump(out, open(RES / "significance.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
