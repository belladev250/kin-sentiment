"""
Error analysis for the best fine-tuned Transformer (T3, AfroXLMR-base) and a
direct comparison with the best classical model (E6).

Questions answered:
  - Per-class precision / recall / F1 and confusion matrix of T3.
  - Is T3 significantly better than E6? (McNemar on paired predictions)
  - On which kinds of tweets does T3 gain most? (same subsets as error_analysis.py)
  - Does performance depend on how much the human annotators agreed?
  - Example tweets that T3 fixes, that both models get wrong, and that T3 breaks.

Run:  python src/transformer_analysis.py   (needs results/transformer_test_predictions.csv)
Outputs: results/transformer_analysis.json, results/transformer_classification_report.txt,
         results/transformer_examples.csv, results/figures/confusion_matrix_T3.png
"""
import json
import re
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from preprocess import LABELS, load_all

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
ENGLISH = {"the", "and", "is", "you", "of", "to", "in", "for", "this", "that", "my", "me", "so",
           "we", "it", "be", "are", "with", "thank", "thanks", "congratulations", "happy", "good",
           "love", "god", "bless", "please", "yes", "no", "not", "but", "very", "all"}
FRENCH = {"le", "la", "les", "de", "des", "et", "est", "je", "pas", "merci", "bonne", "c'est", "une", "un"}


def mcnemar(gold, a, b):
    """Exact McNemar test. Returns (#a right & b wrong, #a wrong & b right, p)."""
    ca, cb = a == gold, b == gold
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


def norm(text):
    """Make raw-annotation tweets comparable with the anonymised test tweets."""
    text = re.sub(r"@\w+", "@user", str(text))
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    train, _, test = load_all()
    c = pd.read_csv(RES / "classical_test_predictions.csv")
    t = pd.read_csv(RES / "transformer_test_predictions.csv")
    assert (c.tweet == t.tweet).all(), "prediction files are not aligned"
    df = c[["tweet", "clean", "gold", "pred_baseline", "pred_E6"]].copy()
    df["pred_T3"] = t.pred_T3.values
    gold, e6, t3 = df.gold.values, df.pred_E6.values, df.pred_T3.values

    out = {"model": "T3 AfroXLMR-base (best seed on dev, the deployed checkpoint)",
           "test_macro_f1": round(f1_score(gold, t3, average="macro"), 4),
           "test_acc": round(float((gold == t3).mean()), 4)}

    rep = classification_report(gold, t3, labels=LABELS, digits=3)
    (RES / "transformer_classification_report.txt").write_text(f"T3 AfroXLMR-base on TEST\n\n{rep}")
    out["per_class"] = {k: {m: round(v, 4) for m, v in d.items() if m != "support"}
                        for k, d in classification_report(gold, t3, labels=LABELS, output_dict=True).items()
                        if k in LABELS}

    cm = confusion_matrix(gold, t3, labels=LABELS)
    fig, ax = plt.subplots(figsize=(3.8, 3.3), dpi=150)
    ax.imshow(cm, cmap="Greens")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title("Confusion matrix (T3, test)")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix_T3.png"); plt.close()
    wrong = df[t3 != gold]
    out["confusions"] = {f"{g}->{p}": int(n) for (g, p), n in
                         wrong.groupby(["gold", "pred_T3"]).size().sort_values(ascending=False).items()}

    for name, other in [("E1", df.pred_baseline.values), ("E6", e6)]:
        a, b, p = mcnemar(gold, other, t3)
        out[f"mcnemar_T3_vs_{name}"] = {f"{name}_right_T3_wrong": a, f"{name}_wrong_T3_right": b, "p_value": round(p, 6)}

    # ---- subsets (same definitions as error_analysis.py) ----
    vocab = set(w for s in train.tweet.str.lower() for w in re.findall(r"\w+", s))
    def feats(row):
        toks = [x for x in re.findall(r"\w+", row.clean) if x not in ("xxuser", "xxurl")]
        return pd.Series({
            "short": len(toks) <= 5,
            "question": "?" in row.tweet,
            "code_switch": any(x in ENGLISH | FRENCH for x in toks),
            "has_emoji": bool(EMOJI.search(row.tweet)),
            "oov_rate": np.mean([x not in vocab for x in toks]) if toks else 0.0})
    df = pd.concat([df, df.apply(feats, axis=1)], axis=1)
    df["ok_E6"], df["ok_T3"] = df.pred_E6 == df.gold, df.pred_T3 == df.gold
    df["oov_bin"] = pd.cut(df.oov_rate, [-0.01, 0.1, 0.3, 1.0], labels=["<=10%", "10-30%", ">30%"])
    subsets = {}
    for col, label in [("short", "short (<=5 words)"), ("question", "question mark"),
                       ("code_switch", "code-switched"), ("has_emoji", "contains emoji")]:
        g = df[df[col]]
        subsets[label] = {"n": len(g), "acc_E6": round(g.ok_E6.mean(), 4), "acc_T3": round(g.ok_T3.mean(), 4)}
    for b, g in df.groupby("oov_bin", observed=True):
        subsets[f"OOV {b}"] = {"n": len(g), "acc_E6": round(g.ok_E6.mean(), 4), "acc_T3": round(g.ok_T3.mean(), 4)}
    subsets["all"] = {"n": len(df), "acc_E6": round(df.ok_E6.mean(), 4), "acc_T3": round(df.ok_T3.mean(), 4)}
    out["subsets"] = subsets

    # ---- annotator agreement (match test tweets to the raw 3-annotator file) ----
    ind = pd.read_csv(ROOT / "data" / "kinyarwanda_individual_labels.csv")
    ind["key"] = ind.text.map(norm)
    lab = ind[["label_1", "label_2", "label_3"]]
    ind["n_agree"] = lab.apply(lambda r: r.value_counts().iloc[0], axis=1)
    ind = ind.drop_duplicates("key").set_index("key")
    df["n_agree"] = df.tweet.map(norm).map(ind.n_agree)
    m = df.dropna(subset=["n_agree"])
    agree = {"matched_test_tweets": int(len(m))}
    for k, g in m.groupby("n_agree"):
        name = {3: "unanimous (3/3)", 2: "majority (2/3)"}.get(int(k), f"{int(k)}/3")
        agree[name] = {"n": int(len(g)), "acc_E6": round(g.ok_E6.mean(), 4), "acc_T3": round(g.ok_T3.mean(), 4)}
    out["by_annotator_agreement"] = agree

    # ---- E6 vs T3 disagreement + examples ----
    out["fixed_by_T3"] = int((~df.ok_E6 & df.ok_T3).sum())
    out["broken_by_T3"] = int((df.ok_E6 & ~df.ok_T3).sum())
    out["both_wrong"] = int((~df.ok_E6 & ~df.ok_T3).sum())
    ex = pd.concat([
        df[~df.ok_E6 & df.ok_T3].assign(kind="fixed_by_T3"),
        df[df.ok_E6 & ~df.ok_T3].assign(kind="broken_by_T3"),
        df[~df.ok_E6 & ~df.ok_T3].assign(kind="both_wrong")])
    ex[["kind", "tweet", "gold", "pred_E6", "pred_T3", "oov_rate", "n_agree"]].to_csv(
        RES / "transformer_examples.csv", index=False)

    json.dump(out, open(RES / "transformer_analysis.json", "w"), indent=2, ensure_ascii=False)
    print(rep)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
