"""
Evaluation figures + error analysis for the best classical model (and baseline).

Run:  python src/error_analysis.py     (after baselines.py and bilstm.py)
Outputs in results/:
  figures/class_distribution.png, figures/experiments.png, figures/confusion_matrix.png
  classification_report.txt, error_samples.csv, error_stats.json
"""
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import LABELS, load_all

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
# Common English words; Kinyarwanda tweets often code-switch with English/French.
ENGLISH = {"the", "and", "is", "you", "of", "to", "in", "for", "this", "that", "my", "me", "so",
           "we", "it", "be", "are", "with", "thank", "thanks", "congratulations", "happy", "good",
           "love", "god", "bless", "please", "yes", "no", "not", "but", "very", "all"}
FRENCH = {"le", "la", "les", "de", "des", "et", "est", "je", "pas", "merci", "bonne", "c'est", "une", "un"}
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def main():
    train, dev, test = load_all()

    # ---------- dataset figure ----------
    counts = pd.DataFrame({n: d.label.value_counts() for n, d in
                           [("train", train), ("dev", dev), ("test", test)]}).loc[LABELS]
    ax = counts.T.plot(kind="bar", stacked=True, figsize=(5, 3.2), color=["#c0504d", "#9e9e9e", "#4f81bd"])
    ax.set_ylabel("tweets"); ax.set_title("AfriSenti-Kinyarwanda label distribution")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIG / "class_distribution.png"); plt.close()

    # ---------- experiments figure ----------
    ce = pd.read_csv(RES / "classical_experiments.csv")
    ne = pd.read_csv(RES / "neural_experiments.csv")
    parts = [ce[["id", "test_macro_f1"]].assign(std=0.0), ne[["id", "test_macro_f1"]].assign(std=ne.test_macro_f1_std)]
    tf = RES / "transformer_experiments.csv"
    if tf.exists():
        t = pd.read_csv(tf)
        parts.append(t[["id", "test_macro_f1"]].assign(std=t.get("test_macro_f1_std", 0.0)))
    allx = pd.concat(parts)
    colors = ["#bbbbbb" if i == "E0" else "#f0a030" if i == "E1" else
              "#4f81bd" if i[0] in "EA" else "#8064a2" if i[0] == "N" else "#2e8b57" for i in allx.id]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(allx.id, allx.test_macro_f1, yerr=allx["std"], color=colors, capsize=3)
    ax.axhline(ce.loc[ce.id == "E1", "test_macro_f1"].item(), ls="--", c="#f0a030", lw=1, label="baseline E1")
    ax.set_ylabel("test macro-F1"); ax.set_ylim(0, 0.85); ax.legend(loc="upper left")
    ax.set_title("Test macro-F1 by experiment (grey=majority, orange=baseline, blue=classical, purple=BiLSTM, green=Transformer)", fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "experiments.png"); plt.close()

    # ---------- best classical model analysis ----------
    best = joblib.load(ROOT / "models" / "best_classical.joblib")
    bid = best["id"]
    pr = pd.read_csv(RES / "classical_test_predictions.csv")
    pred = pr[f"pred_{bid}"]
    rep = classification_report(pr.gold, pred, labels=LABELS, digits=3)
    (RES / "classification_report.txt").write_text(f"Best classical model {bid} on TEST\n\n{rep}")
    print(rep)

    cm = confusion_matrix(pr.gold, pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(f"Confusion matrix ({bid}, test)")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.png"); plt.close()

    # ---------- error categories ----------
    vocab = set(w for t in train.tweet.str.lower() for w in re.findall(r"\w+", t))
    def feats(row):
        toks = re.findall(r"\w+", row.clean)
        toks_nouser = [t for t in toks if t not in ("xxuser", "xxurl")]
        oov = np.mean([t not in vocab for t in toks_nouser]) if toks_nouser else 0
        return pd.Series({
            "n_tokens": len(toks_nouser),
            "oov_rate": oov,
            "has_emoji": bool(EMOJI.search(str(row.tweet))),
            "code_switch": any(t in ENGLISH | FRENCH for t in toks_nouser),
            "short": len(toks_nouser) <= 5,
            "question": "?" in str(row.tweet),
        })
    F = pr.apply(feats, axis=1)
    pr = pd.concat([pr, F], axis=1)
    pr["correct"] = pr.gold == pred
    pr["pred"] = pred

    stats = {"best_model": bid, "test_size": len(pr), "errors": int((~pr.correct).sum())}
    for col in ["short", "has_emoji", "code_switch", "question"]:
        g = pr.groupby(col).correct.mean()
        stats[f"acc_{col}_true"] = round(float(g.get(True, np.nan)), 4)
        stats[f"acc_{col}_false"] = round(float(g.get(False, np.nan)), 4)
        stats[f"n_{col}_true"] = int(pr[col].sum())
    pr["oov_bin"] = pd.cut(pr.oov_rate, [-0.01, 0.1, 0.3, 1.0], labels=["<=10%", "10-30%", ">30%"])
    for b, g in pr.groupby("oov_bin", observed=True):
        stats[f"acc_oov_{b}"] = round(float(g.correct.mean()), 4)
        stats[f"n_oov_{b}"] = int(len(g))
    # most common confusions
    conf = pr[~pr.correct].groupby(["gold", "pred"]).size().sort_values(ascending=False)
    stats["confusions"] = {f"{g}->{p}": int(n) for (g, p), n in conf.items()}
    # baseline vs best disagreement
    stats["fixed_by_best"] = int(((pr.pred_baseline != pr.gold) & pr.correct).sum())
    stats["broken_by_best"] = int(((pr.pred_baseline == pr.gold) & ~pr.correct).sum())
    json.dump(stats, open(RES / "error_stats.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # samples for manual review: errors + a few correct, most "confident" first
    pipe = best["pipeline"]
    if hasattr(pipe, "decision_function"):
        margin = np.sort(pipe.decision_function(pr.clean), axis=1)
        pr["margin"] = margin[:, -1] - margin[:, -2]
    cols = ["tweet", "gold", "pred", "pred_baseline", "n_tokens", "oov_rate", "has_emoji", "code_switch", "margin"]
    errs = pr[~pr.correct].sort_values("margin", ascending=False)[cols]
    good = pr[pr.correct].sort_values("margin", ascending=False)[cols].head(40)
    pd.concat([errs.assign(kind="error"), good.assign(kind="correct")]).to_csv(RES / "error_samples.csv", index=False)


if __name__ == "__main__":
    main()
