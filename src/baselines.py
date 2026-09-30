"""
Baseline + classical ML experiments for Kinyarwanda tweet sentiment.

Run:  python src/baselines.py
Outputs:
  results/classical_experiments.csv   one row per experiment (dev + test scores)
  models/best_classical.joblib        best pipeline (chosen on DEV macro-F1) for the web app

Every model is tuned ONLY on the dev set. The test set is scored once per
experiment and never used for choosing anything.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean, load_all, load_stopwords

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
SEED = 42


def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_p": p,
        "macro_r": r,
        "macro_f1": f,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def word_vec(ngram=(1, 1), stop=None):
    return TfidfVectorizer(ngram_range=ngram, min_df=1, sublinear_tf=True,
                           token_pattern=r"(?u)\S+", stop_words=list(stop) if stop else None)


def char_vec():
    # char_wb = character n-grams inside word boundaries -> captures prefixes/
    # suffixes of Kinyarwanda's agglutinative words (e.g. nta-, -ye, -ze).
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def union():
    return FeatureUnion([("word", word_vec((1, 2))), ("char", char_vec())])


def lr(C):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED)


def tune(name, make_pipe, grid, Xtr, ytr, Xdv, ydv):
    """Pick the hyper-parameter with best DEV macro-F1."""
    best = None
    for g in grid:
        pipe = make_pipe(g)
        pipe.fit(Xtr, ytr)
        f = f1_score(ydv, pipe.predict(Xdv), average="macro")
        if best is None or f > best[0]:
            best = (f, g, pipe)
    print(f"  {name}: best param={best[1]}  dev macro-F1={best[0]:.4f}")
    return best[2], best[1]


def main():
    train, dev, test = load_all()
    stop = load_stopwords()
    ytr, ydv, yte = train.label.values, dev.label.values, test.label.values

    views = {
        "clean": lambda s: s.map(clean),
        "clean_noemoji": lambda s: s.map(lambda t: clean(t, keep_emoji=False)),
        "raw": lambda s: s.astype(str),
    }
    X = {v: (f(train.tweet), f(dev.tweet), f(test.tweet)) for v, f in views.items()}

    Cs = [0.1, 0.3, 1, 3, 10]
    experiments = [
        # id, description, text view, pipeline factory, grid
        ("E0", "Majority class (neutral)", "clean",
         lambda g: DummyClassifier(strategy="most_frequent"), [None]),
        ("E1", "BASELINE: TF-IDF word unigrams + LogReg", "clean",
         lambda C: Pipeline([("v", word_vec()), ("m", lr(C))]), Cs),
        ("E2", "TF-IDF word 1-2grams + LogReg", "clean",
         lambda C: Pipeline([("v", word_vec((1, 2))), ("m", lr(C))]), Cs),
        ("E3", "TF-IDF char 2-5grams + LogReg", "clean",
         lambda C: Pipeline([("v", char_vec()), ("m", lr(C))]), Cs),
        ("E4", "Word 1-2 + char 2-5 + LogReg", "clean",
         lambda C: Pipeline([("v", union()), ("m", lr(C))]), Cs),
        ("E5", "Word 1-2 + char 2-5 + Multinomial NB", "clean",
         lambda a: Pipeline([("v", union()), ("m", MultinomialNB(alpha=a))]), [0.01, 0.03, 0.1, 0.3, 1]),
        ("E6", "Word 1-2 + char 2-5 + Linear SVM", "clean",
         lambda C: Pipeline([("v", union()), ("m", LinearSVC(C=C, class_weight="balanced", random_state=SEED))]),
         [0.03, 0.1, 0.3, 1]),
        # --- preprocessing ablations on the E4 configuration ---
        ("A1", "E4 on RAW text (no preprocessing)", "raw",
         lambda C: Pipeline([("v", union()), ("m", lr(C))]), Cs),
        ("A2", "E4 with emojis REMOVED", "clean_noemoji",
         lambda C: Pipeline([("v", union()), ("m", lr(C))]), Cs),
        ("A3", "E2 with stopwords REMOVED (word only)", "clean",
         lambda C: Pipeline([("v", word_vec((1, 2), stop)), ("m", lr(C))]), Cs),
    ]

    rows, fitted = [], {}
    for eid, desc, view, make, grid in experiments:
        print(f"[{eid}] {desc}")
        Xtr, Xdv, Xte = X[view]
        t0 = time.time()
        pipe, param = tune(eid, make, grid, Xtr, ytr, Xdv, ydv)
        secs = time.time() - t0
        d, t = scores(ydv, pipe.predict(Xdv)), scores(yte, pipe.predict(Xte))
        rows.append({"id": eid, "experiment": desc, "best_param": param,
                     **{f"dev_{k}": v for k, v in d.items()},
                     **{f"test_{k}": v for k, v in t.items()},
                     "tune_seconds": round(secs, 1)})
        fitted[eid] = (pipe, view)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "classical_experiments.csv", index=False)
    show = df[["id", "experiment", "best_param", "dev_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1"]]
    print("\n" + show.round(4).to_string(index=False))

    # Choose the deployable model on DEV only (exclude ablations that use other text views)
    cand = df[df.id.str.startswith("E") & (df.id != "E0")]
    best_id = cand.sort_values("dev_macro_f1", ascending=False).iloc[0].id
    pipe, view = fitted[best_id]
    joblib.dump({"pipeline": pipe, "view": view, "id": best_id}, MODELS / "best_classical.joblib")

    # Save predictions of baseline and best model for error analysis
    Xte = X["clean"][2]
    preds = pd.DataFrame({"tweet": test.tweet, "clean": Xte, "gold": yte,
                          "pred_baseline": fitted["E1"][0].predict(Xte),
                          f"pred_{best_id}": pipe.predict(Xte)})
    preds.to_csv(RESULTS / "classical_test_predictions.csv", index=False)
    json.dump({"best_classical": best_id}, open(RESULTS / "best_classical.json", "w"))
    print(f"\nBest classical model on dev: {best_id} -> saved to models/best_classical.joblib")


if __name__ == "__main__":
    main()
