"""
Fine-tune pretrained Transformers for Kinyarwanda tweet sentiment.
Designed to run on a free Google Colab T4 GPU (see notebooks/finetune_transformers_colab.ipynb).

Models compared (why each one - say this in the video):
  T1 mBERT  (bert-base-multilingual-cased)  - 104 languages, Kinyarwanda NOT in its
     pretraining data -> tests "does generic multilingual pretraining help?"
  T2 AfriBERTa-base (castorini/afriberta_base) - pretrained from scratch on 11 African
     languages incl. Kinyarwanda, small (~111M params), <1GB of text
  T3 AfroXLMR-base (Davlan/afro-xlmr-base) - XLM-R adapted to 17 African languages
     incl. Kinyarwanda via multilingual adaptive fine-tuning (MAFT)
  T4 AfroXLMR-large (optional, --large)  - same, 560M params

Input: tweet -> subword tokens (max 128) -> [CLS] representation -> linear layer -> 3 labels.
All encoder weights are updated (full fine-tuning). Chosen on DEV macro-F1.

Usage:
  python src/finetune_transformer.py                  # T1-T3, 3 seeds each (~25 min on T4)
  python src/finetune_transformer.py --large          # also AfroXLMR-large
  python src/finetune_transformer.py --push-to-hub USERNAME/kin-sentiment-afroxlmr
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from preprocess import LABELS, light_clean, load_all

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
L2I = {l: i for i, l in enumerate(LABELS)}

MODELS = [
    # id, name, hf id, learning rate, batch size
    ("T1", "mBERT-base (no Kinyarwanda in pretraining)", "bert-base-multilingual-cased", 3e-5, 32),
    ("T2", "AfriBERTa-base (African-only pretraining)", "castorini/afriberta_base", 5e-5, 32),
    ("T3", "AfroXLMR-base (XLM-R + African MAFT)", "Davlan/afro-xlmr-base", 3e-5, 32),
]
LARGE = ("T4", "AfroXLMR-large (XLM-R + African MAFT)", "Davlan/afro-xlmr-large", 1e-5, 16)


def metrics(eval_pred):
    logits, labels = eval_pred
    pred = logits.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
    return {"acc": accuracy_score(labels, pred), "macro_p": p, "macro_r": r, "macro_f1": f,
            "weighted_f1": f1_score(labels, pred, average="weighted")}


def run(model_id, hf_id, lr, bs, seed, splits, epochs):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.model_max_length > 10000:        # AfriBERTa ships without a max length
        tok.model_max_length = 512

    def to_ds(df):
        ds = Dataset.from_dict({"text": df.tweet.map(light_clean).tolist(),
                                "label": df.label.map(L2I).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

    tr, dv, te = (to_ds(d) for d in splits)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I)
    out_dir = f"/tmp/{model_id}_s{seed}"
    total_steps = int(np.ceil(len(tr) / bs)) * epochs
    args = TrainingArguments(
        output_dir=out_dir, learning_rate=lr, per_device_train_batch_size=bs,
        per_device_eval_batch_size=64, num_train_epochs=epochs, weight_decay=0.01,
        warmup_steps=int(0.1 * total_steps), eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, fp16=torch.cuda.is_available(), seed=seed, report_to="none", logging_steps=50)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=dv,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    d = trainer.evaluate(dv)
    pt = trainer.predict(te)
    t = pt.metrics
    res = {"dev_macro_f1": d["eval_macro_f1"],
           **{f"test_{k}": t[f"test_{k}"] for k in ["acc", "macro_p", "macro_r", "macro_f1", "weighted_f1"]}}
    return res, trainer, tok, pt.predictions.argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--only", nargs="*", help="run only these ids, e.g. --only T3")
    ap.add_argument("--push-to-hub", default=None, help="HF repo id to upload the best model to")
    a = ap.parse_args()

    splits = load_all()
    models = MODELS + ([LARGE] if a.large else [])
    if a.only:
        models = [m for m in models if m[0] in a.only]

    rows, best = [], None
    for mid, name, hf_id, lr, bs in models:
        runs = []
        for seed in a.seeds:
            r, trainer, tok, pred = run(mid, hf_id, lr, bs, seed, splits, a.epochs)
            runs.append(r)
            print(f"{mid} seed {seed}: dev macro-F1 {r['dev_macro_f1']:.4f} | test macro-F1 {r['test_macro_f1']:.4f}")
            if best is None or r["dev_macro_f1"] > best[0]:
                best = (r["dev_macro_f1"], mid, trainer, tok, pred)
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        df = pd.DataFrame(runs)
        row = {"id": mid, "experiment": name, "hf_model": hf_id, "lr": lr, "batch": bs, "epochs": a.epochs}
        for k in df.columns:
            row[k] = df[k].mean(); row[k + "_std"] = df[k].std()
        rows.append(row)
        pd.DataFrame(rows).to_csv(RES / "transformer_experiments.csv", index=False)

    out = pd.DataFrame(rows)
    print(out[["id", "experiment", "dev_macro_f1", "test_macro_f1", "test_macro_f1_std", "test_weighted_f1"]].round(4))

    _, bid, trainer, tok, pred = best
    test = splits[2]
    pd.DataFrame({"tweet": test.tweet, "gold": test.label, f"pred_{bid}": [LABELS[i] for i in pred]}) \
        .to_csv(RES / "transformer_test_predictions.csv", index=False)
    save = ROOT / "models" / "best_transformer"
    trainer.save_model(save); tok.save_pretrained(save)
    json.dump({"best_transformer": bid}, open(RES / "best_transformer.json", "w"))
    print(f"Best transformer on dev: {bid} -> saved to {save}")
    if a.push_to_hub:
        trainer.model.push_to_hub(a.push_to_hub); tok.push_to_hub(a.push_to_hub)
        print(f"Uploaded to https://huggingface.co/{a.push_to_hub}")


if __name__ == "__main__":
    main()
