"""
Neural model trained from scratch: BiLSTM (+ optional additive attention).

Architecture (explain in video):
  tokens -> Embedding(V, 100) -> dropout -> BiLSTM(hidden 64 per direction)
         -> sentence vector:
              * "maxpool": element-wise max over time of the 128-d states
              * "attention": softmax(v^T tanh(W h_t)) weights, weighted sum of h_t
         -> dropout -> Linear(128, 3) -> softmax over {negative, neutral, positive}

Loss: cross-entropy. Optimizer: Adam (lr 2e-3). Early stopping on DEV macro-F1
(patience 3). Each config is run with 3 random seeds and we report mean ± std,
because with only ~3.3k training tweets a single run is noisy.

Run:  python src/bilstm.py
"""
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from preprocess import LABELS, clean, load_all

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
L2I = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 60
torch.set_num_threads(2)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Vocab:
    def __init__(self, texts, min_freq=2):
        c = Counter(w for t in texts for w in t.split())
        self.itos = ["<pad>", "<unk>"] + [w for w, n in c.most_common() if n >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        ids = [self.stoi.get(w, 1) for w in text.split()][:MAX_LEN]
        return ids or [1]


def batchify(ids_list, labels, bs, shuffle):
    idx = np.arange(len(ids_list))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        L = max(len(ids_list[j]) for j in b)
        x = torch.zeros(len(b), L, dtype=torch.long)
        for k, j in enumerate(b):
            x[k, :len(ids_list[j])] = torch.tensor(ids_list[j])
        yield x, torch.tensor(labels[b])


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=100, hid=64, pooling="maxpool", dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.pooling = pooling
        self.att_W = nn.Linear(2 * hid, 2 * hid)
        self.att_v = nn.Linear(2 * hid, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hid, len(LABELS))

    def forward(self, x, return_attention=False):
        mask = x != 0                                   # (B, T)
        h, _ = self.lstm(self.drop(self.emb(x)))        # (B, T, 2H)
        if self.pooling == "attention":
            scores = self.att_v(torch.tanh(self.att_W(h))).squeeze(-1)  # (B, T)
            scores = scores.masked_fill(~mask, -1e9)
            a = torch.softmax(scores, dim=1)
            s = (a.unsqueeze(-1) * h).sum(1)
        else:
            s = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
            a = None
        logits = self.out(self.drop(s))
        return (logits, a) if return_attention else logits


def predict(model, ids, bs=128):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in batchify(ids, np.zeros(len(ids), dtype=int), bs, False):
            out.append(model(x).argmax(1).numpy())
    return np.concatenate(out)


def train_one(cfg, seed, data):
    set_seed(seed)
    (tr_ids, ytr), (dv_ids, ydv), (te_ids, yte), V = data
    model = BiLSTMClassifier(V, pooling=cfg["pooling"], emb=cfg["emb"], hid=cfg["hid"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, bad = -1, None, 0
    for ep in range(1, 31):
        model.train()
        for x, y in batchify(tr_ids, ytr, 32, True):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        f = f1_score(ydv, predict(model, dv_ids), average="macro")
        if f > best:
            best, best_state, bad, best_ep = f, {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    pte = predict(model, te_ids)
    return {"dev_macro_f1": best, "test_acc": accuracy_score(yte, pte),
            "test_macro_f1": f1_score(yte, pte, average="macro"),
            "test_weighted_f1": f1_score(yte, pte, average="weighted"), "epochs": best_ep}, model, pte


def main():
    train, dev, test = load_all()
    ctr, cdv, cte = (d.tweet.map(clean).tolist() for d in (train, dev, test))
    vocab = Vocab(ctr)
    print(f"Vocab size (min_freq=2, train only): {len(vocab.itos)}")
    enc = lambda texts: [vocab.encode(t) for t in texts]
    lab = lambda d: d.label.map(L2I).values
    data = ((enc(ctr), lab(train)), (enc(cdv), lab(dev)), (enc(cte), lab(test)), len(vocab.itos))

    oov = np.mean([w not in vocab.stoi for t in cte for w in t.split()])
    print(f"Test-set token OOV rate: {oov:.1%}")

    configs = [
        ("N1", "BiLSTM + max-pooling (emb 100, hid 64)", dict(pooling="maxpool", emb=100, hid=64, lr=2e-3)),
        ("N2", "BiLSTM + additive attention (emb 100, hid 64)", dict(pooling="attention", emb=100, hid=64, lr=2e-3)),
        ("N3", "BiLSTM + attention, bigger (emb 200, hid 128)", dict(pooling="attention", emb=200, hid=128, lr=1e-3)),
    ]
    rows, best_models = [], {}
    for cid, desc, cfg in configs:
        runs = []
        for seed in (1, 2, 3):
            r, m, pte = train_one(cfg, seed, data)
            runs.append(r)
            print(f"  {cid} seed {seed}: dev F1 {r['dev_macro_f1']:.4f} test F1 {r['test_macro_f1']:.4f} (ep {r['epochs']})")
            if cid not in best_models or r["dev_macro_f1"] > best_models[cid][0]:
                best_models[cid] = (r["dev_macro_f1"], m, pte)
        df = pd.DataFrame(runs)
        row = {"id": cid, "experiment": desc}
        for k in ["dev_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1"]:
            row[k] = df[k].mean(); row[k + "_std"] = df[k].std()
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "neural_experiments.csv", index=False)
    print("\n" + out[["id", "experiment", "dev_macro_f1", "test_macro_f1", "test_macro_f1_std"]].round(4).to_string(index=False))

    # Keep attention example weights for the report (interpretability)
    _, m, pte = best_models["N2"]
    pd.DataFrame({"tweet": test.tweet, "gold": test.label, "pred_N2": [LABELS[i] for i in pte]}) \
        .to_csv(RESULTS / "neural_test_predictions.csv", index=False)
    torch.save({"state": m.state_dict(), "itos": vocab.itos}, ROOT / "models" / "bilstm_attention.pt")


if __name__ == "__main__":
    main()
