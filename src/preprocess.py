"""
Text preprocessing for Kinyarwanda tweets (AfriSenti-Kin).

Design decisions (explain these in the viva):
- '@user' mentions carry no sentiment and many tweets start with 5-10 of them,
  so we collapse them to a single USER token instead of deleting (keeps the
  "this is a reply" signal).
- URLs -> URL token.
- Lowercasing: Kinyarwanda uses Latin script; case adds sparsity, not meaning.
- Character elongation ("Hhhhhh", "yeeeee") is common in tweets. We cap any
  run of the same character at 2 so "hhhhhh" and "hhhh" become the same token.
- Emojis are KEPT and separated into their own tokens because they are one of
  the strongest sentiment cues in tweets (we test this in an experiment).
- We do NOT remove stopwords by default: Kinyarwanda negation is often a prefix
  or short word (e.g. "nta", "ntabwo", "si") and dropping short words can flip
  meaning. This is also tested as an experiment.
"""
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS = ["negative", "neutral", "positive"]

_USER = re.compile(r"(@\w+\s*)+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_ELONG = re.compile(r"(.)\1{2,}")
# Broad emoji ranges (pictographs, symbols, flags, dingbats)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF⭐❤]"
)
_NON_WORD = re.compile(r"[^\w\s'\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF⭐❤]")
_SPACES = re.compile(r"\s+")


def clean(text: str, keep_emoji: bool = True) -> str:
    t = str(text).lower()
    t = _URL.sub(" xxurl ", t)
    t = _USER.sub(" xxuser ", t)
    t = _ELONG.sub(r"\1\1", t)
    if keep_emoji:
        t = _EMOJI.sub(lambda m: f" {m.group(0)} ", t)
    else:
        t = _EMOJI.sub(" ", t)
    t = _NON_WORD.sub(" ", t)
    t = t.replace("_", " ")
    return _SPACES.sub(" ", t).strip()


def light_clean(text: str) -> str:
    """Minimal cleaning for Transformers: their subword tokenizers handle case,
    punctuation and emojis themselves, so we only remove pure noise."""
    t = _URL.sub("http", str(text))
    t = _USER.sub("@user ", t)
    t = _ELONG.sub(r"\1\1", t)
    return _SPACES.sub(" ", t).strip()


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{split}.tsv", sep="\t")
    df = df.dropna(subset=["tweet", "label"]).reset_index(drop=True)
    return df


def load_all():
    """Returns train, dev, test DataFrames (official AfriSenti splits)."""
    return load_split("train"), load_split("dev"), load_split("test")


def load_stopwords():
    p = DATA_DIR / "kinyarwanda_stopwords.txt"
    if not p.exists():
        return set()
    # File is from Niyongabo et al. (2020), stored as a Python set literal.
    return {w.lower() for w in re.findall(r"'([^']+)'", p.read_text(encoding="utf-8"))}


if __name__ == "__main__":
    tr, dv, te = load_all()
    for name, d in [("train", tr), ("dev", dv), ("test", te)]:
        print(name, len(d), d.label.value_counts().to_dict())
    for s in tr.tweet.head(5):
        print("RAW  :", s)
        print("CLEAN:", clean(s))
