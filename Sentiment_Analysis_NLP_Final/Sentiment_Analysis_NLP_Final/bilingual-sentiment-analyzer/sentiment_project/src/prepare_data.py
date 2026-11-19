"""
prepare_data.py
----------------
Loads the raw English (NLTK movie_reviews) and Arabic (ASTD tweets)
corpora, cleans them with preprocess.py, and writes tidy CSVs to
data/processed/ ready for training.

Run:  python3 src/prepare_data.py
"""

import os
import sys

import nltk
import pandas as pd

try:
    nltk.data.find("corpora/movie_reviews")
except LookupError:
    nltk.download("movie_reviews", quiet=True)
from nltk.corpus import movie_reviews  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_HARD = os.path.join(HERE, "data", "hard_raw", "data", "balanced-reviews.txt")
OUT_DIR = os.path.join(HERE, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

AR_SAMPLE_PER_CLASS = 8000  # balanced subsample size (per class) for speed


def load_english() -> pd.DataFrame:
    """NLTK's movie_reviews corpus: 1000 pos + 1000 neg full-length
    movie reviews (Pang & Lee, 2004) -- a standard sentiment benchmark."""
    rows = []
    for fileid in movie_reviews.fileids():
        label = movie_reviews.categories(fileid)[0]  # 'pos' or 'neg'
        text = movie_reviews.raw(fileid)
        rows.append({"text": text, "label": 1 if label == "pos" else 0})
    df = pd.DataFrame(rows)
    print(f"[EN] loaded {len(df)} reviews  (pos={sum(df.label==1)}, neg={sum(df.label==0)})")
    return df


def load_arabic() -> pd.DataFrame:
    """HARD -- Hotel Arabic-Reviews Dataset (Elnagar et al., 2018).
    Real Booking.com hotel reviews in MSA + dialectal Arabic. Sentiment
    is derived from the 1-5 star rating using the standard split used
    in the literature (e.g. the AraBERT paper): 4-5 stars -> positive,
    1-2 stars -> negative (this "balanced" release already excludes the
    ambiguous 3-star/neutral reviews). Chosen over the alternative ASTD
    tweets corpus because hotel/service reviews are a much closer match
    to the real-world 'customer feedback' use case this tool targets,
    and keeps the training data free of the politically-charged content
    ASTD's tweets contain.
    """
    raw = pd.read_csv(RAW_HARD, sep="\t", encoding="utf-16")
    raw["label"] = raw["rating"].map({1: 0, 2: 0, 4: 1, 5: 1})
    raw = raw.dropna(subset=["label", "review"])
    raw["label"] = raw["label"].astype(int)
    raw = raw.rename(columns={"review": "text"})[["text", "label"]]

    # Balanced random subsample per class, for a lean/fast-to-train artifact
    pos = raw[raw.label == 1].sample(
        n=min(AR_SAMPLE_PER_CLASS, (raw.label == 1).sum()), random_state=42
    )
    neg = raw[raw.label == 0].sample(
        n=min(AR_SAMPLE_PER_CLASS, (raw.label == 0).sum()), random_state=42
    )
    df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[AR] loaded {len(df)} hotel reviews (of {len(raw)} available)  "
          f"(pos={sum(df.label==1)}, neg={sum(df.label==0)})")
    return df


def build(lang: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["clean_text"] = df["text"].apply(lambda t: clean_text(t, lang=lang))
    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[{lang.upper()}] dropped {dropped} rows that were empty after cleaning")
    return df


def main():
    en_df = build("en", load_english())
    ar_df = build("ar", load_arabic())

    en_path = os.path.join(OUT_DIR, "english_reviews.csv")
    ar_path = os.path.join(OUT_DIR, "arabic_reviews.csv")
    en_df.to_csv(en_path, index=False, encoding="utf-8-sig")
    ar_df.to_csv(ar_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {en_path}  ({len(en_df)} rows)")
    print(f"Saved: {ar_path}  ({len(ar_df)} rows)")


if __name__ == "__main__":
    main()
