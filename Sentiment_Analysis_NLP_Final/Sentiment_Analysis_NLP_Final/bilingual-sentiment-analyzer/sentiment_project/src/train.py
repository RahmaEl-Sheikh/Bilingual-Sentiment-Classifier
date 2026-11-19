"""
train.py
--------
Trains and evaluates sentiment classifiers for English and Arabic
independently, compares several algorithms per language, keeps the
best one, and persists:
  - models/{lang}_vectorizer.joblib
  - models/{lang}_model.joblib
  - models/{lang}_confusion_matrix.png
  - models/metrics.json           (full comparison table, for the README)
  - models/model_comparison.png   (bar chart, EN vs AR, per algorithm)

Run:  python3 src/train.py
"""

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data", "processed")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

def make_vectorizer(lang: str) -> TfidfVectorizer:
    """Both languages use word n-grams. This wasn't assumed -- character
    n-grams were tested for Arabic on the assumption that Arabic's rich
    morphology would make them win (they do, dramatically, on a smaller/
    noisier dataset: an earlier exploratory pass on the ASTD tweets corpus
    showed word-level TF-IDF plateauing at F1=0.62 vs char_wb(2,5) reaching
    F1=0.72). But re-tested on the actual production dataset (HARD, 16k
    hotel reviews) the effect reverses: word n-grams score marginally
    *higher* than char n-grams (0.916 vs 0.912 F1, logistic regression).
    With a large, clean, single-domain corpus, word-level vocabulary is
    already comprehensive enough that character n-grams just add noisy
    extra dimensions without upside. Reported here rather than silently
    kept as char-level, because shipping a design decision that no longer
    matches the evidence is worse than not having tested it at all. See
    notebooks/sentiment_analysis.ipynb Section 5.2 for the side-by-side
    comparison this is based on."""
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=10000,
    )


CANDIDATES = {
    "logistic_regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", C=10, random_state=RANDOM_STATE
    ),
    "linear_svm": LinearSVC(class_weight="balanced", C=1, random_state=RANDOM_STATE),
    "multinomial_nb": MultinomialNB(),
    "neural_net_mlp": MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        alpha=1e-4,
        max_iter=200,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
}


def evaluate(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }


def run_for_language(lang: str, csv_name: str, display_name: str):
    print(f"\n{'=' * 60}\n{display_name}\n{'=' * 60}")
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )
    print(f"train={len(X_train)}  test={len(X_test)}  "
          f"(test pos={sum(y_test == 1)}, test neg={sum(y_test == 0)})")

    vectorizer = make_vectorizer(lang)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    results = {}
    fitted_models = {}
    for name, clf in CANDIDATES.items():
        t0 = time.time()
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xte)
        metrics = evaluate(y_test, pred)
        metrics["train_seconds"] = round(time.time() - t0, 2)
        results[name] = metrics
        fitted_models[name] = clf
        print(f"  {name:22s} acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
              f"({metrics['train_seconds']}s)")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]
    print(f"  -> best: {best_name}")

    # Persist winning vectorizer + model
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, f"{lang}_vectorizer.joblib"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{lang}_model.joblib"))

    # Confusion matrix plot for the winner
    cm = confusion_matrix(y_test, best_model.predict(Xte))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{display_name}\nbest model: {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, f"{lang}_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    return {
        "language": display_name,
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "vocab_size": len(vectorizer.vocabulary_),
        "best_model": best_name,
        "all_results": results,
    }


def plot_comparison(all_metrics: dict):
    algos = list(CANDIDATES.keys())
    en_f1 = [all_metrics["en"]["all_results"][a]["f1"] for a in algos]
    ar_f1 = [all_metrics["ar"]["all_results"][a]["f1"] for a in algos]

    x = np.arange(len(algos))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, en_f1, width, label="English")
    ax.bar(x + width / 2, ar_f1, width, label="Arabic")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in algos], fontsize=8)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison — English vs Arabic sentiment classifiers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    all_metrics = {}
    all_metrics["en"] = run_for_language("en", "english_reviews.csv", "English (movie reviews)")
    all_metrics["ar"] = run_for_language("ar", "arabic_reviews.csv", "Arabic (HARD hotel reviews)")

    plot_comparison(all_metrics)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nSaved models, plots, and metrics.json to {MODEL_DIR}")


if __name__ == "__main__":
    main()
