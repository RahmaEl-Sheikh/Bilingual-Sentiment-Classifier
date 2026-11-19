"""
predict.py
----------
Inference-time module: detects the language of an input text, routes
it to the matching pipeline (English or Arabic), and returns a
sentiment prediction with a confidence score.

Can be used as a library (see `analyze()`) or run directly for a
quick CLI smoke test.
"""

import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text, detect_language  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models")

_CACHE = {}

_FRIENDLY_NAMES = {
    "LogisticRegression": "logistic regression",
    "LinearSVC": "linear SVM",
    "MultinomialNB": "naive bayes",
    "MLPClassifier": "neural net (MLP)",
}


def _load(lang: str):
    if lang not in _CACHE:
        vec_path = os.path.join(MODEL_DIR, f"{lang}_vectorizer.joblib")
        model_path = os.path.join(MODEL_DIR, f"{lang}_model.joblib")
        if not (os.path.exists(vec_path) and os.path.exists(model_path)):
            raise FileNotFoundError(
                f"No trained model found for '{lang}'. Run `python3 src/train.py` first."
            )
        _CACHE[lang] = (joblib.load(vec_path), joblib.load(model_path))
    return _CACHE[lang]


def _confidence(model, X) -> float:
    """Both LogisticRegression and LinearSVC are supported. LogReg has
    predict_proba; LinearSVC doesn't, so we squash its decision_function
    margin through a sigmoid to get a comparable 0-1 confidence value."""
    if hasattr(model, "predict_proba"):
        return float(np.max(model.predict_proba(X)))
    margin = float(model.decision_function(X)[0])
    return float(1 / (1 + np.exp(-abs(margin))))


def analyze(text: str, lang: str = None) -> dict:
    if not text or not text.strip():
        return {"error": "empty input"}

    lang = lang or detect_language(text)
    lang = "ar" if lang == "ar" else "en"  # only two pipelines are trained

    cleaned = clean_text(text, lang=lang)
    vectorizer, model = _load(lang)

    if not cleaned.strip():
        return {
            "error": (
                "That text didn't have enough meaningful words to analyze "
                "(e.g. it was only a link or common filler words) — try something "
                "with a bit more content."
            ),
            "language": "Arabic" if lang == "ar" else "English",
        }

    X = vectorizer.transform([cleaned])
    pred = int(model.predict(X)[0])
    conf = _confidence(model, X)

    model_class = type(model).__name__

    return {
        "language": "Arabic" if lang == "ar" else "English",
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": round(conf, 4),
        "cleaned_text": cleaned,
        "model_name": f"{_FRIENDLY_NAMES.get(model_class, model_class)} (word n-grams)",
    }


if __name__ == "__main__":
    samples = [
        "This movie was a complete waste of time, terrible acting.",
        "I absolutely loved this restaurant, best service ever!",
        "الخدمة كانت سيئة جدا والاكل بارد",
        "تجربة رائعة والفريق محترم جدا وسريع في الرد",
    ]
    for s in samples:
        print(s)
        print(" ->", analyze(s))
        print()
