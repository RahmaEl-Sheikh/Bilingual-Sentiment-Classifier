"""
app.py
------
Flask backend for the bilingual (Arabic + English) sentiment analyzer.

Routes:
  GET  /            -> the web UI
  POST /api/analyze -> {"text": "..."} -> sentiment JSON
  GET  /api/health   -> readiness check (are both models loaded?)

Run:
  python3 app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import analyze  # noqa: E402

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > 4000:
        return jsonify({"error": "Text is too long (max 4000 characters)."}), 400

    try:
        result = analyze(text)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    """Serves the real numbers from the last training run (models/metrics.json)
    so the UI never shows a hand-typed figure that could drift from reality."""
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "no metrics yet — run src/train.py"}), 404
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({
        "en_f1": data["en"]["all_results"][data["en"]["best_model"]]["f1"],
        "en_model": data["en"]["best_model"],
        "en_train_size": data["en"]["dataset_size"],
        "ar_f1": data["ar"]["all_results"][data["ar"]["best_model"]]["f1"],
        "ar_model": data["ar"]["best_model"],
        "ar_train_size": data["ar"]["dataset_size"],
    })


@app.route("/api/health")
def health():
    en_ready = os.path.exists(os.path.join(MODEL_DIR, "en_model.joblib"))
    ar_ready = os.path.exists(os.path.join(MODEL_DIR, "ar_model.joblib"))
    return jsonify({"english_model": en_ready, "arabic_model": ar_ready})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Bilingual Sentiment Analyzer running -> http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
