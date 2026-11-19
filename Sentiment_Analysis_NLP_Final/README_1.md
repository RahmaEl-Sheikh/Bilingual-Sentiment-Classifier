# Bilingual Sentiment Analyzer — Arabic & English

A sentiment analysis engine that handles **Arabic and English independently and well**, rather than bolting Arabic onto an English-first pipeline. Paste a review, tweet, or piece of feedback in either language and it tells you whether the writer is happy or not, with a confidence score and a transparent look at exactly what the model read.

![Python](https://img.shields.io/badge/python-3.12-blue) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange) ![Flask](https://img.shields.io/badge/flask-3.x-black) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

```
"الفندق رائع جدا والموظفين متعاونين"                →  Positive · 97%
"This film was a waste of time, the plot made no sense" →  Negative · 79%
```

## Why two languages, and why it's harder than it looks

Most tutorial sentiment-analysis projects handle one language and stop. Arabic is genuinely harder than English here: it's morphologically rich (one root can surface as dozens of inflected forms), orthographic conventions vary between writers (إ/أ/آ/ا are frequently used interchangeably, as are ى/ي), and real-world text is a mix of Modern Standard Arabic and regional dialect. A pipeline built for English and run on Arabic as an afterthought typically underperforms — this project measures that gap and closes most of it.

**The concrete finding — and a correction along the way:** an early experiment on a smaller, noisier Arabic dataset (political tweets, ASTD) showed character-level TF-IDF outperforming word-level (F1 0.62 → 0.71, +9 points) — a plausible result, since Arabic's morphology means many surface word-forms share a root that only character n-grams can see. But after switching to the larger, cleaner HARD hotel-reviews dataset actually used for the production model, re-running the same comparison reversed the result: word-level n-grams score marginally *higher* (91.6% vs 91.2% F1). With enough clean in-domain data, word-level vocabulary is already comprehensive, and character n-grams just add noisy extra dimensions. The pipeline ships with whichever representation the evidence actually supports for its real training data — word n-grams for both languages — rather than keeping an earlier conclusion that no longer held once the dataset changed. Both experiments are in [`notebooks/sentiment_analysis.ipynb`](notebooks/sentiment_analysis.ipynb) Section 5.2, and the full per-model numbers are in [`models/metrics.json`](models/metrics.json).

## Results

| Language | Model (best of 4) | Train / Test size | Accuracy | F1 |
|---|---|---|---|---|
| English | Linear SVM, word n-grams (1,2) | 1,600 / 400 | 87.5% | **0.877** |
| Arabic | Logistic Regression, word n-grams (1,2) | 12,709 / 3,178 | 91.6% | **0.916** |

Both numbers are held-out test-set results from a stratified 80/20 split, not training accuracy. Four algorithms (Logistic Regression, Linear SVM, Multinomial Naive Bayes, and a small neural net / MLP) were trained and compared per language; full results for all four are in `models/metrics.json`.

<table>
<tr>
<td><img src="models/en_confusion_matrix.png" width="330"></td>
<td><img src="models/ar_confusion_matrix.png" width="330"></td>
</tr>
</table>

<img src="models/model_comparison.png" width="600">

## What makes the pipeline non-trivial

- **A real Arabic normalization pipeline** (`src/preprocess.py`): strips diacritics and tatweel, unifies alef/hamza/ya variants, collapses emphatic letter elongation ("رائععع" → "رائع"), and filters a curated Egyptian-dialect-aware stopword list — this is the difference between a toy regex and something that survives real user-generated text.
- **Feature representation chosen by evidence, and re-checked after the data changed.** Both languages end up using word-level TF-IDF — not because that was assumed, but because character-level features were tested for Arabic, won decisively on a smaller/noisier dataset, then were re-tested after the training data changed and *lost*. The final choice tracks the second result, not the first.
- **Model selection by F1, not accuracy**, because the Arabic dataset is naturally imbalanced — optimizing for raw accuracy would quietly reward a model that just predicts the majority class.
- **Dataset choice was deliberate and revised.** An early version trained Arabic sentiment on political tweets (ASTD); testing showed it generalized poorly to everyday service/product feedback and pulled in politically-charged training content, so the project moved to the HARD hotel-reviews corpus — a closer match to real customer-feedback use cases, at 8x the data. That kind of mid-project correction is exactly what's documented in the commit history / code comments rather than hidden.
- **No black-box inference.** Every prediction returns the exact cleaned tokens the model scored, so a user (or an interviewer) can see *why* a prediction came out the way it did.

## Notebook

[`notebooks/sentiment_analysis.ipynb`](notebooks/sentiment_analysis.ipynb) is a self-contained, already-executed walkthrough — data loading, EDA, preprocessing (with before/after examples), the word-vs-character n-gram experiment, model training and comparison, evaluation, and error analysis — with every plot and result baked in, so it opens and reads without needing to be re-run. Open it in Jupyter (`pip install jupyter`, then `jupyter notebook`), VS Code's notebook viewer, or by uploading it to [Google Colab](https://colab.research.google.com/) directly (it re-fetches its own data via `git clone`/NLTK, so it doesn't depend on the rest of this repo).

## Try it

```bash
git clone <your-repo-url>
cd sentiment-analyzer
pip install -r requirements.txt
python3 app.py
# → open http://127.0.0.1:5000
```

The trained models are included in `models/`, so it runs immediately — no training step required. The web UI lets you paste text in either language (or click a sample) and see the sentiment, confidence, detected language, and the cleaned tokens the model actually used.

To retrain from scratch (e.g. on your own labeled data):
```bash
python3 src/prepare_data.py   # rebuilds data/processed/*.csv
python3 src/train.py          # retrains + re-evaluates + overwrites models/
```
`prepare_data.py` auto-downloads the English corpus via NLTK. The Arabic source (HARD dataset) needs to be fetched once — see [Data sources](#data-sources) below.

## API

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "الخدمة ممتازة والفريق محترم جدا"}'
```
```json
{
  "language": "Arabic",
  "sentiment": "positive",
  "confidence": 0.9769,
  "cleaned_text": "الخدمه ممتازه والفريق محترم جدا",
  "model_name": "logistic regression (word n-grams)"
}
```
Language is auto-detected from the input, so a single endpoint handles both languages — no `lang` parameter needed.

## Project structure

```
├── app.py                    # Flask app: serves the UI + /api/analyze, /api/metrics, /api/health
├── requirements.txt
├── src/
│   ├── preprocess.py         # Arabic + English cleaning/normalization, language detection
│   ├── prepare_data.py       # loads raw corpora → data/processed/*.csv
│   ├── train.py              # trains + compares 4 models per language, saves best + metrics
│   └── predict.py            # inference: detect language → route → predict + confidence
├── data/processed/           # cleaned, ready-to-train CSVs (included)
├── models/                   # trained models, vectorizers, metrics.json, plots (included)
├── notebooks/sentiment_analysis.ipynb  # full walkthrough: EDA, preprocessing, the
│                              # word-vs-char n-gram experiment, training, error analysis
├── templates/index.html      # bilingual RTL/LTR-aware UI
└── static/{style.css,script.js}
```

## Data sources

| Language | Dataset | Size used | Citation |
|---|---|---|---|
| English | NLTK `movie_reviews` (Pang & Lee) | 2,000 reviews | Pang, B. & Lee, L. (2004) |
| Arabic | HARD — Hotel Arabic-Reviews Dataset | 16,000 reviews (balanced sample) | Elnagar, A., Khalifa, Y.S., Einea, A. (2018) |

To reproduce the Arabic data from source (cross-platform — works on Windows too, since it avoids `unzip`/`git`):
```bash
python3 -c "
import os, urllib.request, zipfile
os.makedirs('data/hard_raw/data', exist_ok=True)
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/elnagara/HARD-Arabic-Dataset/master/data/balanced-reviews.zip',
    'data/hard_raw/data/balanced-reviews.zip')
zipfile.ZipFile('data/hard_raw/data/balanced-reviews.zip').extractall('data/hard_raw/data')
"
```

## Known limitations

- Both models are trained on **written reviews** (hotel/film). They generalize reasonably to other opinionated text (comments, feedback forms) but confidence will be lower on very different domains (e.g. legal or medical text) — retraining on in-domain labeled data via `prepare_data.py` + `train.py` is the documented path to fix that.
- Binary positive/negative only — no neutral class. Text that's genuinely neutral or mixed will be forced into whichever side edges out, which is why the UI always surfaces the confidence score alongside the label rather than the label alone.
- The language detector is a lightweight script-based heuristic (Arabic-vs-Latin character counting), not a full language-ID model — it's reliable for Arabic/English but wouldn't extend to a third language without more work.

## Possible extensions

- Fine-tune a compact transformer (e.g. AraBERT for Arabic, DistilBERT for English) and compare against these classical baselines — the modular `train.py` structure is designed to make swapping in a new model straightforward.
- Add a neutral/mixed class using 3-star HARD reviews (currently excluded).
- Batch-mode endpoint for scoring a CSV of reviews at once (e.g. for a client's product review export).



