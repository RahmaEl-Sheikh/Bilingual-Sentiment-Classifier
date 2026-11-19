"""
preprocess.py
--------------
Text cleaning & normalization utilities for the bilingual
(Arabic + English) sentiment analysis pipeline.

Two independent cleaning pipelines are provided because Arabic and
English need fundamentally different treatment (script, morphology,
diacritics, stopword lists, elongation patterns, etc.).
"""

import re
import string

import nltk


def _ensure_nltk_data():
    """First-run convenience: fetch the small NLTK resources this module
    needs if they aren't already present, instead of making the user run
    a separate manual download step."""
    required = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for path, pkg in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------
ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def detect_language(text: str) -> str:
    """Cheap, dependency-free language detector.

    Counts Arabic-script characters vs Latin letters. Good enough for
    routing short-to-medium review/feedback text between the two
    pipelines; not meant as a general-purpose language ID system.
    """
    if not text:
        return "en"
    arabic_chars = len(ARABIC_RANGE.findall(text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    if arabic_chars == 0 and latin_chars == 0:
        return "en"
    return "ar" if arabic_chars >= latin_chars else "en"


# --------------------------------------------------------------------------
# Arabic pipeline
# --------------------------------------------------------------------------
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # tashkeel + tatweel
ARABIC_PUNCT_EXTRA = "،؛؟”“«»…"

# A small, curated Arabic stopword list (MSA + common Egyptian-dialect
# function words). Kept local instead of pulling a large external list
# so behaviour is transparent and doesn't depend on an extra download.
ARABIC_STOPWORDS = set("""
من في على و او أو ان إن أن ثم إلى الى عن مع هذا هذه ذلك تلك هؤلاء التي الذي
الذين كان كانت يكون تكون لم لن لا ما لكن غير حتى إذا اذا كل بعض بين قد
هو هي هم انا أنا انت أنت انتم أنتم احنا إحنا احنا ده دي دى دة اللي علشان
عشان عايز عايزة عاوز عاوزة كده كدا بس فقط ايه إيه ليه لية فين امتى إمتى
ازاي إزاي مش ايوه أيوه لا آه اه يا يعني برضو كمان لسه دلوقتي هنا هناك
""".split())


def normalize_arabic(text: str) -> str:
    """Normalize Arabic orthographic variants that are semantically
    identical but surface as different tokens (this alone typically
    reduces an Arabic TF-IDF vocabulary by 15-25%)."""
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)
    return text


def reduce_elongation(text: str) -> str:
    """Collapse character repetition used for emphasis in social text,
    e.g. 'حلوووة' -> 'حلوة', 'sooooo good' -> 'soo good'."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def clean_arabic(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                       # mentions
    text = re.sub(r"#", " ", text)                          # keep hashtag word, drop symbol
    text = normalize_arabic(text)
    text = reduce_elongation(text)
    text = re.sub(f"[{re.escape(ARABIC_PUNCT_EXTRA)}]", " ", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)          # keep Arabic letters only
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in ARABIC_STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# English pipeline
# --------------------------------------------------------------------------
_EN_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_english(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"<.*?>", " ", text)                      # html tags (movie_reviews has some)
    text = reduce_elongation(text)
    text = text.translate(str.maketrans("", "", string.digits))
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(t) for t in tokens
        if t not in _EN_STOPWORDS and len(t) > 1
    ]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------
def clean_text(text: str, lang: str = None) -> str:
    if lang is None:
        lang = detect_language(text)
    return clean_arabic(text) if lang == "ar" else clean_english(text)


if __name__ == "__main__":
    samples = [
        ("The movie was absolutely wonderful, I loved it!!!", "en"),
        ("الفيلم ده تحفففففة حقيقي وممثلين رائعين", "ar"),
        ("خدمة العملاء سيئة جدااا ومحدش بيرد", "ar"),
        ("Worst experience ever, do not recommend this place.", "en"),
    ]
    for text, lang in samples:
        print(f"[{lang}] {text!r}\n  -> {clean_text(text, lang)!r}\n")
