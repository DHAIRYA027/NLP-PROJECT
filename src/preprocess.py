"""Text preprocessing: cleaning, tokenization, stopword removal, lemmatization."""

from __future__ import annotations

import hashlib
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_REQUIRED_NLTK_DATA = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

# Negations change meaning ("wifi is not working"), so they are kept even though
# NLTK lists them as stopwords. Question words (what/how/when/...) are deliberately
# still removed: keeping them raised CV accuracy by under one point (within noise)
# but made off-topic queries like "what is the capital of france" look like
# in-scope ones, cutting the dev out-of-scope rejection score from 0.87 to 0.79.
KEPT_STOPWORDS = frozenset({"no", "not", "nor"})


def _nltk_resource_available(path: str) -> bool:
    # NLTK ships some corpora (wordnet, omw-1.4) only as .zip files, and
    # nltk.data.find() does not resolve the extension-less name to them.
    for candidate in (path, f"{path}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_data() -> None:
    """Download the NLTK resources this project needs, but only if they are missing."""
    for path, package in _REQUIRED_NLTK_DATA:
        if not _nltk_resource_available(path):
            nltk.download(package, quiet=True)


ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stopwords = frozenset(stopwords.words("english")) - KEPT_STOPWORDS


def config_signature() -> str:
    """Short hash of the preprocessing configuration, used to invalidate cached models."""
    payload = "|".join(sorted(_stopwords))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return word_tokenize(clean_text(text))


def preprocess(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> list[str]:
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _stopwords]
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def preprocess_to_string(text: str, **kwargs) -> str:
    return " ".join(preprocess(text, **kwargs))


if __name__ == "__main__":
    sample = "Hi! When does the Library open on Sundays?"
    print("raw:      ", sample)
    print("tokens:   ", tokenize(sample))
    print("processed:", preprocess(sample))
