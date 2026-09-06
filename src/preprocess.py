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


def ensure_nltk_data():
    for path, package in _REQUIRED_NLTK_DATA:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))


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
