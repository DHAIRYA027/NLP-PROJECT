"""Intent classification: TF-IDF features plus a choice of linear classifiers."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import config_signature, preprocess_to_string

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "intent_model.joblib"

AVAILABLE_MODELS = {
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, C=10),
    "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(), cv=3),
}
DEFAULT_MODEL = "logistic_regression"

# "word"      : word-level TF-IDF only (the original baseline).
# "word+char" : word TF-IDF plus character 2-5-gram TF-IDF. The character features
#               give partial credit to typos and unseen word forms ("libary",
#               "timings") that a pure word vocabulary scores as all-zero.
FEATURE_SETS = ("word", "word+char")
DEFAULT_FEATURES = "word+char"

UNKNOWN_INTENT = "unknown"


def build_features(feature_set: str = DEFAULT_FEATURES):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Choose from {list(FEATURE_SETS)}")
    word = TfidfVectorizer()
    if feature_set == "word":
        return word
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    return FeatureUnion([("word", word), ("char", char)])


def build_pipeline(model_name: str = DEFAULT_MODEL, features: str = DEFAULT_FEATURES) -> Pipeline:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(AVAILABLE_MODELS)}")
    return Pipeline([
        ("features", build_features(features)),
        ("clf", AVAILABLE_MODELS[model_name]()),
    ])


def load_intents(path: Path = DATA_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_training_data(intents: dict) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            texts.append(preprocess_to_string(pattern))
            labels.append(intent["tag"])
    return texts, labels


def fingerprint(intents: dict, model_name: str, features: str) -> str:
    """Identify everything a trained model depends on, so stale caches are detected."""
    parts = [
        json.dumps(intents, sort_keys=True),
        repr(build_pipeline(model_name, features)),
        sklearn.__version__,
        config_signature(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class IntentClassifier:
    def __init__(self, model_name: str = DEFAULT_MODEL, features: str = DEFAULT_FEATURES):
        self.model_name = model_name
        self.features = features
        self.pipeline = build_pipeline(model_name, features)
        self.responses: dict[str, list[str]] = {}
        self.labels: dict[str, str] = {}
        self.smalltalk: set[str] = set()
        self.fingerprint: str | None = None

    def train(self, intents: dict | None = None) -> IntentClassifier:
        intents = intents or load_intents()
        texts, labels = build_training_data(intents)
        self.pipeline.fit(texts, labels)
        self.responses = {i["tag"]: i["responses"] for i in intents["intents"]}
        self.labels = {i["tag"]: i.get("label", i["tag"].replace("_", " ")) for i in intents["intents"]}
        self.smalltalk = {i["tag"] for i in intents["intents"] if i.get("smalltalk")}
        self.fingerprint = fingerprint(intents, self.model_name, self.features)
        return self

    @property
    def topics(self) -> list[str]:
        """Human-readable names of the substantive (non-smalltalk) intents."""
        return [label for tag, label in self.labels.items() if tag not in self.smalltalk]

    def _probabilities(self, text: str):
        """Class probabilities, or None if the text shares no features with the training data.

        An all-zero feature vector carries no evidence; a linear model would then
        answer from its class prior alone (favouring the largest class), so we
        report "no evidence" explicitly instead of a misleading prediction.
        """
        features = self.pipeline[:-1].transform([preprocess_to_string(text)])
        if features.nnz == 0:
            return None
        return self.pipeline[-1].predict_proba(features)[0]

    def predict_top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        probs = self._probabilities(text)
        if probs is None:
            return []
        classes = self.pipeline.classes_
        order = probs.argsort()[::-1][:k]
        return [(str(classes[i]), float(probs[i])) for i in order]

    def predict(self, text: str) -> tuple[str, float]:
        top = self.predict_top(text, k=1)
        return top[0] if top else (UNKNOWN_INTENT, 0.0)

    def response_for(self, tag: str, rng: random.Random | None = None) -> str:
        return (rng or random).choice(self.responses[tag])

    def save(self, path: Path = MODEL_PATH) -> None:
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "responses": self.responses,
                "labels": self.labels,
                "smalltalk": self.smalltalk,
                "model_name": self.model_name,
                "features": self.features,
                "fingerprint": self.fingerprint,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> IntentClassifier:
        data = joblib.load(path)
        clf = cls(
            model_name=data.get("model_name", DEFAULT_MODEL),
            features=data.get("features", DEFAULT_FEATURES),
        )
        clf.pipeline = data["pipeline"]
        clf.responses = data["responses"]
        clf.labels = data.get("labels", {})
        clf.smalltalk = data.get("smalltalk", set())
        clf.fingerprint = data.get("fingerprint")
        return clf

    @classmethod
    def load_or_train(
        cls,
        model_name: str = DEFAULT_MODEL,
        features: str = DEFAULT_FEATURES,
        path: Path = MODEL_PATH,
        intents_path: Path = DATA_PATH,
    ) -> IntentClassifier:
        """Return the cached model if it matches the current data and config, else retrain."""
        intents = load_intents(intents_path)
        expected = fingerprint(intents, model_name, features)
        path = Path(path)
        if path.exists():
            try:
                cached = cls.load(path)
                if cached.fingerprint == expected:
                    return cached
                logger.info("Cached intent model is stale; retraining.")
            except Exception:  # corrupt file or incompatible scikit-learn version
                logger.warning("Could not load cached intent model; retraining.", exc_info=True)
        clf = cls(model_name, features).train(intents)
        try:
            clf.save(path)
        except OSError:
            logger.warning("Could not write model cache to %s", path)
        return clf


if __name__ == "__main__":
    clf = IntentClassifier().train()
    for query in [
        "hey what's up",
        "when is the library open",
        "libary timings on saturday",
        "how much do I need to pay this semester",
        "tell me a joke",
        "???",
    ]:
        tag, confidence = clf.predict(query)
        print(f"{query!r:45} -> {tag:16} ({confidence:.2f})")
