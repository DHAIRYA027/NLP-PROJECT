from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.preprocess import preprocess_to_string

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "intent_model.joblib"

AVAILABLE_MODELS = {
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(), cv=3),
}


def build_pipeline(model_name: str = "naive_bayes") -> Pipeline:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(AVAILABLE_MODELS)}")
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
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


class IntentClassifier:
    def __init__(self, model_name: str = "naive_bayes"):
        self.model_name = model_name
        self.pipeline = build_pipeline(model_name)
        self.responses: dict[str, list[str]] = {}

    def train(self, intents: dict | None = None):
        intents = intents or load_intents()
        texts, labels = build_training_data(intents)
        self.pipeline.fit(texts, labels)
        self.responses = {i["tag"]: i["responses"] for i in intents["intents"]}
        return self

    def predict(self, text: str) -> tuple[str, float]:
        processed = preprocess_to_string(text)
        probs = self.pipeline.predict_proba([processed])[0]
        classes = self.pipeline.classes_
        best_idx = probs.argmax()
        return classes[best_idx], float(probs[best_idx])

    def response_for(self, tag: str) -> str:
        import random
        return random.choice(self.responses[tag])

    def save(self, path: Path = MODEL_PATH):
        joblib.dump(
            {"pipeline": self.pipeline, "responses": self.responses, "model_name": self.model_name},
            path,
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "IntentClassifier":
        data = joblib.load(path)
        clf = cls(model_name=data.get("model_name", "naive_bayes"))
        clf.pipeline = data["pipeline"]
        clf.responses = data["responses"]
        return clf


if __name__ == "__main__":
    clf = IntentClassifier().train()
    for query in [
        "hey what's up",
        "when is the library open",
        "how much do I need to pay this semester",
        "tell me a joke",
    ]:
        tag, confidence = clf.predict(query)
        print(f"{query!r:55} -> {tag:15} ({confidence:.2f})")
