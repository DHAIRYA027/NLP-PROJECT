from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.intent_classifier import build_training_data, load_intents
from src.ngram_model import NgramLanguageModel, load_sentences

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate_classifier():
    intents = load_intents()
    texts, labels = build_training_data(intents)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB()),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    predictions = cross_val_predict(pipeline, texts, labels, cv=cv)

    print("=== Intent Classifier: 3-fold Cross-Validation ===")
    print(classification_report(labels, predictions))

    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        labels, predictions, xticks_rotation=45, ax=ax, colorbar=False
    )
    plt.tight_layout()
    out_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")


def evaluate_language_model():
    sentences = load_sentences()
    split = int(len(sentences) * 0.85)
    train_sents, test_sents = sentences[:split], sentences[split:]

    print("\n=== N-gram Language Model: Perplexity ===")
    for n in (1, 2, 3):
        lm = NgramLanguageModel(n=n, add_k=1.0).fit(train_sents)
        pp = lm.perplexity(test_sents)
        print(f"n={n}: perplexity = {pp:.2f}")


if __name__ == "__main__":
    evaluate_classifier()
    evaluate_language_model()
