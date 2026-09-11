from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.intent_classifier import AVAILABLE_MODELS, build_pipeline, build_training_data, load_intents
from src.ngram_model import NgramLanguageModel, load_sentences

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate_classifiers() -> dict:
    intents = load_intents()
    texts, labels = build_training_data(intents)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = {}
    print("=== Intent Classifier Comparison: 3-fold Cross-Validation ===")
    for model_name in AVAILABLE_MODELS:
        pipeline = build_pipeline(model_name)
        predictions = cross_val_predict(pipeline, texts, labels, cv=cv)
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        results[model_name] = {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
        print(f"\n--- {model_name} ---")
        print(classification_report(labels, predictions, zero_division=0))

        if model_name == "naive_bayes":
            RESULTS_DIR.mkdir(exist_ok=True)
            fig, ax = plt.subplots(figsize=(9, 9))
            ConfusionMatrixDisplay.from_predictions(
                labels, predictions, xticks_rotation=45, ax=ax, colorbar=False
            )
            plt.tight_layout()
            out_path = RESULTS_DIR / "confusion_matrix.png"
            plt.savefig(out_path, dpi=150)
            print(f"Confusion matrix saved to {out_path}")

    return results


def evaluate_language_model() -> dict:
    sentences = load_sentences()
    split = int(len(sentences) * 0.85)
    train_sents, test_sents = sentences[:split], sentences[split:]

    print("\n=== N-gram Language Model: Perplexity by order and smoothing ===")
    results = {}
    for n in (1, 2, 3):
        results[str(n)] = {}
        for smoothing in ("add_k", "interpolation"):
            lm = NgramLanguageModel(n=n, add_k=1.0, smoothing=smoothing).fit(train_sents)
            pp = lm.perplexity(test_sents)
            results[str(n)][smoothing] = pp
            print(f"n={n} smoothing={smoothing:13} perplexity={pp:.2f}")

    return results


if __name__ == "__main__":
    classifier_results = evaluate_classifiers()
    lm_results = evaluate_language_model()

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"classifiers": classifier_results, "language_model": lm_results}, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_DIR / 'metrics.json'}")
