"""Reproducible evaluation: `python -m src.evaluate`.

Writes results/metrics.json, results/confusion_matrix.png and
results/threshold_curve.png. Everything is seeded, so repeated runs give
identical numbers.

Evaluation protocol (see REPORT.md):
  * Classifier model/feature comparison: 5x repeated stratified 3-fold CV on the
    training patterns (mean +- std over the 15 folds).
  * Held-out test: 42 fresh in-scope questions and 52 out-of-scope queries from
    data/eval_queries.json. Nothing was tuned on the test split.
  * Confidence threshold: chosen on in-scope CV probabilities plus the
    out-of-scope *dev* split, then reported on the *test* split.
  * Language model: 5-fold CV over corpus sentences; add-k and interpolation
    hyperparameters are tuned on a dev slice of each training portion.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from src.intent_classifier import (
    AVAILABLE_MODELS,
    DEFAULT_FEATURES,
    DEFAULT_MODEL,
    FEATURE_SETS,
    IntentClassifier,
    build_pipeline,
    build_training_data,
    load_intents,
)
from src.ngram_model import (
    NgramLanguageModel,
    interpolation_weight_grid,
    load_sentences,
    tune_add_k,
    tune_interpolation_weights,
)

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
EVAL_PATH = ROOT / "data" / "eval_queries.json"

SEED = 42
# Thresholds shown in the held-out operating curve. Reported for transparency only;
# the shipped threshold is chosen on dev data, never from this curve.
OPERATING_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# The configuration the project started from (Naive Bayes, word TF-IDF, threshold 0.2),
# scored here with the *current* preprocessing. REPORT.md also quotes the exact original
# code's numbers, measured separately on commit 470c19c.
BASELINE_CONFIG = ("naive_bayes", "word", 0.2)
# Interpolation weights that were set by hand before they were tuned.
HAND_SET_WEIGHTS = {1: (1.0,), 2: (0.25, 0.75), 3: (0.1, 0.3, 0.6)}


def load_eval_queries(path: Path = EVAL_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean_std(values) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


# --------------------------------------------------------------------------- classifier


def evaluate_classifiers(texts: list[str], labels: list[str]) -> dict:
    """Repeated CV for every (model, feature set) pair."""
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=SEED)
    results: dict = {}
    for model_name in AVAILABLE_MODELS:
        results[model_name] = {}
        for features in FEATURE_SETS:
            scores = cross_validate(
                build_pipeline(model_name, features), texts, labels, cv=cv,
                scoring=("accuracy", "f1_macro"),
            )
            results[model_name][features] = {
                "accuracy": _mean_std(scores["test_accuracy"]),
                "macro_f1": _mean_std(scores["test_f1_macro"]),
            }
    return results


def zero_evidence_diagnostic(texts: list[str], labels: list[str]) -> dict:
    """How many CV errors come from test items that share no feature with the training fold?

    Such items give the classifier nothing to go on, so it answers from the class
    prior (the biggest class, `greeting`). This is the main source of the
    `* -> greeting` confusions.
    """
    texts_arr, labels_arr = np.array(texts, dtype=object), np.array(labels)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    out = {}
    for features in FEATURE_SETS:
        zero = errors = errors_zero = errors_to_greeting = errors_to_greeting_zero = 0
        for train_idx, test_idx in cv.split(texts_arr, labels_arr):
            pipe = build_pipeline(DEFAULT_MODEL, features).fit(texts_arr[train_idx], labels_arr[train_idx])
            feats = pipe[:-1].transform(texts_arr[test_idx])
            no_evidence = np.asarray(feats.sum(axis=1)).ravel() == 0
            pred = pipe.predict(texts_arr[test_idx])
            wrong = pred != labels_arr[test_idx]
            to_greeting = wrong & (pred == "greeting")
            zero += int(no_evidence.sum())
            errors += int(wrong.sum())
            errors_zero += int((wrong & no_evidence).sum())
            errors_to_greeting += int(to_greeting.sum())
            errors_to_greeting_zero += int((to_greeting & no_evidence).sum())
        out[features] = {
            "items_with_no_known_features": zero,
            "errors": errors,
            "errors_with_no_known_features": errors_zero,
            "errors_predicted_greeting": errors_to_greeting,
            "errors_predicted_greeting_with_no_known_features": errors_to_greeting_zero,
        }
    return out


def save_confusion_matrix(texts: list[str], labels: list[str], path: Path) -> dict:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    predictions = cross_val_predict(build_pipeline(DEFAULT_MODEL, DEFAULT_FEATURES), texts, labels, cv=cv)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay.from_predictions(
        labels, predictions, xticks_rotation=45, ax=ax, colorbar=False
    )
    ax.set_title(f"{DEFAULT_MODEL} / {DEFAULT_FEATURES}: 3-fold CV predictions")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    wrong = [(t, p) for t, p in zip(labels, predictions) if t != p]
    return {"errors": len(wrong), "accuracy": accuracy_score(labels, predictions)}


def error_analysis(intents: dict, texts: list[str], labels: list[str], queries: dict) -> dict:
    """Where the default model goes wrong: top CV confusions and every held-out miss."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    predictions = cross_val_predict(build_pipeline(DEFAULT_MODEL, DEFAULT_FEATURES), texts, labels, cv=cv)
    confusions = Counter((t, p) for t, p in zip(labels, predictions) if t != p)

    clf = IntentClassifier(DEFAULT_MODEL, DEFAULT_FEATURES).train(intents)
    misses = []
    for q in queries["in_scope_test"]:
        tag, conf = clf.predict(q["text"])
        if tag != q["intent"]:
            misses.append(
                {"text": q["text"], "expected": q["intent"], "predicted": tag, "confidence": conf}
            )
    return {
        "cv_top_confusions": [
            {"true": t, "predicted": p, "count": c} for (t, p), c in confusions.most_common(8)
        ],
        "held_out_misses": misses,
    }


# ----------------------------------------------------------------------- held-out test


def evaluate_held_out(intents: dict, queries: dict) -> dict:
    """Train on all patterns and score the untouched in-scope test questions."""
    test = queries["in_scope_test"]
    y_true = [q["intent"] for q in test]
    results: dict = {}
    for model_name in AVAILABLE_MODELS:
        results[model_name] = {}
        for features in FEATURE_SETS:
            clf = IntentClassifier(model_name, features).train(intents)
            y_pred = [clf.predict(q["text"])[0] for q in test]
            results[model_name][features] = {
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
    return results


def _confidences(clf: IntentClassifier, texts: list[str]) -> np.ndarray:
    return np.array([clf.predict(t)[1] for t in texts])


def _rates(threshold: float, in_scope_conf, in_scope_correct, oos_conf) -> dict:
    answered = in_scope_conf >= threshold
    return {
        "in_scope_answered_correctly": float(np.mean(answered & in_scope_correct)),
        "in_scope_answered_wrongly": float(np.mean(answered & ~in_scope_correct)),
        "in_scope_sent_to_fallback": float(np.mean(~answered)),
        "out_of_scope_rejected": float(np.mean(oos_conf < threshold)),
    }


def _select_threshold(in_conf, in_correct, oos_conf):
    """Threshold maximising the mean of (in-scope answered correctly, out-of-scope rejected)."""
    grid = np.round(np.arange(0.0, 1.0, 0.01), 2)
    scores = np.array([
        0.5 * (np.mean((in_conf >= t) & in_correct) + np.mean(oos_conf < t)) for t in grid
    ])
    plateau = grid[scores >= scores.max() - 1e-12]
    return float(np.round(np.median(plateau), 2)), grid, scores  # middle of the plateau


def _in_scope_outcomes(clf: IntentClassifier, queries: list[dict]):
    preds = [clf.predict(q["text"]) for q in queries]
    conf = np.array([c for _, c in preds])
    correct = np.array([tag == q["intent"] for (tag, _), q in zip(preds, queries)])
    return conf, correct


def _score_queries(clf: IntentClassifier, in_scope: list[dict], far: list[str], near: list[str],
                   threshold: float) -> dict:
    conf, correct = _in_scope_outcomes(clf, in_scope)
    far_conf, near_conf = _confidences(clf, far), _confidences(clf, near)
    out = _rates(threshold, conf, correct, np.concatenate([far_conf, near_conf]))
    out["out_of_scope_far_rejected"] = float(np.mean(far_conf < threshold))
    out["out_of_scope_near_rejected"] = float(np.mean(near_conf < threshold))
    return out


def threshold_analysis(intents: dict, texts: list[str], labels: list[str], queries: dict) -> dict:
    """Pick the confidence threshold on dev queries, then report it on the test queries."""
    oos = queries["out_of_scope"]
    oos_dev = [q["text"] for q in oos if q["split"] == "dev"]
    far_test = [q["text"] for q in oos if q["split"] == "test" and q["kind"] == "far"]
    near_test = [q["text"] for q in oos if q["split"] == "test" and q["kind"] == "near"]
    in_dev, in_test = queries["in_scope_dev"], queries["in_scope_test"]

    clf = IntentClassifier(DEFAULT_MODEL, DEFAULT_FEATURES).train(intents)
    oos_dev_conf = _confidences(clf, oos_dev)

    dev_conf, dev_correct = _in_scope_outcomes(clf, in_dev)
    chosen, grid, scores = _select_threshold(dev_conf, dev_correct, oos_dev_conf)

    # Rejected alternative: use out-of-fold probabilities of the *training patterns* as the
    # in-scope signal. Patterns resemble each other far more than real questions do, so
    # this overestimates confidence and picks a threshold that is too high.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    proba = cross_val_predict(
        build_pipeline(DEFAULT_MODEL, DEFAULT_FEATURES), texts, labels, cv=cv, method="predict_proba"
    )
    classes = np.array(sorted(set(labels)))
    pattern_chosen, _, _ = _select_threshold(
        proba.max(axis=1), classes[proba.argmax(axis=1)] == np.array(labels), oos_dev_conf
    )

    base_model, base_features, base_threshold = BASELINE_CONFIG
    base = IntentClassifier(base_model, base_features).train(intents)

    _plot_threshold_curve(grid, dev_conf, dev_correct, oos_dev_conf, chosen,
                          RESULTS_DIR / "threshold_curve.png")
    return {
        "chosen": chosen,
        "dev_balanced_score_at_chosen": float(scores[int(np.where(grid == chosen)[0][0])]),
        "test_at_chosen": _score_queries(clf, in_test, far_test, near_test, chosen),
        "pattern_cv_alternative": {
            "chosen": pattern_chosen,
            "test": _score_queries(clf, in_test, far_test, near_test, pattern_chosen),
        },
        "test_operating_curve": {
            f"{t:.2f}": _score_queries(clf, in_test, far_test, near_test, t) for t in OPERATING_GRID
        },
        "test_baseline_config": {
            "model": base_model, "features": base_features, "threshold": base_threshold,
            **_score_queries(base, in_test, far_test, near_test, base_threshold),
        },
        "counts": {
            "in_scope_dev": len(in_dev), "in_scope_test": len(in_test), "oos_dev": len(oos_dev),
            "oos_test_far": len(far_test), "oos_test_near": len(near_test),
        },
    }


def _plot_threshold_curve(grid, in_conf, in_correct, oos_conf, chosen, path: Path) -> None:
    answered_ok = [np.mean((in_conf >= t) & in_correct) for t in grid]
    oos_rejected = [np.mean(oos_conf < t) for t in grid]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(grid, answered_ok, label="in-scope answered correctly (dev)")
    ax.plot(grid, oos_rejected, label="out-of-scope rejected (dev)")
    ax.axvline(chosen, color="grey", linestyle="--", label=f"chosen threshold = {chosen:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("fraction of queries")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- language model


def _lm_folds(n_items: int, k: int = 5, seed: int = 0) -> list[list[int]]:
    order = list(range(n_items))
    random.Random(seed).shuffle(order)
    return [order[i::k] for i in range(k)]


def evaluate_language_model(sentences: list[list[str]] | None = None) -> dict:
    """5-fold CV perplexity, with hyperparameters tuned on a dev slice inside each fold."""
    sentences = sentences or load_sentences()
    folds = _lm_folds(len(sentences))
    perplexity: dict = {}
    selected: dict = {}

    for n in (1, 2, 3):
        runs = {"add_k_1": [], "add_k_tuned": [], "interpolation_hand_set": [], "interpolation_tuned": []}
        for held_out, test_idx in enumerate(folds):
            rest = [i for g, fold in enumerate(folds) if g != held_out for i in fold]
            test = [sentences[i] for i in test_idx]
            full = [sentences[i] for i in rest]
            dev_size = max(1, len(rest) // 5)
            dev, sub = full[:dev_size], full[dev_size:]

            runs["add_k_1"].append(NgramLanguageModel(n=n, add_k=1.0).fit(full).perplexity(test))
            k = tune_add_k(sub, dev, n)
            runs["add_k_tuned"].append(NgramLanguageModel(n=n, add_k=k).fit(full).perplexity(test))
            hand = NgramLanguageModel(
                n=n, smoothing="interpolation", interpolation_weights=HAND_SET_WEIGHTS[n]
            )
            runs["interpolation_hand_set"].append(hand.fit(full).perplexity(test))
            w = (1.0,) if n == 1 else tune_interpolation_weights(sub, dev, n)
            tuned = NgramLanguageModel(n=n, smoothing="interpolation", interpolation_weights=w)
            runs["interpolation_tuned"].append(tuned.fit(full).perplexity(test))
        perplexity[str(n)] = {name: _mean_std(vals) for name, vals in runs.items()}

        # Weights to ship: the grid point with the lowest mean held-out perplexity over the folds.
        if n > 1:
            def mean_pp(weights, n=n):
                vals = []
                for held_out, test_idx in enumerate(folds):
                    rest = [sentences[i] for g, fold in enumerate(folds) if g != held_out for i in fold]
                    model = NgramLanguageModel(n=n, smoothing="interpolation", interpolation_weights=weights)
                    vals.append(model.fit(rest).perplexity([sentences[i] for i in test_idx]))
                return statistics.fmean(vals)
            selected[str(n)] = list(min(interpolation_weight_grid(n), key=mean_pp))

    return {
        "corpus": {
            "sentences": len(sentences),
            "tokens": sum(len(s) for s in sentences),
            "folds": len(folds),
        },
        "perplexity": perplexity,
        "selected_interpolation_weights": selected,
    }


# ------------------------------------------------------------------------------ main


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} +- {stat['std']:.3f}"


def main() -> None:
    intents = load_intents()
    queries = load_eval_queries()
    texts, labels = build_training_data(intents)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Dataset: {len(intents['intents'])} intents, {len(texts)} training patterns")
    print("\n=== Intent classifiers: 5x repeated stratified 3-fold CV (mean +- std over 15 folds) ===")
    classifiers = evaluate_classifiers(texts, labels)
    print(f"{'model':22}{'features':11}{'accuracy':>18}{'macro F1':>18}")
    for model_name, by_features in classifiers.items():
        for features, m in by_features.items():
            print(f"{model_name:22}{features:11}{_fmt(m['accuracy']):>18}{_fmt(m['macro_f1']):>18}")

    diagnostic = zero_evidence_diagnostic(texts, labels)
    print("\n=== Why do errors land in 'greeting'? (3-fold CV, default model) ===")
    for features, d in diagnostic.items():
        print(f"{features:10} {d['errors_predicted_greeting_with_no_known_features']} of "
              f"{d['errors_predicted_greeting']} errors predicted 'greeting' had no known features "
              f"({d['errors_with_no_known_features']} of {d['errors']} errors overall)")

    cm = save_confusion_matrix(texts, labels, RESULTS_DIR / "confusion_matrix.png")
    print(f"\nConfusion matrix ({DEFAULT_MODEL}/{DEFAULT_FEATURES}) saved: {cm['errors']} errors")

    held_out = evaluate_held_out(intents, queries)
    print("\n=== Held-out test: 42 fresh in-scope questions ===")
    for model_name, by_features in held_out.items():
        for features, m in by_features.items():
            print(f"{model_name:22}{features:11} accuracy={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f}")

    errors = error_analysis(intents, texts, labels, queries)
    print("\n=== Error analysis (default model) ===")
    print("Top CV confusions:", ", ".join(
        f"{c['true']}->{c['predicted']} x{c['count']}" for c in errors["cv_top_confusions"]))
    for miss in errors["held_out_misses"]:
        print(f"Held-out miss: {miss['text']!r} expected={miss['expected']} "
              f"predicted={miss['predicted']} ({miss['confidence']:.2f})")

    thr = threshold_analysis(intents, texts, labels, queries)
    print(f"\n=== Confidence threshold: {thr['chosen']:.2f} (chosen on dev; rows are held-out test) ===")
    rows = (
        (f"baseline @{thr['test_baseline_config']['threshold']:.2f}", thr["test_baseline_config"]),
        (f"default @{thr['chosen']:.2f}", thr["test_at_chosen"]),
        (f"pattern-CV @{thr['pattern_cv_alternative']['chosen']:.2f}", thr["pattern_cv_alternative"]["test"]),
    )
    for label, r in rows:
        print(
            f"{label:18} correct={r['in_scope_answered_correctly']:.3f} "
            f"wrong={r['in_scope_answered_wrongly']:.3f} "
            f"fallback={r['in_scope_sent_to_fallback']:.3f} | OOS rejected: "
            f"all={r['out_of_scope_rejected']:.3f} far={r['out_of_scope_far_rejected']:.3f} "
            f"near={r['out_of_scope_near_rejected']:.3f}"
        )

    lm = evaluate_language_model()
    print(f"\n=== Language model: 5-fold CV perplexity ({lm['corpus']['sentences']} sentences) ===")
    print(f"{'n':3}{'add-k (k=1)':>18}{'add-k (tuned)':>18}{'interp (hand)':>18}{'interp (tuned)':>18}")
    for n, p in lm["perplexity"].items():
        print(f"{n:3}" + "".join(f"{p[k]['mean']:>11.1f} +-{p[k]['std']:>4.0f}" for k in (
            "add_k_1", "add_k_tuned", "interpolation_hand_set", "interpolation_tuned")))
    print("Selected interpolation weights:", lm["selected_interpolation_weights"])

    metrics = {
        "dataset": {"intents": len(intents["intents"]), "patterns": len(texts)},
        "classifiers_cv": classifiers,
        "zero_evidence_diagnostic": diagnostic,
        "classifiers_held_out": held_out,
        "error_analysis": errors,
        "threshold": thr,
        "language_model": lm,
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
