import json

import numpy as np
import pytest

from src import evaluate
from src.chatbot import DEFAULT_CONFIDENCE_THRESHOLD
from src.intent_classifier import build_training_data, load_intents
from src.ngram_model import DEFAULT_INTERPOLATION_WEIGHTS


@pytest.fixture(scope="module")
def intents():
    return load_intents()


@pytest.fixture(scope="module")
def queries():
    return evaluate.load_eval_queries()


@pytest.fixture(scope="module")
def training_data(intents):
    return build_training_data(intents)


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """Evaluation writes plots; keep the tests from touching the committed results/."""
    path = tmp_path_factory.mktemp("results")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluate, "RESULTS_DIR", path)
        yield path


# Each of these is computed once for the module and shared by the tests below.


@pytest.fixture(scope="module")
def cv_results(training_data):
    texts, labels = training_data
    return evaluate.evaluate_classifiers(texts, labels)


@pytest.fixture(scope="module")
def held_out_results(intents, queries):
    return evaluate.evaluate_held_out(intents, queries)


@pytest.fixture(scope="module")
def threshold_results(intents, training_data, queries, results_dir):
    texts, labels = training_data
    return evaluate.threshold_analysis(intents, texts, labels, queries)


@pytest.fixture(scope="module")
def lm_results():
    return evaluate.evaluate_language_model()


# ------------------------------------------------------------------ helpers (pure logic)


def test_rates_partition_the_in_scope_queries():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([True, False, True, True])
    r = evaluate._rates(0.5, conf, correct, np.array([0.1, 0.7]))
    assert r["in_scope_answered_correctly"] == 0.25
    assert r["in_scope_answered_wrongly"] == 0.25
    assert r["in_scope_sent_to_fallback"] == 0.5
    assert r["out_of_scope_rejected"] == 0.5
    total = (
        r["in_scope_answered_correctly"]
        + r["in_scope_answered_wrongly"]
        + r["in_scope_sent_to_fallback"]
    )
    assert total == pytest.approx(1.0)


def test_select_threshold_separates_two_clean_groups():
    in_conf = np.array([0.8, 0.85, 0.9, 0.95])
    oos_conf = np.array([0.1, 0.15, 0.2, 0.25])
    chosen, grid, scores = evaluate._select_threshold(in_conf, np.ones(4, dtype=bool), oos_conf)
    assert 0.25 < chosen <= 0.8
    assert scores.max() == pytest.approx(1.0)
    assert len(grid) == len(scores)


def test_select_threshold_penalises_in_scope_queries_answered_wrongly():
    in_conf = np.array([0.9, 0.9])
    oos = np.array([0.1])
    _, _, right = evaluate._select_threshold(in_conf, np.array([True, True]), oos)
    _, _, wrong = evaluate._select_threshold(in_conf, np.array([False, False]), oos)
    assert right.max() > wrong.max()


def test_language_model_folds_form_a_partition():
    folds = evaluate._lm_folds(87, k=5, seed=0)
    flat = [i for fold in folds for i in fold]
    assert sorted(flat) == list(range(87))
    assert max(len(f) for f in folds) - min(len(f) for f in folds) <= 1


# ------------------------------------------------------------------- evaluation functions


def test_zero_evidence_diagnostic_is_internally_consistent(training_data):
    texts, labels = training_data
    result = evaluate.zero_evidence_diagnostic(texts, labels)
    for stats in result.values():
        assert stats["errors_with_no_known_features"] <= stats["errors"]
        assert (
            stats["errors_predicted_greeting_with_no_known_features"]
            <= stats["errors_predicted_greeting"]
        )
        assert stats["errors_predicted_greeting"] <= stats["errors"]
    # Character n-grams give every query some features to work with.
    key = "items_with_no_known_features"
    assert result["word+char"][key] <= result["word"][key]


def test_threshold_analysis_reports_consistent_rates(threshold_results, queries):
    assert 0.0 < threshold_results["chosen"] < 1.0
    at = threshold_results["test_at_chosen"]
    total = (
        at["in_scope_answered_correctly"]
        + at["in_scope_answered_wrongly"]
        + at["in_scope_sent_to_fallback"]
    )
    assert total == pytest.approx(1.0)
    assert threshold_results["counts"]["in_scope_test"] == len(queries["in_scope_test"])


# ------------------------------------------------- quality floors (regression guards)
#
# These are deliberately loose: they exist to catch a broken pipeline or a bad data edit,
# not to pin exact numbers (those live in results/metrics.json).


def test_default_classifier_clears_the_held_out_accuracy_floor(held_out_results):
    assert held_out_results["logistic_regression"]["word+char"]["accuracy"] >= 0.80


def test_character_features_do_not_hurt_held_out_accuracy(held_out_results):
    for by_features in held_out_results.values():
        assert by_features["word+char"]["accuracy"] >= by_features["word"]["accuracy"] - 0.02


def test_shipped_threshold_answers_most_questions_and_rejects_most_off_topic(threshold_results):
    at_chosen = threshold_results["test_at_chosen"]
    assert at_chosen["in_scope_answered_correctly"] >= 0.60
    assert at_chosen["out_of_scope_far_rejected"] >= 0.80


# ------------------------- shipped constants must equal what the evaluation selects


def test_default_threshold_matches_the_dev_selection(threshold_results):
    assert threshold_results["chosen"] == pytest.approx(DEFAULT_CONFIDENCE_THRESHOLD, abs=1e-9)


def test_default_interpolation_weights_match_the_cross_validated_selection(lm_results):
    selected = lm_results["selected_interpolation_weights"]
    for n in ("2", "3"):
        assert list(DEFAULT_INTERPOLATION_WEIGHTS[int(n)]) == pytest.approx(selected[n])


# -------------------------------------- committed metrics.json must be reproducible


def test_committed_metrics_are_reproducible(
    cv_results, held_out_results, threshold_results, lm_results
):
    """results/metrics.json (quoted by README and REPORT) must match what the code produces."""
    committed = json.loads((evaluate.ROOT / "results" / "metrics.json").read_text())

    for model, by_features in committed["classifiers_cv"].items():
        for features, stats in by_features.items():
            for metric in ("accuracy", "macro_f1"):
                got = cv_results[model][features][metric]["mean"]
                assert got == pytest.approx(stats[metric]["mean"], abs=1e-4)

    for model, by_features in committed["classifiers_held_out"].items():
        for features, stats in by_features.items():
            assert held_out_results[model][features]["accuracy"] == pytest.approx(
                stats["accuracy"], abs=1e-4
            )

    assert threshold_results["chosen"] == pytest.approx(committed["threshold"]["chosen"], abs=1e-9)
    for key, value in committed["threshold"]["test_at_chosen"].items():
        assert threshold_results["test_at_chosen"][key] == pytest.approx(value, abs=1e-4)

    committed_lm = committed["language_model"]
    assert lm_results["selected_interpolation_weights"] == committed_lm["selected_interpolation_weights"]
    for n, runs in committed_lm["perplexity"].items():
        for name, stats in runs.items():
            assert lm_results["perplexity"][n][name]["mean"] == pytest.approx(stats["mean"], rel=1e-4)
