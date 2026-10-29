"""README.md and REPORT.md quote numbers from results/metrics.json; keep them in sync.

If one of these fails after you changed the data or the model: run `python -m src.evaluate`,
then update the tables in README.md and REPORT.md from results/metrics.json.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
DOCS = {name: (ROOT / name).read_text(encoding="utf-8").splitlines() for name in ("README.md", "REPORT.md")}


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def lines_with(doc: str, *needles: str) -> list:
    return [line for line in DOCS[doc] if all(n in line for n in needles)]


def assert_row(doc: str, *needles: str) -> None:
    assert lines_with(doc, *needles), f"{doc} has no line containing all of {needles}"


@pytest.mark.parametrize("doc", ["README.md", "REPORT.md"])
def test_classifier_table_matches_metrics(doc):
    for model, by_features in METRICS["classifiers_cv"].items():
        for features, cv in by_features.items():
            held_out = METRICS["classifiers_held_out"][model][features]
            cv_acc = f"{cv['accuracy']['mean']:.3f} ± {cv['accuracy']['std']:.3f}"
            needles = [cv_acc, f"{held_out['accuracy']:.3f}"]
            if doc == "REPORT.md":
                needles += [
                    f"{cv['macro_f1']['mean']:.3f} ± {cv['macro_f1']['std']:.3f}",
                    f"{held_out['macro_f1']:.3f}",
                ]
            assert_row(doc, *needles)


def test_dataset_sizes_are_quoted_correctly():
    d = METRICS["dataset"]
    assert_row("README.md", f"{d['intents']} intents", f"{d['patterns']} training patterns")
    assert_row("REPORT.md", f"{d['intents']} intents", f"{d['patterns']} patterns")
    corpus = METRICS["language_model"]["corpus"]
    assert_row("REPORT.md", f"{corpus['sentences']} sentences", f"{corpus['tokens']:,} tokens")
    counts = METRICS["threshold"]["counts"]
    assert_row("REPORT.md", "In-scope dev questions", f"| {counts['in_scope_dev']} ")
    assert_row("REPORT.md", "In-scope test questions", f"| {counts['in_scope_test']} ")


@pytest.mark.parametrize("doc", ["README.md", "REPORT.md"])
def test_original_vs_current_behaviour_matches_metrics(doc):
    threshold = METRICS["threshold"]
    for key in ("test_baseline_config", "test_at_chosen"):
        r = threshold[key]
        assert_row(
            doc,
            pct(r["in_scope_answered_correctly"]),
            pct(r["in_scope_answered_wrongly"]),
            pct(r["in_scope_sent_to_fallback"]),
            pct(r["out_of_scope_far_rejected"], 0),
            pct(r["out_of_scope_near_rejected"], 0),
        )


def test_readme_quotes_the_shipped_threshold_and_the_conservative_option():
    chosen = METRICS["threshold"]["chosen"]
    assert_row("README.md", f"threshold {chosen:.2f}")
    conservative = METRICS["threshold"]["test_operating_curve"]["0.70"]
    assert conservative["in_scope_answered_wrongly"] == 0.0
    paragraph = " ".join(DOCS["README.md"])  # the sentence wraps across lines
    assert "raising it to 0.70 gives 0% wrong answers" in paragraph
    assert f"rejects {pct(conservative['out_of_scope_near_rejected'], 0)}" in paragraph
    assert f"answers only {pct(conservative['in_scope_answered_correctly'], 0)}" in paragraph


def test_report_operating_curve_matches_metrics():
    assert_row("REPORT.md", f"**0.40 (≈ shipped {METRICS['threshold']['chosen']:.2f})**")
    for t, r in METRICS["threshold"]["test_operating_curve"].items():
        assert_row(
            "REPORT.md",
            f"| {t}" if t != "0.40" else "| **0.40",
            pct(r["in_scope_answered_correctly"]),
            pct(r["in_scope_answered_wrongly"]),
            pct(r["in_scope_sent_to_fallback"]),
            pct(r["out_of_scope_far_rejected"], 0),
            pct(r["out_of_scope_near_rejected"], 0),
        )


@pytest.mark.parametrize("doc", ["README.md", "REPORT.md"])
def test_language_model_table_matches_metrics(doc):
    columns = ("add_k_1", "add_k_tuned", "interpolation_hand_set", "interpolation_tuned")
    for n, runs in METRICS["language_model"]["perplexity"].items():
        if doc == "README.md" and n == "1":
            continue  # the README table starts at n=2
        cells = [f"{runs[c]['mean']:.1f} ± {runs[c]['std']:.0f}" for c in columns]
        assert_row(doc, f"| {n} |", *cells)


def test_report_quotes_the_selected_interpolation_weights():
    for weights in METRICS["language_model"]["selected_interpolation_weights"].values():
        assert_row("REPORT.md", "/".join(str(w) for w in weights))


def test_report_diagnostic_table_matches_metrics():
    for features, d in METRICS["zero_evidence_diagnostic"].items():
        label = features.replace("+", " + ")
        assert_row(
            "REPORT.md",
            f"| {label} | {d['items_with_no_known_features']} | {d['errors']} | "
            f"{d['errors_predicted_greeting']} | {d['errors_predicted_greeting_with_no_known_features']} |",
        )


def test_report_lists_every_held_out_mistake():
    for miss in METRICS["error_analysis"]["held_out_misses"]:
        confidence = f"{miss['confidence']:.2f}"
        assert_row("REPORT.md", miss["text"], miss["expected"], miss["predicted"], confidence)
    for conf in METRICS["error_analysis"]["cv_top_confusions"][:4]:
        count = f"×{conf['count']}"
        assert_row("REPORT.md", f"`{conf['true']} → {conf['predicted']}` {count}")
