"""Guards on the data files: label noise and train/test leakage silently ruin evaluations."""

from collections import Counter

import pytest

from src.evaluate import load_eval_queries
from src.intent_classifier import load_intents
from src.ngram_model import load_sentences


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


@pytest.fixture(scope="module")
def intents():
    return load_intents()["intents"]


@pytest.fixture(scope="module")
def queries():
    return load_eval_queries()


def test_intent_tags_are_unique_and_labelled(intents):
    tags = [i["tag"] for i in intents]
    assert len(tags) == len(set(tags))
    assert all(i.get("label") for i in intents)


def test_every_intent_has_enough_patterns_and_a_response(intents):
    for intent in intents:
        assert len(intent["patterns"]) >= 8, intent["tag"]
        assert intent["responses"], intent["tag"]


def test_no_pattern_appears_twice_within_or_across_intents(intents):
    seen = Counter(_norm(p) for i in intents for p in i["patterns"])
    duplicates = [p for p, count in seen.items() if count > 1]
    assert not duplicates, f"duplicate training patterns (label noise): {duplicates}"


def test_in_scope_eval_queries_use_known_intents_and_cover_all_of_them(intents, queries):
    tags = {i["tag"] for i in intents}
    for split in ("in_scope_dev", "in_scope_test"):
        used = {q["intent"] for q in queries[split]}
        assert used == tags, split


def test_eval_queries_do_not_leak_from_the_training_patterns(intents, queries):
    train = {_norm(p) for i in intents for p in i["patterns"]}
    everything = (
        [q["text"] for q in queries["in_scope_dev"]]
        + [q["text"] for q in queries["in_scope_test"]]
        + [q["text"] for q in queries["out_of_scope"]]
    )
    assert [t for t in everything if _norm(t) in train] == []


def test_dev_and_test_splits_are_disjoint_and_free_of_duplicates(queries):
    dev = [_norm(q["text"]) for q in queries["in_scope_dev"]]
    dev += [_norm(q["text"]) for q in queries["out_of_scope"] if q["split"] == "dev"]
    test = [_norm(q["text"]) for q in queries["in_scope_test"]]
    test += [_norm(q["text"]) for q in queries["out_of_scope"] if q["split"] == "test"]
    assert len(dev) == len(set(dev)) and len(test) == len(set(test))
    assert not set(dev) & set(test)


def test_out_of_scope_queries_are_well_formed(queries):
    counts = Counter((q["split"], q["kind"]) for q in queries["out_of_scope"])
    assert set(counts) == {("dev", "far"), ("dev", "near"), ("test", "far"), ("test", "near")}
    assert all(count >= 5 for count in counts.values())


def test_corpus_is_large_enough_to_train_a_language_model():
    sentences = load_sentences()
    assert len(sentences) >= 50
    assert sum(len(s) for s in sentences) >= 1500
