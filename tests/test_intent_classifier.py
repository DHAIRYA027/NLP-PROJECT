import json
import random

import pytest

from src.intent_classifier import (
    AVAILABLE_MODELS,
    UNKNOWN_INTENT,
    IntentClassifier,
    build_features,
    build_pipeline,
    build_training_data,
    load_intents,
)


@pytest.fixture(scope="module")
def intents():
    return load_intents()


@pytest.fixture(scope="module")
def trained_classifier(intents):
    return IntentClassifier().train(intents)


def test_load_intents_has_expected_shape(intents):
    assert "intents" in intents
    assert len(intents["intents"]) >= 10
    for intent in intents["intents"]:
        assert "tag" in intent
        assert len(intent["patterns"]) > 0
        assert len(intent["responses"]) > 0


def test_build_training_data_produces_matching_lengths(intents):
    texts, labels = build_training_data(intents)
    assert len(texts) == len(labels)
    assert len(texts) > 0


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError):
        build_pipeline("not_a_real_model")


@pytest.mark.parametrize("model_name", list(AVAILABLE_MODELS))
def test_each_available_model_trains_and_predicts(intents, model_name):
    clf = IntentClassifier(model_name=model_name).train(intents)
    tag, confidence = clf.predict("what time does the library close")
    assert tag == "library_hours"
    assert 0.0 <= confidence <= 1.0


def test_predict_returns_tag_and_confidence(trained_classifier):
    tag, confidence = trained_classifier.predict("hello there")
    assert tag == "greeting"
    assert 0.0 <= confidence <= 1.0


def test_response_for_returns_a_configured_response(trained_classifier):
    reply = trained_classifier.response_for("greeting")
    assert reply in trained_classifier.responses["greeting"]


def test_save_and_load_round_trip(tmp_path, trained_classifier):
    path = tmp_path / "model.joblib"
    trained_classifier.save(path)
    loaded = IntentClassifier.load(path)
    assert loaded.model_name == trained_classifier.model_name
    tag, _ = loaded.predict("what is the admission process")
    assert tag == "admissions"


# ---------------------------------------------------------------- behaviour added later


def test_build_features_rejects_unknown_feature_set():
    with pytest.raises(ValueError):
        build_features("not_a_feature_set")


@pytest.mark.parametrize("text", ["", "   ", "???", "!!!"])
def test_text_without_any_evidence_gets_zero_confidence(trained_classifier, text):
    assert trained_classifier.predict(text) == (UNKNOWN_INTENT, 0.0)
    assert trained_classifier.predict_top(text) == []


def test_word_only_features_report_no_evidence_for_unseen_vocabulary(intents):
    clf = IntentClassifier(features="word").train(intents)
    assert clf.predict("xyzzy plugh") == (UNKNOWN_INTENT, 0.0)


def test_character_features_recover_a_misspelling(trained_classifier):
    tag, confidence = trained_classifier.predict("libary timings")
    assert tag == "library_hours"
    assert confidence > 0.0


def test_predict_top_is_sorted_and_respects_k(trained_classifier):
    top = trained_classifier.predict_top("what is the semester fee", k=3)
    assert len(top) == 3
    confidences = [c for _, c in top]
    assert confidences == sorted(confidences, reverse=True)
    assert sum(confidences) <= 1.0 + 1e-9
    assert top[0][0] == "fees"


def test_topics_exclude_smalltalk_and_use_labels(trained_classifier):
    topics = trained_classifier.topics
    assert "Library hours" in topics
    assert "Greeting" not in topics and "Goodbye" not in topics and "Thanks" not in topics


def test_response_for_is_reproducible_with_a_seeded_rng(trained_classifier):
    first = [trained_classifier.response_for("greeting", random.Random(7)) for _ in range(5)]
    second = [trained_classifier.response_for("greeting", random.Random(7)) for _ in range(5)]
    assert first == second


def _write_intents(path, extra_pattern=None):
    data = load_intents()
    if extra_pattern:
        data["intents"][0]["patterns"].append(extra_pattern)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_or_train_reuses_a_matching_cache(tmp_path):
    intents_path = _write_intents(tmp_path / "intents.json")
    cache = tmp_path / "model.joblib"
    first = IntentClassifier.load_or_train(path=cache, intents_path=intents_path)
    assert cache.exists()
    mtime = cache.stat().st_mtime_ns
    second = IntentClassifier.load_or_train(path=cache, intents_path=intents_path)
    assert second.fingerprint == first.fingerprint
    assert cache.stat().st_mtime_ns == mtime  # not rewritten


def test_load_or_train_retrains_when_the_intents_change(tmp_path):
    cache = tmp_path / "model.joblib"
    old = IntentClassifier.load_or_train(
        path=cache, intents_path=_write_intents(tmp_path / "a.json")
    )
    new = IntentClassifier.load_or_train(
        path=cache, intents_path=_write_intents(tmp_path / "b.json", extra_pattern="howdy partner")
    )
    assert new.fingerprint != old.fingerprint
    assert IntentClassifier.load(cache).fingerprint == new.fingerprint


def test_load_or_train_retrains_when_the_model_type_changes(tmp_path):
    intents_path = _write_intents(tmp_path / "intents.json")
    cache = tmp_path / "model.joblib"
    nb = IntentClassifier.load_or_train("naive_bayes", path=cache, intents_path=intents_path)
    lr = IntentClassifier.load_or_train("logistic_regression", path=cache, intents_path=intents_path)
    assert nb.model_name == "naive_bayes" and lr.model_name == "logistic_regression"
    assert nb.fingerprint != lr.fingerprint


def test_load_or_train_recovers_from_a_corrupt_cache(tmp_path):
    cache = tmp_path / "model.joblib"
    cache.write_bytes(b"this is not a joblib file")
    clf = IntentClassifier.load_or_train(
        path=cache, intents_path=_write_intents(tmp_path / "intents.json")
    )
    assert clf.predict("what is the admission process")[0] == "admissions"
