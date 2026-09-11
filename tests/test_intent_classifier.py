import pytest

from src.intent_classifier import (
    AVAILABLE_MODELS,
    IntentClassifier,
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
