from src.preprocess import (
    clean_text,
    ensure_nltk_data,
    preprocess,
    preprocess_to_string,
    tokenize,
)


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hi! When does the Library open??") == "hi when does the library open"


def test_clean_text_collapses_whitespace():
    assert clean_text("hello    world") == "hello world"


def test_tokenize_splits_into_words():
    assert tokenize("hello world") == ["hello", "world"]


def test_preprocess_removes_stopwords_by_default():
    tokens = preprocess("what is the admission process")
    assert "is" not in tokens
    assert "the" not in tokens
    assert "admission" in tokens


def test_preprocess_can_keep_stopwords():
    tokens = preprocess("what is the admission process", remove_stopwords=False)
    assert "is" in tokens
    assert "the" in tokens


def test_preprocess_lemmatizes():
    tokens = preprocess("running processes", lemmatize=True)
    assert "process" in tokens


def test_preprocess_to_string_returns_joined_tokens():
    result = preprocess_to_string("hello world")
    assert isinstance(result, str)
    assert result == "hello world"


def test_preprocess_handles_empty_string():
    assert preprocess("") == []


def test_ensure_nltk_data_does_not_download_when_resources_are_installed(monkeypatch):
    # wordnet and omw-1.4 exist only as .zip files; a naive nltk.data.find() reports them
    # missing and re-downloads on every start, which hangs when the network is slow or offline.
    calls = []
    monkeypatch.setattr("nltk.download", lambda *args, **kwargs: calls.append(args))
    ensure_nltk_data()
    assert calls == []


def test_ensure_nltk_data_downloads_a_genuinely_missing_resource(monkeypatch):
    calls = []
    monkeypatch.setattr("nltk.download", lambda package, **kwargs: calls.append(package))
    monkeypatch.setattr("src.preprocess._nltk_resource_available", lambda path: "wordnet" not in path)
    ensure_nltk_data()
    assert calls == ["wordnet"]
