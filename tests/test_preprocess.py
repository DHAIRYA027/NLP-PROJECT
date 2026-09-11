from src.preprocess import clean_text, preprocess, preprocess_to_string, tokenize


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
