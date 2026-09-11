import math

import pytest

from src.ngram_model import END, NgramLanguageModel, START, UNK, load_sentences


TOY_SENTENCES = [
    ["the", "library", "is", "open"],
    ["the", "library", "is", "quiet"],
    ["the", "cafeteria", "is", "open"],
]


def test_load_sentences_returns_nonempty_list_of_token_lists():
    sentences = load_sentences()
    assert len(sentences) > 0
    assert all(isinstance(s, list) and len(s) > 0 for s in sentences)


def test_fit_builds_a_nonempty_vocabulary():
    lm = NgramLanguageModel(n=2).fit(TOY_SENTENCES)
    assert "library" in lm.vocab
    assert UNK in lm.vocab
    assert END in lm.vocab


@pytest.mark.parametrize("smoothing", ["add_k", "interpolation"])
def test_prob_is_a_valid_probability(smoothing):
    lm = NgramLanguageModel(n=2, smoothing=smoothing).fit(TOY_SENTENCES)
    p = lm.prob("library", (START,))
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("smoothing", ["add_k", "interpolation"])
def test_unseen_word_never_gives_zero_probability(smoothing):
    lm = NgramLanguageModel(n=2, smoothing=smoothing).fit(TOY_SENTENCES)
    p = lm.prob("spaceship", ("the",))
    assert p > 0.0


@pytest.mark.parametrize("smoothing", ["add_k", "interpolation"])
def test_perplexity_is_finite_and_positive(smoothing):
    lm = NgramLanguageModel(n=2, smoothing=smoothing).fit(TOY_SENTENCES)
    pp = lm.perplexity(TOY_SENTENCES)
    assert pp > 0
    assert math.isfinite(pp)


def test_higher_order_interpolation_beats_add_k_on_real_corpus():
    sentences = load_sentences()
    split = int(len(sentences) * 0.85)
    train_sents, test_sents = sentences[:split], sentences[split:]

    add_k = NgramLanguageModel(n=3, smoothing="add_k").fit(train_sents)
    interpolated = NgramLanguageModel(n=3, smoothing="interpolation").fit(train_sents)

    assert interpolated.perplexity(test_sents) < add_k.perplexity(test_sents)


def test_generate_returns_a_nonempty_string():
    lm = NgramLanguageModel(n=2).fit(TOY_SENTENCES)
    text = lm.generate(seed=["the"], max_length=10)
    assert isinstance(text, str)
    assert len(text) > 0
