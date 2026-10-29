import math
import random

import pytest

from src.ngram_model import (
    DEFAULT_K_GRID,
    END,
    START,
    UNK,
    NgramLanguageModel,
    interpolation_weight_grid,
    load_sentences,
    tune_add_k,
    tune_interpolation_weights,
)

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


# ---------------------------------------------------------------- behaviour added later


@pytest.mark.parametrize("smoothing", ["add_k", "interpolation"])
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("context", [(), (START,), ("the", "library"), ("never", "seen")])
def test_probabilities_over_the_vocabulary_sum_to_one(smoothing, n, context):
    lm = NgramLanguageModel(n=n, smoothing=smoothing).fit(TOY_SENTENCES)
    total = sum(lm.prob(word, context) for word in lm.vocab)
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "weights",
    [
        (0.5, 0.5),             # wrong length for n=3
        (-0.1, 0.6, 0.5),       # negative weight
        (0.2, 0.2, 0.2),        # does not sum to 1
        (0.0, 0.4, 0.6),        # zero unigram weight -> zero probability for unseen words
    ],
)
def test_invalid_interpolation_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        NgramLanguageModel(n=3, smoothing="interpolation", interpolation_weights=weights)


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"add_k": 0}, {"add_k": -1.0}, {"smoothing": "nope"}])
def test_invalid_constructor_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        NgramLanguageModel(**kwargs)


def test_default_weights_are_valid_for_every_order():
    for n in (1, 2, 3):
        lm = NgramLanguageModel(n=n, smoothing="interpolation").fit(TOY_SENTENCES)
        assert sum(lm.interpolation_weights) == pytest.approx(1.0)
        assert math.isfinite(lm.perplexity(TOY_SENTENCES))


def test_generation_is_reproducible_with_a_seeded_rng():
    lm = NgramLanguageModel(n=3, smoothing="interpolation").fit(load_sentences())
    first = lm.generate(seed=["the", "library"], rng=random.Random(3), top_k=8)
    second = lm.generate(seed=["the", "library"], rng=random.Random(3), top_k=8)
    assert first == second


def test_generation_never_repeats_a_word_immediately():
    lm = NgramLanguageModel(n=2, smoothing="interpolation").fit(load_sentences())
    for seed in range(20):
        words = lm.generate(max_length=30, rng=random.Random(seed)).split()
        assert all(a != b for a, b in zip(words, words[1:]))


def test_top_k_restricts_sampling_to_the_most_probable_words():
    lm = NgramLanguageModel(n=2, smoothing="interpolation").fit(TOY_SENTENCES)
    # One generation step after "the": candidates exclude <unk> and the previous word,
    # and are ranked by (probability, word), exactly as generate() does.
    candidates = [w for w in lm.vocab if w not in (UNK, "the")]
    ranked = sorted(candidates, key=lambda w: (lm.prob(w, ("the",)), w), reverse=True)
    allowed = set(ranked[:3])

    observed = set()
    for seed in range(60):
        words = lm.generate(seed=["the"], max_length=1, rng=random.Random(seed), top_k=3).split()
        observed.add(words[1] if len(words) > 1 else END)  # END ends the text without a word
    assert observed <= allowed
    assert len(observed) > 1  # it samples among the top k rather than always taking the best


def test_generation_rejects_non_positive_temperature():
    lm = NgramLanguageModel(n=2).fit(TOY_SENTENCES)
    with pytest.raises(ValueError):
        lm.generate(temperature=0)


def test_generation_can_start_from_words_outside_the_vocabulary():
    lm = NgramLanguageModel(n=3, smoothing="interpolation").fit(TOY_SENTENCES)
    text = lm.generate(seed=["zebra", "quantum"], max_length=5, rng=random.Random(0))
    assert text.startswith("zebra quantum")


def test_interpolation_weight_grid_has_expected_shape():
    for n, expected in ((1, 1), (2, 10), (3, 55)):
        grid = interpolation_weight_grid(n, step=0.1)
        assert len(grid) == expected
        assert all(abs(sum(w) - 1.0) < 1e-9 and w[0] > 0 for w in grid)


def test_tuning_helpers_return_valid_hyperparameters():
    sentences = load_sentences()
    train, dev = sentences[:60], sentences[60:]
    assert tune_add_k(train, dev, n=2) in DEFAULT_K_GRID
    weights = tune_interpolation_weights(train, dev, n=3, step=0.25)
    assert len(weights) == 3 and sum(weights) == pytest.approx(1.0) and weights[0] > 0


def test_tuned_hyperparameters_beat_the_naive_defaults_on_dev_data():
    sentences = load_sentences()
    train, dev = sentences[:60], sentences[60:]
    k = tune_add_k(train, dev, n=3)
    tuned = NgramLanguageModel(n=3, add_k=k).fit(train).perplexity(dev)
    laplace = NgramLanguageModel(n=3, add_k=1.0).fit(train).perplexity(dev)
    assert tuned <= laplace
