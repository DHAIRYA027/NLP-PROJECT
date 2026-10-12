"""N-gram language model with add-k and linear-interpolation smoothing."""

from __future__ import annotations

import heapq
import itertools
import math
import random
import re
from collections import Counter
from pathlib import Path

import nltk

from src.preprocess import ensure_nltk_data

ensure_nltk_data()

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"

START = "<s>"
END = "</s>"
UNK = "<unk>"

# Weights for (unigram, bigram, trigram, ...) in linear interpolation. The n=2 and
# n=3 rows are the grid points with the lowest mean held-out perplexity in 5-fold
# cross-validation (`python -m src.evaluate` -> language_model.selected_interpolation_weights
# in results/metrics.json). They lean on the unigram term because the corpus is tiny
# (~2,000 words), so most bigram/trigram contexts are rarely seen.
DEFAULT_INTERPOLATION_WEIGHTS = {
    1: (1.0,),
    2: (0.7, 0.3),
    3: (0.6, 0.2, 0.2),
}

DEFAULT_K_GRID = (1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001)


def load_sentences(path: Path = CORPUS_PATH) -> list[list[str]]:
    raw = Path(path).read_text(encoding="utf-8")
    sentences = nltk.sent_tokenize(raw.replace("\n", " "))
    tokenized = []
    for sent in sentences:
        words = re.findall(r"[a-z']+", sent.lower())
        if words:
            tokenized.append(words)
    return tokenized


def _validate_weights(weights: tuple[float, ...], n: int) -> tuple[float, ...]:
    if len(weights) != n:
        raise ValueError(f"interpolation_weights needs {n} values (orders 1..{n}), got {len(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError("interpolation_weights must be non-negative")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"interpolation_weights must sum to 1, got {sum(weights):.4f}")
    if weights[0] <= 0:
        # Only the (add-one smoothed) unigram term gives unseen words non-zero
        # probability; a zero weight there makes perplexity infinite.
        raise ValueError("the unigram weight (first value) must be positive")
    return weights


class NgramLanguageModel:
    """Markov-assumption n-gram model, with either add-k (Laplace) smoothing
    or linear interpolation across orders 1..n as the smoothing strategy."""

    def __init__(
        self,
        n: int = 3,
        add_k: float = 1.0,
        smoothing: str = "add_k",
        interpolation_weights: tuple[float, ...] | None = None,
    ):
        if smoothing not in {"add_k", "interpolation"}:
            raise ValueError("smoothing must be 'add_k' or 'interpolation'")
        if n < 1:
            raise ValueError("n must be at least 1")
        if add_k <= 0:
            raise ValueError("add_k must be positive")
        self.n = n
        self.add_k = add_k
        self.smoothing = smoothing
        self.interpolation_weights = _validate_weights(
            tuple(interpolation_weights or DEFAULT_INTERPOLATION_WEIGHTS.get(n, (1 / n,) * n)), n
        )
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.order_counts: dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        self.order_context_counts: dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        self.vocab: set[str] = set()
        self._candidates: list[str] = []

    def _pad(self, sentence: list[str], order: int | None = None) -> list[str]:
        order = order or self.n
        return [START] * (order - 1) + sentence + [END]

    def fit(self, sentences: list[list[str]]) -> NgramLanguageModel:
        for sentence in sentences:
            self.vocab.update(sentence)
        self.vocab.add(UNK)
        self.vocab.add(END)
        # Sorted so that seeded generation is reproducible across processes
        # (set iteration order depends on PYTHONHASHSEED).
        self._candidates = sorted(self.vocab - {UNK})

        for sentence in sentences:
            padded = self._pad(sentence)
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1:i])
                word = padded[i]
                self.ngram_counts[context + (word,)] += 1
                self.context_counts[context] += 1

        if self.smoothing == "interpolation":
            for order in range(1, self.n + 1):
                for sentence in sentences:
                    padded = self._pad(sentence, order=order)
                    for i in range(order - 1, len(padded)):
                        context = tuple(padded[i - order + 1:i])
                        word = padded[i]
                        self.order_counts[order][context + (word,)] += 1
                        self.order_context_counts[order][context] += 1
        return self

    def _mle_prob(self, order: int, word: str, context: tuple[str, ...]) -> float:
        context = context[-(order - 1):] if order > 1 else ()
        v = len(self.vocab)
        num = self.order_counts[order].get(context + (word,), 0)
        den = self.order_context_counts[order].get(context, 0)
        if order == 1:
            return (num + 1) / (den + v)
        if den == 0:
            return 1.0 / v
        return num / den

    def _interpolated_prob(self, word: str, context: tuple[str, ...]) -> float:
        context = context[-(self.n - 1):] if self.n > 1 else ()
        return sum(
            weight * self._mle_prob(order, word, context[-(order - 1):] if order > 1 else ())
            for order, weight in zip(range(1, self.n + 1), self.interpolation_weights)
        )

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        if word not in self.vocab:
            word = UNK
        if self.smoothing == "interpolation":
            return self._interpolated_prob(word, context)
        context = context[-(self.n - 1):] if self.n > 1 else ()
        v = len(self.vocab)
        num = self.ngram_counts.get(context + (word,), 0) + self.add_k
        den = self.context_counts.get(context, 0) + self.add_k * v
        return num / den

    def sentence_logprob(self, sentence: list[str]) -> float:
        padded = self._pad(sentence)
        log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            log_prob += math.log2(self.prob(word, context))
        return log_prob

    def perplexity(self, sentences: list[list[str]]) -> float:
        total_log_prob = 0.0
        total_tokens = 0
        for sentence in sentences:
            total_log_prob += self.sentence_logprob(sentence)
            total_tokens += len(sentence) + 1  # +1 for </s>
        return 2 ** (-total_log_prob / total_tokens)

    def generate(
        self,
        seed: list[str] | None = None,
        max_length: int = 25,
        rng: random.Random | None = None,
        top_k: int | None = None,
        temperature: float = 1.0,
    ) -> str:
        """Sample a continuation word by word.

        top_k restricts each step to the k most probable words (much less word
        salad than sampling the whole vocabulary); temperature < 1 sharpens the
        distribution, > 1 flattens it.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        rng = rng or random
        context = [START] * (self.n - 1)
        if seed:
            context = (context + seed)[-(self.n - 1):] if self.n > 1 else []
        output = list(seed) if seed else []

        for _ in range(max_length):
            last = output[-1] if output else None
            scored = [
                (w, self.prob(w, tuple(context))) for w in self._candidates if w != last
            ]
            if top_k:
                scored = heapq.nlargest(top_k, scored, key=lambda item: (item[1], item[0]))
            words = [w for w, _ in scored]
            weights = [p ** (1.0 / temperature) for _, p in scored]
            next_word = rng.choices(words, weights=weights, k=1)[0]
            if next_word == END:
                break
            output.append(next_word)
            context = (context + [next_word])[-(self.n - 1):] if self.n > 1 else []

        return " ".join(output)


def tune_add_k(
    train: list[list[str]],
    dev: list[list[str]],
    n: int,
    candidates: tuple[float, ...] = DEFAULT_K_GRID,
) -> float:
    """Return the add-k value with the lowest perplexity on the dev sentences."""
    return min(
        candidates,
        key=lambda k: NgramLanguageModel(n=n, add_k=k).fit(train).perplexity(dev),
    )


def interpolation_weight_grid(n: int, step: float = 0.1) -> list[tuple[float, ...]]:
    """All weight vectors on a `step` grid that sum to 1 with a positive unigram weight."""
    units = round(1 / step)
    grid = []
    for combo in itertools.product(range(units + 1), repeat=n):
        if sum(combo) == units and combo[0] > 0:
            grid.append(tuple(c / units for c in combo))
    return grid


def tune_interpolation_weights(
    train: list[list[str]],
    dev: list[list[str]],
    n: int,
    step: float = 0.1,
) -> tuple[float, ...]:
    """Return the interpolation weights with the lowest perplexity on the dev sentences."""
    return min(
        interpolation_weight_grid(n, step),
        key=lambda w: NgramLanguageModel(
            n=n, smoothing="interpolation", interpolation_weights=w
        ).fit(train).perplexity(dev),
    )


if __name__ == "__main__":
    sentences = load_sentences()
    split = int(len(sentences) * 0.85)
    train_sents, test_sents = sentences[:split], sentences[split:]
    print(f"Trained on {len(train_sents)} sentences, tested on {len(test_sents)}")

    for n in (1, 2, 3):
        for smoothing in ("add_k", "interpolation"):
            lm = NgramLanguageModel(n=n, add_k=1.0, smoothing=smoothing).fit(train_sents)
            pp = lm.perplexity(test_sents)
            print(f"n={n} smoothing={smoothing:13} perplexity={pp:.2f}")

    print()
    lm = NgramLanguageModel(n=3, smoothing="interpolation").fit(train_sents)
    rng = random.Random(0)
    for seed in [["the", "library"], ["students", "can"], ["the", "college"]]:
        print(f"seed={seed!r:30} -> {lm.generate(seed=seed, max_length=15, rng=rng, top_k=8)}")
