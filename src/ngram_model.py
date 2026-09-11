from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import nltk

from src.preprocess import ensure_nltk_data

ensure_nltk_data()

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"

START = "<s>"
END = "</s>"
UNK = "<unk>"


def load_sentences(path: Path = CORPUS_PATH) -> list[list[str]]:
    raw = Path(path).read_text(encoding="utf-8")
    sentences = nltk.sent_tokenize(raw.replace("\n", " "))
    tokenized = []
    for sent in sentences:
        words = re.findall(r"[a-z']+", sent.lower())
        if words:
            tokenized.append(words)
    return tokenized


DEFAULT_INTERPOLATION_WEIGHTS = {
    1: (1.0,),
    2: (0.25, 0.75),
    3: (0.1, 0.3, 0.6),
}


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
        self.n = n
        self.add_k = add_k
        self.smoothing = smoothing
        self.interpolation_weights = (
            interpolation_weights or DEFAULT_INTERPOLATION_WEIGHTS.get(n, tuple([1 / n] * n))
        )
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.order_counts: dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        self.order_context_counts: dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        self.vocab: set[str] = set()

    def _pad(self, sentence: list[str], order: int | None = None) -> list[str]:
        order = order or self.n
        return [START] * (order - 1) + sentence + [END]

    def fit(self, sentences: list[list[str]]) -> "NgramLanguageModel":
        for sentence in sentences:
            self.vocab.update(sentence)
        self.vocab.add(UNK)
        self.vocab.add(END)

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

    def generate(self, seed: list[str] | None = None, max_length: int = 25) -> str:
        context = [START] * (self.n - 1)
        if seed:
            context = (context + seed)[-(self.n - 1):] if self.n > 1 else []
        output = list(seed) if seed else []

        for _ in range(max_length):
            candidates = [w for w in self.vocab if w not in (START, UNK)]
            weights = [self.prob(w, tuple(context)) for w in candidates]
            next_word = random.choices(candidates, weights=weights, k=1)[0]
            if next_word == END:
                break
            output.append(next_word)
            context = (context + [next_word])[-(self.n - 1):] if self.n > 1 else []

        return " ".join(output)


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
    lm = NgramLanguageModel(n=3, add_k=1.0, smoothing="interpolation").fit(train_sents)
    for seed in [["the", "library"], ["students", "can"], ["the", "college"]]:
        print(f"seed={seed!r:30} -> {lm.generate(seed=seed, max_length=15)}")
