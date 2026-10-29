"""Chatbot orchestration: answer from a template when the intent is confident,
otherwise say so plainly and fall back to the n-gram language model."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

from src.intent_classifier import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    MODEL_PATH,
    UNKNOWN_INTENT,
    IntentClassifier,
)
from src.ngram_model import NgramLanguageModel, load_sentences
from src.preprocess import tokenize

# Chosen on the dev queries (in-scope dev + out-of-scope dev) by
# `python -m src.evaluate`; see REPORT.md section 5. Raising it rejects more
# off-topic questions but sends more valid ones to the fallback. It is specific to
# the default model and features: re-run the analysis if you change either.
DEFAULT_CONFIDENCE_THRESHOLD = 0.37

_EMPTY_INPUT_REPLY = "Please type a question and I'll do my best to help."

# Language-model samples shorter than this read as fragments ("tell me a"), so they are
# resampled a few times and dropped if none is long enough.
_MIN_SAMPLE_WORDS = 6
_SAMPLE_ATTEMPTS = 5
# Suggest the top intent as "Did you mean...?" once its confidence is within this
# fraction of the answer threshold.
_SUGGESTION_FRACTION = 0.5


@dataclass
class Reply:
    text: str
    intent: str
    confidence: float
    used_fallback: bool
    generated: str = ""
    alternatives: list = field(default_factory=list)  # [(intent, confidence), ...]


class Chatbot:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        smoothing: str = "interpolation",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        seed: int | None = None,
        classifier: IntentClassifier | None = None,
        lm: NgramLanguageModel | None = None,
        cache_path: Path | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.rng = random.Random(seed)
        self.classifier = classifier or IntentClassifier.load_or_train(
            model_name, path=cache_path or MODEL_PATH
        )
        self.lm = lm or NgramLanguageModel(n=3, smoothing=smoothing).fit(load_sentences())
        self.history: list[tuple[str, str]] = []

    def _language_model_sample(self, user_input: str) -> str:
        """A short n-gram continuation of the user's first words, or "" if none is long enough."""
        seed = [t for t in tokenize(user_input) if t.isalpha()][:2] or None
        for _ in range(_SAMPLE_ATTEMPTS):
            sample = self.lm.generate(seed=seed, max_length=18, rng=self.rng, top_k=8)
            if len(sample.split()) >= _MIN_SAMPLE_WORDS:
                return sample
        return ""

    def _fallback_text(self, generated: str, intent: str, confidence: float) -> str:
        topics = ", ".join(self.classifier.topics)
        text = "I'm not sure I can answer that. "
        # A near-miss (e.g. a typo that pulled the confidence just under the threshold)
        # is worth surfacing instead of discarding.
        if (
            intent in self.classifier.labels
            and intent not in self.classifier.smalltalk
            and confidence >= self.confidence_threshold * _SUGGESTION_FRACTION
        ):
            text += f"Did you mean to ask about {self.classifier.labels[intent]}? "
        text += f"I can help with: {topics}. Try rephrasing, or ask about one of those."
        if generated:
            text += f'\n\nLanguage-model sample (generated text, not an answer): "{generated}"'
        return text

    def reply(self, user_input: str) -> Reply:
        """Answer one message and return the reply together with how it was produced."""
        if not user_input.strip():
            result = Reply(_EMPTY_INPUT_REPLY, UNKNOWN_INTENT, 0.0, used_fallback=True)
        else:
            top = self.classifier.predict_top(user_input, k=3)
            intent, confidence = top[0] if top else (UNKNOWN_INTENT, 0.0)
            if confidence >= self.confidence_threshold:
                result = Reply(
                    self.classifier.response_for(intent, self.rng),
                    intent, confidence, used_fallback=False, alternatives=top[1:],
                )
            else:
                generated = self._language_model_sample(user_input)
                result = Reply(
                    self._fallback_text(generated, intent, confidence),
                    intent, confidence, used_fallback=True,
                    generated=generated, alternatives=top[1:],
                )
        self.history.append((user_input, result.text))
        return result

    def respond(self, user_input: str) -> str:
        return self.reply(user_input).text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Campus helpdesk chatbot (command line)")
    parser.add_argument("--model", choices=list(AVAILABLE_MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--seed", type=int, help="seed for reproducible replies")
    parser.add_argument("--debug", action="store_true", help="show intent and confidence")
    args = parser.parse_args(argv)

    bot = Chatbot(model_name=args.model, confidence_threshold=args.threshold, seed=args.seed)
    print("Campus Helpdesk Bot (type 'quit' to exit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        reply = bot.reply(user_input)
        print("Bot:", reply.text)
        if args.debug:
            print(
                f"     [intent={reply.intent} confidence={reply.confidence:.2f} "
                f"fallback={reply.used_fallback}]"
            )


if __name__ == "__main__":
    main()
