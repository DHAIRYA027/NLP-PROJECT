from src.intent_classifier import IntentClassifier, MODEL_PATH
from src.ngram_model import NgramLanguageModel, load_sentences
from src.preprocess import tokenize

# With 10 intents and short patterns, MultinomialNB posteriors rarely exceed
# ~0.35 even for correct predictions; tuned empirically against data/intents.json.
CONFIDENCE_THRESHOLD = 0.2


class Chatbot:
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

        if MODEL_PATH.exists():
            self.classifier = IntentClassifier.load()
        else:
            self.classifier = IntentClassifier().train()
            self.classifier.save()

        self.lm = NgramLanguageModel(n=3, add_k=1.0).fit(load_sentences())
        self.history: list[tuple[str, str]] = []

    def respond(self, user_input: str) -> str:
        tag, confidence = self.classifier.predict(user_input)

        if confidence >= self.confidence_threshold:
            reply = self.classifier.response_for(tag)
        else:
            seed_tokens = tokenize(user_input)[:2] or None
            generated = self.lm.generate(seed=seed_tokens, max_length=20)
            reply = (
                "I'm not fully sure about that, but here's something related from what I know: "
                f"{generated}."
            )

        self.history.append((user_input, reply))
        return reply


if __name__ == "__main__":
    bot = Chatbot()
    print("Campus Helpdesk Bot (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        print("Bot:", bot.respond(user_input))
