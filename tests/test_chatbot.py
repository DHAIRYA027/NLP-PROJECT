import pytest

from src.chatbot import DEFAULT_CONFIDENCE_THRESHOLD, Chatbot, Reply, main
from src.intent_classifier import UNKNOWN_INTENT


@pytest.fixture(scope="module")
def bot(tmp_path_factory):
    cache = tmp_path_factory.mktemp("model") / "intent_model.joblib"
    return Chatbot(seed=0, cache_path=cache)


def test_respond_returns_nonempty_string(bot):
    reply = bot.respond("hi there")
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_confident_intent_uses_template_response(bot):
    reply = bot.respond("what time does the library close")
    assert "library" in reply.lower() or "8" in reply or "9" in reply


def test_respond_appends_to_history(bot):
    bot.history.clear()
    bot.respond("hello")
    assert len(bot.history) == 1
    assert bot.history[0][0] == "hello"


def test_low_confidence_input_falls_back_to_language_model(bot):
    reply = bot.respond("purple elephants dance on tuesdays maybe")
    assert isinstance(reply, str)
    assert len(reply) > 0


# ------------------------------------------------------------ structured replies (real models)


def test_reply_describes_how_a_confident_answer_was_produced(bot):
    reply = bot.reply("when does the library open")
    assert isinstance(reply, Reply)
    assert reply.intent == "library_hours"
    assert reply.confidence >= bot.confidence_threshold
    assert not reply.used_fallback
    assert reply.generated == ""
    assert reply.text in bot.classifier.responses["library_hours"]


def test_off_topic_question_gets_an_honest_fallback(bot):
    reply = bot.reply("tell me a joke")
    assert reply.used_fallback
    assert reply.text.startswith("I'm not sure I can answer that.")
    for topic in bot.classifier.topics:
        assert topic in reply.text


def test_fallback_labels_generated_text_as_not_an_answer(bot):
    reply = bot.reply("tell me a joke")
    if reply.generated:
        assert "generated text, not an answer" in reply.text
        assert reply.generated in reply.text


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_input_asks_for_a_question(bot, text):
    reply = bot.reply(text)
    assert reply.used_fallback
    assert reply.intent == UNKNOWN_INTENT
    assert "type a question" in reply.text


def test_punctuation_only_input_does_not_crash_or_guess(bot):
    reply = bot.reply("???")
    assert reply.used_fallback
    assert reply.confidence == 0.0
    assert "Did you mean" not in reply.text


def test_same_seed_gives_identical_replies(tmp_path):
    cache = tmp_path / "intent_model.joblib"
    questions = ["tell me a joke", "hello", "purple elephants dance", "what is the fee deadline"]
    first = Chatbot(seed=5, cache_path=cache)
    second = Chatbot(seed=5, cache_path=cache)
    assert [first.respond(q) for q in questions] == [second.respond(q) for q in questions]


def test_default_threshold_is_a_probability():
    assert 0.0 < DEFAULT_CONFIDENCE_THRESHOLD < 1.0


# ------------------------------------------------ decision logic (deterministic stubs)


class StubClassifier:
    labels = {"library_hours": "Library hours", "greeting": "Greeting"}
    smalltalk = {"greeting"}
    topics = ["Library hours"]
    responses = {"library_hours": ["The library opens at 8."], "greeting": ["Hello!"]}

    def __init__(self, top):
        self._top = top

    def predict_top(self, text, k=3):
        return self._top[:k]

    def response_for(self, tag, rng=None):
        return self.responses[tag][0]


class StubLM:
    def __init__(self, sample):
        self.sample = sample

    def generate(self, **kwargs):
        return self.sample


def make_bot(top, sample="a long enough generated sample for the reader", threshold=0.4):
    return Chatbot(
        classifier=StubClassifier(top), lm=StubLM(sample), confidence_threshold=threshold
    )


def test_confidence_at_the_threshold_is_answered():
    reply = make_bot([("library_hours", 0.4)]).reply("anything")
    assert not reply.used_fallback
    assert reply.text == "The library opens at 8."


def test_confidence_just_below_the_threshold_falls_back():
    reply = make_bot([("library_hours", 0.39)]).reply("anything")
    assert reply.used_fallback


def test_near_miss_suggests_the_closest_topic():
    reply = make_bot([("library_hours", 0.25)]).reply("libary timings")
    assert "Did you mean to ask about Library hours?" in reply.text


def test_far_miss_does_not_suggest_anything():
    reply = make_bot([("library_hours", 0.1)]).reply("tell me a joke")
    assert "Did you mean" not in reply.text


def test_smalltalk_near_miss_is_not_suggested():
    reply = make_bot([("greeting", 0.3)]).reply("recommend a good movie")
    assert reply.used_fallback
    assert "Did you mean" not in reply.text


def test_generated_samples_that_are_too_short_are_dropped():
    reply = make_bot([("library_hours", 0.1)], sample="tell me a").reply("tell me a joke")
    assert reply.generated == ""
    assert "Language-model sample" not in reply.text


def test_alternatives_exclude_the_chosen_intent():
    top = [("library_hours", 0.8), ("greeting", 0.1)]
    reply = make_bot(top).reply("anything")
    assert reply.alternatives == [("greeting", 0.1)]


# ---------------------------------------------------------------------------------- CLI


def test_cli_answers_then_exits(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("src.chatbot.MODEL_PATH", tmp_path / "intent_model.joblib")
    inputs = iter(["when does the library open", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    main(["--seed", "0", "--debug"])
    out = capsys.readouterr().out
    assert "Bot:" in out and "library" in out.lower()
    assert "intent=library_hours" in out
    assert "Goodbye!" in out


def test_cli_exits_cleanly_on_end_of_input(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("src.chatbot.MODEL_PATH", tmp_path / "intent_model.joblib")

    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    main([])  # must return rather than raise
    assert "Campus Helpdesk Bot" in capsys.readouterr().out
