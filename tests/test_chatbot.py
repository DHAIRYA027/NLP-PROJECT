import pytest

from src.chatbot import Chatbot


@pytest.fixture(scope="module")
def bot():
    return Chatbot()


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
