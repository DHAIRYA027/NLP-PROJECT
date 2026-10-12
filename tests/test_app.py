from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def _clear_button(at):
    return next(b for b in at.sidebar.button if b.label == "Clear chat")


def test_app_starts_empty_with_example_questions(app):
    assert len(app.chat_message) == 0
    labels = [b.label for b in app.sidebar.button]
    assert "Clear chat" in labels and len(labels) >= 4


def test_typed_question_gets_a_labelled_answer(app):
    app.chat_input[0].set_value("when does the library open").run()
    assert not app.exception
    user, assistant = app.chat_message
    assert user.markdown[0].value == "when does the library open"
    assert "library" in assistant.markdown[0].value.lower()
    assert "classifier answer" in assistant.caption[0].value


def test_off_topic_question_is_marked_as_a_fallback(app):
    app.chat_input[0].set_value("tell me a joke").run()
    assistant = app.chat_message[1]
    assert "I'm not sure I can answer that." in assistant.markdown[0].value
    assert "language-model fallback" in assistant.caption[0].value


def test_example_button_asks_that_question(app):
    example = next(b for b in app.sidebar.button if b.label != "Clear chat")
    example.click().run()
    assert not app.exception
    assert app.chat_message[0].markdown[0].value == example.label


def test_clear_chat_empties_the_conversation(app):
    app.chat_input[0].set_value("hello").run()
    assert len(app.chat_message) == 2
    _clear_button(app).click().run()
    assert len(app.chat_message) == 0
    assert not app.exception
