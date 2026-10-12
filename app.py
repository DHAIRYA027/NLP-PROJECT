import streamlit as st

from src.chatbot import Chatbot

EXAMPLES = [
    "When does the library open?",
    "How do I apply for a scholarship?",
    "My wifi is not working in the hostel",
    "How do I report ragging?",
    "Tell me a joke",
]

st.set_page_config(page_title="Campus Helpdesk Bot", page_icon="🎓")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Train/load the models once per server process; they are read-only afterwards."""
    bot = Chatbot()
    return bot.classifier, bot.lm


def _queue(text: str) -> None:
    st.session_state.pending = text


def _clear() -> None:
    st.session_state.messages = []
    st.session_state.bot.history.clear()


if "bot" not in st.session_state:
    classifier, lm = load_models()
    # One lightweight Chatbot per browser session so chat histories never mix.
    st.session_state.bot = Chatbot(classifier=classifier, lm=lm)
if "messages" not in st.session_state:
    st.session_state.messages = []

bot: Chatbot = st.session_state.bot

st.title("🎓 Campus Helpdesk Chatbot")
st.caption(
    "Intent classification (TF-IDF + logistic regression) answers questions it is "
    "confident about; otherwise it says so and adds a sample from an n-gram language model."
)

prompt = st.chat_input("Ask about admissions, library, fees, hostel...")
user_text = prompt or st.session_state.pop("pending", None)
if user_text:
    reply = bot.reply(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": reply.text, "reply": reply})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        reply = message.get("reply")
        if reply is not None:
            how = "language-model fallback" if reply.used_fallback else "classifier answer"
            label = bot.classifier.labels.get(reply.intent, reply.intent)
            st.caption(f"{how} · closest intent: {label} ({reply.confidence:.0%})")

with st.sidebar:
    st.header("Try asking")
    for example in EXAMPLES:
        st.button(example, on_click=_queue, args=(example,), use_container_width=True)
    st.button("Clear chat", on_click=_clear, use_container_width=True)

    st.header("About")
    st.write(
        f"**Classifier:** {bot.classifier.model_name.replace('_', ' ')} on "
        f"{bot.classifier.features} TF-IDF features.\n\n"
        f"**Confidence threshold:** {bot.confidence_threshold:.2f}. Below it the bot "
        "declines to answer and adds a trigram language-model sample "
        "(interpolation smoothing), clearly labelled as generated text."
    )
    last = next((m["reply"] for m in reversed(st.session_state.messages) if m.get("reply")), None)
    if last is not None:
        st.header("Last message")
        st.metric(
            "Closest intent",
            bot.classifier.labels.get(last.intent, last.intent),
            f"{last.confidence:.0%} confidence",
        )
        if last.alternatives:
            st.caption(
                "Next closest: "
                + ", ".join(f"{bot.classifier.labels.get(i, i)} ({c:.0%})" for i, c in last.alternatives)
            )
