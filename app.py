import streamlit as st

from src.chatbot import Chatbot

st.set_page_config(page_title="Campus Helpdesk Bot", page_icon="🎓")
st.title("🎓 Campus Helpdesk Chatbot")
st.caption(
    "TF-IDF + Naive Bayes intent classification, with an n-gram language "
    "model fallback for open-ended questions."
)

if "bot" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.bot = Chatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if user_input := st.chat_input("Ask about admissions, library, fees, hostel..."):
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    reply = st.session_state.bot.respond(user_input)
    st.session_state.messages.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)

with st.sidebar:
    st.header("About")
    st.write(
        "This bot classifies your message into an intent (greeting, "
        "admissions, library hours, fees, etc.) using a Naive Bayes "
        "classifier trained on TF-IDF features. If it isn't confident "
        "about the intent, it falls back to a trigram language model "
        "trained on a small campus-related corpus to generate a reply."
    )
    tag, confidence = (None, None)
    if st.session_state.messages:
        last_user_msg = [m for r, m in st.session_state.messages if r == "user"][-1]
        tag, confidence = st.session_state.bot.classifier.predict(last_user_msg)
    if tag:
        st.metric("Last predicted intent", tag, f"{confidence:.0%} confidence")
