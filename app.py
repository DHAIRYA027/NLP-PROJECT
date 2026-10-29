import html
import json
import math
import re
import time
from pathlib import Path

import streamlit as st

from src.knowledge_base import STOPWORDS, SYNONYMS, analyze, normalize, raw_tokens
from src.smart_bot import STRONG_MATCH, TENTATIVE_MATCH, SmartBot

EVAL_PATH = Path(__file__).resolve().parent / "data" / "eval_set.json"

st.set_page_config(page_title="Campus Helpdesk", page_icon="🎓", layout="wide")

STYLE = """
<style>
[data-testid="stAppDeployButton"], .stDeployButton, #MainMenu, footer {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.stApp {
    background-image:
        radial-gradient(900px 480px at 15% -8%, rgba(99, 102, 241, 0.20), transparent 60%),
        radial-gradient(800px 420px at 100% 0%, rgba(14, 165, 233, 0.16), transparent 55%);
    background-attachment: fixed;
}
.block-container {
    max-width: 960px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 1.7rem 1.9rem;
    border-radius: 22px;
    background: linear-gradient(120deg, #4338ca 0%, #7c3aed 55%, #0ea5e9 100%);
    color: #ffffff;
    box-shadow: 0 18px 40px rgba(79, 70, 229, 0.32);
    margin-bottom: 1.4rem;
}
.hero::after {
    content: "";
    position: absolute;
    right: -60px;
    top: -60px;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.12);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.hero-sub {
    margin-top: 0.45rem;
    font-size: 1rem;
    opacity: 0.92;
    max-width: 38rem;
}
.chips {
    margin-top: 1rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}
.chips span {
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.28);
    font-size: 0.78rem;
    font-weight: 600;
    backdrop-filter: blur(4px);
}
.section-label {
    margin: 0.4rem 0 0.6rem 0.15rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    opacity: 0.6;
}
div.stButton > button {
    min-height: 3rem;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.28);
    background: rgba(128, 128, 128, 0.07);
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    border-color: #6366f1;
    box-shadow: 0 10px 22px rgba(99, 102, 241, 0.25);
}
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.7rem;
    border: 1px solid rgba(128, 128, 128, 0.18);
    background: rgba(128, 128, 128, 0.06);
}
[data-testid="stChatInput"] {
    border-radius: 16px;
}
[data-testid="stExpander"] {
    margin-top: 0.8rem;
    border-radius: 12px;
}
.meta {
    margin-top: 0.7rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
}
.b-knowledge { background: rgba(16, 185, 129, 0.16); color: #10b981; }
.b-template { background: rgba(99, 102, 241, 0.18); color: #818cf8; }
.b-tentative { background: rgba(245, 158, 11, 0.16); color: #f59e0b; }
.b-generative { background: rgba(236, 72, 153, 0.16); color: #ec4899; }
.b-fallback { background: rgba(148, 163, 184, 0.2); color: #94a3b8; }
.b-topic { background: rgba(128, 128, 128, 0.16); }
.conf {
    font-size: 0.78rem;
    opacity: 0.72;
    margin-left: 0.2rem;
}
.meter {
    flex-basis: 100%;
    height: 6px;
    border-radius: 999px;
    background: rgba(128, 128, 128, 0.2);
    overflow: hidden;
}
.meter > span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}
.stat {
    padding: 0.7rem 0.9rem;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    background: rgba(128, 128, 128, 0.07);
}
.stat b {
    display: block;
    font-size: 1.45rem;
    line-height: 1.2;
}
.stat span {
    font-size: 0.74rem;
    opacity: 0.7;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.15);
}
.hero.compact {
    padding: 0.9rem 1.4rem;
    margin-bottom: 0.8rem;
}
.hero.compact .hero-title {
    font-size: 1.35rem;
}
.hero.compact .hero-sub, .hero.compact .chips {
    display: none;
}
.hero.compact::after {
    display: none;
}
button[data-baseweb="tab"] {
    font-weight: 700;
    font-size: 0.95rem;
}
.step {
    display: flex;
    gap: 0.9rem;
    padding: 1rem 1.1rem;
    margin-bottom: 0.7rem;
    border-radius: 16px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    background: rgba(128, 128, 128, 0.06);
}
.step-n {
    flex: 0 0 auto;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.9rem;
    color: #ffffff;
    background: linear-gradient(135deg, #6366f1, #0ea5e9);
}
.step-body {
    flex: 1 1 auto;
    min-width: 0;
}
.step-title {
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.step-note {
    font-size: 0.8rem;
    opacity: 0.65;
    margin-top: 0.35rem;
}
.tok {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border-radius: 8px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
}
.tok-keep { background: rgba(16, 185, 129, 0.16); color: #10b981; }
.tok-drop { background: rgba(148, 163, 184, 0.14); color: #94a3b8; text-decoration: line-through; }
.tok-map { background: rgba(99, 102, 241, 0.18); color: #818cf8; }
.tok-fix { background: rgba(245, 158, 11, 0.16); color: #f59e0b; }
.bar-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0.35rem 0;
    font-size: 0.86rem;
}
.bar-label {
    flex: 0 0 46%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.bar-track {
    flex: 1 1 auto;
    height: 10px;
    border-radius: 999px;
    background: rgba(128, 128, 128, 0.2);
    overflow: hidden;
}
.bar-track > span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}
.bar-value {
    flex: 0 0 3.2rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    opacity: 0.8;
}
.kpi {
    padding: 1rem 1.1rem;
    border-radius: 16px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    background: rgba(128, 128, 128, 0.07);
}
.kpi b {
    display: block;
    font-size: 1.9rem;
    line-height: 1.15;
    background: linear-gradient(90deg, #818cf8, #22d3ee);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.kpi span {
    font-size: 0.78rem;
    opacity: 0.7;
}
.foot {
    text-align: center;
    margin-top: 1.5rem;
    font-size: 0.78rem;
    opacity: 0.55;
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

TOPICS = [
    ("🎓", "Admissions", "How do I apply for admission?"),
    ("💳", "Fees", "What is the fee structure?"),
    ("📚", "Library", "What are the library timings?"),
    ("🏠", "Hostel", "Tell me about hostel facilities"),
    ("📝", "Exams", "When are the exams?"),
    ("💼", "Placements", "Tell me about placements"),
]

CUSTOM_EXAMPLES = [
    "Can I pay my fees by UPI?",
    "is there wifi in the hostal?",
    "What happens if I fail a subject?",
    "Is there a bus from the city?",
]

SOURCES = {
    "knowledge": ("Knowledge base", "b-knowledge"),
    "template": ("Intent classifier", "b-template"),
    "tentative": ("Best guess", "b-tentative"),
    "generative": ("N-gram model", "b-generative"),
    "fallback": ("Not sure", "b-fallback"),
}


def queue_query(text):
    st.session_state.pending = text


def clear_chat():
    st.session_state.messages = []
    st.session_state.animate = None


def stream_words(text):
    for chunk in re.findall(r"\S+\s*", text):
        yield chunk
        time.sleep(0.018)


def meta_html(message):
    label, css = SOURCES.get(message["source"], SOURCES["fallback"])
    parts = [f'<span class="badge {css}">{html.escape(label)}</span>']
    if message["source"] in ("knowledge", "template", "tentative") and message["label"]:
        parts.append(f'<span class="badge b-topic">{html.escape(str(message["label"]))}</span>')
    if message["confidence_label"]:
        percent = int(round(message["confidence"] * 100))
        percent = max(0, min(percent, 100))
        parts.append(
            f'<span class="conf">{percent}% {html.escape(message["confidence_label"])}</span>'
        )
        parts.append(f'<div class="meter"><span style="width:{percent}%"></span></div>')
    return f'<div class="meta">{"".join(parts)}</div>'


def render_trace(trace):
    query = trace["query"].replace("`", "'")
    lines = [f"**Your question:** `{query}`"]
    if trace["changes"]:
        fixes = ", ".join(f"`{old}` → `{new}`" for old, new in trace["changes"])
        lines.append(f"**Spelling corrected:** {fixes}")
    tokens = " ".join(f"`{token}`" for token in trace["tokens"]) or "none"
    lines.append(f"**After preprocessing:** {tokens}")
    if trace["hits"]:
        rows = ["| Closest FAQ question | Cosine similarity |", "|---|---|"]
        for hit in trace["hits"]:
            rows.append(f"| {hit['question']} | {hit['score']:.3f} |")
        lines.append("\n".join(rows))
    else:
        lines.append("**Retrieval:** no FAQ question shares a meaningful word with this query.")
    if trace["intent"]:
        lines.append(
            f"**Naive Bayes intent:** `{trace['intent']}` at {trace['intent_confidence']:.0%} confidence"
        )
    with st.expander("How I got this answer"):
        st.markdown("\n\n".join(lines))


def step(number, title, body, note=""):
    note_html = f'<div class="step-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="step"><div class="step-n">{number}</div><div class="step-body">'
        f'<div class="step-title">{html.escape(title)}</div>{body}{note_html}</div></div>',
        unsafe_allow_html=True,
    )


def bar_row(label, fraction, value):
    width = max(0, min(int(round(fraction * 100)), 100))
    return (
        f'<div class="bar-row"><div class="bar-label" title="{html.escape(label)}">'
        f'{html.escape(label)}</div><div class="bar-track"><span style="width:{width}%"></span></div>'
        f'<div class="bar-value">{html.escape(value)}</div></div>'
    )


def render_lab(bot):
    st.markdown("Type any question and watch it move through every stage of the pipeline.")
    text = st.text_input(
        "Question",
        value="when does the libary close",
        key="lab_text",
        label_visibility="collapsed",
    )
    if not text.strip():
        return
    kb = bot.kb
    result = bot.ask(text, allow_generative=False)
    trace = result["trace"]
    corrected = trace["corrected"]

    step(1, "Raw input", f'<span class="tok tok-keep">{html.escape(text.strip())}</span>')

    if trace["changes"]:
        chips = "".join(
            f'<span class="tok tok-fix">{html.escape(old)} → {html.escape(new)}</span>'
            for old, new in trace["changes"]
        )
    else:
        chips = '<span class="tok tok-keep">no corrections needed</span>'
    step(
        2,
        "Spelling correction",
        chips,
        "Levenshtein edit distance, at most 1 edit for short words and 2 for long ones. Only words of 5 or more letters are corrected.",
    )

    chips = ""
    for token in raw_tokens(corrected):
        if token in STOPWORDS:
            chips += f'<span class="tok tok-drop">{html.escape(token)}</span>'
        else:
            mapped = normalize(token)
            if mapped != token:
                chips += f'<span class="tok tok-map">{html.escape(token)} → {html.escape(mapped)}</span>'
            else:
                chips += f'<span class="tok tok-keep">{html.escape(token)}</span>'
    step(
        3,
        "Tokenize, remove stopwords, normalize",
        chips or "nothing left after cleaning",
        "Struck-through words are stopwords. Arrows show a synonym or stemming rule being applied.",
    )

    tokens = analyze(corrected)
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    rows = ""
    for term, count in counts.items():
        weight_tf = 1 + math.log(count)
        if term in kb.idf:
            idf_text = f"{kb.idf[term]:.3f}"
            weight_text = f"{weight_tf * kb.idf[term]:.3f}"
        else:
            idf_text = "unseen"
            weight_text = "-"
        rows += (
            f"<tr><td>{html.escape(term)}</td><td>{count}</td>"
            f"<td>{idf_text}</td><td>{weight_text}</td></tr>"
        )
    table = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
        '<tr style="text-align:left;opacity:0.7"><th>Term</th><th>Count</th><th>IDF</th><th>TF-IDF weight</th></tr>'
        f"{rows}</table>"
    ) if rows else "no terms"
    step(
        4,
        "TF-IDF vector of the query",
        table,
        "Words the knowledge base has never seen still lengthen the query vector, which lowers similarity. That is how out-of-scope questions get low scores.",
    )

    hits = kb.search(corrected, k=5)
    if hits:
        bars = "".join(bar_row(hit["question"], hit["score"], f"{hit['score']:.3f}") for hit in hits)
    else:
        bars = "No FAQ question shares a meaningful word with this query."
    step(5, "Cosine similarity with the FAQ", bars, "The top 5 closest FAQ questions.")

    if trace["intent"]:
        classifier = (
            f'<span class="badge b-template">{html.escape(str(trace["intent"]))}</span> '
            f'<span class="conf">{trace["intent_confidence"]:.0%} confidence</span>'
        )
    else:
        classifier = "no prediction"
    step(6, "Naive Bayes intent classifier", classifier, "The original model from the course project.")

    explanations = {
        "knowledge": f"Top similarity is at least {STRONG_MATCH:.2f}, so the answer comes straight from the knowledge base.",
        "template": "Retrieval was weak, but the classifier was confident, so a template reply is used.",
        "tentative": f"Similarity is between {TENTATIVE_MATCH:.2f} and {STRONG_MATCH:.2f}, so a best guess is shown with a caveat.",
        "generative": "Nothing matched, so the n-gram model generated a reply.",
        "fallback": "Nothing matched well, so the bot admits it and offers suggestions.",
    }
    step(7, "Routing decision", explanations.get(result["source"], ""))
    st.markdown("**Final answer**")
    st.markdown(result["reply"])
    st.markdown(meta_html(result), unsafe_allow_html=True)


def run_evaluation(kb):
    with open(EVAL_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    in_scope = []
    for item in data["in_scope"]:
        corrected, _ = kb.correct(item["q"])
        hits = kb.search(corrected, k=1)
        top = hits[0] if hits else None
        in_scope.append(
            {
                "query": item["q"],
                "expected": item["id"],
                "predicted": top["id"] if top else "none",
                "score": round(top["score"], 3) if top else 0.0,
                "correct": bool(top and top["id"] == item["id"]),
            }
        )
    out_of_scope = []
    for query in data["out_of_scope"]:
        corrected, _ = kb.correct(query)
        hits = kb.search(corrected, k=1)
        score = hits[0]["score"] if hits else 0.0
        out_of_scope.append({"query": query, "score": round(score, 3)})
    return in_scope, out_of_scope


def kpi(column, value, label):
    column.markdown(
        f'<div class="kpi"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>',
        unsafe_allow_html=True,
    )


def render_evaluation(bot):
    if not EVAL_PATH.exists():
        st.info("data/eval_set.json was not found.")
        return
    if "eval_rows" not in st.session_state:
        st.session_state.eval_rows = run_evaluation(bot.kb)
    in_scope, out_of_scope = st.session_state.eval_rows

    total = len(in_scope)
    correct = sum(row["correct"] for row in in_scope)
    confident = [row for row in in_scope if row["score"] >= STRONG_MATCH]
    confident_correct = sum(row["correct"] for row in confident)
    rejected = sum(1 for row in out_of_scope if row["score"] < TENTATIVE_MATCH)

    columns = st.columns(4)
    kpi(columns[0], f"{correct / total:.0%}", f"Top-1 accuracy ({correct}/{total})")
    kpi(
        columns[1],
        f"{confident_correct / len(confident):.0%}" if confident else "n/a",
        f"Precision when confident ({confident_correct}/{len(confident)})",
    )
    kpi(
        columns[2],
        f"{rejected / len(out_of_scope):.0%}" if out_of_scope else "n/a",
        f"Out-of-scope not answered ({rejected}/{len(out_of_scope)})",
    )
    kpi(columns[3], str(total + len(out_of_scope)), "Evaluation questions")

    st.markdown('<div class="section-label">Accuracy by similarity score</div>', unsafe_allow_html=True)
    bands = [
        (f"Score at least {STRONG_MATCH:.2f}", STRONG_MATCH, 1.01),
        (f"Score {TENTATIVE_MATCH:.2f} to {STRONG_MATCH:.2f}", TENTATIVE_MATCH, STRONG_MATCH),
        (f"Score below {TENTATIVE_MATCH:.2f}", -1.0, TENTATIVE_MATCH),
    ]
    html_bars = ""
    for label, low, high in bands:
        members = [row for row in in_scope if low <= row["score"] < high]
        if not members:
            continue
        hit = sum(row["correct"] for row in members)
        html_bars += bar_row(f"{label}  ({len(members)} questions)", hit / len(members), f"{hit / len(members):.0%}")
    st.markdown(html_bars, unsafe_allow_html=True)
    st.caption(
        "Higher scores are more reliable, which is why the bot answers confidently above "
        f"{STRONG_MATCH:.2f} and adds a caveat between {TENTATIVE_MATCH:.2f} and {STRONG_MATCH:.2f}."
    )

    misses = [row for row in in_scope if not row["correct"]]
    with st.expander(f"Questions answered incorrectly ({len(misses)})"):
        if misses:
            st.dataframe(misses, use_container_width=True, hide_index=True)
        else:
            st.write("None.")
    with st.expander("Out-of-scope questions and their best similarity"):
        st.dataframe(out_of_scope, use_container_width=True, hide_index=True)
    st.caption(
        "The in-scope questions are paraphrases and typos written separately from the ones used while tuning the system. "
        "Edit data/eval_set.json to add your own, and results update on the next run."
    )


if "bot" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.bot = SmartBot()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "animate" not in st.session_state:
    st.session_state.animate = None

bot = st.session_state.bot

typed = st.chat_input("Ask anything about the campus...")
query = st.session_state.pop("pending", None) or typed

if query:
    with st.spinner("Thinking..."):
        result = bot.ask(query, allow_generative=st.session_state.get("use_ngram", False))
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["reply"],
            "source": result["source"],
            "label": result["label"],
            "confidence": result["confidence"],
            "confidence_label": result["confidence_label"],
            "suggestions": result["suggestions"],
            "trace": result["trace"],
        }
    )
    st.session_state.animate = len(st.session_state.messages) - 1

assistant_indexes = [i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"]
last_assistant = assistant_indexes[-1] if assistant_indexes else None

with st.sidebar:
    st.markdown("## 🎓 Campus Helpdesk")
    st.caption("A classical NLP assistant. No pretrained LLM.")

    entries, questions, terms = bot.kb.stats()
    left, right = st.columns(2)
    left.markdown(
        f'<div class="stat"><b>{entries}</b><span>FAQ topics</span></div>',
        unsafe_allow_html=True,
    )
    right.markdown(
        f'<div class="stat"><b>{questions}</b><span>Indexed questions</span></div>',
        unsafe_allow_html=True,
    )

    toggle = getattr(st, "toggle", st.checkbox)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    toggle(
        "Show NLP internals",
        key="show_internals",
        help="Show tokens, spelling fixes, cosine scores and the classifier output for every answer.",
    )
    toggle(
        "Allow n-gram fallback",
        key="use_ngram",
        help="When nothing else matches, let the trigram language model generate a reply. It is experimental and often rough.",
    )

    st.markdown("### Pipeline")
    st.markdown(
        "1. Spell-correct using edit distance\n"
        "2. Tokenize, drop stopwords, map synonyms, stem\n"
        "3. TF-IDF retrieval over the FAQ with cosine similarity\n"
        "4. Naive Bayes intent classifier with template replies\n"
        "5. Suggestions, or the trigram model as a last resort"
    )

    with st.expander("Theory behind the models"):
        st.markdown("**TF-IDF weight**")
        st.latex(
            r"w_{t,d}=(1+\log \mathrm{tf}_{t,d})\left(\ln\frac{1+N}{1+\mathrm{df}_t}+1\right)"
        )
        st.markdown("**Cosine similarity**")
        st.latex(r"\cos(q,d)=\frac{q\cdot d}{\lVert q\rVert\,\lVert d\rVert}")
        st.markdown("**Edit distance (spelling)**")
        st.latex(
            r"d_{i,j}=\min\big(d_{i-1,j}+1,\;d_{i,j-1}+1,\;d_{i-1,j-1}+[a_i\ne b_j]\big)"
        )
        st.markdown("**Naive Bayes**")
        st.latex(r"\hat{c}=\arg\max_{c}\;P(c)\prod_{i=1}^{n}P(w_i\mid c)")
        st.latex(r"P(w\mid c)=\frac{\mathrm{count}(w,c)+1}{\mathrm{count}(c)+|V|}")
        st.markdown("**Trigram language model**")
        st.latex(
            r"P(w_i\mid w_{i-2},w_{i-1})=\frac{C(w_{i-2}w_{i-1}w_i)+k}{C(w_{i-2}w_{i-1})+k|V|}"
        )
        st.markdown("**Perplexity**")
        st.latex(r"PP(W)=P(w_1\ldots w_N)^{-1/N}")

    if last_assistant is not None:
        latest = st.session_state.messages[last_assistant]
        label, _ = SOURCES.get(latest["source"], SOURCES["fallback"])
        st.markdown("### Last answer")
        st.markdown(f"**Source:** {label}")
        trace = latest["trace"]
        if trace["intent"]:
            st.markdown(
                f"**Classifier intent:** `{trace['intent']}` ({trace['intent_confidence']:.0%})"
            )

    st.button("Clear chat", on_click=clear_chat, use_container_width=True)

compact = " compact" if st.session_state.messages else ""
st.markdown(
    f"""
    <div class="hero{compact}">
        <div class="hero-title">🎓 Campus Helpdesk</div>
        <div class="hero-sub">Ask about admissions, fees, the library, hostel, exams and placements.
        Typos and rephrasing are fine.</div>
        <div class="chips">
            <span>No LLM</span>
            <span>TF-IDF retrieval</span>
            <span>Naive Bayes</span>
            <span>Spell correction</span>
            <span>Trigram LM</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_lab, tab_eval = st.tabs(["💬 Chat", "🔬 NLP Lab", "📊 Evaluation"])

with tab_chat:
    if not st.session_state.messages:
        st.markdown('<div class="section-label">Popular topics</div>', unsafe_allow_html=True)
        for row_start in (0, 3):
            columns = st.columns(3)
            for offset, column in enumerate(columns):
                emoji, name, question = TOPICS[row_start + offset]
                column.button(
                    f"{emoji}  {name}",
                    key=f"topic_{row_start + offset}",
                    on_click=queue_query,
                    args=(question,),
                    use_container_width=True,
                )

        st.markdown('<div class="section-label">Or try your own wording</div>', unsafe_allow_html=True)
        columns = st.columns(2)
        for index, example in enumerate(CUSTOM_EXAMPLES):
            columns[index % 2].button(
                example,
                key=f"example_{index}",
                on_click=queue_query,
                args=(example,),
                use_container_width=True,
            )

    for index, message in enumerate(st.session_state.messages):
        avatar = "🙋" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant" and index == st.session_state.animate:
                if hasattr(st, "write_stream"):
                    st.write_stream(stream_words(message["content"]))
                else:
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])

            if message["role"] == "assistant":
                st.markdown(meta_html(message), unsafe_allow_html=True)
                if st.session_state.get("show_internals", False):
                    render_trace(message["trace"])
                if index == last_assistant and message["suggestions"]:
                    st.caption("Did you mean:")
                    suggestion_columns = st.columns(len(message["suggestions"]))
                    for position, suggestion in enumerate(message["suggestions"]):
                        suggestion_columns[position].button(
                            suggestion,
                            key=f"suggest_{index}_{position}",
                            on_click=queue_query,
                            args=(suggestion,),
                            use_container_width=True,
                        )

    st.session_state.animate = None

    if st.session_state.messages:
        st.markdown(
            '<div class="foot">Built with classical NLP. No pretrained language model is used.</div>',
            unsafe_allow_html=True,
        )

with tab_lab:
    render_lab(bot)

with tab_eval:
    render_evaluation(bot)