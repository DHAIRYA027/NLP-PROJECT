# Campus Helpdesk Chatbot (NLP Course Project)

A conversational chatbot that answers common college queries (admissions, library
hours, fees, hostel, exams, placements) and demonstrates core NLP fundamentals
end to end rather than wrapping a pretrained LLM.

## Architecture

```
user input
   |
   v
preprocessing (tokenize, remove stopwords, lemmatize)   -- src/preprocess.py
   |
   v
intent classification (TF-IDF + Multinomial Naive Bayes) -- src/intent_classifier.py
   |
   +-- confidence >= threshold --> template response (from data/intents.json)
   |
   +-- confidence <  threshold --> n-gram language model generates a reply
                                    (Markov assumption, add-k smoothing)      -- src/ngram_model.py
```

Orchestration lives in `src/chatbot.py`; `app.py` is a Streamlit chat UI on top of it.

## NLP concepts this covers

- **Tokenization, stopword removal, lemmatization** (`preprocess.py`)
- **Bag-of-Words / TF-IDF** vectorization (`intent_classifier.py`)
- **Naive Bayes classification** — Bayes' theorem, MLE with Laplace smoothing under the hood
- **N-gram language modeling** — Markov assumption, add-k (Laplace) smoothing, perplexity as an evaluation metric (`ngram_model.py`)
- **Text generation** via sampling from a probability distribution over next words

## Setup

```bash
cd nlp-chatbot-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

First run downloads a few small NLTK data packages automatically (punkt, stopwords, wordnet).

## Usage

**Command-line chat:**
```bash
python -m src.chatbot
```

**Web UI:**
```bash
streamlit run app.py
```

**Train/inspect the classifier alone:**
```bash
python -m src.intent_classifier
```

**Train/inspect the language model alone:**
```bash
python -m src.ngram_model
```

**Run evaluation (classifier accuracy/F1/confusion matrix + LM perplexity):**
```bash
python -m src.evaluate
```
Confusion matrix image is saved to `results/confusion_matrix.png`.

## Project layout

```
nlp-chatbot-project/
├── app.py                  # Streamlit chat UI
├── data/
│   ├── intents.json        # intents: patterns + template responses
│   └── corpus.txt          # freeform text for the n-gram LM
├── src/
│   ├── preprocess.py       # tokenization, stopwords, lemmatization
│   ├── intent_classifier.py# TF-IDF + Naive Bayes intent classifier
│   ├── ngram_model.py      # n-gram LM with smoothing, perplexity, generation
│   ├── chatbot.py           # ties everything together
│   └── evaluate.py          # accuracy/F1/confusion matrix + perplexity
└── requirements.txt
```

## Extending it (if you have extra time / want a stronger report)

- Add **Kneser-Ney or interpolated smoothing** to `ngram_model.py` and compare perplexity against add-k.
- Swap Naive Bayes for an **SVM** or **logistic regression** and compare accuracy.
- Add more intents/patterns — the classifier is only as good as `data/intents.json`.
- Log misclassified examples from `evaluate.py` for an error-analysis section in your report.
- Add a simple **spell-correction** step before classification (edit distance) as another classical-NLP component.

## Notes

- The dataset in `data/intents.json` is small and hand-written — for a real report, consider
  expanding each intent to 15-20 varied patterns for a more convincing classifier evaluation.
- The n-gram model's generated fallback text is intentionally rough — it's meant to demonstrate
  statistical language modeling, not to be fluent. Say so plainly in your report; it's the honest
  tradeoff of not using a pretrained model.
