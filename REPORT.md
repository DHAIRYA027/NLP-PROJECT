# Campus Helpdesk Chatbot — Evaluation Report

This report documents the design choices, evaluation methodology, and results
for the intent classifier and the n-gram language model, and discusses where
each one fails and why. All numbers below come directly from running
`python -m src.evaluate` (3-fold stratified cross-validation for the
classifiers, an 85/15 train/test split of `data/corpus.txt` for the language
model) — see `results/metrics.json` for the raw numbers and
`results/confusion_matrix.png` for the full confusion matrix.

## 1. Dataset

14 intents, 174 total training patterns (~12 patterns/intent on average).
Small by NLP-industry standards, but large enough to expose real classifier
behavior rather than trivially separable toy classes — several intents share
vocabulary (`fees` / `hostel` / `admissions` all mention money and deadlines,
`wifi_it_support` / `contact` both mention "who do I contact"), which is what
makes the comparison below meaningful instead of every model scoring 100%.

## 2. Intent classification: three models compared

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Multinomial Naive Bayes (baseline) | 0.741 | 0.742 | 0.738 |
| Logistic Regression | **0.782** | **0.792** | **0.786** |
| Linear SVM (calibrated) | 0.776 | 0.777 | 0.774 |

Logistic Regression edges out both Naive Bayes and the SVM here. This tracks
with the standard bias/variance story: Naive Bayes' independence assumption
between TF-IDF features hurts it when intents share vocabulary (exactly the
overlap built into this dataset), while a discriminative linear model can
weigh those shared words down. The gap isn't huge (~4 points of accuracy),
which is expected given how small the dataset is — with 174 examples across
14 classes, no model has much signal to work with.

### Error analysis

The confusion matrix (Naive Bayes) shows one dominant failure mode: **`greeting`
acts as a dumping ground.** Of all misclassifications, the largest single
block is `goodbye → greeting` (9 of 15 `goodbye` examples), followed by
`wifi_it_support → greeting` (5 of 10) and smaller leaks from `thanks`,
`complaint`, `contact`, `exam_schedule`, `placement`, and `transport` — all
into `greeting`.

Why: `greeting` has the most patterns (18) and the most common, low-IDF words
("hi", "hey", "what's up"), so under TF-IDF weighting it has the broadest,
lowest-magnitude decision boundary — anything the model is *unsure* about
tends to fall into it by default. This is a direct, reproducible illustration
of a known weakness of Naive Bayes with sparse, short-text TF-IDF features:
classes with generic vocabulary become an attractor for uncertain predictions.
Logistic Regression and the SVM both reduce (but don't eliminate) this,
because they can push down the weight of individual common tokens instead of
just counting frequencies.

**What would fix this in a real system:** more patterns per intent (especially
more `goodbye` and `wifi_it_support` examples that don't overlap lexically
with greetings), and/or word embeddings instead of TF-IDF so the model can
use semantic similarity rather than raw token overlap.

## 3. Language model: add-k vs. linear interpolation

| Order (n) | Add-k (Laplace) | Linear interpolation |
|---|---|---|
| Unigram (n=1) | 401.77 | 401.77 *(identical — no lower order to back off to)* |
| Bigram (n=2) | 480.96 | **322.77** |
| Trigram (n=3) | 559.16 | **261.07** |

The add-k model's perplexity **gets worse** as n increases — the classic
symptom of add-k over-penalizing unseen n-grams as the context grows (the `+k`
term dominates when most trigram contexts in a ~2,000-word corpus were seen
zero or one times). Linear interpolation (mixing unigram, bigram, and trigram
MLE estimates with weights `0.1 / 0.3 / 0.6`) fixes this by falling back to
the more reliable lower-order estimate whenever the higher-order context is
sparse, and perplexity correctly **improves** with n instead.

This is the more interesting and more honest finding to report than "our
model gets X perplexity" in isolation — it shows *why* smoothing choice
matters, not just that a formula was implemented.

**Caveat to state plainly in class:** ~2,000 words is still small for a
language model. The generated fallback text (see `src/ngram_model.py`
`__main__` block) is grammatical in short bursts but wanders — that's
expected and is the honest tradeoff of a from-scratch statistical LM instead
of a pretrained one, not a bug.

## 4. Reproducing these results

```
python -m src.evaluate
```

regenerates `results/metrics.json` and `results/confusion_matrix.png` from
scratch. Numbers will vary slightly run to run (the interpolation weights are
fixed, but `StratifiedKFold(shuffle=True, random_state=42)` is seeded, so
classifier numbers should be stable; try different `random_state` values to
see the effect of the small dataset size on variance).
