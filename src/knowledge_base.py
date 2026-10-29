import json
import math
import re
from collections import Counter
from pathlib import Path

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "i", "me", "my", "mine", "you", "your", "we", "our", "us", "they", "them", "their",
    "he", "she", "it", "its", "this", "that", "these", "those", "there", "here",
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "to", "of", "for", "in", "on", "at", "by", "with", "from", "into", "about", "as",
    "and", "or", "but", "if", "so", "than", "then", "also", "just", "very", "too",
    "can", "could", "will", "would", "should", "shall", "may", "might", "must",
    "have", "has", "had", "get", "got", "tell", "give", "show", "know", "want", "need",
    "please", "kindly", "any", "some", "more", "much", "many", "all", "again",
    "am", "im", "ive", "u", "ur", "pls", "plz",
    "come", "comes", "coming", "go", "goes", "going", "back", "let", "lets", "make",
    "take", "find", "see", "check", "look", "college", "university", "like",
}

SYNONYMS = {
    "dorm": "hostel", "dorms": "hostel", "dormitory": "hostel", "dormitories": "hostel",
    "accommodation": "hostel", "residence": "hostel", "lodging": "hostel", "hostal": "hostel",
    "tuition": "fee", "charge": "fee", "charges": "fee", "cost": "fee", "costs": "fee",
    "price": "fee", "pricing": "fee", "expense": "fee", "expenses": "fee",
    "timing": "hour", "timings": "hour", "hours": "hour", "opening": "hour",
    "examination": "exam", "examinations": "exam", "test": "exam", "tests": "exam",
    "job": "placement", "jobs": "placement", "recruitment": "placement",
    "recruit": "placement", "recruiter": "placement", "recruiters": "placement",
    "hiring": "placement", "hire": "placement",
    "cafeteria": "canteen", "mess": "canteen", "food": "canteen", "dining": "canteen",
    "eat": "canteen", "lunch": "canteen", "snack": "canteen", "snacks": "canteen",
    "internet": "wifi", "network": "wifi", "broadband": "wifi",
    "shuttle": "transport", "bus": "transport", "buses": "transport",
    "transportation": "transport", "commute": "transport", "cab": "transport",
    "doctor": "medical", "clinic": "medical", "health": "medical",
    "ambulance": "medical", "hospital": "medical", "dispensary": "medical",
    "gym": "sport", "sports": "sport", "playground": "sport", "games": "sport",
    "ground": "sport",
    "society": "club", "societies": "club", "activities": "club", "activity": "club",
    "grant": "scholarship", "aid": "scholarship", "bursary": "scholarship",
    "reimbursement": "refund",
    "gpa": "cgpa", "percentage": "cgpa",
    "mark": "result", "marks": "result", "grade": "result", "grades": "result",
    "scorecard": "result", "score": "result", "scores": "result",
    "backlog": "backlog", "supplementary": "backlog", "reappear": "backlog",
    "arrear": "backlog", "arrears": "backlog", "fail": "backlog", "failed": "backlog",
    "failing": "backlog",
    "recheck": "revaluation", "reevaluation": "revaluation",
    "guest": "visitor", "guests": "visitor", "parent": "visitor", "parents": "visitor",
    "phone": "contact", "call": "contact", "email": "contact", "mail": "contact",
    "application": "apply", "applications": "apply", "register": "apply",
    "registration": "apply",
    "procedure": "process", "steps": "process", "step": "process",
    "deadline": "date", "due": "date", "schedule": "date", "timetable": "date",
    "datesheet": "date",
    "fine": "penalty", "fines": "penalty", "overdue": "penalty", "late": "penalty",
    "issu": "borrow", "issue": "borrow", "issued": "borrow", "lend": "borrow",
    "loan": "borrow",
    "ebook": "digital", "ebooks": "digital", "eresource": "digital",
    "eresources": "digital", "journal": "digital", "journals": "digital",
    "ieee": "digital",
    "identity": "id",
    "cutoff": "eligibility", "eligible": "eligibility", "qualify": "eligibility",
    "leave": "withdraw", "quit": "withdraw", "cancel": "withdraw",
    "discontinue": "withdraw", "withdrawal": "withdraw",
    "replacement": "duplicate",
    "ticket": "admit", "hallticket": "admit",
    "wifi": "wifi",
    "hey": "hello", "hi": "hello", "hii": "hello", "hiya": "hello", "namaste": "hello",
    "thanks": "thank", "thankyou": "thank", "thx": "thank", "ty": "thank",
    "goodbye": "bye", "cya": "bye",
}

TEXT_REPLACEMENTS = (
    ("cut off", "cutoff"),
    ("cut-off", "cutoff"),
    ("wi-fi", "wifi"),
    ("wi fi", "wifi"),
    ("e-book", "ebook"),
    ("e-resource", "eresource"),
    ("re-evaluation", "revaluation"),
    ("re evaluation", "revaluation"),
    ("hall ticket", "admit"),
    ("campus drive", "placement"),
    ("thank you", "thank"),
    ("money back", "refund"),
    ("good morning", "hello"),
    ("good afternoon", "hello"),
    ("good evening", "hello"),
    ("see you", "bye"),
    ("talk to you later", "bye"),
)


def raw_tokens(text):
    text = text.lower().replace("'s", " ").replace("\u2019s", " ")
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    return re.findall(r"[a-z0-9]+", text)


def stem(word):
    if len(word) <= 3:
        return word
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize(token):
    if token in SYNONYMS:
        return SYNONYMS[token]
    stemmed = stem(token)
    return SYNONYMS.get(stemmed, stemmed)


def analyze(text):
    return [normalize(token) for token in raw_tokens(text) if token not in STOPWORDS]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


class KnowledgeBase:
    def __init__(self, entries):
        self.entries = entries
        self.doc_text = []
        self.doc_tokens = []
        self.doc_entry = []
        vocabulary = Counter()
        question_vocabulary = Counter()
        for index, entry in enumerate(entries):
            texts = list(entry["questions"]) + list(entry.get("keywords", []))
            for text in texts:
                tokens = analyze(text)
                if not tokens:
                    continue
                self.doc_text.append(text)
                self.doc_tokens.append(tokens)
                self.doc_entry.append(index)
                vocabulary.update(raw_tokens(text))
                question_vocabulary.update(raw_tokens(text))
            vocabulary.update(raw_tokens(entry.get("answer", "")))
        vocabulary.update(SYNONYMS.keys())
        vocabulary.update(SYNONYMS.values())
        self.vocab = vocabulary
        self.question_vocab = question_vocabulary
        self.known_stems = {stem(word) for word in vocabulary}
        self.vocab_by_initial = {}
        for word in question_vocabulary:
            if len(word) >= 5 and word not in STOPWORDS:
                self.vocab_by_initial.setdefault(word[0], []).append(word)

        total = len(self.doc_tokens)
        document_frequency = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))
        self.idf = {
            term: math.log((1 + total) / (1 + count)) + 1
            for term, count in document_frequency.items()
        }
        self.max_idf = math.log(1 + total) + 1
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.doc_tokens]

    @classmethod
    def from_json(cls, path):
        with open(Path(path), encoding="utf-8") as handle:
            data = json.load(handle)
        entries = data["entries"] if isinstance(data, dict) else data
        return cls(entries)

    def stats(self):
        return len(self.entries), len(self.doc_tokens), len(self.idf)

    def _vectorize(self, tokens):
        counts = Counter(tokens)
        vector = {}
        unknown = 0.0
        for term, count in counts.items():
            weight = 1 + math.log(count)
            if term in self.idf:
                vector[term] = weight * self.idf[term]
            else:
                unknown += (weight * self.max_idf) ** 2
        norm = math.sqrt(sum(value * value for value in vector.values()) + unknown)
        return vector, norm

    def correct(self, text):
        corrected = []
        changes = []
        for word in raw_tokens(text):
            if (
                word in self.vocab
                or stem(word) in self.known_stems
                or word in STOPWORDS
                or len(word) < 5
                or word.isdigit()
            ):
                corrected.append(word)
                continue
            limit = 1 if len(word) <= 7 else 2
            best = None
            for candidate in self.vocab_by_initial.get(word[0], []):
                if abs(len(candidate) - len(word)) > limit:
                    continue
                distance = edit_distance(word, candidate)
                if distance <= limit:
                    key = (distance, -self.question_vocab[candidate])
                    if best is None or key < best[0]:
                        best = (key, candidate)
            if best:
                corrected.append(best[1])
                changes.append((word, best[1]))
            else:
                corrected.append(word)
        return " ".join(corrected), changes

    def search(self, text, k=3):
        tokens = analyze(text)
        if not tokens:
            return []
        query_vector, query_norm = self._vectorize(tokens)
        if query_norm == 0:
            return []
        best = {}
        for position, (doc_vector, doc_norm) in enumerate(self.doc_vectors):
            dot = sum(
                weight * doc_vector.get(term, 0.0)
                for term, weight in query_vector.items()
            )
            if dot == 0:
                continue
            score = dot / (query_norm * doc_norm)
            entry_index = self.doc_entry[position]
            if entry_index not in best or score > best[entry_index][0]:
                best[entry_index] = (score, position)
        ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)[:k]
        results = []
        for entry_index, (score, position) in ranked:
            entry = self.entries[entry_index]
            results.append(
                {
                    "id": entry["id"],
                    "category": entry.get("category", ""),
                    "answer": entry["answer"],
                    "question": self.doc_text[position],
                    "score": score,
                }
            )
        return results
