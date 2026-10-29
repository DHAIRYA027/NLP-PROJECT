import json
from pathlib import Path

from src.chatbot import Chatbot
from src.knowledge_base import KnowledgeBase, analyze

ROOT = Path(__file__).resolve().parent.parent

STRONG_MATCH = 0.50
TENTATIVE_MATCH = 0.30
SUGGEST_MATCH = 0.15

DEFAULT_SUGGESTIONS = [
    "How do I apply for admission?",
    "What is the fee structure?",
    "What are the library timings?",
    "Tell me about hostel facilities",
    "When are the exams?",
    "Tell me about placements",
]


def load_templates(path):
    templates = set()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return templates
    intents = data.get("intents", []) if isinstance(data, dict) else data
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        for response in intent.get("responses", []):
            if isinstance(response, str):
                templates.add(response.strip())
    return templates


class SmartBot:
    def __init__(self):
        self.base = Chatbot()
        self.kb = KnowledgeBase.from_json(ROOT / "data" / "faq.json")
        self.templates = load_templates(ROOT / "data" / "intents.json")

    def _classify(self, text):
        try:
            tag, confidence = self.base.classifier.predict(text)
            return tag, float(confidence)
        except Exception:
            return None, 0.0

    def _suggestions(self, hits, limit=3):
        picked = []
        for hit in hits:
            if hit["score"] >= SUGGEST_MATCH and hit["category"] != "Chat":
                picked.append(hit["question"])
        if not picked:
            return DEFAULT_SUGGESTIONS[:limit]
        return picked[:limit]

    def ask(self, text, allow_generative=False):
        cleaned = text.strip()
        trace = {
            "query": cleaned,
            "corrected": cleaned,
            "changes": [],
            "tokens": [],
            "hits": [],
            "intent": None,
            "intent_confidence": 0.0,
        }
        if not cleaned:
            return {
                "reply": "Type a question and I'll do my best to help.",
                "source": "fallback",
                "label": "Empty",
                "intent": None,
                "confidence": 0.0,
                "confidence_label": "",
                "suggestions": DEFAULT_SUGGESTIONS[:3],
                "trace": trace,
            }

        corrected, changes = self.kb.correct(cleaned)
        hits = self.kb.search(corrected, k=3)
        intent, intent_confidence = self._classify(corrected)
        trace.update(
            corrected=corrected,
            changes=changes,
            tokens=analyze(corrected),
            hits=[
                {"question": h["question"], "score": h["score"], "category": h["category"]}
                for h in hits
            ],
            intent=intent,
            intent_confidence=intent_confidence,
        )
        best = hits[0] if hits else None

        if best and best["score"] >= STRONG_MATCH:
            return {
                "reply": best["answer"],
                "source": "knowledge",
                "label": best["category"],
                "intent": intent,
                "confidence": best["score"],
                "confidence_label": "match",
                "suggestions": [],
                "trace": trace,
            }

        base_reply = ""
        try:
            base_reply = self.base.respond(corrected)
        except Exception:
            base_reply = ""

        is_template = base_reply.strip() in self.templates if self.templates else intent_confidence >= 0.5
        if base_reply and is_template:
            return {
                "reply": base_reply,
                "source": "template",
                "label": intent or "intent",
                "intent": intent,
                "confidence": intent_confidence,
                "confidence_label": "confidence",
                "suggestions": [],
                "trace": trace,
            }

        if best and best["score"] >= TENTATIVE_MATCH:
            return {
                "reply": "I'm not completely sure, but this might be what you're looking for:\n\n"
                + best["answer"],
                "source": "tentative",
                "label": best["category"],
                "intent": intent,
                "confidence": best["score"],
                "confidence_label": "match",
                "suggestions": self._suggestions(hits[1:] if len(hits) > 1 else hits),
                "trace": trace,
            }

        if allow_generative and base_reply:
            return {
                "reply": base_reply,
                "source": "generative",
                "label": "n-gram model",
                "intent": intent,
                "confidence": intent_confidence,
                "confidence_label": "confidence",
                "suggestions": self._suggestions(hits),
                "trace": trace,
            }

        return {
            "reply": "I couldn't find a confident answer to that. I can help with admissions, "
            "fees, the library, hostel, exams, placements and campus life. "
            "Try rephrasing, or pick one of these:",
            "source": "fallback",
            "label": "Not sure",
            "intent": intent,
            "confidence": best["score"] if best else 0.0,
            "confidence_label": "match",
            "suggestions": self._suggestions(hits),
            "trace": trace,
        }
