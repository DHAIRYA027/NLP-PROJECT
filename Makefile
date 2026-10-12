PYTHON ?= venv/bin/python

.PHONY: install test lint eval chat app clean

install:
	python3 -m venv venv
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

eval:
	$(PYTHON) -m src.evaluate

chat:
	$(PYTHON) -m src.chatbot --debug

app:
	$(PYTHON) -m streamlit run app.py

clean:
	rm -rf .pytest_cache .ruff_cache data/intent_model.joblib
	find . -name __pycache__ -not -path "./venv/*" -prune -exec rm -rf {} +
