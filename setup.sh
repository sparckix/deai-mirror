#!/usr/bin/env bash
# One-time setup: venv + libs + spaCy model + nltk data.
set -e
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -c "import nltk; [nltk.download(x, quiet=True) for x in ['averaged_perceptron_tagger_eng','punkt','punkt_tab']]"
echo "setup done. run:  .venv/bin/python deai.py mirror your_draft.md corpora/ai_and_society"
