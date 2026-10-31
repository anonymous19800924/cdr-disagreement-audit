PY ?= python
.PHONY: all repro check consensus agreement sensitivity theory symmetry

all: repro check

repro: consensus agreement sensitivity theory symmetry
	@echo "Reproduction complete: audit, human agreement, held-out/sensitivity, theory, symmetry."

check:
	$(PY) scripts/check_release.py

consensus:
	$(PY) scripts/verify_consensus.py

agreement:
	$(PY) scripts/verify_agreement_v2.py --a1 data/anchor_annotator1.csv --a2 data/anchor_annotator2.csv --key data/anchor_key.jsonl

sensitivity:
	$(PY) scripts/sensitivity.py

theory:
	$(PY) scripts/verify_theory.py

symmetry:
	$(PY) scripts/verify_symmetry.py
