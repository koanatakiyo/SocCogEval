PYTHON ?= python3
.PHONY: test reproduce paper
test:
	$(PYTHON) -m pytest -q
reproduce:
	$(PYTHON) scripts/reproduce_paper.py
paper:
	$(MAKE) -C paper
