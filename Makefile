# Makefile LocalMind OS (этап E0/E1)
PYTHON ?= python3

.PHONY: detect detect-json doctor test lint clean

detect:
	./localmind-detect

detect-json:
	./localmind-detect --json

doctor:
	./localmind-detect lm-doctor

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m py_compile scripts/localmind-detect.py tests/test_detect.py tests/test_detect_v02.py
	$(PYTHON) -c "import json; json.load(open('config/report-schema.json')); print('schema: valid JSON')"

clean:
	rm -rf __pycache__ tests/__pycache__ scripts/__pycache__
