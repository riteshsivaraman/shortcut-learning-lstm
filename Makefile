.PHONY: test test-models test-trigger lint install

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-models:
	pytest tests/test_models.py -v

test-trigger:
	pytest tests/test_trigger.py -v