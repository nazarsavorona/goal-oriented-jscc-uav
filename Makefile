.PHONY: install train test lint format typecheck clean

install:
	pip install -e ".[dev]"

train:
	python scripts/train.py --model deepjscc --channel awgn --data-dir data/frames

test:
	python -m pytest tests/ -v

lint:
	ruff check src/

format:
	ruff format src/ scripts/ tests/

typecheck:
	mypy src/semvideo/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .mypy_cache .ruff_cache .pytest_cache dist build *.egg-info
