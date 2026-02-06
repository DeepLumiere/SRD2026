.PHONY: help install test lint clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install the package and dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run code linting"
	@echo "  make clean      - Clean build artifacts"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

lint:
	flake8 src/ scripts/
	black --check src/ scripts/

format:
	black src/ scripts/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
