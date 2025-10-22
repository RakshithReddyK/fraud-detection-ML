.PHONY: setup train api test clean

setup:
	@echo "Installing dependencies..."
	poetry install
	mkdir -p data models logs
	poetry run python src/data/generator.py

train:
	@echo "Training model..."
	poetry run python scripts/train.py

api:
	@echo "Starting API..."
	poetry run uvicorn src.api.main:app --reload --port 8000

test:
	@echo "Running tests..."
	poetry run pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type f -name "*.pyc" -delete
