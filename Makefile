.PHONY: setup train api test lint docker-build docker-run clean

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

lint:
	@echo "Linting..."
	poetry run ruff check .
	poetry run black --check .

docker-build:
	docker build -t fraud-detector:latest .

docker-run:
	docker run -p 8000:8000 fraud-detector:latest

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type f -name "*.pyc" -delete
