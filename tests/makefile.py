.PHONY: setup train api test clean

setup:
	poetry install
	mkdir -p data models logs
	poetry run python src/data/generator.py

train:
	poetry run python -c "from src.models.trainer import FraudModel; \
	import pandas as pd; \
	df = pd.read_csv('data/transactions.csv'); \
	model = FraudModel(); \
	model.train(df)"

api:
	poetry run uvicorn src.api.main:app --reload --port 8000

test:
	poetry run pytest tests/ -v

docker-build:
	docker build -t fraud-detector:latest .

docker-run:
	docker run -p 8000:8000 fraud-detector:latest

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type f -name "*.pyc" -delete