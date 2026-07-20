#!/usr/bin/env python3
"""Simple manual load test against a locally running fraud-detection API.

This is a script, not a pytest test: it makes live HTTP calls and needs the
API server up at http://localhost:8000. Run it explicitly:

    poetry run python scripts/load_test.py

It previously lived at tests/test_load.py, where pytest would try to collect
it as a test module and execute this top-level code (including live network
calls) during collection, breaking `pytest` when no server was running.
"""
import random
import time

import requests

URL = "http://localhost:8000/predict"
NUM_REQUESTS = 100


def main():
    print(f"Running {NUM_REQUESTS} predictions...")
    times = []

    for i in range(NUM_REQUESTS):
        data = {
            "amount": random.uniform(10, 1000),
            "merchant_risk_score": random.random(),
            "days_since_last_transaction": random.uniform(0, 30),
            "hour_of_day": random.randint(0, 23),
            "is_weekend": random.choice([0, 1]),
            "num_transactions_today": random.randint(1, 20),
            "location_risk": random.random(),
        }

        start = time.time()
        response = requests.post(URL, json=data)
        response.raise_for_status()
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        if i % 20 == 0:
            print(f"Completed {i} requests...")

    avg_time = sum(times) / len(times)
    p95_time = sorted(times)[int(len(times) * 0.95) - 1]

    print("\nPerformance results:")
    print(f"Average latency: {avg_time:.2f}ms")
    print(f"P95 latency: {p95_time:.2f}ms")
    print(f"Throughput: {1000 / avg_time:.1f} requests/second")


if __name__ == "__main__":
    main()
