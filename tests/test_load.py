import requests
import time
import random

url = "http://localhost:8000/predict"

print("Running 100 predictions...")
times = []

for i in range(100):
    data = {
        "amount": random.uniform(10, 1000),
        "merchant_risk_score": random.random(),
        "days_since_last_transaction": random.uniform(0, 30),
        "hour_of_day": random.randint(0, 23),
        "is_weekend": random.choice([0, 1]),
        "num_transactions_today": random.randint(1, 20),
        "location_risk": random.random()
    }
    
    start = time.time()
    response = requests.post(url, json=data)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    
    if i % 20 == 0:
        print(f"Completed {i} requests...")

avg_time = sum(times) / len(times)
p95_time = sorted(times)[95]

print(f"\n📊 Performance Results:")
print(f"Average latency: {avg_time:.2f}ms")
print(f"P95 latency: {p95_time:.2f}ms")
print(f"Throughput: {1000/avg_time:.1f} requests/second")
