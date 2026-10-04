"""Benchmark one local Uvicorn process over real loopback HTTP, without paid inference."""

import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "REDIS_URL": "", "MODELS_DIR": str(ROOT / "models")}
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-access-log",
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    payload = {
        "amount": 150.0,
        "merchant_risk_score": 0.3,
        "days_since_last_transaction": 1.0,
        "hour_of_day": 14,
        "is_weekend": 0,
        "num_transactions_today": 3,
        "location_risk": 0.2,
    }
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=20, trust_env=False
        ) as client:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("API process exited before readiness")
                try:
                    if client.get("/ready").status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                time.sleep(0.05)
            else:
                raise RuntimeError("API did not become ready")
            results = {}
            for path in ["/predict"]:
                for _ in range(5):
                    client.post(path, json=payload).raise_for_status()
                samples = []
                for _ in range(100):
                    start = time.perf_counter()
                    response = client.post(path, json=payload)
                    response.raise_for_status()
                    samples.append((time.perf_counter() - start) * 1000)
                samples.sort()
                results[path] = {
                    "requests": len(samples),
                    "p50_ms": statistics.median(samples),
                    "p95_ms": samples[math.ceil(0.95 * len(samples)) - 1],
                    "max_ms": max(samples),
                }
            report = {
                "transport": "loopback HTTP, persistent client, one Uvicorn worker",
                "concurrency": 1,
                "warmup_per_endpoint": 5,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "provider_calls": 0,
                "provider_cost_usd": 0,
                "model": client.get("/ready").json(),
                "results": results,
                "scope": (
                    "Synthetic trained model, repeated transaction, cache disabled; "
                    "not a production load benchmark"
                ),
            }
            path = ROOT / "reports/latency.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
