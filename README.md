# Challenge: Caching CSV with Redis

This project demonstrates how to use **Redis** for caching CSV data in Python.  
It includes utilities for loading, querying, and benchmarking cache performance.

## Features

- Load CSV data into Redis.
- Query cached data efficiently.
- Benchmark dictionary lookup HIT vs MISS times.
- Auto-update this README with fresh benchmark results.

## Usage

1. Create and activate your virtual environment:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Run the benchmark:
   python run_benchmark.py

   Or update this README automatically:
   python run_benchmark.py --update-readme

## Benchmark

Below is the latest benchmark (auto-generated):

<!-- BENCHMARK:START -->

**Dictionary lookup benchmark** — iters=100,000, miss_ratio=0.50

| Operation | n | Avg (ns) | Median (ns) | Std (ns) |
|---|---:|---:|---:|---:|
| HIT | 50,214 | 143.5 | 125.0 | 54.9 |
| MISS | 49,786 | 131.4 | 125.0 | 80.1 |

<!-- BENCHMARK:END -->

---
💡 The benchmark compares **HITs** (keys found in the cache) vs **MISSes** (keys not found).  
Times are measured in nanoseconds using Python’s `time.perf_counter_ns`.
