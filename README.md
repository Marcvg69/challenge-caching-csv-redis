# Challenge – Caching Large CSV Queries with Redis

Author: **Marc Van Goolen** (solo project)

---

## 🚀 Overview

This project implements a **caching layer on top of heavy CSV queries** using **Redis**.  
The main goal is to **speed up repeated aggregations** on large airline flight datasets  
(USDOT Flight Delays dataset).  

Queries such as:

- Average **arrival delay per airline**  
- Total **flights per origin airport**  
- Average **departure delay per month**  

are supported. Results are **cached in Redis** so that subsequent runs are **instantaneous**.

---

## ✅ Features

### Must-haves
- [x] Read very large CSVs in **chunked mode** (streaming, not full load in memory).  
- [x] Run aggregations on flight delays and counts.  
- [x] Cache expensive queries in Redis with **TTL**.  
- [x] CLI interface with `--query`, `--limit`, etc.  
- [x] Schema detection for **Kaggle vs BTS** datasets.  

### Nice-to-haves
- [x] **Environment variables** (`.env`) for config (CSV path, Redis host, etc).  
- [x] Pretty JSON output with optional airline/airport enrichment.  
- [x] `--show-cache` flag to inspect what’s stored in Redis.  
- [x] Benchmark script to compare **cache HIT vs MISS**.  
- [x] Dockerized Redis via `docker-compose.yml`.  

---

## 📂 Repository Layout

```
challenge-caching-csv-redis/
│
├── app/
│   ├── main.py        # CLI for running queries
│   ├── cache.py       # Redis caching utilities
│   └── __init__.py
│
├── data/              # Place small test CSVs here if needed
│
├── docker-compose.yml # Redis service
├── Dockerfile         # (optional, for containerised app)
├── run_benchmark.py   # Benchmark HIT vs MISS performance
├── README.md          # This file
└── .env               # Configuration (CSV path, Redis config, TTL, etc.)
```

---

## ⚙️ Environment Setup

### 1. Clone repo & create venv
```bash
git clone https://github.com/yourusername/challenge-caching-csv-redis.git
cd challenge-caching-csv-redis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
Edit `.env` with your CSV paths and Redis config:
```ini
CSV_PATH=/Users/Marc/Datasets/usdot-flight-delays/flights.csv
AIRLINES_CSV=/Users/Marc/Datasets/usdot-flight-delays/airlines.csv
AIRPORTS_CSV=/Users/Marc/Datasets/usdot-flight-delays/airports.csv
REDIS_HOST=localhost
REDIS_PORT=6380
CACHE_TTL_SECONDS=60
CHUNKSIZE=1000000
```

### 3. Start Redis with Docker
```bash
docker compose up -d
```
Redis will run on port **6380**.

---

## ▶️ Running Queries

**Example 1: Average arrival delay per airline**
```bash
python app/main.py --query avg_arr_delay_by_airline --limit=5
```
First run → **CACHE_MISS** (computes and stores in Redis).  
Second run → **CACHE_HIT** (instant retrieval).

**Example 2: Flights per origin airport**
```bash
python app/main.py --query flights_by_origin --limit=5
```

**Example 3: Average departure delay per month**
```bash
python app/main.py --query avg_dep_delay_by_month --limit=5
```

---

## 🔍 Inspecting Cache

See what’s stored in Redis:
```bash
python app/main.py --show-cache --limit=10
```

Or pretty print all entries:
```bash
python app/main.py --show-cache --full
```

---

## 📊 Benchmarking

Run the benchmark script to compare MISS vs HIT:
```bash
python run_benchmark.py
```

Output is a small table comparing timings, also auto-inserts results into README between special markers:

```markdown
<!-- BENCHMARK -->
(benchmark results will be inserted here automatically)
<!-- END -->
```

---

## 🏆 Outcome

This project demonstrates:

- Streaming CSV processing with pandas chunks.  
- Smart caching with Redis to avoid recomputation.  
- CLI interface for reproducible queries.  
- Dockerised environment for easy setup.  

It fulfills all required goals and several nice-to-haves.

---

## 👤 Author

**Marc Van Goolen**  
Data Science Bootcamp, Belgium  
_Solo project_
