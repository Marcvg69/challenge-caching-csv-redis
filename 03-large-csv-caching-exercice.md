# Challenge: Caching CSV Data Processing with Redis in Python  

- **Repository:** `challenge-caching-csv-redis`  
- **Type of Challenge:** Consolidation  
- **Duration:** 1 day  
- **Deadline:** 21/08/2025 - 5:00 PM  
- **Team Challenge:** Solo  


## 🎯 Mission Objectives
Practice **caching techniques** with **Redis** by optimizing expensive data transformations on a large CSV dataset.  


## 📚 Learning Objectives
- Work with a real-world dataset in Python.  
- Implement caching of computed results in Redis.  
- Use cache expiration (TTL) to simulate fresh data ingestion.  
- Compare performance with and without caching.  
- Understand real-world applications of caching in data pipelines.  


## 📝 The Mission
You are a **data engineer** tasked with analyzing the **Airline On-Time Performance dataset** (flight delays in the US).  

Your team frequently needs aggregated statistics, such as **average delay per airline** or **number of flights per airport**. However, computing these directly from the CSV each time is slow.  

Your task is to build a **Python app with Redis caching** that:  
- Reads the CSV dataset.  
- Performs **expensive aggregations** (e.g., average arrival delay per airline).  
- Caches results in Redis.  
- Returns cached results instantly if the same query is requested again.  
- Applies a **TTL (time-to-live)** so cached results refresh after a set time.  
- Logs whether results came from Redis or CSV.  


## ✅ Must-Have Features
- Python script that:  
  - Loads and processes the Airline On-Time dataset.  
  - Computes an aggregation (e.g., average delay per airline, total flights per airport).  
  - Caches results in Redis with a TTL (e.g., 60 seconds).  
  - Returns cached data if available.  
  - Logs whether the result came from Redis or CSV.  
- Demonstrate performance difference (first run slower, subsequent runs faster).  


## 🌟 Nice-to-Have Features
- Support **multiple queries** (e.g., by airline, by airport, by month).  
- Store results as **Redis hashes** for structured data.  
- Add a function to **clear or invalidate cache**.  
- Track **cache hits vs misses** in Redis.  
- Dockerize the app and Redis with `docker-compose`.  


## ⚙️ Constraints
- Use **pandas** for CSV processing.  
- Use **Redis** (local install or Docker).  
- Cache TTL must be configurable (via `.env` file or config).  
- Use **Python logging** to clearly indicate cache vs CSV results.  


## 📂 Repository Structure

- `README.md` → Project documentation (description, usage, screenshots)  
- `app/` → Application code  
  - `main.py` → Entry point script with caching + CSV logic  
  - `cache.py` → Helper functions for Redis interactions  
- `docker-compose.yml` → (Optional) run Redis + app easily  


## 📊 Dataset
For this challenge, use the **Airline On-Time Performance dataset**:  

- **Source:** [Kaggle – Airline Delay Dataset](https://www.kaggle.com/datasets/usdot/flight-delays)  
- **Size:** ~5M+ rows, includes flight times, delays, airlines, and airports.  
- **Suggested Columns for Aggregations:**  
  - `OP_CARRIER` → Airline  
  - `ORIGIN`, `DEST` → Airports  
  - `DEP_DELAY`, `ARR_DELAY` → Delays  
  - `FL_DATE` → Date  

👉 Example queries:  
- Average **arrival delay per airline**.  
- Total **flights per origin airport**.  
- Average **departure delay per month**.  


## 📦 Deliverables
- Public GitHub repo with working code.  
- README must include:  
  - How to install & run Redis.  
  - How to run the app.  
  - Example logs showing cache vs CSV computation.  
  - Performance notes (e.g., “First run took 4.8s, second run took 0.03s”).  


## 🧮 Evaluation Criteria

| Criteria    | Indicator |
|-------------|-----------|
| **Complete** | Script reads CSV, processes, caches results with TTL, logs cache vs CSV source. |
| **Correct** | Cache behaves as expected (first query slow, subsequent queries fast, TTL works). |
| **Great**   | Repo is clean and documented, extra features (multi-query, analytics, Docker) implemented. |  